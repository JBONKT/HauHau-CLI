"""Agent orchestration: system prompt, tool loop, context compression.

The loop is a plain OpenAI-style tool-call cycle. Two model-specific guards:

* after ``max_steps`` the tools are withheld for one final call, so the turn
  always terminates in prose rather than a dangling tool call;
* context trimming only cuts at a ``user`` boundary, never between an assistant
  ``tool_calls`` message and its ``tool`` result (that would invalidate the
  template).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .llm import Completion, LlamaClient, LLMError, Usage
from .tools import Registry, ToolResult, fail
from .util import estimate_messages_tokens, estimate_tokens


class UI:
    """No-op UI sink; the CLI supplies a rich implementation."""

    def token(self, text: str) -> None: ...
    def reasoning(self, text: str) -> None: ...
    def tool_call(self, name: str, args: str) -> None: ...
    def tool_result(self, name: str, result: ToolResult) -> None: ...
    def status(self, text: str) -> None: ...
    def newline(self) -> None: ...
    def turn_start(self, label: str = "thinking…") -> None: ...
    def turn_end(self, usage: Usage | None = None, elapsed: float = 0.0) -> None: ...
    def interrupt(self) -> None: ...


class Agent:
    def __init__(
        self,
        cfg,
        client: LlamaClient,
        registry: Registry,
        brain=None,
        audit=None,
        ui: UI | None = None,
        on_confirm=None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.registry = registry
        self.brain = brain
        self.audit = audit
        self.ui = ui or UI()
        self.on_confirm = on_confirm
        self.messages: list[dict[str, Any]] = []
        self.summary = ""
        self.last_completion: Completion | None = None
        self._turn_usage = Usage()

    # --- prompt --------------------------------------------------------
    def system_prompt(self, query: str = "") -> str:
        a = self.cfg.agent
        parts = [self._personality(a.personality, a.name)]
        parts.append(f"Working directory: {self.cfg.workspace}")
        parts.append(f"Today: {time.strftime('%Y-%m-%d %H:%M %Z')}")
        parts.append(
            "You can call tools. Prefer acting over describing: if the user asks about files, "
            "the system, or code, inspect it with a tool before answering. Tool results are "
            "authoritative. If a command fails, read the error and adapt - do not repeat the same "
            "call. Keep final answers to a few lines unless asked for detail."
        )
        if self.brain is not None and self.cfg.brain.enabled:
            block = self.brain.context_block(query or self._last_user_text())
            if block:
                parts.append(block)
        if self.summary:
            parts.append(f"Summary of the earlier conversation in this session:\n{self.summary}")
        return "\n\n".join(parts)

    @staticmethod
    def _personality(template: str, name: str) -> str:
        """Fill ``{name}`` without dying on any other brace.

        A hand-edited personality containing "{topic}" raised KeyError out of
        every single turn; the text is still usable, so keep it.
        """
        try:
            return template.format(name=name)
        except (KeyError, IndexError, ValueError):
            return template.replace("{name}", name)

    def _last_user_text(self) -> str:
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    def _request(self, query: str, with_tools: bool) -> tuple[list[dict], list[dict] | None]:
        messages = [{"role": "system", "content": self.system_prompt(query)}, *self.messages]
        return messages, (self.registry.schemas() if with_tools else None)

    # --- turn ----------------------------------------------------------
    def run_turn(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self.compress_if_needed()
        self._turn_usage = Usage()
        started = time.monotonic()
        self.ui.turn_start()
        try:
            return self._run_steps(user_text)
        finally:
            self.ui.turn_end(self._turn_usage, time.monotonic() - started)

    def _run_steps(self, user_text: str) -> str:
        for _ in range(self.cfg.agent.max_steps):
            completion = self._chat(user_text, with_tools=True, stream=True)
            if completion.tool_calls:
                self._record_assistant(completion)
                # A call cut off at the token limit has incomplete arguments;
                # running it would mean guessing the rest. Make the model redo it.
                truncated = completion.finish_reason == "length"
                for call in completion.tool_calls:
                    self._execute(call, truncated=truncated)
                continue
            text = completion.content.strip()
            self.messages.append({"role": "assistant", "content": text})
            self._warn_if_truncated(completion)
            return text

        self.ui.status(f"step budget ({self.cfg.agent.max_steps}) reached - answering without tools")
        self.ui.newline()
        completion = self._chat(user_text, with_tools=False, stream=True)
        text = completion.content.strip()
        self.messages.append({"role": "assistant", "content": text})
        self._warn_if_truncated(completion)
        return text

    def _warn_if_truncated(self, completion: Completion) -> None:
        """Never let a cut-off answer look complete."""
        if completion.finish_reason == "length":
            self.ui.status(
                f"answer hit the {self.cfg.generation.max_tokens}-token cap and was cut off; "
                "raise [generation] max_tokens to finish it"
            )

    def _chat(self, query: str, with_tools: bool, stream: bool) -> Completion:
        messages, tools = self._request(query, with_tools)
        g = self.cfg.generation
        completion = self.client.chat(
            messages,
            tools,
            stream=stream,
            think=g.think,
            temperature=g.temperature,
            top_p=g.top_p,
            max_tokens=g.max_tokens,
            on_token=self.ui.token,
            on_reasoning=self.ui.reasoning,
        )
        self.last_completion = completion
        self._turn_usage.prompt_tokens += completion.usage.prompt_tokens
        self._turn_usage.completion_tokens += completion.usage.completion_tokens
        if completion.usage.tok_per_second:
            self._turn_usage.tok_per_second = completion.usage.tok_per_second
        if completion.usage.prompt_per_second:
            self._turn_usage.prompt_per_second = completion.usage.prompt_per_second
        if completion.reasoning and not completion.content and not completion.tool_calls:
            self.ui.status("the model spent its whole budget thinking; raise max_tokens or disable think")
        return completion

    def _record_assistant(self, completion: Completion) -> None:
        message: dict[str, Any] = {"role": "assistant", "content": completion.content or ""}
        if completion.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in completion.tool_calls
            ]
        self.messages.append(message)

    def _execute(self, call, truncated: bool = False) -> None:
        self.ui.tool_call(call.name, call.arguments)
        tool = self.registry.get(call.name)
        if truncated:
            result = fail(
                f"{call.name}: this call was cut off at the {self.cfg.generation.max_tokens}-token "
                "limit, so its arguments are incomplete and it was NOT run. Retry with less content "
                "per call (for example write_file in several smaller writes)."
            )
        elif tool is None:
            result = fail(f"unknown tool {call.name!r}; available: {', '.join(self.registry.names())}")
        else:
            try:
                args = call.parsed()
            except LLMError as exc:
                result = fail(str(exc))
            else:
                try:
                    result = tool.run(args)
                except Exception as exc:  # a tool crash must not kill the turn
                    result = fail(f"{call.name} crashed: {exc.__class__.__name__}: {exc}")
                if not isinstance(result, ToolResult):
                    result = fail(f"{call.name} returned {type(result).__name__}, not a ToolResult")
        content = result.clipped(self.cfg.tools.max_output_chars)
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
        )
        self.ui.tool_result(call.name, result)
        if self.audit:
            self.audit.log("tool", name=call.name, ok=result.ok, args=call.arguments[:400])

    # --- context management --------------------------------------------
    def summary_cap_tokens(self) -> int:
        """How large the running summary may get, in tokens.

        The summary is re-injected on every call, so an uncapped one crowds out
        the conversation: each compression append grew it until the prompt was
        over budget no matter how many messages were trimmed away.
        """
        return max(120, self.cfg.agent.context_budget_tokens // 3)

    def compress_if_needed(self, force: bool = False) -> bool:
        a = self.cfg.agent
        budget = a.context_budget_tokens
        used = estimate_messages_tokens(self.messages) + estimate_tokens(self.summary)
        # Fold an oversized summary first: trimming messages cannot help when
        # the summary alone is over budget, and that was a permanent dead end.
        folded = self._cap_summary()
        if not force and used <= budget:
            return folded
        cut = self._safe_split(a.keep_recent_turns * 2)
        if cut <= 0:
            return folded
        dropped, self.messages = self.messages[:cut], self.messages[cut:]
        self.summary = self._summarise(dropped) or self.summary
        self._cap_summary()
        return True

    def _cap_summary(self) -> bool:
        """Compress the summary into itself when it exceeds the cap."""
        cap = self.summary_cap_tokens()
        if estimate_tokens(self.summary) <= cap:
            return False
        folded = self._fold_summary(self.summary, cap)
        if folded == self.summary:
            return False
        self.summary = folded
        return True

    def _fold_summary(self, text: str, cap: int) -> str:
        """Return a shorter version of ``text``. Always makes progress."""
        prompt = (
            "Merge these notes into at most 8 dense bullet points, oldest first. Keep names, paths, "
            "commands and decisions. Do not add anything new.\n\n" + text
        )
        folded = ""
        try:
            completion = self.client.chat(
                [{"role": "user", "content": prompt}],
                None,
                think=False,
                max_tokens=max(120, min(400, cap)),
                temperature=0.2,
            )
        except LLMError:
            folded = ""
        else:
            folded = completion.content.strip()
        if folded and estimate_tokens(folded) < estimate_tokens(text):
            return folded
        # No model, a refusal, or a reply that grew: keep the newest notes.
        marker = "…earlier notes dropped…\n"
        room = max(80, cap * 4 - len(marker))
        return marker + text[-room:].lstrip()

    def _safe_split(self, keep: int) -> int:
        """Index at which it is safe to cut the transcript.

        A turn always starts with a ``user`` message, so cutting there can never
        separate an assistant ``tool_calls`` message from its ``tool`` results.
        Prefers the latest such boundary that still leaves ``keep`` messages, and
        otherwise falls back to the earliest one: a context over budget must make
        progress, or it grows until the window overflows.
        """
        if len(self.messages) <= keep:
            return 0
        boundaries = [
            i for i, m in enumerate(self.messages) if i > 0 and m.get("role") == "user"
        ]
        if not boundaries:
            return 0
        target = max(0, len(self.messages) - keep)
        for index in boundaries:
            if index >= target:
                return index
        return boundaries[0]

    def _summarise(self, dropped: list[dict]) -> str:
        transcript = []
        for msg in dropped:
            role = msg.get("role")
            if role == "tool":
                transcript.append(f"tool {msg.get('name')}: {str(msg.get('content'))[:200]}")
            else:
                transcript.append(f"{role}: {str(msg.get('content'))[:400]}")
        if not transcript:
            return ""
        prompt = (
            "Compress this conversation fragment into at most 8 dense bullet points. Keep names, "
            "paths, commands, decisions and outcomes. Drop pleasantries.\n\n" + "\n".join(transcript)
        )
        try:
            completion = self.client.chat(
                [{"role": "user", "content": prompt}],
                None,
                think=False,
                max_tokens=400,
                temperature=0.2,
            )
        except LLMError:
            return ""
        summary = completion.content.strip()
        return (self.summary + "\n" + summary).strip() if self.summary else summary

    # --- session -------------------------------------------------------
    def reset(self) -> None:
        self.messages.clear()
        self.summary = ""

    def persist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary, "messages": self.messages}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    def load(self, path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        messages = payload.get("messages") or []
        summary = payload.get("summary") or ""
        # A hand-edited or half-written session used to load as a dict or a list
        # of strings and crash the first turn with AttributeError.
        if not isinstance(messages, list) or not all(
            isinstance(message, dict) and isinstance(message.get("role"), str) for message in messages
        ):
            return False
        if not isinstance(summary, str):
            return False
        self.messages = messages
        self.summary = summary
        return True

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "user")
