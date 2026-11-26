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

from .llm import Completion, LLMError, LlamaClient
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

    # --- prompt --------------------------------------------------------
    def system_prompt(self, query: str = "") -> str:
        a = self.cfg.agent
        parts = [a.personality.format(name=a.name)]
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

        for step in range(self.cfg.agent.max_steps):
            completion = self._chat(user_text, with_tools=True, stream=True)
            if completion.tool_calls:
                self._record_assistant(completion)
                for call in completion.tool_calls:
                    self._execute(call)
                continue
            text = completion.content.strip()
            self.messages.append({"role": "assistant", "content": text})
            return text

        self.ui.status(f"step budget ({self.cfg.agent.max_steps}) reached - answering without tools")
        self.ui.newline()
        completion = self._chat(user_text, with_tools=False, stream=True)
        text = completion.content.strip()
        self.messages.append({"role": "assistant", "content": text})
        return text

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
            max_tokens=g.max_tokens if not tools else g.tool_max_tokens,
            on_token=self.ui.token,
            on_reasoning=self.ui.reasoning,
        )
        self.last_completion = completion
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

    def _execute(self, call) -> None:
        self.ui.tool_call(call.name, call.arguments)
        tool = self.registry.get(call.name)
        if tool is None:
            result = fail(f"unknown tool {call.name!r}; available: {', '.join(self.registry.names())}")
        else:
            try:
                args = call.parsed()
            except LLMError as exc:
                result = fail(str(exc))
            else:
                try:
                    result = tool.run(args)
                except Exception as exc:  # noqa: BLE001 - a tool crash must not kill the turn
                    result = fail(f"{call.name} crashed: {exc.__class__.__name__}: {exc}")
        content = result.clipped(self.cfg.tools.max_output_chars)
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}
        )
        self.ui.tool_result(call.name, result)
        if self.audit:
            self.audit.log("tool", name=call.name, ok=result.ok, args=call.arguments[:400])

    # --- context management --------------------------------------------
    def compress_if_needed(self, force: bool = False) -> bool:
        a = self.cfg.agent
        budget = a.context_budget_tokens
        used = estimate_messages_tokens(self.messages) + estimate_tokens(self.summary)
        if not force and used <= budget:
            return False
        cut = self._safe_split(a.keep_recent_turns * 2)
        if cut <= 0:
            return False
        dropped, self.messages = self.messages[:cut], self.messages[cut:]
        self.summary = self._summarise(dropped) or self.summary
        return True

    def _safe_split(self, keep: int) -> int:
        """Index at which it is safe to cut: the next ``user`` message."""
        if len(self.messages) <= keep:
            return 0
        idx = len(self.messages) - keep
        while idx < len(self.messages) and self.messages[idx].get("role") != "user":
            idx += 1
        return idx if idx < len(self.messages) else 0

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
        except (OSError, json.JSONDecodeError):
            return False
        self.messages = payload.get("messages") or []
        self.summary = payload.get("summary") or ""
        return True

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "user")
