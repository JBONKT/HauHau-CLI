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
import re
import time
from pathlib import Path
from typing import Any

from .llm import Completion, LlamaClient, LLMError, Usage
from .tools import Registry, ToolResult, fail
from .util import estimate_messages_tokens, estimate_tokens

# --- autonomy -------------------------------------------------------------
# Two things a small model does instead of working: ask the user for something it
# could have looked up, or announce that it cannot do the task and stop. Both are
# answerable inside the turn, so the loop nudges and continues rather than handing
# the problem back. The rules are spelled out because a 2B model does not infer
# "be autonomous" from one personality sentence.
AUTONOMY_RULES = """\
Work independently, and ask only as a last resort.
- Missing a fact? Get it yourself: run the command, read `--help` or the man page,
  grep the repo, find the file, or rag_query your own notes. Never ask for
  something you can look up in one step.
- Ambiguous request? Pick the most likely reading, do it, and state the assumption
  in one short line ("assuming X"). Ask only when the action would destroy or
  publish something.
- Cannot do it? Do not stop at "I can't". Work out what is missing, learn it in
  this turn - look it up, try it, read the error - and finish the job. Once it
  works, record it with note_lesson so the next attempt is instant."""


NUDGE_AMBIGUOUS = (
    "You asked the user a question instead of acting. Do not ask again. Choose the most likely "
    "reading of their request, carry it out with tools now, and state the assumption in one line. "
    "Ask only when the action would destroy or publish something."
)
NUDGE_ACT = (
    "You printed a command for the user to copy instead of running it. Run it yourself with the "
    "shell tool now and report the result. If it would install, uninstall, delete or overwrite "
    "something, say which command and why you stopped, and ask once - otherwise just run it."
)
NUDGE_STUCK = (
    "You said you cannot do this. Do not stop there: work out what you are missing, look it up "
    "yourself (run the command with --help, read the man page, grep the repo, rag_query your own "
    "notes), try it, and finish the task. Then record what you learned with note_lesson."
)

# A refusal only reads as one at the start of a sentence and in the agent's own
# voice. Matched anywhere, "the build said unable to find crate, then I fixed it"
# and "du kan köra make om den inte är installerad" both looked like a refusal and
# spent a step telling the model to go and finish work it had already finished.
_EN_CANNOT = (
    "i cannot",
    "i can't",
    "i can not",
    "i'm unable to",
    "i am unable to",
    "unable to",
    "not able to",
    "i don't know how",
    "i do not know how",
    "i have no access",
    "i don't have access",
)

# Swedish puts the pronoun inside the modal ("det kan jag inte"), so the subject
# is optional and the phrase is matched by shape rather than by position.
_SV_CANNOT = re.compile(
    r"^(?:det |den |det här |jag |vi ){0,2}(?:kan|vet|har)\b[^.!?]{0,25}\binte\b"
    r"|^jag saknar\b"
)

# Code spans, quotes and fenced blocks are evidence being described, not the
# agent's own claim.
_QUOTED = re.compile(r"```[\s\S]*?```|`[^`]*`|\"[^\"\n]*\"|'[^'\n]*'")
_SENTENCE = re.compile(r"[^.!?\n]+")
_LEAD = " \t-*•\u2013123456789.)>#"


def asks_instead_of_acting(text: str) -> bool:
    """A short reply that is nothing but a question hands the turn back.

    An answer with a closing offer ("Klart, filen är borta. Vill du att jag
    återställer den?") is an answer: only a reply whose every sentence asks
    something is treated as one the user has to finish.
    """
    body = (text or "").strip()
    if not body.endswith("?"):
        return False
    # A long answer that happens to end in "?" is an answer, not a question.
    if len(body) > 300 or len(body.split()) > 45:
        return False
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]
    return bool(sentences) and all(part.endswith("?") for part in sentences)


_FENCED = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n([\s\S]*?)```")
_INLINE = re.compile(r"`([^`\n]+)`")
_SHELL_LANGS = frozenset({"", "bash", "sh", "shell", "zsh", "console", "terminal", "fish"})

# Enough of a whitelist to recognise a pasted command without flagging a file
# name, a tool name or a phrase in backticks.
_COMMAND_WORDS = """
    awk bash brew cat cd chmod chown cp crontab curl cut defaults df dig diskutil
    docker du echo env export find git grep head htop ifconfig kill kubectl launchctl
    ln ls lsof make md5 mdfind mdls mkdir mount mv netstat node npm open patch pbpaste
    pbcopy perl pgrep ping pip pkill pmset ps pytest python python3 readlink rm rsync
    scp sed sh sleep sort source ssh stat sudo sw_vers sysctl system_profiler tail
    tar tee top touch tr uname unzip vi wc wget which whoami xargs xattr yq zip
"""

_COMMANDS = frozenset(_COMMAND_WORDS.split())


def _looks_like_command(code: str) -> bool:
    for raw in code.splitlines():
        line = raw.strip().lstrip("$ ").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        first = parts[0].casefold()
        if first in _COMMANDS:
            return True
        # "pmset -g" style: an unknown verb with a short flag is still a command.
        return any(part.startswith("-") and 1 < len(part) <= 6 for part in parts[1:])
    return False


def commands_to_run(text: str) -> bool:
    """Whether the reply pastes a shell command instead of executing it.

    A small model answers "let's check" and quotes `du -sh *` - inline, or in a
    fence with or without a language tag - leaving the work to the user. A pasted
    command is only legitimate when it changes the system, and the nudge tells the
    model to run it unless that is the case.
    """
    body = text or ""
    for lang, code in _FENCED.findall(body):
        if lang.casefold() in _SHELL_LANGS and _looks_like_command(code):
            return True
    return any(_looks_like_command(span) for span in _INLINE.findall(body))


def says_it_cannot(text: str) -> bool:
    """Whether the reply is a refusal dressed up as a finished answer."""
    plain = _QUOTED.sub(" ", text or "")
    for sentence in _SENTENCE.findall(plain):
        stripped = sentence.strip(_LEAD).casefold()
        if not stripped:
            continue
        if stripped.startswith(_EN_CANNOT) or _SV_CANNOT.search(stripped):
            return True
    return False


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
        skills: list | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.registry = registry
        self.brain = brain
        self.audit = audit
        self.ui = ui or UI()
        self.on_confirm = on_confirm
        self.skills = skills or []
        self.messages: list[dict[str, Any]] = []
        self.summary = ""
        self.last_completion: Completion | None = None
        self._turn_usage = Usage()
        self._nudge: str | None = None
        self.nudges_this_turn = 0
        self._tool_failed = False

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
        if getattr(a, "autonomy", True):
            parts.append(AUTONOMY_RULES)
        if self._nudge:
            # The nudge lives for exactly one request, so it cannot leak into the
            # rest of the conversation or into the session summary.
            parts.append(self._nudge)
        if self.brain is not None and self.cfg.brain.enabled:
            block = self.brain.context_block(query or self._last_user_text())
            if block:
                parts.append(block)
        if self.skills:
            from .skills import prompt_block

            block = prompt_block(self.skills)
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
        self._nudge = None  # spent on this request
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
        self.nudges_this_turn = 0
        self._tool_failed = False
        # A nudge left over from a turn that was interrupted would otherwise be
        # injected into the first request of the next one.
        self._nudge = None
        for _ in range(self.cfg.agent.max_steps):
            # Tool results are the bulk of what grows the context; checking only
            # once per turn let a long tool loop overrun n_ctx mid-turn and the
            # backend answered 400, killing the whole turn.
            self.compress_if_needed()
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
            if self._nudge_needed(text):
                continue
            return text

        self.ui.status(f"step budget ({self.cfg.agent.max_steps}) reached - answering without tools")
        self.ui.newline()
        # This call has no tools, so a nudge set by the last step ("carry it out
        # with tools now") could not be obeyed and would only confuse the answer.
        self._nudge = None
        completion = self._chat(user_text, with_tools=False, stream=True)
        text = completion.content.strip()
        self.messages.append({"role": "assistant", "content": text})
        self._warn_if_truncated(completion)
        return text

    def _nudge_needed(self, text: str) -> bool:
        """Whether to spend one more step instead of handing the turn back.

        A small model answers "vill du att jag kollar diskarna?" or "det kan jag
        inte" and stops - and the user, who asked for the thing to be *done*, has
        to reply "ja, gör det". One nudge turns that into the work. Capped so a
        model that keeps asking cannot loop, and skipped once a tool has failed:
        then the question is about a real obstacle and the user must see it.
        """
        if not getattr(self.cfg.agent, "autonomy", True):
            return False
        if self._tool_failed:
            # A tool already failed in this turn: whatever the model says now is
            # about a real obstacle, and the user must get to see it.
            return False
        if self.nudges_this_turn >= 2:
            return False
        reason: str | None = None
        if says_it_cannot(text):
            reason = NUDGE_STUCK
        elif commands_to_run(text):
            reason = NUDGE_ACT
        elif asks_instead_of_acting(text):
            reason = NUDGE_AMBIGUOUS
        if reason is None:
            return False
        self.nudges_this_turn += 1
        self._nudge = reason
        if self.audit:
            # Visible in audit.jsonl, so "is it thinking for itself?" has an answer.
            self.audit.log(
                "nudge",
                kind={NUDGE_STUCK: "stuck", NUDGE_ACT: "act"}.get(reason, "ambiguous"),
                count=self.nudges_this_turn,
                text=text[:200],
            )
        self.ui.status("not asking - working out what was meant and doing it")
        return True

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
        try:
            result = self._call_tool(call, truncated)
        except BaseException:
            # The assistant message that announced this call is already in the
            # history, so unwinding without a matching tool message leaves a
            # transcript every later request rejects. Answer the call, then pass
            # the interrupt on - Ctrl-C must still stop the turn.
            self._append_tool_result(
                call, fail(f"{call.name}: interrupted before it finished, nothing was reported")
            )
            raise
        self._append_tool_result(call, result)

    def _call_tool(self, call, truncated: bool) -> ToolResult:
        tool = self.registry.get(call.name)
        if truncated:
            return fail(
                f"{call.name}: this call was cut off at the {self.cfg.generation.max_tokens}-token "
                "limit, so its arguments are incomplete and it was NOT run. Retry with less content "
                "per call (for example write_file in several smaller writes)."
            )
        if tool is None:
            return fail(f"unknown tool {call.name!r}; available: {', '.join(self.registry.names())}")
        try:
            args = call.parsed()
        except LLMError as exc:
            return fail(str(exc))
        try:
            result = tool.run(args)
        except Exception as exc:  # a tool crash must not kill the turn
            return fail(f"{call.name} crashed: {exc.__class__.__name__}: {exc}")
        if not isinstance(result, ToolResult):
            return fail(f"{call.name} returned {type(result).__name__}, not a ToolResult")
        return result

    def _append_tool_result(self, call, result: ToolResult) -> None:
        if not result.ok:
            self._tool_failed = True
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": result.clipped(self.cfg.tools.max_output_chars),
            }
        )
        self.ui.tool_result(call.name, result)
        if self.audit:
            # Arguments arrive as a string normally, but a model can send an
            # object: slicing that raised KeyError out of the whole turn.
            self.audit.log("tool", name=call.name, ok=result.ok, args=str(call.arguments)[:400])

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
        # Fold an oversized summary first: trimming messages cannot help when
        # the summary alone is over budget, and that was a permanent dead end.
        folded = self._cap_summary()
        used = estimate_messages_tokens(self.messages) + estimate_tokens(self.summary)
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
        # The latest boundary that still keeps `keep` messages: taking the first
        # one at or after the target kept a single message where ten were meant.
        late = [index for index in boundaries if index <= target]
        return late[-1] if late else boundaries[0]

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
