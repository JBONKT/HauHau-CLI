"""Post-session self-learning: distil the transcript into durable memories.

This is the part that makes the agent cumulative. It runs on demand (``/learn``)
and on exit, never mid-turn - a separate model pass per user turn would cost more
latency than it saves on a 27 tok/s machine.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ..llm import LLMError
from .store import KINDS

# Below this, a session is too thin to be worth a model pass.
MIN_TRANSCRIPT_CHARS = 40
# A greeting with a one-line answer can exceed the threshold above, and a model
# pass costs ~20s at 27 tok/s on this machine. Only a session with real work in
# it - a tool call, a second turn, or a long exchange - earns the exit delay.
MIN_SIGNAL_CHARS = 400

PROMPT = """You are reviewing a finished terminal session to write long-term memory.

Extract only durable, reusable knowledge:
- preference: how this user wants things done
- semantic: stable facts about their machine, projects, paths, accounts, versions
- procedural: a command sequence or approach that worked and should be reused
- episodic: one line on what this session accomplished, only if it matters later

Rules: one self-contained sentence per memory, written in the third person.
No narration, no transient details (temp paths, one-off PIDs), no secrets.
Never record that a command succeeded, a test string, an echo, or a one-off
verification: "ran echo OK to check the shell" is not knowledge.
Record only what is specific to this machine, this project, or this user - never
generic advice that any assistant already knows. A procedural memory must name
the actual command or path that worked.
Anything listed under ALREADY REMEMBERED is known: do not restate or reword it.

ALREADY REMEMBERED:
{known}

Return ONLY a JSON array, max 8 items, no prose and no code fences:
[{"kind":"preference","text":"...","tags":["..."],"importance":0.7}]

If there is nothing durable and new to record, return [].

TRANSCRIPT:
{transcript}"""


@dataclass
class Draft:
    kind: str
    text: str
    tags: str = ""
    importance: float = 0.5


# A fact or procedure is worth keeping only if it names something concrete: a
# path, a filename, a version, a port, a URL, a flag. Without this guard a 2B
# model happily records "ran echo OK to confirm the shell worked" as a procedure
# - generic, useless, and it crowds out the real memories at retrieval time.
_SPECIFIC = re.compile(
    r"(~/|\s/(?:Users|opt|etc|usr|var|srv|home|private|Applications)/"
    r"|\b[\w.-]+\.(?:py|toml|json|md|txt|gguf|sh|zsh|bash|db|ya?ml|cfg|ini|env|toml)\b"
    r"|\b\d+\.\d+(?:\.\d+)?\b"
    r"|https?://"
    r"|(?<!\w)--?[a-z][a-z0-9-]+"
    r"|:\d{2,5}\b)",
    re.IGNORECASE,
)

# Kinds that must be concrete; a preference may legitimately be a plain sentence
# ("the user wants answers in Swedish").
_NEEDS_SPECIFICS = {"semantic", "procedural"}

# Naming a real tool, service or platform is as specific as naming a path: the
# old rule rejected "the CI provider is GitHub Actions" while accepting
# "remember to be brief at 10:30".
_TOOLING = re.compile(
    r"\b(?:github|gitlab|bitbucket|actions|jenkins|circleci|homebrew|brew|docker|podman"
    r"|kubernetes|postgres(?:ql)?|mysql|sqlite|redis|nginx|apache|macos|osx|linux|ubuntu"
    r"|debian|windows|wsl|macbook|imac|emacs|vim|neovim|vscode|pycharm|zsh|bash|fish"
    r"|python|node|rust|cargo|uv|poetry|pipx|ruff|pytest|mypy|llama\.cpp|ollama"
    r"|staging|production|release branch|ci\b|gpu|cuda|metal|rocm)\b",
    re.IGNORECASE,
)


def _anchors(cfg) -> tuple[str, ...]:
    """Words that tie a memory to this machine, project or user."""
    words = [
        getattr(cfg.agent, "name", ""),
        Path(cfg.workspace).name if getattr(cfg, "workspace", None) else "",
        Path(os.path.expanduser(cfg.model.gguf)).stem if getattr(cfg.model, "gguf", "") else "",
    ]
    with contextlib.suppress(Exception):
        words.append(getpass.getuser())
    return tuple(w.lower() for w in words if w and len(w) > 2)


def _kind(value: object) -> str:
    """Only the four stored kinds survive; anything else is a plain fact."""
    kind = str(value or "semantic").strip().lower()
    return kind if kind in KINDS else "semantic"


def durable(draft: Draft, anchors: tuple[str, ...] = ()) -> bool:
    """Reject generic advice the agent should not need a memory for.

    An unrecognised kind fails *closed*: ``parse_drafts`` accepts whatever string
    the model sends, so "procedure" (or "Semantic") used to bypass the guard and
    store exactly the noise it exists to drop.
    """
    kind = (draft.kind or "").strip().lower()
    if kind in KINDS and kind not in _NEEDS_SPECIFICS:
        return True
    if _SPECIFIC.search(draft.text) or _TOOLING.search(draft.text):
        return True
    low = draft.text.lower()
    return any(anchor in low for anchor in anchors)


def build_transcript(messages: list[dict], max_chars: int = 6000) -> str:
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            lines.append(f"[tool {msg.get('name')}] {str(msg.get('content'))[:300]}")
        elif role == "assistant":
            calls = [c["function"]["name"] for c in msg.get("tool_calls") or [] if c.get("function")]
            text = str(msg.get("content") or "")[:500]
            if calls:
                lines.append(f"[assistant -> {', '.join(calls)}] {text}")
            elif text.strip():
                lines.append(f"[assistant] {text}")
        elif role == "user":
            lines.append(f"[user] {str(msg.get('content'))[:500]}")
    transcript = "\n".join(lines)
    if len(transcript) > max_chars:
        transcript = transcript[-max_chars:]
    return transcript


def worth_distilling(messages: list[dict]) -> bool:
    """Cheap pre-check: is there anything a model pass could learn from?"""
    if any(m.get("role") == "tool" for m in messages):
        return True
    if sum(1 for m in messages if m.get("role") == "user") >= 2:
        return True
    return len(build_transcript(messages)) >= MIN_SIGNAL_CHARS


def parse_drafts(raw: str) -> list[Draft]:
    """Tolerantly pull a JSON array out of a model reply."""
    if not raw:
        return []
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        # Salvage individual objects if the array itself is malformed.
        items = []
        for blob in re.findall(r"\{[^{}]*\}", raw[start : end + 1]):
            try:
                items.append(json.loads(blob))
            except json.JSONDecodeError:
                continue
    drafts: list[Draft] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if len(text) < 12:
            continue
        tags = item.get("tags") or ""
        if isinstance(tags, list):
            tags = ",".join(str(t) for t in tags)
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        if not math.isfinite(importance):
            importance = 0.5
        drafts.append(
            Draft(
                kind=_kind(item.get("kind")),
                text=text,
                tags=str(tags),
                importance=max(0.0, min(1.0, importance)),
            )
        )
    return drafts[:8]


class Learner:
    def __init__(self, cfg, client, brain, audit=None) -> None:
        self.cfg = cfg
        self.client = client
        self.brain = brain
        self.audit = audit

    def _known_block(self, limit: int = 25, max_chars: int = 1500) -> str:
        """Existing memories, so the model stops restating what is already stored."""
        rows = self.brain.store.recent(limit=limit)
        if not rows:
            return "(nothing yet)"
        lines, budget = [], max_chars
        for hit in rows:
            line = f"- [{hit.kind}] {hit.text}"
            if len(line) > budget:
                break
            lines.append(line)
            budget -= len(line)
        return "\n".join(lines) or "(nothing yet)"

    def distill(self, messages: list[dict], min_turns: int = 1) -> tuple[list[Draft], list[tuple[int, bool]]]:
        """Ask the model what was worth remembering; write it to the brain."""
        if self.brain is None or not self.cfg.brain.enabled:
            return [], []
        turns = sum(1 for m in messages if m.get("role") == "user")
        if turns < min_turns:
            return [], []
        transcript = build_transcript(messages)
        if len(transcript) < MIN_TRANSCRIPT_CHARS or not worth_distilling(messages):
            return [], []
        # Transcript first: memory text is model-written and may itself contain
        # "{known}" or "{transcript}", which the old order spliced in.
        prompt = PROMPT.replace("{transcript}", transcript).replace("{known}", self._known_block())
        try:
            completion = self.client.chat(
                [{"role": "user", "content": prompt}],
                None,
                think=False,
                temperature=0.2,
                max_tokens=700,
            )
        except LLMError as exc:
            if self.audit:
                self.audit.log("learn.error", error=str(exc))
            return [], []
        parsed = parse_drafts(completion.content)
        anchors = _anchors(self.cfg)
        drafts = [d for d in parsed if durable(d, anchors)]
        rejected = len(parsed) - len(drafts)
        written: list[tuple[int, bool]] = []
        for draft in drafts:
            try:
                written.append(
                    self.brain.remember(draft.kind, draft.text, draft.tags, draft.importance)
                )
            except ValueError:
                continue
        if self.audit:
            self.audit.log("learn", drafts=len(drafts), written=len(written), rejected=rejected)
        return drafts, written
