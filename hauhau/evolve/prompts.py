"""Versioned system prompt that the agent may rewrite - under a gate.

Every version is a plain file under ``<data_dir>/evolve/prompts/v<N>.md`` with a
JSONL index, so a bad edit is one ``/prompt v2`` away from being undone, and the
whole history can be committed to git if the user wants an audit trail outside
``audit.jsonl``.

A candidate is adopted only if it scores strictly better on the behavioural
suite. That gate is the whole point: a 2B model asked to improve its own
instructions will happily produce something that sounds wiser and acts worse.
"""

from __future__ import annotations

import contextlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..llm import LLMError
from .suite import SuiteResult


def _number_of(path: Path) -> int | None:
    """Version number encoded in a ``v<N>.md`` filename."""
    match = re.fullmatch(r"v(\d+)\.md", path.name)
    return int(match.group(1)) if match else None


def _score(value: object) -> float | None:
    """A score read back from JSONL, which may have been hand-edited."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None

# A floor and a ceiling, but the floor is relative: a rewrite that keeps the same
# instructions in fewer words is a legitimate improvement, and rejecting it for
# being 199 characters long (as a fixed 200 did) is an arbitrary loss.
MIN_CHARS = 80
MAX_CHARS = 6000
MIN_RATIO = 0.5

REWRITE_PROMPT = """You are editing the system prompt of a terminal agent. It runs on a small local
model and it must act with tools rather than talk about acting.

CURRENT PROMPT:
---
{current}
---

Tasks from the behavioural test suite that the current prompt FAILS:
{failures}

Rewrite the prompt so those failures stop happening. Keep what already works.
Hard rules:
- Third person instructions to the agent, no examples of conversations.
- Keep it under {limit} characters.
- Keep the `{{name}}` placeholder if it is present.
- Never mention specific task prompts or test names.
- Never weaken the rules about claiming success without evidence.

Return ONLY the new prompt text, no commentary and no code fences."""


@dataclass(frozen=True)
class Version:
    number: int
    path: Path
    score: float | None
    parent: int | None
    created: str

    @property
    def label(self) -> str:
        return f"v{self.number}"


@dataclass
class Evolution:
    baseline: float
    candidate: float
    accepted: bool
    reason: str
    version: int | None = None
    text: str = ""


class PromptStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.index_path = self.root / "index.jsonl"
        self.active_path = self.root / "active"

    # --- reads ---------------------------------------------------------
    def _index(self) -> list[dict]:
        if not self.index_path.is_file():
            return []
        entries: list[dict] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    def versions(self) -> list[Version]:
        out: list[Version] = []
        for entry in self._index():
            try:
                number = int(entry["number"])
            except (KeyError, TypeError, ValueError):
                continue
            path = self.versions_dir / f"v{number}.md"
            if not path.is_file():
                continue
            out.append(
                Version(
                    number=number,
                    path=path,
                    score=_score(entry.get("score")),
                    parent=entry.get("parent"),
                    created=str(entry.get("created", "")),
                )
            )
        return sorted(out, key=lambda v: v.number)

    def text(self, number: int) -> str:
        """Return the prompt exactly as it was stored (file-ending aside)."""
        path = self.versions_dir / f"v{number}.md"
        if not path.is_file():
            raise FileNotFoundError(f"no prompt version v{number}")
        return path.read_text(encoding="utf-8").rstrip("\n")

    def active_number(self) -> int | None:
        if not self.active_path.is_file():
            return None
        try:
            return int(self.active_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None

    def active(self) -> Version | None:
        number = self.active_number()
        if number is None:
            return None
        for version in self.versions():
            if version.number == number:
                return version
        return None

    # --- writes --------------------------------------------------------
    def add(self, text: str, *, parent: int | None = None, score: float | None = None) -> Version:
        text = text.strip()
        if not text:
            raise ValueError("refusing to store an empty prompt")
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        # `versions()` is the *intersection* of index.jsonl and the files on
        # disk, so deriving the next number from it re-issued a live number
        # whenever either side was lost - and overwrote a rollback point.
        numbers = [v.number for v in self.versions()]
        numbers += [_number_of(path) for path in self.versions_dir.glob("v*.md")]
        for entry in self._index():
            with contextlib.suppress(KeyError, TypeError, ValueError):
                numbers.append(int(entry["number"]))
        number = max([n for n in numbers if n is not None] or [0]) + 1
        path = self.versions_dir / f"v{number}.md"
        while True:
            try:
                with path.open("x", encoding="utf-8") as fh:
                    fh.write(text + "\n")
                break
            except FileExistsError:
                number += 1
                path = self.versions_dir / f"v{number}.md"
        entry = {
            "number": number,
            "score": score,
            "parent": parent,
            "chars": len(text),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        return Version(number, path, score, parent, entry["created"])

    def activate(self, number: int) -> Version:
        version = next((v for v in self.versions() if v.number == number), None)
        if version is None:
            raise FileNotFoundError(f"no prompt version v{number}")
        self.active_path.write_text(f"{number}\n", encoding="utf-8")
        return version


def _clean(raw: str) -> str:
    """Strip code fences and chatter a small model wraps around the prompt."""
    text = (raw or "").strip()
    fence = re.search(r"```(?:[a-zA-Z]*)\n(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    return text.strip()


def validate(text: str, *, current: str = "") -> str | None:
    """Return a rejection reason, or None when the text is usable as a prompt."""
    floor = MIN_CHARS
    if current:
        floor = max(MIN_CHARS, int(len(current) * MIN_RATIO))
    if len(text) < floor:
        return f"too short ({len(text)} chars, need {floor})"
    if len(text) > MAX_CHARS:
        return f"too long ({len(text)} chars, limit {MAX_CHARS})"
    try:
        text.format(name="Hauhau")
    except (KeyError, IndexError, ValueError) as exc:
        # The agent formats the personality with name=..., and a stray brace in
        # the rewrite would crash every turn after it is adopted.
        return f"breaks the {{name}} template: {exc.__class__.__name__}"
    return None


def evolve_prompt(
    client,
    store: PromptStore,
    run: Callable[[str], SuiteResult],
    current: str,
    *,
    max_chars: int = MAX_CHARS,
    progress: Callable[[str], None] | None = None,
) -> Evolution:
    """Ask the model for a better prompt; adopt it only if the suite improves."""
    if progress is not None:
        progress("scoring the current prompt")
    baseline = run(current)
    if baseline.total == 0:
        return Evolution(0.0, 0.0, False, "the suite has no runnable tasks")
    if baseline.passed == baseline.total:
        return Evolution(baseline.score, baseline.score, False, "the suite already passes every task")

    failures = "\n".join(f"- {name}: {detail}" for name, detail in baseline.failures) or "- (none)"
    prompt = REWRITE_PROMPT.format(current=current, failures=failures, limit=max_chars - 100)
    if progress is not None:
        progress("asking the model for a new prompt")
    try:
        completion = client.chat(
            [{"role": "user", "content": prompt}], None, think=False, temperature=0.4, max_tokens=900
        )
    except LLMError as exc:
        return Evolution(baseline.score, 0.0, False, f"rewrite failed: {exc}")

    candidate = _clean(completion.content)
    reason = validate(candidate, current=current)
    if reason is not None:
        return Evolution(baseline.score, 0.0, False, f"candidate rejected: {reason}")
    if candidate.strip() == current.strip():
        return Evolution(baseline.score, baseline.score, False, "the model returned the same prompt")

    if progress is not None:
        progress("scoring the candidate prompt")
    scored = run(candidate)
    if scored.score <= baseline.score:
        return Evolution(
            baseline.score,
            scored.score,
            False,
            f"no improvement ({baseline.summary()} → {scored.summary()})",
            text=candidate,
        )

    parent = store.active_number()
    version = store.add(candidate, parent=parent, score=scored.score)
    store.activate(version.number)
    return Evolution(
        baseline.score,
        scored.score,
        True,
        f"adopted v{version.number} ({baseline.summary()} → {scored.summary()})",
        version=version.number,
        text=candidate,
    )
