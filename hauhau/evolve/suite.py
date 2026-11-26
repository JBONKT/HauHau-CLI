"""Behavioural eval suite: does the prompt still make the agent *act*.

Prompt self-evolution is meaningless without a pass/fail signal, and the only
honest signal available offline is observable behaviour: given a task, does the
model reach for the right tool, and does it answer when no tool is needed? That
needs one short completion per task - about a minute for the whole suite at
27 tok/s - so it runs on request (``/eval``), never mid-turn.

The suite deliberately tests *behaviour*, not knowledge: a 2B model's facts are
unreliable, but whether it calls ``grep`` for a code search or chats instead is
stable enough to compare two prompts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from ..llm import LlamaClient


@dataclass(frozen=True)
class Task:
    """One observable requirement.

    ``tools_any``: pass if the model's first action is one of these tools.
    ``args_contains``: additionally require one of these strings in some call's
    arguments - a model that picks ``shell`` but runs the wrong command has not
    understood the request, and that is the difference a clearer prompt makes.
    ``contains``: pass if the final answer contains one of these strings
    (whole words, case-insensitive) and no tool was called - answering with a
    tool call is exactly the failure a chatty prompt causes.
    ``exact``: the answer must *be* one of ``contains``, not merely mention it -
    "Reply with exactly: pong" was passed by "I will not reply pong."
    """

    name: str
    prompt: str
    tools_any: tuple[str, ...] = ()
    args_contains: tuple[str, ...] = ()
    contains: tuple[str, ...] = ()
    exact: bool = False

    @property
    def kind(self) -> str:
        return "tool" if self.tools_any else "text"


# Written against the tool names in hauhau.tools. "shell" is accepted nearly
# everywhere because it is the general-purpose escape hatch a small model
# reaches for first; requiring a specific tool would measure the model's taste
# rather than the prompt's clarity.
DEFAULT_TASKS: tuple[Task, ...] = (
    Task("list-files", "List the files in the current directory.", ("list_dir", "shell"), ("ls", ".", "list")),
    Task("read-config", "Show me what is in config.toml.", ("read_file", "shell"), ("config.toml",)),
    Task(
        "code-search",
        "Find every place in the code that mentions context_budget_tokens.",
        ("grep", "shell"),
        ("context_budget_tokens",),
    ),
    Task("run-tests", "Run the unit tests and tell me how many passed.", ("shell",), ("unittest", "pytest", "test")),
    Task("hostname", "What is this machine's hostname?", ("shell",), ("hostname", "uname", "scutil")),
    Task("disk-space", "How much disk space is free right now?", ("shell",), ("df", "diskutil")),
    Task(
        "find-files",
        "Find the Python files under the hauhau directory.",
        ("find", "list_dir", "shell"),
        (".py", "hauhau", "python"),
    ),
    # Measuring beats guessing: a prompt that encourages action makes the model
    # read the file, a chatty one invents a plausible number.
    Task(
        "line-count",
        "How many lines are in README.md?",
        ("shell", "read_file"),
        ("wc", "readme", "line"),
    ),
    Task(
        "python-version",
        "Which Python version does pyproject.toml require?",
        ("read_file", "shell", "grep"),
        ("pyproject", "python"),
    ),
    Task("recall", "What do you remember about my release checklist?", ("rag_query",)),
    Task("store-fact", "Remember that deploys go through the staging pipeline.", ("rag_write",)),
    Task("exact-reply", "Reply with exactly: pong", contains=("pong",), exact=True),
    Task("no-tool-answer", "In one short sentence: what does the ls command do?", contains=("list",)),
    Task("arithmetic", "What is 1+1? Answer with the number only.", contains=("2",)),
)


@dataclass
class SuiteResult:
    passed: int = 0
    total: int = 0
    seconds: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        return f"{self.passed}/{self.total} tasks ({self.score:.0%})"


def available(tasks: tuple[Task, ...], tool_names: list[str]) -> tuple[Task, ...]:
    """Keep only tasks whose required tools exist (brain off drops rag_*)."""
    return tuple(t for t in tasks if not t.tools_any or any(n in tool_names for n in t.tools_any))


def _mentions(want: str, haystack: str) -> bool:
    """Whole-token match: "12" must not pass for "2", "false" is not "ls".

    The boundary is only required where the expected string has one: ".py" has
    to keep matching "mod.py", where a leading word character precedes it.
    """
    want, haystack = want.lower(), haystack.lower()
    if not want:
        return False
    left = r"(?<!\w)" if (want[0].isalnum() or want[0] == "_") else ""
    right = r"(?!\w)" if (want[-1].isalnum() or want[-1] == "_") else ""
    return re.search(f"{left}{re.escape(want)}{right}", haystack) is not None


def _judge(task: Task, completion: Any) -> tuple[bool, str]:
    raw = list(getattr(completion, "tool_calls", None) or [])
    calls = [c.name for c in raw]
    text = str(getattr(completion, "content", "") or "").strip()
    if task.tools_any:
        if not calls:
            return False, f"answered in prose instead of calling {'/'.join(task.tools_any)}"
        if calls[0] not in task.tools_any:
            return False, f"called {calls[0]} (wanted {'/'.join(task.tools_any)})"
        if task.args_contains:
            arguments = " ".join(str(c.arguments) for c in raw).lower()
            if not any(_mentions(want, arguments) for want in task.args_contains):
                return False, f"{calls[0]} ran the wrong command: {arguments[:80]}"
        return True, calls[0]
    if calls:
        return False, f"called {calls[0]} instead of answering"
    if task.exact:
        folded = re.sub(r"[^\w]+", "", text.lower())
        if any(folded == re.sub(r"[^\w]+", "", want.lower()) for want in task.contains):
            return True, "answered"
        return False, f"answer was not exactly {task.contains!r}"
    if all(want.strip().isdigit() for want in task.contains if want.strip()):
        # "Answer with the number only" means exactly that; "It is not 2." and
        # "12" both contained the expected digit and passed the old substring test.
        allowed = "|".join(re.escape(want.strip()) for want in task.contains if want.strip())
        if re.fullmatch(rf"[\s\W]*(?:{allowed})[\s\W]*", text):
            return True, "answered"
        return False, f"answer was not the number {task.contains!r}"
    if any(_mentions(want, text) for want in task.contains):
        return True, "answered"
    return False, f"answer did not contain {task.contains!r}"


def run_suite(
    client: LlamaClient,
    tasks: tuple[Task, ...],
    *,
    system_prompt: str,
    tools: list[dict] | None,
    max_tokens: int = 96,
    temperature: float = 0.0,
    progress=None,
) -> SuiteResult:
    """Run every task once. Raises LLMError if the backend itself fails."""
    started = time.monotonic()
    result = SuiteResult(total=len(tasks))
    for task in tasks:
        if progress is not None:
            progress(task.name)
        completion = client.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.prompt},
            ],
            tools,
            think=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        ok, detail = _judge(task, completion)
        if ok:
            result.passed += 1
        else:
            result.failures.append((task.name, detail))
    result.seconds = time.monotonic() - started
    return result
