"""Terminal UI: streaming renderer, tool display, panels, prompt chrome.

Kept out of ``cli.py`` so the presentation layer can be exercised without a
REPL, and so every visual decision lives in one place.

Design notes:
* Answers are rendered as Markdown. The model emits ``**bold**``, lists and code
  fences constantly; printing those raw (as the first version did) looked broken.
  Tokens accumulate into a Live region that re-renders the block, throttled,
  because a re-parse per token at 27 tok/s is wasted work.
* Nothing renders through Rich markup from model output - tool output and
  answers can contain square brackets and angle brackets that would be eaten.
* Width is respected explicitly: tool lines never wrap, they ellipsize.
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import deque
from typing import Any

from prompt_toolkit import HTML, PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import UI
from .tools import ToolResult

# One accent, one muted tone, plus semantic colours. Tokyo-night-ish so it reads
# well on both dark and light terminal backgrounds.
ACCENT = "#7aa2f7"
OK = "green"
WARN = "yellow"
ERR = "red"
# Rich accepts named colours; prompt_toolkit needs hex, so keep both.
MUTED = "grey58"
MUTED_HEX = "#565f89"
TOOLBAR_BG = "#26263a"

PROMPT_GLYPH = "❯"  # noqa: RUF001 - the prompt glyph is deliberate
STREAM_MIN_INTERVAL = 1 / 8  # seconds between markdown re-renders

# Grouped so that /help answers "what can I do here", not "here are N strings".
COMMAND_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "this session",
        [
            ("/help", "commands, or details for one: /help tune"),
            ("/status", "one screen: model, memory, prompt, session"),
            ("/config", "effective configuration and where it came from"),
            ("/context", "token usage of this conversation"),
            ("/clear", "forget the conversation, keep memory"),
            ("/save", "persist the session now"),
            ("/audit", "tail the audit log"),
            ("/quit", "leave (runs a /learn pass first)"),
        ],
    ),
    (
        "working",
        [
            ("/tools", "tools the model can call"),
            ("/show", "the last tool output in full: /show, /show 2"),
            ("/skills", "installed skills: list, show one, or reload the folders"),
        ],
    ),
    (
        "memory",
        [
            ("/mem", "memory stats, or search memories"),
            ("/compact", "merge near-duplicate memories"),
            ("/prune", "forget stale memories that were never reused"),
            ("/remember", "store a fact yourself"),
            ("/forget", "delete a memory by id"),
            ("/learn", "distil this session into long-term memory"),
        ],
    ),
    (
        "model",
        [
            ("/model", "list local .gguf files, or switch: /model 2"),

            ("/perf", "measure real tokens/second"),
        ],
    ),
    (
        "self-tuning",
        [
            ("/eval", "play 14 behavioural tasks and score them"),
            ("/tune", "hill-climb threads/context for speed"),
            ("/evolve", "rewrite the system prompt, gated by /eval"),
            ("/prompt", "list prompt versions, or activate one"),
        ],
    ),
    (
        "journal",
        [
            ("/sync", "pull mails, messages, calls, events, reminders into the journal"),
            ("/journal", "journal stats, or search it: /journal Dana"),
            ("/brief", "today's digest: what arrived, what is due, what is next"),
        ],
    ),
]

COMMANDS: list[tuple[str, str]] = [
    (command, description) for _group, items in COMMAND_GROUPS for command, description in items
]

# The commands that take an argument get a usage line, shown by /help <command>.
COMMAND_USAGE: dict[str, str] = {
    "/help": "/help            list everything\n/help tune       detail for one command",
    "/show": (
        "/show            the most recent tool output, unabridged\n"
        "/show 2          the one before it (up to 5 back)"
    ),
    "/mem": "/mem             totals, kinds, database size\n/mem postgres    search memories",
    "/compact": "/compact         merge near-duplicate memories",
    "/prune": (
        "/prune           drop memories that are stale and were never recalled\n"
        "/prune 0.1       stricter: drop anything below 0.1 weight"
    ),
    "/remember": "/remember <text> store a fact by hand, deduplicated like any other",
    "/forget": "/forget <id>     delete one memory by id (see /mem for ids)",
    "/learn": "/learn           distil this session into memory now (also runs on exit)",
    "/eval": "/eval            run the behavioural suite; about a second per task",
    "/tune": (
        "/tune            try a few values, keep only a faster one\n"
        "/tune 6           allow up to 6 trials (each restarts the backend)"
    ),
    "/evolve": "/evolve          ask the model for a better prompt, keep it only if /eval improves",
    "/prompt": (
        "/prompt          list versions with their scores\n"
        "/prompt v2       activate a version (that is the rollback)"
    ),
    "/sync": (
        "/sync            pull every source into the journal\n"
        "/sync mail calls only those two (mail, messages, calls, calendar, reminders)"
    ),
    "/journal": "/journal         what the journal holds\n/journal Dana    search it for 'Dana'",
    "/brief": "/brief           today's digest: arrivals, due reminders, upcoming events",
    "/audit": "/audit           tail the last 10 audit entries\n/audit 40        the last 40",
}

# Tools whose arguments have one obviously interesting field. Showing
# {"cmd":"ls -la"} verbatim wasted the whole line on syntax.
_ARG_KEYS = ("cmd", "query", "pattern", "path", "text")

# "total 993272" as printed by ls -l / ls -lhS
_LS_TOTAL = re.compile(r"^total \d+$")


def make_console() -> Console:
    """A console that never auto-highlights or mangles arbitrary model output."""
    return Console(highlight=False, emoji=False, soft_wrap=False)


def short_name(name: str, width: int = 34) -> str:
    """Middle-ellipsize a long file name, cutting at a hyphen, not mid-word."""
    if len(name) <= width:
        return name
    tail = name[-11:]
    head = name[: max(6, width - len(tail) - 1)]
    if "-" in head:
        head = head[: head.rfind("-")]
    return f"{head}…{tail}"


def summarize_args(name: str, args: str) -> str:
    """One readable line for a tool call: the salient argument, not raw JSON."""
    try:
        parsed = json.loads(args or "{}")
    except (json.JSONDecodeError, TypeError):
        return " ".join(str(args).split())[:100]
    if not isinstance(parsed, dict):
        return str(args)[:100]
    for key in _ARG_KEYS:
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            main = " ".join(value.split())[:90]
            rest = {
                k: v
                for k, v in parsed.items()
                if k != key and v not in (None, "", [], {}, False)
            }
            tail = "".join(f" {k}={_scalar(v)}" for k, v in rest.items())
            return (main + tail)[:140]
    return (" ".join(json.dumps(parsed, ensure_ascii=False).split()))[:100]


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())[:20]
    if isinstance(value, list):
        return f"[{len(value)}]"
    return str(value)


def result_preview(result: ToolResult) -> tuple[str, int]:
    """First meaningful line of a tool result, plus how many lines remain hidden.

    One line only: tool output is usually multi-line, and joining lines with a
    separator made df/du output read like a path list.
    """
    lines = [ln.strip() for ln in (result.content or "").strip().splitlines() if ln.strip()]
    # Drop noise the tools add in front of the useful output: the shell tool's
    # "exit=N" and the block-total line "ls -l" prints first.
    informative = [
        ln for ln in lines if not ln.startswith("exit=") and not _LS_TOTAL.match(ln)
    ]
    lines = informative or lines
    if not lines:
        return "(no output)", 0
    return lines[0][:400], max(0, len(lines) - 1)


class StreamRenderer:
    """Accumulates tokens and renders the growing block as Markdown."""

    def __init__(self, console: Console, style: str = "") -> None:
        self.console = console
        self.style = style
        self.buffer = ""
        self._live: Live | None = None
        self._last = 0.0

    def _body(self):
        text = self.buffer
        if not text.strip():
            return Text("", style=MUTED)
        return Markdown(text, style="none" if not self.style else self.style)

    def feed(self, chunk: str) -> None:
        self.buffer += chunk
        if not self.console.is_terminal:
            # Piped output: keep it greppable, no escape sequences.
            self.console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)
            return
        now = time.monotonic()
        if self._live is None:
            self._live = Live(
                self._body(),
                console=self.console,
                refresh_per_second=8,
                vertical_overflow="visible",
            )
            self._live.start(refresh=True)
            self._last = now
        elif now - self._last >= STREAM_MIN_INTERVAL:
            self._live.update(self._body(), refresh=True)
            self._last = now

    def finish(self) -> str:
        if self._live is not None:
            self._live.update(self._body(), refresh=True)
            self._live.stop()
            self._live = None
        elif not self.console.is_terminal:
            self.console.print()
        return self.buffer

    def abort(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None


class Progress:
    """Live feedback for a command that takes longer than a moment.

    Rich suppresses its spinner when output is piped, so this prints plain
    one-line steps instead - a piped run should still read like a transcript
    rather than silence followed by a table.
    """

    def __init__(self, console: Console, label: str) -> None:
        self.console = console
        self.label = label
        self._status: Any = None

    def __enter__(self) -> Progress:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def _spin(self) -> None:
        self._status = self.console.status(Text("  " + self.label, style=MUTED), spinner="dots")
        self._status.start()

    def start(self) -> None:
        if self.console.is_terminal:
            self._spin()
        else:
            self.console.print(Text(self.label, style=MUTED))

    def update(self, label: str) -> None:
        """Change the spinner text without printing a line."""
        self.label = label
        if self._status is not None:
            self._status.update(Text("  " + label, style=MUTED))

    def step(self, line: str, style: str = MUTED) -> None:
        """Print a finished step, keeping the spinner running underneath."""
        if self._status is not None:
            self._status.stop()
            self._status = None
        self.console.print(Text(line, style=style))
        if self.console.is_terminal:
            self._spin()

    def stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None


class RichUI(UI):
    """Streams a turn to the terminal: spinner, markdown answer, tool lines."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._stream: StreamRenderer | None = None
        self._spinner: Any = None
        self._thinking = False
        self._reasoning_header = False
        self._needs_gap = False
        # The one-line preview is not enough when a command fails inside a long
        # output: /show replays the last few results in full.
        self.recent: deque[tuple[str, ToolResult]] = deque(maxlen=5)

    # --- lifecycle -----------------------------------------------------
    def turn_start(self, label: str = "thinking…") -> None:
        self._stop_spinner()
        if self.console.is_terminal:
            self._spinner = self.console.status(Text(label, style=MUTED), spinner="dots")
            self._spinner.start()

    def _stop_spinner(self) -> None:
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def turn_end(self, usage=None, elapsed: float = 0.0) -> None:
        self.newline()
        self._stop_spinner()
        if not self.console.is_terminal or usage is None:
            return
        parts = [f"{elapsed:.1f}s"]
        if getattr(usage, "completion_tokens", 0):
            parts.append(f"{usage.completion_tokens} tok")
        if getattr(usage, "tok_per_second", 0.0):
            parts.append(f"{usage.tok_per_second:.0f} tok/s")
        self.console.print(Text("  " + " · ".join(parts), style=MUTED))

    def interrupt(self) -> None:
        self._needs_gap = False
        if self._stream is not None:
            self._stream.abort()
            self._stream = None
        self._stop_spinner()
        self._thinking = self._reasoning_header = False

    # --- streaming -----------------------------------------------------
    def token(self, text: str) -> None:
        self._stop_spinner()
        self._close_reasoning()
        if self._stream is None:
            if self._needs_gap:
                self.console.print()
                self._needs_gap = False
            self._stream = StreamRenderer(self.console)
        self._stream.feed(text)

    def _close_stream(self) -> None:
        if self._stream is not None:
            self._stream.finish()
            self._stream = None

    def reasoning(self, text: str) -> None:
        self._stop_spinner()
        self._close_stream()
        if not self._reasoning_header:
            self.console.print(Text("· thinking", style=MUTED))
            self._reasoning_header = True
            self._thinking = True
        self.console.print(Text(text, style="italic " + MUTED), end="", soft_wrap=True)

    def _close_reasoning(self) -> None:
        if self._thinking:
            self.console.print()
            self._thinking = self._reasoning_header = False

    def newline(self) -> None:
        self._close_stream()
        self._close_reasoning()

    # --- tool + status -------------------------------------------------
    def status(self, text: str) -> None:
        self.newline()
        self.console.print(Text(f"· {text}", style=MUTED))

    def tool_call(self, name: str, args: str) -> None:
        self.newline()
        detail = summarize_args(name, args)
        line = Text("  ")
        line.append("⚙ ", style=ACCENT)
        line.append(name, style=f"bold {ACCENT}")
        if detail:
            line.append("  " + detail, style="default")
        self.console.print(line, overflow="ellipsis", no_wrap=True)

    def tool_result(self, name: str, result: ToolResult) -> None:
        self.recent.append((name, result))
        preview, more = result_preview(result)
        line = Text("    ")
        line.append("✓ " if result.ok else "✗ ", style=OK if result.ok else ERR)
        line.append(preview, style="" if result.ok else ERR)
        if more:
            line.append(f"  +{more} lines", style=MUTED)
            line.append(" · /show", style=MUTED)
        # A failure is the one line the user must not lose to an ellipsis, so it
        # wraps instead of being cut off.
        self.console.print(line, overflow="fold" if not result.ok else "ellipsis",
                           no_wrap=result.ok)
        self._needs_gap = True

    def recent_results(self) -> list[tuple[str, ToolResult]]:
        """Most recent tool result first."""
        return list(reversed(self.recent))


# --- static pieces -----------------------------------------------------


def banner(
    console: Console,
    *,
    name: str,
    model: str,
    ctx: int,
    threads: int,
    ram: float,
    memories: int,
    url: str,
    workspace: str = "",
    prompt: str = "",
) -> None:
    title = Text(name, style=f"bold {ACCENT}")
    facts = (
        f"{ctx} ctx · {threads} threads · {ram:g} GB RAM · "
        f"{memories} {'memory' if memories == 1 else 'memories'}"
    )
    if prompt:
        facts += f" · prompt {prompt}"
    lines = [Text(short_name(model) if model else "no model", style="bold"), Text(facts, style=MUTED)]
    if workspace:
        lines.append(Text(workspace, style=MUTED))
    lines.append(Text(f"{url} · /help for commands", style=MUTED))
    body = Group(*lines)
    console.print(
        Panel(body, title=title, title_align="left", border_style=ACCENT, box=ROUNDED, expand=False)
    )


def kv_table(console: Console, title: str, rows: list[tuple[str, str]]) -> None:
    """The one table style used by every /command."""
    table = Table(
        box=ROUNDED,
        border_style=MUTED,
        title=Text(title, style=f"bold {ACCENT}"),
        title_justify="left",
        show_header=False,
        padding=(0, 1),
        expand=False,
    )
    table.add_column(style=MUTED, no_wrap=True)
    table.add_column(overflow="fold")
    for key, value in rows:
        # Plain strings go through Rich's markup parser, so a journal hit called
        # "Re: [urgent] thing" rendered as "Re: thing".
        table.add_row(Text(str(key)), Text(str(value)))
    console.print(table)


def data_table(
    console: Console, title: str, columns: list[str], rows: list[list[str]], no_wrap: int = 1
) -> None:
    table = Table(
        box=ROUNDED,
        border_style=MUTED,
        title=Text(title, style=f"bold {ACCENT}"),
        title_justify="left",
        expand=False,
        header_style=MUTED,
    )
    for index, column in enumerate(columns):
        table.add_column(column, no_wrap=index < no_wrap, overflow="fold")
    for row in rows:
        table.add_row(*[Text(str(cell)) for cell in row])
    console.print(table)


def nearest_command(name: str) -> str:
    """Best guess for a mistyped command, or '' when nothing is close."""
    known = [command for command, _description in COMMANDS]
    if not name.startswith("/"):
        name = "/" + name
    matches = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    return matches[0] if matches else ""


def help_panel(console: Console, command: str = "") -> None:
    """All commands by group, or the detail for one."""
    wanted = command.strip()
    if wanted:
        name = wanted if wanted.startswith("/") else "/" + wanted
        known = dict(COMMANDS)
        if name not in known:
            hint = nearest_command(name)
            tail = f" - did you mean {hint}?" if hint else ""
            note(console, f"no such command {name}{tail}", WARN)
            return
        body = Text()
        body.append(known[name] + "\n", style="default")
        usage = COMMAND_USAGE.get(name)
        if usage:
            body.append("\n" + usage, style=MUTED)
        console.print(
            Panel(
                body,
                title=Text(name, style=f"bold {ACCENT}"),
                title_align="left",
                border_style=MUTED,
                box=ROUNDED,
                expand=False,
            )
        )
        return

    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column(style=f"bold {ACCENT}", no_wrap=True)
    table.add_column(style="default", overflow="fold")
    for group, items in COMMAND_GROUPS:
        table.add_row(Text(group, style=f"italic {MUTED}"), "")
        for item_command, description in items:
            table.add_row("  " + item_command, description)
    console.print(
        Panel(
            table,
            title=Text("commands", style=f"bold {ACCENT}"),
            title_align="left",
            border_style=MUTED,
            box=ROUNDED,
            expand=False,
            subtitle=Text(" /help <command> for usage · esc+enter for a new line ", style=MUTED),
            subtitle_align="right",
        )
    )


def error_panel(console: Console, text: str) -> None:
    console.print(
        Panel(
            Text(text, style=ERR),
            title=Text("error", style=f"bold {ERR}"),
            title_align="left",
            border_style=ERR,
            box=ROUNDED,
            expand=False,
        )
    )


def note(console: Console, text: str, style: str = MUTED) -> None:
    console.print(Text(text, style=style))


# --- prompt chrome -----------------------------------------------------


class SlashCompleter(Completer):
    """Complete /commands, and their arguments where the CLI can offer some.

    ``extra`` returns (fragment, note) pairs for the command being typed - the
    prompt versions for ``/prompt``, the command names for ``/help`` - so the
    user does not have to remember which v-numbers exist.
    """

    def __init__(self, extra: Any = None) -> None:
        self.extra = extra

    def get_completions(self, document: Any, complete_event: Any) -> Any:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        if " " not in text:
            for command, description in COMMANDS:
                if command.startswith(text):
                    yield Completion(
                        command, start_position=-len(text), display=command, display_meta=description
                    )
            return
        if self.extra is None:
            return
        word = text.split(" ")[-1]
        for fragment, meta in self.extra(text):
            if fragment.startswith(word):
                yield Completion(fragment, start_position=-len(word), display=fragment, display_meta=meta)


PROMPT_STYLE = Style.from_dict(
    {
        "prompt": f"{ACCENT} bold",
        "continuation": MUTED_HEX,
        "bottom-toolbar": f"bg:{TOOLBAR_BG} {MUTED_HEX}",
    }
)


def make_prompt_session(history_path, toolbar, completer: SlashCompleter | None = None) -> PromptSession:
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event) -> None:  # newline without submitting
        event.current_buffer.insert_text("\n")

    return PromptSession(
        history=FileHistory(str(history_path)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=completer or SlashCompleter(),
        complete_while_typing=True,
        style=PROMPT_STYLE,
        key_bindings=bindings,
        bottom_toolbar=toolbar,
        prompt_continuation=lambda width, line_number, is_soft_wrap: "… ".rjust(width),
    )


def prompt_glyph(width: int = 0) -> HTML:
    return HTML(f"<prompt>{PROMPT_GLYPH}</prompt> ")


def toolbar_text(*, ctx: int, memories: int, ram: float, shell_mode: str, prompt: str = "") -> str:
    mem = f"{memories} mem" if memories else "no memory"
    version = f" · prompt {prompt}" if prompt else ""
    return f" {ram:g}GB · {ctx} ctx · {mem}{version} · shell {shell_mode} · ctrl-c cancels · /help "
