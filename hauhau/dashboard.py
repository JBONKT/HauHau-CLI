"""A widget dashboard for the agent, built on Textual.

The other two interfaces exist for different reasons: the stream one is what a
pipe or an editor gets, and the prompt_toolkit one owns the terminal in place.
This one is for watching the agent work: the conversation on the left, live
numbers on the right, and a real scrollback you can drag.

Everything the agent does is funnelled through :class:`DashboardUI`, which
implements the same ``agent.UI`` protocol as ``ui.RichUI`` and ``tui.ScreenUI``
- so ``cli.py`` wires it up exactly like the others and no agent code knows
which front end is running.

Threading: a turn runs on a worker thread while Textual owns the event loop.
Every mutation of a widget therefore goes through :meth:`DashboardUI._call`,
which marshals to the app thread (``App.call_from_thread`` refuses to be called
*from* that thread, so the same-thread case is handled separately). Rich output
from ``/status`` and friends arrives through a ``rich`` Console writing into
:class:`_Sink`, which buffers until the app is mounted - the config warnings are
printed before there is anything to print them into.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any, ClassVar

from rich.console import Console as RichConsole
from rich.text import Text

try:  # the dashboard is an optional extra
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.css.query import NoMatches
    from textual.suggester import Suggester
    from textual.widgets import Footer, Header, Input, Markdown, Sparkline, Static
except ImportError as exc:  # pragma: no cover - exercised by the CLI message
    raise ImportError(
        "the dashboard needs textual: .venv/bin/pip install 'hauhau[tui]'"
    ) from exc

from . import suggest
from .agent import Usage
from .tools import ToolResult
from .ui import ACCENT, ERR, MUTED_HEX, OK, result_preview, summarize_args

# Textual CSS. Colours come from ui.py so all three interfaces look related.
CSS = f"""
Screen {{
    layers: base;
}}
#body {{
    height: 1fr;
}}
#chat {{
    width: 3fr;
    padding: 0 1;
    scrollbar-size-vertical: 1;
}}
#side {{
    width: 34;
    padding: 0 1;
    border-left: solid $panel;
}}
#side Static {{
    padding: 0 0 1 0;
}}
.user {{
    text-align: right;
    color: {ACCENT};
    text-style: bold;
    margin: 1 0 0 6;
}}
.tool {{
    color: {ACCENT};
    text-wrap: nowrap;
    text-overflow: ellipsis;
}}
.result {{
    color: $text-muted;
    text-wrap: nowrap;
    text-overflow: ellipsis;
}}
.failed {{
    color: {ERR};
}}
.note {{
    color: $text-muted;
}}
.out {{
    text-wrap: nowrap;
    text-overflow: ellipsis;
}}
.reasoning {{
    color: {MUTED_HEX};
    text-style: italic;
}}
#prompt {{
    dock: bottom;
}}
#speed {{
    height: 2;
    color: {ACCENT};
}}
"""

SPARK_POINTS = 40
# Idle time before the model is asked for a completion. A 2B model needs about a
# second; waiting for the typing to stop is what makes the answer arrive while it
# is still the answer to the current line.
AI_DELAY = 0.55
# Streaming: re-parse the answer at most this often, and only when it grew by
# enough to be visible. Markdown parsing is O(document), so re-running it for
# every token is quadratic - it cost 5% of a core while a long answer arrived.
STREAM_MIN_CHARS = 400  # a jump this large renders at once (first chunk, a paste)
STREAM_MAX_WAIT = 0.2  # otherwise at most 5 reparses a second


def _graphic(name: str) -> str | None:
    return shutil.which(name)


def write_clipboard(text: str) -> bool:
    """Copy text, on macOS through pbcopy as well as OSC 52.

    Textual's ``copy_to_clipboard`` speaks OSC 52, which Terminal.app ignores;
    ``pbcopy`` is what actually works there. Returns whether a system tool took
    it, so the UI can say "copied" honestly.
    """
    took = False
    tool = _graphic("pbcopy")
    if tool:
        try:
            subprocess.run([tool], input=text.encode("utf-8"), timeout=3, check=False)
            took = True
        except (OSError, subprocess.SubprocessError):
            pass
    return took


def read_clipboard() -> str:
    tool = _graphic("pbpaste")
    if not tool:
        return ""
    try:
        done = subprocess.run([tool], capture_output=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.decode("utf-8", "replace")


class _Sink(io.TextIOBase):
    """A ``rich`` Console output that lands in the chat log, line by line."""

    def __init__(self, ui: DashboardUI) -> None:
        self.ui = ui
        self.pending = ""

    def write(self, text: str) -> int:
        if "\r" in text:
            # A rich Live/Status frame: the spinner redraws itself many times a
            # second, and every repaint became its own chat widget (twelve for a
            # 0.8 s spinner). It is transient by design, so it is dropped;
            # Progress.step prints finished lines, and those carry a newline.
            return len(text)
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.ui.append_ansi(line)
        return len(text)

    def flush(self) -> None:
        if self.pending and "\r" not in self.pending:
            line, self.pending = self.pending, ""
            self.ui.append_ansi(line)


class DashboardUI:
    """The agent-facing UI: everything becomes a widget in the chat column."""

    def __init__(self) -> None:
        self.app: HauhauDashboard | None = None
        self.ui_thread_id: int | None = None
        self.recent: deque[tuple[str, ToolResult]] = deque(maxlen=5)
        self.busy = False
        self.busy_label = ""
        self.last_usage: Usage | None = None
        self.last_elapsed = 0.0
        # Kept so ctrl-y can copy the answer when nothing is selected.
        self.last_answer = ""
        # Tokens arrive faster than a Markdown re-parse is worth; they land here
        # and the app's timer flushes them at ~8 fps.
        self._answer: list[str] = []
        self._stream_widget: Any = None
        self._rendered_chars = 0
        self._rendered_at = 0.0
        # Seconds between two Markdown re-parses while an answer streams. Raised
        # for terminals that redraw on the CPU and cannot sync their repaints.
        self.stream_wait = STREAM_MAX_WAIT
        self._reasoning: list[str] = []
        self._reasoning_text = ""
        self._reasoning_widget: Any = None
        # Widgets built before the app is mounted (config warnings land here).
        # Stored as widgets, not text: an unmounted widget can still be updated
        # and is mounted with whatever it holds at that point.
        self._early: list[Any] = []
        self._sink = _Sink(self)
        self.console = RichConsole(
            file=self._sink,
            force_terminal=True,
            color_system="truecolor",
            width=100,
            soft_wrap=True,
        )

    # --- marshalling ------------------------------------------------------
    def _call(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the app thread, or now when there is no app yet."""
        app = self.app
        if app is None or not app.is_running or getattr(app, "_loop", None) is None:
            fn()  # not mounted yet: the widget calls buffer themselves
            return
        if threading.get_ident() == self.ui_thread_id:
            fn()
            return
        # The app may shut down between the check and the call.
        with contextlib.suppress(RuntimeError):
            app.call_from_thread(fn)

    def _chat(self) -> Any:
        app = self.app
        if app is None:
            return None
        try:
            return app.query_one("#chat", VerticalScroll)
        except NoMatches:
            # During shutdown the dashboard screen (and #chat with it) is
            # gone while the REPL still prints its /learn goodbyes; buffer
            # them instead of crashing the exit path.
            return None

    def _mount(self, widget: Any, *, scroll: bool = True) -> Any:
        chat = self._chat()
        if chat is None:
            self._early.append(widget)
            return widget
        chat.mount(widget)
        if scroll:
            self.app.call_after_refresh(lambda: chat.scroll_end(animate=False))
        return widget

    # --- the agent.UI protocol --------------------------------------------
    def turn_start(self, label: str = "thinking…") -> None:
        def draw() -> None:
            self.busy = True
            self.busy_label = label or "thinking…"
            self._close_stream()

        self._call(draw)

    def turn_end(self, usage: Usage | None = None, elapsed: float = 0.0) -> None:
        def draw() -> None:
            self.busy = False
            self.busy_label = ""
            if usage is not None:
                self.last_usage = usage
            self.last_elapsed = elapsed
            self._close_stream()

        self._call(draw)

    def token(self, text: str) -> None:
        # Appended from the worker thread; the timer renders it. The whole answer
        # is kept, not just the new chunk: a Markdown widget re-parses its full
        # source, so updating it with only the latest tokens would erase the rest.
        self._answer.append(text)

    def reasoning(self, text: str) -> None:
        # The model's visible thinking is not the answer; it renders into its own
        # dim block so a later Markdown re-parse cannot eat it.
        self._reasoning.append(text)

    def tool_call(self, name: str, args: str) -> None:
        detail = summarize_args(name, args)
        line = f"⚙ {name}" + (f"  {detail}" if detail else "")

        def draw() -> None:
            self._close_stream()
            self._mount(Static(Text(line, style=ACCENT), classes="tool"))

        self._call(draw)

    def tool_result(self, name: str, result: ToolResult) -> None:
        preview, more = result_preview(result)
        mark = "✓" if result.ok else "✗"
        # The hidden-line count goes *before* the preview: this row is clipped
        # with an ellipsis at the column width, and a suffix would be the first
        # thing the ellipsis eats.
        head = f"  {mark} (+{more} lines)" if more else f"  {mark}"
        budget = max(20, 92 - len(head))
        if len(preview) > budget:
            preview = preview[: budget - 1].rstrip() + "…"
        css = "result" if result.ok else "failed"

        def draw() -> None:
            self._close_stream()
            self.recent.append((name, result))
            self._mount(Static(Text(f"{head} {preview}", style=OK if result.ok else ERR), classes=css))

        self._call(draw)

    def status(self, text: str) -> None:
        def draw() -> None:
            self._close_stream()
            self._mount(Static(Text(f"· {text}", style=MUTED_HEX), classes="note"))

        self._call(draw)

    def note(self, text: str, error: bool = False) -> None:
        def draw() -> None:
            self._close_stream()
            mark = "✗" if error else "·"
            style = ERR if error else MUTED_HEX
            self._mount(Static(Text(f"{mark} {text}", style=style), classes="failed" if error else "note"))

        self._call(draw)

    def newline(self) -> None:
        self._call(self._close_stream)

    def interrupt(self) -> None:
        self._call(self._close_stream)

    def echo(self, text: str) -> None:
        def draw() -> None:
            self._close_stream()
            self._mount(Static(Text(text), classes="user"))

        self._call(draw)

    def append_ansi(self, text: str) -> None:
        """Rich panels, tables and help boxes, already rendered."""
        if not text.strip():
            return
        rendered = Text.from_ansi(text)

        def draw() -> None:
            self._mount(Static(rendered, markup=False, classes="out"))

        self._call(draw)

    def recent_results(self) -> list[tuple[str, ToolResult]]:
        return list(reversed(self.recent))

    # --- streaming --------------------------------------------------------
    def flush_stream(self, *, force: bool = False) -> None:
        """Render buffered tokens (called by the app's timer, on its thread)."""
        if self._reasoning:
            chat = self._chat()
            if chat is not None:
                if self._reasoning_widget is None:
                    self._reasoning_widget = self._mount(Static("· thinking", markup=False, classes="reasoning"))
                # Keep the whole block: showing only the newest chunk made the
                # thinking widget read like the last tenth of a second.
                self._reasoning_text += "".join(self._reasoning)
                self._reasoning.clear()
                self._reasoning_widget.update(Text(f"· thinking {self._reasoning_text}", style=MUTED_HEX))
        if not self._answer:
            return
        chat = self._chat()
        if chat is None:
            return
        text = "".join(self._answer)
        now = time.monotonic()
        if self._stream_widget is not None and not force:
            grew = len(text) - self._rendered_chars
            if grew < STREAM_MIN_CHARS and now - self._rendered_at < self.stream_wait:
                return
        self._rendered_chars = len(text)
        self._rendered_at = now
        if self._stream_widget is None:
            # Mounted with its first content already in place: a Markdown widget
            # created empty and updated before it is attached keeps the empty
            # source, so the answer would render blank.
            self._stream_widget = self._mount(Markdown(text), scroll=False)
        else:
            self._stream_widget.update(text)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        """Keep the newest line in view, without a layout pass when it already is."""
        app = self.app
        chat = self._chat()
        if app is None or chat is None:
            return
        if chat.is_vertical_scroll_end:
            app.call_after_refresh(lambda: chat.scroll_end(animate=False))

    def _close_stream(self) -> None:
        self.flush_stream(force=True)
        if self._answer:
            self.last_answer = "".join(self._answer).strip()
        self._answer.clear()
        self._stream_widget = None
        self._reasoning_widget = None
        self._reasoning_text = ""


class TaskSuggester(Suggester):
    """The instant half of the ghost text: commands and keyword continuations.

    Deliberately model-free. Textual runs this as a worker per keystroke, and a
    suggestion has to be there before the next character arrives; the model half
    is debounced by the app instead (:meth:`HauhauDashboard._ask_model`), because
    a request fired per keystroke is answered for a prefix the user has already
    typed past - and Textual drops a suggestion whose value no longer matches.
    """

    def __init__(self, app: HauhauDashboard) -> None:
        super().__init__(use_cache=True, case_sensitive=False)
        self.app = app

    async def get_suggestion(self, value: str) -> str | None:
        if not value.strip():
            self.app.set_suggestion("")
            return None
        quick = suggest.complete_command(value) or suggest.continue_task(value)
        if quick is not None:
            self.app.set_suggestion(quick.source)
            return quick.text
        self.app.set_suggestion("")
        return None


class HauhauDashboard(App):
    """The window: chat column, stats column, one input line."""

    CSS = CSS
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+q", "quit_app", "quit"),
        Binding("ctrl+c", "interrupt", "interrupt"),
        Binding("f2", "focus_prompt", "prompt"),
        # priority: the focused Input binds Tab (focus_next) and Ctrl-V (its own
        # paste from Textual's clipboard, which knows nothing about pbpaste).
        Binding("tab", "accept_suggestion", "accept", show=False, priority=True),
        Binding("ctrl+y", "copy", "copy", priority=True),
        Binding("ctrl+v", "paste", "paste", priority=True),
        Binding("pageup", "scroll_chat('up')", "scroll", show=False),
        Binding("pagedown", "scroll_chat('down')", "scroll", show=False),
    ]

    def __init__(
        self,
        ui: DashboardUI,
        *,
        submit: Callable[[str], None],
        restart_backend: Callable[[], Any],
        should_exit: Callable[[], bool],
        stats: Callable[[], dict[str, Any]],
        propose: Callable[[str], str | None] | None = None,
    ) -> None:
        super().__init__()
        self.ui = ui
        self.submit = submit
        self.restart_backend = restart_backend
        self.should_exit = should_exit
        self.stats = stats
        self.propose = propose
        self.suggestion_source = ""
        self._ai_timer: Any = None
        self._panel_drawn: tuple | None = None
        self._speeds: deque[float] = deque(maxlen=SPARK_POINTS)
        # The usage object the last sparkline sample came from: a repaint must
        # not add the same reading again.
        self._last_speed_usage: Any = None
        self.title = "Hauhau"
        ui.app = self

    # --- layout -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield VerticalScroll(id="chat")
            with Vertical(id="side"):
                yield Static("", id="model")
                yield Static("", id="runtime")
                yield Static("", id="speed-label")
                yield Sparkline([], id="speed", summary_function=max)
                yield Static("", id="memory")
                yield Static("", id="keys")
        yield Input(
            placeholder="write a message…  (/help lists commands)",
            suggester=TaskSuggester(self),
            id="prompt",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.ui.ui_thread_id = threading.get_ident()
        chat = self.query_one("#chat", VerticalScroll)
        for widget in self.ui._early:
            chat.mount(widget)
        self.ui._early.clear()
        self.query_one("#prompt", Input).focus()
        self._tick()  # draws once and records the signature, so idle is free
        self.set_interval(0.12, self._tick)
        if os.environ.get("TERM_PROGRAM") == "Apple_Terminal":
            # A CPU-rendered terminal also gets fewer, bigger frames: it has to
            # redraw every character we write, and it cannot batch them.
            self.ui.stream_wait = 0.4
            # Textual asks the terminal for DEC 2026 (synchronized output) and
            # skips Apple Terminal, which does not support it: every repaint is
            # written as several small diffs and re-rendered on the CPU, which is
            # where the remaining stickiness comes from. Nothing in the app can
            # fix that; a GPU terminal can.
            self.notify(
                "Apple Terminal cannot synchronise repaints - Ghostty, kitty or WezTerm "
                "will scroll and stream noticeably smoother.",
                title="slower terminal",
                severity="warning",
                timeout=8,
            )

    def _tick(self) -> None:
        """The 8 Hz clock: stream the answer, then repaint the panel if it changed.

        Nothing else repaints while the app is idle, and the panel's signature is
        compared first - six widget updates plus a layout pass, eight times a
        second, for numbers that only change between turns, cost ~6% of a core on
        an M1 Air and are why the dashboard felt sticky.
        """
        if not self.is_running:
            return
        self.ui.flush_stream()
        signature = self._panel_signature()
        if signature == self._panel_drawn:
            return
        self._panel_drawn = signature
        self._refresh_stats()

    # --- suggestions ------------------------------------------------------
    def set_suggestion(self, source: str) -> None:
        """Record where the current ghost text came from, for the panel."""
        self.suggestion_source = source

    # --- the side panel ---------------------------------------------------
    def _panel_signature(self) -> tuple:
        """Everything the panel shows, cheaply comparable."""
        try:
            stats = self._stats()
        except Exception:
            stats = {}
        usage = self.ui.last_usage
        return (
            stats.get("model"),
            stats.get("backend"),
            stats.get("ctx"),
            stats.get("threads"),
            stats.get("prompt"),
            stats.get("memories"),
            stats.get("journal"),
            stats.get("tuner"),
            stats.get("session"),
            stats.get("workspace"),
            self.ui.busy,
            self.ui.busy_label,
            self.ui.last_elapsed,
            usage.tok_per_second if usage else None,
            usage.completion_tokens if usage else None,
            usage.prompt_tokens if usage else None,
            self.suggestion_source,
            len(self._speeds),
        )

    def _stats(self) -> dict[str, Any]:
        try:
            return self.stats()
        except Exception:
            return {}

    def _refresh_stats(self) -> None:
        """Redraw the numbers column, once, when something in it changed."""
        if not self.is_running:
            return
        stats = self._stats()
        usage = self.ui.last_usage
        if usage is not None and usage.tok_per_second and usage is not self._last_speed_usage:
            # Once per finished turn: appending on every repaint filled the
            # sparkline with forty copies of the same reading.
            self._last_speed_usage = usage
            self._speeds.append(float(usage.tok_per_second))

        speed = f"{usage.tok_per_second:.0f} tok/s" if usage and usage.tok_per_second else "-"
        took = f"{self.ui.last_elapsed:.1f}s" if self.ui.last_elapsed else "-"
        state = self.ui.busy_label if self.ui.busy else "ready"
        rows: list[tuple[str, str]] = [
            ("context", str(stats.get("ctx", "-"))),
            ("threads", str(stats.get("threads", "-"))),
            ("state", state),
            ("last turn", f"{took} · {speed}"),
            ("prompt", str(stats.get("prompt", "-"))),
            ("suggest", self.suggestion_source or "-"),
        ]
        facts: list[tuple[str, str]] = [
            ("memories", str(stats.get("memories", "-"))),
            ("journal", str(stats.get("journal", "-"))),
            ("tuner", str(stats.get("tuner", "-"))),
            ("session", str(stats.get("session", "-"))),
            ("workdir", str(stats.get("workspace", "-"))),
        ]
        keys = Text()
        for key, what in (
            ("ctrl-q", "quit"),
            ("ctrl-c", "interrupt"),
            ("f2", "prompt"),
            ("tab", "accept"),
            ("ctrl-y", "copy"),
            ("ctrl-v", "paste"),
            ("pgup/pgdn", "scroll"),
        ):
            keys.append(key + " ", style=f"bold {ACCENT}")
            keys.append(what + "\n", style=MUTED_HEX)

        try:
            # One batch, no layout pass: the column has a fixed width, so only the
            # characters change. Six separate updates each triggered a screen
            # layout, which is most of what made an idle dashboard cost 6% CPU.
            with self.batch_update():
                self.query_one("#model", Static).update(
                    Text.assemble(
                        ("Hauhau\n", f"bold {ACCENT}"),
                        (str(stats.get("model", "-")), "bold"),
                        ("\n", ""),
                        (str(stats.get("backend", "")), MUTED_HEX),
                    ),
                    layout=False,
                )
                self.query_one("#runtime", Static).update(
                    Text("\n".join(f"{label:<10}{value}" for label, value in rows)), layout=False
                )
                self.query_one("#memory", Static).update(
                    Text("\n".join(f"{label:<10}{value}" for label, value in facts)), layout=False
                )
                self.query_one("#keys", Static).update(keys, layout=False)
                spark = self.query_one("#speed", Sparkline)
                data = list(self._speeds)
                if list(spark.data) != data:
                    spark.data = data
        except NoMatches:
            # The app is shutting down and the widgets are already gone; a pending
            # timer callback must not raise into the event loop.
            return

    # --- input ------------------------------------------------------------
    def on_input_changed(self, event: Input.Changed) -> None:
        """Restart the model-completion clock on every keystroke."""
        if self._ai_timer is not None:
            self._ai_timer.stop()
            self._ai_timer = None
        if self.propose is None or not suggest.should_ask(event.value):
            return
        # A keyword answer is already on the line; asking the model as well would
        # spend a second of CPU to overwrite a suggestion that cost nothing.
        if suggest.complete_command(event.value) or suggest.continue_task(event.value):
            return
        self._ai_timer = self.set_timer(AI_DELAY, lambda: self._ask_model(event.value))

    def _ask_model(self, value: str) -> None:
        self._ai_timer = None
        field = self.query_one("#prompt", Input)
        if field.value != value:
            return  # already out of date by the time the timer fired
        self.run_worker(self._model_suggestion(value), group="suggest", exclusive=True)

    async def _model_suggestion(self, value: str) -> None:
        try:
            # Blocking request on a thread: the prompt line keeps drawing.
            answer = await asyncio.to_thread(self.propose, value)
        except Exception:
            return
        if not answer:
            return
        field = self.query_one("#prompt", Input)
        if field.value != value:
            return  # typed past it; a newer call owns the line now
        field._suggestion = answer
        self.set_suggestion("ai")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        field = self.query_one("#prompt", Input)
        field.value = ""
        if not text:
            return
        self.ui.echo(text)
        self._set_busy(True)
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _set_busy(self, busy: bool) -> None:
        if busy:
            self.set_suggestion("")
        field = self.query_one("#prompt", Input)
        field.disabled = busy
        field.placeholder = "working… (ctrl-c interrupts)" if busy else "write a message…  (/help lists commands)"

    def _run(self, text: str) -> None:
        try:
            self.submit(text)
        except Exception as exc:  # a worker thread would otherwise swallow it
            self.ui.note(f"turn failed: {exc.__class__.__name__}: {exc}", error=True)
        finally:
            self.call_from_thread(self._turn_done)

    def _turn_done(self) -> None:
        self._set_busy(False)
        self.query_one("#prompt", Input).focus()
        self.ui.flush_stream()
        if self.should_exit():
            self.exit()

    # --- actions ----------------------------------------------------------
    def action_quit_app(self) -> None:
        self.exit()

    def action_focus_prompt(self) -> None:
        self.query_one("#prompt", Input).focus()

    def action_interrupt(self) -> None:
        if self.ui.busy:
            # Restarting the backend aborts the in-flight request; it blocks, so
            # it cannot run on the event loop. Only claim what actually happened.
            def stop() -> None:
                if self.restart_backend():
                    self.ui.note("interrupted - the backend was restarted", error=False)
                else:
                    self.ui.note("interrupted - the answer may still be generating", error=False)

            threading.Thread(target=stop, daemon=True).start()
            return
        field = self.query_one("#prompt", Input)
        if field.value:
            field.value = ""

    def action_accept_suggestion(self) -> None:
        """Tab takes the ghost text; Right already does this (Textual's own)."""
        field = self.query_one("#prompt", Input)
        ghost = getattr(field, "_suggestion", "") or ""
        if len(ghost) > len(field.value):
            field.value = ghost
            field.cursor_position = len(ghost)
            self.set_suggestion("")

    def action_copy(self) -> None:
        """Copy the selection, or the last answer when nothing is selected."""
        selected = ""
        with contextlib.suppress(Exception):
            selected = self.screen.get_selected_text() or ""
        text = selected or self.ui.last_answer
        if not text:
            self.notify("nothing to copy", severity="warning", timeout=2)
            return
        self.copy_to_clipboard(text)  # OSC 52: iTerm2, WezTerm, Ghostty, kitty
        took = write_clipboard(text)  # pbcopy: includes Terminal.app
        what = "selection" if selected else "last answer"
        self.notify(f"copied {len(text)} chars from the {what}" + ("" if took else " (OSC 52 only)"), timeout=2)

    def action_paste(self) -> None:
        """Insert the system clipboard at the caret, as one line."""
        text = read_clipboard()
        if not text:
            self.notify("clipboard is empty", severity="warning", timeout=2)
            return
        field = self.query_one("#prompt", Input)
        flat = " ".join(text.split("\n")).strip()
        position = field.cursor_position
        value = field.value
        field.value = value[:position] + flat + value[position:]
        field.cursor_position = position + len(flat)
        field.focus()

    def action_scroll_chat(self, direction: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        if direction == "up":
            chat.scroll_page_up(animate=False)
        else:
            chat.scroll_page_down(animate=False)


def available() -> bool:
    """Whether textual is importable, for the CLI's error message."""
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True

