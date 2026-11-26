"""Fixed-size full-screen interface: status header, chat window, input line.

The stream interface prints and the terminal scrolls: a few turns later the top
of the conversation is gone. This module is the alternative for a real
terminal - the application owns exactly ``$LINES`` rows and never writes past
them:

    row 1      header   model · context · threads · RAM · workspace
    row 2      status   working/thinking or the last turn's speed and tokens
    middle     chat     the conversation, scrolled *inside* its own window
    row -2     hint     the keys worth knowing
    row -1     input    the prompt, always the last line

Everything the agent renders (streaming answers, tool lines, rich panels from
``/status`` and friends) is captured as styled text and appended to the chat
window, so no command can push the layout off the screen. The chat follows the
newest line; PageUp/PageDown scroll inside the window; typing snaps back to
the bottom.

A turn runs on a worker thread so the interface keeps drawing while the model
streams, and every fragment is marshalled back to the UI thread (the layout
must never read a list that another thread is changing). Ctrl-C during a turn
restarts the backend - the same recovery the stream interface uses - and
Ctrl-D or ``/exit`` leaves.
"""

from __future__ import annotations

import io
import re
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style
from rich.console import Console as RichConsole

from .agent import Usage
from .tools import ToolResult
from .ui import ACCENT, MUTED_HEX, result_preview, summarize_args

# prompt_toolkit parses its own inline styles: hex colours and "ansi..." names.
# Rich's names ("green", "grey58") are rejected here - they became, of all
# things, a crash in the renderer.
STYLE_OK = "ansigreen"
STYLE_ERR = "ansired"
STYLE_MUTED = MUTED_HEX

STYLE_USER = "bold"
STYLE_USER_MARK = f"fg:{ACCENT}"
STYLE_ANSWER = ""
STYLE_TOOL = f"fg:{ACCENT}"
STYLE_TOOL_OUT = f"fg:{STYLE_MUTED}"
STYLE_REASONING = f"italic fg:{STYLE_MUTED}"
STYLE_NOTE = f"fg:{STYLE_MUTED}"
STYLE_ERROR = STYLE_ERR
STYLE_HEADER = f"bold fg:{ACCENT}"
STYLE_STATUS = f"fg:{STYLE_MUTED}"
STYLE_PROMPT = f"bold fg:{ACCENT}"

_TOKEN = re.compile(r"\S+\s*|\s+")


class _Wrapper:
    """Incremental word wrap of the transcript.

    The chat window grows by one fragment per streamed token, so re-wrapping
    the whole history on every repaint would be quadratic; only new fragments
    are fed in. ``prompt_toolkit`` ignores ``get_vertical_scroll`` on windows
    with ``wrap_lines=True`` (``_scroll_when_linewrapping`` never consults it),
    so wrapping and slicing happen here and the window renders one screenful.
    """

    def __init__(self) -> None:
        self.width = 0
        self.consumed = 0
        self.lines: list[list[tuple[str, str]]] = [[]]
        self.col = 0

    def reset(self) -> None:
        self.width = 0
        self.consumed = 0
        self.lines = [[]]
        self.col = 0

    def wrap(self, fragments: list[tuple[str, str]], width: int) -> list[list[tuple[str, str]]]:
        if width != self.width or len(fragments) < self.consumed:
            self.reset()
            self.width = width
        if len(fragments) > self.consumed:
            self._feed(fragments[self.consumed :])
            self.consumed = len(fragments)
        return self.lines

    # --- internals --------------------------------------------------------
    def _trim(self) -> None:
        """Drop the spaces a wrap left dangling at the end of the line."""
        line = self.lines[-1]
        while line:
            style, text = line[-1]
            stripped = text.rstrip(" ")
            if stripped == text:
                break
            self.col -= len(text) - len(stripped)
            if stripped:
                line[-1] = (style, stripped)
                break
            line.pop()

    def _newline(self) -> None:
        self._trim()
        self.lines.append([])
        self.col = 0

    def _emit(self, style: str, text: str) -> None:
        if not text:
            return
        line = self.lines[-1]
        if line and line[-1][0] == style:
            line[-1] = (style, line[-1][1] + text)
        else:
            line.append((style, text))
        self.col += len(text)

    def _feed(self, fragments: list[tuple[str, str]]) -> None:
        for style, text in fragments:
            for index, part in enumerate(text.split("\n")):
                if index:
                    self._newline()
                for token in _TOKEN.findall(part):
                    word = token.rstrip()
                    pad = token[len(word) :]
                    if word:
                        self._wrap_word(style, word)
                    if pad and self.col + len(pad) <= self.width:
                        self._emit(style, pad)

    def _wrap_word(self, style: str, word: str) -> None:
        if self.col and self.col + len(word) > self.width:
            word = word.lstrip()
            self._newline()
        while len(word) > self.width:  # a word longer than the window
            take = max(1, self.width - self.col)
            self._emit(style, word[:take])
            word = word[take:]
            self._newline()
        self._emit(style, word)



class _Router(io.TextIOBase):
    """Feeds a rich Console's output into the chat window, line by line."""

    def __init__(self, ui: ScreenUI) -> None:
        self.ui = ui
        self.pending = ""

    def write(self, text: str) -> int:
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.ui.append_ansi(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self.pending:
            line, self.pending = self.pending, ""
            self.ui.append_ansi(line)


class ScreenUI:
    """The agent-facing UI, plus the live state the full screen renders.

    All public methods may be called from the worker thread; every mutation of
    ``fragments`` happens on the UI thread via :meth:`_call`, because the layout
    reads the list while it draws.
    """

    def __init__(self) -> None:
        self.fragments: list[tuple[str, str]] = []
        # The five most recent tool results, for /show.
        self.recent: deque[tuple[str, ToolResult]] = deque(maxlen=5)
        self.busy = False
        self.busy_label = ""
        self.last_usage: Usage | None = None
        self.last_elapsed = 0.0
        self.follow = True
        self.scroll_offset = 0
        self.app: Application | None = None
        self.ui_thread_id: int | None = None
        self._stream_open = False
        self._reasoning_open = False

        self._router = _Router(self)
        self.console = RichConsole(
            file=self._router,
            force_terminal=True,
            color_system="truecolor",
            width=100,
            soft_wrap=True,
        )

    # --- routing between the worker thread and the UI thread --------------
    def _call(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` now when called on the UI thread, else marshal it there."""
        app = self.app
        if app is None or not app.is_running or app.loop is None:
            fn()  # app not running yet (or already closed): nothing to marshal to
            return
        if threading.get_ident() == self.ui_thread_id:
            # call_soon_threadsafe from inside the event loop deadlocks; the
            # accept handler and the typing callbacks live on that thread.
            fn()
        else:
            # prompt_toolkit 3.0.53 dropped Application.call_from_executor;
            # the app's asyncio loop is the supported way to marshal over.
            app.loop.call_soon_threadsafe(fn)
        app.invalidate()  # thread-safe repaint trigger

    def append_ansi(self, text: str) -> None:
        """Append pre-rendered ANSI output (rich panels, tables, markdown)."""
        if not text:
            return
        parsed = [(("" if frag[0] is None else str(frag[0])), frag[1]) for frag in to_formatted_text(ANSI(text))]
        self._call(lambda: self._extend(parsed))

    def _extend(self, fragments: list[tuple[str, str]]) -> None:
        self.fragments.extend(fragments)

    def _push(self, style: str, text: str) -> None:
        self.fragments.append((style, text))

    def _at_line_start(self) -> bool:
        """True when the next fragment would start a fresh line."""
        for _style, text in reversed(self.fragments):
            if text:
                return text.endswith("\n")
        return True

    def _break_line(self) -> None:
        """Open a new line, unless the last fragment already did."""
        if not self._at_line_start():
            self._push("", "\n")

    def _close_block(self) -> None:
        if self._stream_open or self._reasoning_open:
            self._push("", "\n")
        self._stream_open = self._reasoning_open = False

    def invalidate(self) -> None:
        app = self.app
        if app is not None and app.is_running:
            app.invalidate()

    # --- the agent.UI protocol --------------------------------------------
    def turn_start(self, label: str = "thinking…") -> None:
        def draw() -> None:
            self._close_block()
            self.busy_label = label or "thinking…"
            self.busy = True
            self.follow = True

        self._call(draw)
        self.invalidate()

    def turn_end(self, usage: Usage | None = None, elapsed: float = 0.0) -> None:
        def draw() -> None:
            self._close_block()
            self.busy = False
            self.busy_label = ""
            if usage is not None:
                self.last_usage = usage
            self.last_elapsed = elapsed

        self._call(draw)
        self.invalidate()

    def token(self, text: str) -> None:
        def draw() -> None:
            if not self._stream_open:
                self._close_block()
                self._break_line()
            self._stream_open = True
            self._push(STYLE_ANSWER, text)

        self._call(draw)
        self.invalidate()

    def reasoning(self, text: str) -> None:
        def draw() -> None:
            if not self._reasoning_open:
                self._close_block()
                self._break_line()
                self._push(STYLE_REASONING, "· thinking ")
                self._reasoning_open = True
            self._push(STYLE_REASONING, text)

        self._call(draw)
        self.invalidate()

    def tool_call(self, name: str, args: str) -> None:
        detail = summarize_args(name, args)

        def draw() -> None:
            self._close_block()
            self._break_line()
            self._push(STYLE_TOOL, f"  ⚙ {name}")
            if detail:
                self._push("", f"  {detail}")

        self._call(draw)
        self.invalidate()

    def tool_result(self, name: str, result: ToolResult) -> None:
        preview, more = result_preview(result)

        def draw() -> None:
            self._close_block()
            self.recent.append((name, result))
            self._break_line()
            self._push(STYLE_OK if result.ok else STYLE_ERR, f"    {'✓' if result.ok else '✗'} ")
            self._push(STYLE_TOOL_OUT if result.ok else STYLE_ERROR, preview)
            if more:
                self._push(f"fg:{STYLE_MUTED}", f"  +{more} lines · /show")

        self._call(draw)
        self.invalidate()

    def status(self, text: str) -> None:
        def draw() -> None:
            self._close_block()
            self._break_line()
            self._push(STYLE_NOTE, f"· {text}")

        self._call(draw)
        self.invalidate()

    def note(self, text: str, error: bool = False) -> None:
        def draw() -> None:
            self._close_block()
            mark = "✗" if error else "·"
            self._break_line()
            self._push(STYLE_ERROR if error else STYLE_NOTE, f"{mark} {text}")

        self._call(draw)
        self.invalidate()

    def newline(self) -> None:
        self._call(self._close_block)
        self.invalidate()

    def interrupt(self) -> None:
        self._call(self._close_block)
        self.invalidate()

    def echo(self, text: str) -> None:
        def draw() -> None:
            self._close_block()
            self._push(STYLE_USER_MARK, "\n❯ ")  # noqa: RUF001 - the prompt glyph
            self._push(STYLE_USER, text)
            self._push("", "\n")

        self._call(draw)
        self.invalidate()

    def recent_results(self) -> list[tuple[str, ToolResult]]:
        """Most recent tool result first, for /show."""
        return list(reversed(self.recent))


class ScreenApp:
    """Draws :class:`ScreenUI` as a fixed-size full-screen layout."""

    def __init__(
        self,
        ui: ScreenUI,
        *,
        submit: Callable[[str], None],
        restart_backend: Callable[[], Any],
        should_exit: Callable[[], bool],
        stats: Callable[[], dict[str, Any]],
    ) -> None:
        self.ui = ui
        self.submit = submit
        self.restart_backend = restart_backend
        self.should_exit = should_exit
        self.stats = stats
        self._chat_height = (0, 0)  # (wrapped lines, window rows) of the last frame
        self._wrapper = _Wrapper()

        # multiline must be False: prompt_toolkit 3.0.53 defaults Buffer to
        # multiline, which makes Enter insert a newline and disables the
        # built-in accept-line binding.
        self.input_buffer = Buffer(
            accept_handler=self._on_enter, on_text_changed=self._on_typed, multiline=False
        )

        keys = KeyBindings()

        @keys.add("c-d")
        def exit_now(_event: KeyPressEvent) -> None:
            self._exit()

        @keys.add("c-c")
        def interrupt(_event: KeyPressEvent) -> None:
            if self.ui.busy:
                # The stream interface recovers from a stuck turn by restarting
                # the backend, which aborts the in-flight request; the worker
                # thread's error path prints the note. Restarting blocks, so it
                # happens on yet another thread.
                threading.Thread(target=self.restart_backend, daemon=True).start()
            elif self.input_buffer.text:
                self.input_buffer.text = ""
            else:
                self._exit()

        @keys.add("pageup")
        @keys.add("c-b")
        def page_up(_event: KeyPressEvent) -> None:
            self._scroll_page(+1)

        @keys.add("pagedown")
        @keys.add("c-f")
        def page_down(_event: KeyPressEvent) -> None:
            self._scroll_page(-1)

        @keys.add("up")
        def line_up(_event: KeyPressEvent) -> None:
            self._scroll_lines(+1)

        @keys.add("down")
        def line_down(_event: KeyPressEvent) -> None:
            self._scroll_lines(-1)

        header_window = Window(
            FormattedTextControl(self._header, show_cursor=False),
            height=1,
            dont_extend_height=True,
            wrap_lines=False,
        )
        status_window = Window(
            FormattedTextControl(self._status, show_cursor=False),
            height=1,
            dont_extend_height=True,
            wrap_lines=False,
        )
        self.chat_window = Window(
            FormattedTextControl(self._chat_view, show_cursor=False),
            wrap_lines=False,
        )
        hint_window = Window(
            FormattedTextControl(self._hint, show_cursor=False),
            height=1,
            dont_extend_height=True,
            wrap_lines=False,
        )
        glyph_window = Window(
            FormattedTextControl(lambda: "… " if self.ui.busy else "❯ "),  # noqa: RUF001 - the prompt glyph
            width=2,
            wrap_lines=False,
        )
        input_window = Window(BufferControl(buffer=self.input_buffer), wrap_lines=False, dont_extend_height=True)
        body = HSplit(
            [header_window, status_window, self.chat_window, hint_window, VSplit([glyph_window, input_window])]
        )

        self.app = Application(
            layout=Layout(body),
            key_bindings=keys,
            full_screen=True,
            style=Style({}),
        )
        ui.app = self.app
        self.app.layout.focus(input_window)

    # --- input handling ---------------------------------------------------
    def _set_read_only(self, flag: bool) -> None:
        # prompt_toolkit stores read_only as a callable filter; a plain bool
        # would break filter evaluation on every keystroke.
        self.input_buffer.read_only = to_filter(flag)

    def _on_enter(self, buffer: Buffer) -> bool:
        text = buffer.text.strip()
        if not text:
            return False
        buffer.text = ""
        self.ui.echo(text)
        self._set_read_only(True)
        threading.Thread(target=self._run_turn, args=(text,), daemon=True).start()
        return False

    def _run_turn(self, text: str) -> None:
        try:
            self.submit(text)
        except Exception as exc:  # the worker thread would otherwise swallow it
            self.ui.note(f"turn failed: {exc.__class__.__name__}: {exc}", error=True)
        finally:
            self.ui._call(self._turn_done)

    def _turn_done(self) -> None:
        self._set_read_only(False)
        self.invalidate()
        if self.should_exit():
            self.app.exit()

    def _on_typed(self, _buffer: Buffer) -> None:
        if not self.ui.follow:
            self.ui.follow = True
            self.ui.scroll_offset = 0
        self.invalidate()

    def _exit(self) -> None:
        self._set_read_only(True)
        self.app.exit()

    def invalidate(self) -> None:
        app = self.ui.app
        if app is not None and app.is_running:
            app.invalidate()

    # --- chat scrolling: inside the window, never the terminal -----------
    def _output_size(self) -> tuple[int, int]:
        """``(columns, rows)`` of the terminal, with a fallback for odd ttys."""
        try:
            size = self.app.output.get_size() if self.app is not None else None
            if size is not None and size.columns > 0 and size.rows > 0:
                return size.columns, size.rows
        except Exception:
            pass
        return 100, 24

    def _chat_rows(self) -> int:
        """Rows left for the transcript once the chrome is taken out."""
        return max(3, self._output_size()[1] - 4)

    def _chat_view(self) -> list[tuple[str, str]]:
        """The one screenful of transcript the window shows."""
        width = self._output_size()[0]
        height = self._chat_rows()
        lines = self._wrapper.wrap(self.ui.fragments, max(8, width))
        self._chat_height = (len(lines), height)
        bottom = max(0, len(lines) - height)
        start = bottom if self.ui.follow else max(0, min(self.ui.scroll_offset, bottom))
        visible: list[tuple[str, str]] = []
        for index, line in enumerate(lines[start : start + height]):
            if index:
                visible.append(("", "\n"))
            visible.extend(line)
        return visible

    def _bottom(self) -> int:
        content, window = self._chat_height
        return max(0, content - window)

    def _leave_follow(self) -> None:
        """Stop following the newest line, remembering where it was."""
        if self.ui.follow:
            self.ui.follow = False
            self.ui.scroll_offset = self._bottom()

    def _scroll_page(self, direction: int) -> None:
        """+1 towards older lines, -1 towards the newest."""
        page = max(1, self._chat_height[1] - 2)
        if direction > 0:
            self._leave_follow()
            self.ui.scroll_offset = max(0, self.ui.scroll_offset - page)
        elif not self.ui.follow:
            self.ui.scroll_offset = min(self._bottom(), self.ui.scroll_offset + page)
            if self.ui.scroll_offset >= self._bottom():
                self.ui.follow = True
                self.ui.scroll_offset = 0
        self.invalidate()

    def _scroll_lines(self, direction: int) -> None:
        """+1 towards older lines, -1 towards the newest, one line at a time."""
        if direction > 0:
            self._leave_follow()
            self.ui.scroll_offset = max(0, self.ui.scroll_offset - direction)
        elif not self.ui.follow:
            self.ui.scroll_offset = min(self._bottom(), self.ui.scroll_offset - direction)
            if self.ui.scroll_offset >= self._bottom():
                self.ui.follow = True
                self.ui.scroll_offset = 0
        self.invalidate()

    # --- the chrome -------------------------------------------------------
    def _header(self) -> list[tuple[str, str]]:
        try:
            stats = self.stats()
        except Exception:
            stats = {}
        parts: list[tuple[str, str]] = [(STYLE_HEADER, f" {stats.get('name', 'Hauhau')} ")]
        for label, value in (
            ("model", stats.get("model")),
            ("ctx", stats.get("ctx")),
            ("threads", stats.get("threads")),
            ("ram", stats.get("ram")),
            ("dir", stats.get("workspace")),
        ):
            if value:
                parts.append((STYLE_STATUS, f"· {label} {value} "))
        return parts

    def _status(self) -> list[tuple[str, str]]:
        ui = self.ui
        if ui.busy:
            return [
                (f"bold fg:{ACCENT}", f" {ui.busy_label or 'working…'}"),
                (STYLE_STATUS, " · Ctrl-C restarts the backend"),
            ]
        usage = ui.last_usage
        metrics: list[str] = []
        if usage is not None:
            if ui.last_elapsed:
                metrics.append(f"{ui.last_elapsed:.1f}s")
            if usage.completion_tokens:
                metrics.append(f"{usage.completion_tokens} tok")
            if usage.tok_per_second:
                metrics.append(f"{usage.tok_per_second:.0f} tok/s")
            if usage.prompt_tokens:
                metrics.append(f"prompt {usage.prompt_tokens}")
        parts: list[tuple[str, str]] = (
            [(STYLE_STATUS, " " + " · ".join(metrics))] if metrics else [(STYLE_STATUS, " ready")]
        )
        parts.append((STYLE_STATUS, " · /help for commands · Ctrl-D exits · PgUp/PgDn scrolls"))
        return parts

    def _hint(self) -> list[tuple[str, str]]:
        if self.ui.follow:
            return [(STYLE_STATUS, f" {self.stats().get('state', 'ready')} · write a message and press Enter")]
        return [(STYLE_STATUS, " scrolled up - press PgDn or type to jump back to the newest line")]

    def run(self) -> None:
        self.ui.ui_thread_id = threading.get_ident()
        self.app.run()
