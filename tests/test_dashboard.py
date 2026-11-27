"""The Textual dashboard: mounting, streaming and the input round-trip.

These run the real app, headless, through Textual's own pilot - a widget tree
that composes is not the same as one that updates, and the marshalling between
the worker thread and the event loop is exactly where this kind of UI fails.
"""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import patch

from textual.widgets import Markdown, Sparkline, Static

from hauhau.agent import Usage
from hauhau.dashboard import DashboardUI, HauhauDashboard, available
from hauhau.tools.base import ToolResult

STATS = {
    "name": "Hauhau",
    "model": "LFM3-1.7B",
    "backend": "http://127.0.0.1:8080",
    "ctx": "16k",
    "threads": "7t",
    "memories": "41 mem",
    "journal": "47 events",
    "tuner": "idle",
    "session": "s1",
    "workspace": "~",
    "prompt": "v2",
}


def text_of(widget: object) -> str:
    """The plain text a Static is showing."""
    return str(getattr(widget, "content", ""))


def make_app(
    ui: DashboardUI | None = None,
    submitted: list[str] | None = None,
    propose=None,
) -> HauhauDashboard:
    ui = ui or DashboardUI()
    seen = submitted if submitted is not None else []

    def submit(text: str) -> None:
        seen.append(text)

    return HauhauDashboard(
        ui,
        submit=submit,
        restart_backend=lambda: None,
        should_exit=lambda: False,
        stats=lambda: STATS,
        propose=propose,
    )


async def settle(pilot, times: int = 4) -> None:
    """Give the suggester worker a chance to answer and the ghost to land."""
    for _ in range(times):
        await pilot.pause()


async def wait_until(pilot, predicate, timeout: float = 5.0) -> bool:
    """Poll until ``predicate`` holds.

    Fixed sleeps made these tests fail on a loaded machine: the debounce, the
    worker thread and the model call are all asynchronous, and how long they take
    is not a property of the code under test.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
        await pilot.pause()
    return predicate()


@unittest.skipUnless(available(), "textual is not installed")
class DashboardShutdownTestCase(unittest.TestCase):
    """The exit path: /learn prints its goodbyes while Textual is already
    tearing down the screen that holds #chat - observed as NoMatches on quit."""

    def test_printing_after_the_screen_is_gone_does_not_crash(self) -> None:
        from textual.css.query import NoMatches

        class DeadApp:
            is_running = False  # so _call() runs the draw inline

            def query_one(self, *args, **kwargs):
                raise NoMatches("the dashboard screen is gone")

        ui = DashboardUI()
        ui.app = DeadApp()
        ui.append_ansi("distilling this session into memory…\n")  # must not raise


@unittest.skipUnless(available(), "textual is not installed")
class DashboardTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        # The model half is debounced; a test should not wait 0.55 s per case.
        patcher = patch("hauhau.dashboard.AI_DELAY", 0.01)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_it_mounts_with_both_columns(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            self.assertIsNotNone(app.query_one("#chat"))
            self.assertIsNotNone(app.query_one("#side"))
            self.assertIsNotNone(app.query_one("#prompt"))
            self.assertIsNotNone(app.query_one("#speed", Sparkline))

    async def test_the_side_panel_shows_the_stats(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            side = text_of(app.query_one("#model", Static))
            self.assertIn("LFM3-1.7B", side)
            runtime = text_of(app.query_one("#runtime", Static))
            self.assertIn("context", runtime)
            self.assertIn("16k", runtime)
            memory = text_of(app.query_one("#memory", Static))
            self.assertIn("41 mem", memory)
            self.assertIn("47 events", memory)

    async def test_submitting_a_message_runs_the_turn(self) -> None:
        submitted: list[str] = []
        app = make_app(submitted=submitted)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.query_one("#prompt").value = "how much disk is left?"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(submitted, ["how much disk is left?"])

    async def test_the_input_is_cleared_and_left_enabled_after_a_turn(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.query_one("#prompt").value = "hello"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            field = app.query_one("#prompt")
            self.assertEqual(field.value, "")
            self.assertFalse(field.disabled)

    async def test_a_user_message_echoes_into_the_chat(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.ui.echo("hej")
            await pilot.pause()
            users = app.query(".user")
            self.assertEqual(len(users), 1)

    async def test_tokens_become_one_markdown_block(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.ui.turn_start("thinking…")
            for chunk in ("The volume ", "is 11% ", "full."):
                app.ui.token(chunk)
            app.ui.turn_end(Usage(prompt_tokens=20, completion_tokens=5, tok_per_second=40.0), 1.5)
            await pilot.pause()
            await pilot.pause()
            blocks = app.query(Markdown)
            self.assertEqual(len(blocks), 1)
            self.assertIn("The volume is 11% full.", blocks.first().source)

    async def test_a_later_flush_keeps_the_earlier_tokens(self) -> None:
        """A Markdown widget re-parses its whole source, so a flush that carries
        only the newest chunk would replace the answer with its own tail."""
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.ui.token("first half ")
            app.ui.flush_stream()
            await pilot.pause()
            app.ui.token("second half")
            app.ui.flush_stream(force=True)  # the throttle holds non-forced flushes
            await pilot.pause()
            await pilot.pause()
            blocks = app.query(Markdown)
            self.assertEqual(len(blocks), 1)
            self.assertEqual(str(blocks.first().source), "first half second half")

    async def test_a_tool_call_and_result_render_as_rows(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.ui.tool_call("shell", '{"cmd": "df -h /"}')
            app.ui.tool_result("shell", ToolResult(ok=True, content="13Gi used\nmore\nmore"))
            await pilot.pause()
            rows = [text_of(widget) for widget in app.query(Static)]
            self.assertTrue(any("⚙ shell" in row for row in rows), rows)
            self.assertTrue(any("✓" in row and "+2 lines" in row for row in rows), rows)

    async def test_a_failed_tool_result_is_marked(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.ui.tool_result("shell", ToolResult(ok=False, content="boom"))
            await pilot.pause()
            self.assertTrue(any("✗ boom" in text_of(w) for w in app.query(Static)))

    async def test_rich_output_from_a_command_lands_in_the_chat(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            # What /status does: a rich table rendered into the UI's console.
            app.ui.console.print("│ model │ LFM3 │")
            await pilot.pause()
            rows = [text_of(widget) for widget in app.query(Static)]
            self.assertTrue(any("LFM3" in row for row in rows), rows)

    async def test_output_printed_before_mount_is_not_lost(self) -> None:
        ui = DashboardUI()
        ui.note("config: unknown setting foo.bar", error=False)
        app = make_app(ui=ui)
        self.assertEqual(len(ui._early), 1)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            self.assertEqual(ui._early, [])
            self.assertTrue(any("unknown setting" in text_of(w) for w in app.query(Static)))

    async def test_the_speed_history_feeds_the_sparkline(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.ui.last_usage = Usage(completion_tokens=40, tok_per_second=42.0)
            app._refresh_stats()
            await pilot.pause()
            spark = app.query_one("#speed", Sparkline)
            # The 0.12s timer refreshes the panel too, so the same reading can be
            # recorded more than once; what matters is that it was recorded.
            self.assertTrue(spark.data)
            self.assertTrue(all(value == 42.0 for value in spark.data), list(spark.data))

    async def test_the_busy_state_shows_in_the_panel_and_the_prompt(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.ui.turn_start("running a tool")
            app._set_busy(True)
            app._refresh_stats()
            await pilot.pause()
            self.assertIn("running a tool", text_of(app.query_one("#runtime", Static)))
            self.assertTrue(app.query_one("#prompt").disabled)

    # --- suggestions ------------------------------------------------------
    async def test_a_keyword_completion_appears_as_ghost_text(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one("#prompt")
            field.value = "hur mycket disk"
            await wait_until(pilot, lambda: bool(getattr(field, "_suggestion", "")))
            ghost = getattr(field, "_suggestion", "")
            self.assertEqual(ghost, "hur mycket disk är det kvar och vad tar mest plats?")
            self.assertEqual(app.suggestion_source, "keyword")

    async def test_a_slash_command_is_completed(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one("#prompt")
            field.value = "/mod"
            await wait_until(pilot, lambda: getattr(field, "_suggestion", "") == "/model")
            self.assertEqual(getattr(field, "_suggestion", ""), "/model")
            self.assertEqual(app.suggestion_source, "command")

    async def test_tab_accepts_the_ghost_text(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one("#prompt")
            field.value = "kolla loggen"
            await wait_until(pilot, lambda: bool(getattr(field, "_suggestion", "")))
            await pilot.press("tab")
            await pilot.pause()
            self.assertEqual(field.value, "kolla loggen och säg vad som faktiskt hände")

    async def test_the_model_finishes_what_the_table_does_not_know(self) -> None:
        seen: list[str] = []

        def propose(prefix: str) -> str:
            seen.append(prefix)
            return prefix + " och sortera dem nyast först"

        app = make_app(propose=propose)
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one("#prompt")
            field.value = "mina semesterbilder"
            await wait_until(pilot, lambda: bool(getattr(field, "_suggestion", "")))
            ghost = getattr(field, "_suggestion", "")
            self.assertTrue(ghost.endswith("och sortera dem nyast först"), ghost)
            self.assertEqual(app.suggestion_source, "ai")
            self.assertTrue(seen)

    async def test_the_model_is_asked_once_per_burst_not_per_keystroke(self) -> None:
        """Every keystroke restarts the debounce clock, so only the value the user
        stopped on is sent - a request per character would be answered for a
        prefix they have already typed past."""
        calls: list[str] = []

        def propose(prefix: str) -> str:
            calls.append(prefix)
            return prefix + " och gör klart"

        # A long delay, so the burst below cannot finish the timer by accident
        # even when the machine is busy.
        with patch("hauhau.dashboard.AI_DELAY", 1.0):
            app = make_app(propose=propose)
            async with app.run_test(size=(100, 30)) as pilot:
                field = app.query_one("#prompt")
                for value in ("mina", "mina semester", "mina semesterbilder", "mina semesterbilder i"):
                    field.value = value
                    await pilot.pause()
                self.assertEqual(calls, [], "nothing is asked while the typing continues")
                await wait_until(pilot, lambda: bool(calls), timeout=6.0)
                self.assertEqual(calls, ["mina semesterbilder i"], calls)

    async def test_a_short_value_never_reaches_the_model(self) -> None:
        calls: list[str] = []
        app = make_app(propose=lambda prefix: calls.append(prefix) or prefix + " mer")
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one("#prompt").value = "kort"
            await wait_until(pilot, lambda: bool(getattr(app.query_one("#prompt"), "_suggestion", "")), timeout=1.0)
            self.assertEqual(calls, [])

    # --- cost ------------------------------------------------------------
    async def test_an_idle_tick_does_not_touch_the_panel(self) -> None:
        """Six widget updates plus a layout, 8 times a second, cost ~6% of a core
        on an M1 Air. Nothing may be painted while the numbers are unchanged."""
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app._tick()  # the first frame draws and records the signature
            await pilot.pause()
            painted: list[int] = []
            original = app._refresh_stats
            app._refresh_stats = lambda: painted.append(1)  # type: ignore[method-assign]
            try:
                for _ in range(10):
                    app._tick()
            finally:
                app._refresh_stats = original  # type: ignore[method-assign]
            self.assertEqual(painted, [])

    async def test_a_changed_number_repaints_the_panel(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app._tick()
            await pilot.pause()
            painted: list[int] = []
            original = app._refresh_stats
            app._refresh_stats = lambda: painted.append(1)  # type: ignore[method-assign]
            try:
                app.ui.turn_start("running a tool")  # changes the "state" row
                app._tick()
            finally:
                app._refresh_stats = original  # type: ignore[method-assign]
            self.assertEqual(len(painted), 1)

    async def test_streaming_flushes_are_throttled(self) -> None:
        """Markdown re-parses the whole answer, so a flush per token is quadratic."""
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.ui.token("a few characters ")
            app.ui.flush_stream()
            await pilot.pause()
            first = str(app.query(Markdown).first().source)
            app.ui.token("more")  # below STREAM_MIN_CHARS and inside the window
            app.ui.flush_stream()
            await pilot.pause()
            self.assertEqual(str(app.query(Markdown).first().source), first)
            app.ui.flush_stream(force=True)
            await pilot.pause()
            await pilot.pause()
            self.assertEqual(str(app.query(Markdown).first().source), first + "more")

    # --- clipboard --------------------------------------------------------
    async def test_ctrl_y_copies_the_last_answer(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.ui.last_answer = "The volume is 11% full."
            copied: list[str] = []
            with patch("hauhau.dashboard.write_clipboard", side_effect=lambda t: copied.append(t) or True):
                await pilot.press("ctrl+y")
            await pilot.pause()
            self.assertEqual(copied, ["The volume is 11% full."])

    async def test_ctrl_y_prefers_the_selection(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.ui.last_answer = "the whole answer"
            with (
                patch.object(type(app.screen), "get_selected_text", return_value="a fragment"),
                patch("hauhau.dashboard.write_clipboard") as writer,
            ):
                await pilot.press("ctrl+y")
            await pilot.pause()
            self.assertEqual(writer.call_args.args[0], "a fragment")

    async def test_ctrl_y_with_nothing_to_copy_says_so(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            with patch("hauhau.dashboard.write_clipboard") as writer:
                await pilot.press("ctrl+y")
            await pilot.pause()
            self.assertFalse(writer.called)

    async def test_ctrl_v_pastes_at_the_caret_on_one_line(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one("#prompt")
            field.value = "abc"
            field.cursor_position = 1
            with patch("hauhau.dashboard.read_clipboard", return_value="X\nY"):
                await pilot.press("ctrl+v")
            await pilot.pause()
            self.assertEqual(field.value, "aX Ybc")

    async def test_ctrl_v_with_an_empty_clipboard_changes_nothing(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            field = app.query_one("#prompt")
            field.value = "abc"
            with patch("hauhau.dashboard.read_clipboard", return_value=""):
                await pilot.press("ctrl+v")
            await pilot.pause()
            self.assertEqual(field.value, "abc")

    async def test_the_answer_is_kept_for_copying_after_it_ends(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            app.ui.token("the answer")
            app.ui.turn_end(None, 1.0)
            await pilot.pause()
            self.assertEqual(app.ui.last_answer, "the answer")

    async def test_f2_refocuses_the_prompt(self) -> None:
        app = make_app()
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.query_one("#chat").focus()
            await pilot.press("f2")
            await pilot.pause()
            self.assertIs(app.focused, app.query_one("#prompt"))


if __name__ == "__main__":
    unittest.main()
