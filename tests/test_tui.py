"""The full-screen interface: transcript wrapping, slicing and chrome."""

from __future__ import annotations

import unittest

from hauhau.llm import Usage
from hauhau.tools.base import ToolResult
from hauhau.tui import ScreenApp, ScreenUI, _Wrapper


def text_of(fragments: list[tuple[str, str]]) -> str:
    return "".join(text for _style, text in fragments)


def lines_of(fragments: list[tuple[str, str]]) -> list[str]:
    return list(text_of(fragments).split("\n"))


class TestWrapper(unittest.TestCase):
    def wrap(self, fragments, width):
        return [text_of(line) for line in _Wrapper().wrap(fragments, width)]

    def test_explicit_newlines_split_lines(self):
        self.assertEqual(self.wrap([("", "one\ntwo\nthree")], 20), ["one", "two", "three"])

    def test_words_wrap_at_the_width(self):
        wrapped = self.wrap([("", "alpha beta gamma delta")], 12)
        self.assertEqual(wrapped, ["alpha beta", "gamma delta"])
        self.assertTrue(all(len(line) <= 12 for line in wrapped))

    def test_a_word_longer_than_the_window_is_split(self):
        wrapped = self.wrap([("", "x" * 25)], 10)
        self.assertEqual(wrapped, ["x" * 10, "x" * 10, "x" * 5])

    def test_styles_survive_wrapping(self):
        wrapped = _Wrapper().wrap([("bold", "hello "), ("", "world")], 40)
        self.assertEqual(wrapped[0], [("bold", "hello "), ("", "world")])

    def test_incremental_feed_matches_one_shot(self):
        fragments = [("", "one two three "), ("", "four five six seven"), ("", "\neight")]
        one_shot = [text_of(line) for line in _Wrapper().wrap(fragments, 11)]

        wrapper = _Wrapper()
        for index in range(1, len(fragments) + 1):
            incremental = [text_of(line) for line in wrapper.wrap(fragments[:index], 11)]
        self.assertEqual(incremental, one_shot)

    def test_a_width_change_rewraps_from_scratch(self):
        fragments = [("", "alpha beta gamma")]
        wrapper = _Wrapper()
        self.assertEqual(len(wrapper.wrap(fragments, 40)), 1)
        self.assertEqual(len(wrapper.wrap(fragments, 6)), 3)

    def test_shrinking_history_resets(self):
        wrapper = _Wrapper()
        wrapper.wrap([("", "a"), ("", "\nb"), ("", "\nc")], 20)
        self.assertEqual(len(wrapper.wrap([("", "just one")], 20)), 1)


class TestScreenUI(unittest.TestCase):
    def setUp(self):
        self.ui = ScreenUI()

    def test_echo_opens_and_closes_its_line(self):
        self.ui.echo("hello")
        self.assertEqual(text_of(self.ui.fragments), "\n❯ hello\n")  # noqa: RUF001 - prompt glyph

    def test_tokens_do_not_add_a_blank_line_after_the_echo(self):
        self.ui.echo("hello")
        self.ui.token("an ")
        self.ui.token("answer")
        self.assertEqual(text_of(self.ui.fragments), "\n❯ hello\nan answer")  # noqa: RUF001

    def test_turn_start_and_end_clear_the_busy_state(self):
        self.ui.turn_start()
        self.assertTrue(self.ui.busy)
        self.assertTrue(self.ui.follow)
        self.ui.turn_end(Usage(prompt_tokens=3, completion_tokens=7), 1.5)
        self.assertFalse(self.ui.busy)
        self.assertEqual(self.ui.last_usage.completion_tokens, 7)
        self.assertEqual(self.ui.last_elapsed, 1.5)

    def test_tool_call_and_result_render(self):
        self.ui.echo("run it")
        self.ui.tool_call("shell", '{"cmd": "echo hi"}')
        self.ui.tool_result("shell", ToolResult(ok=True, content="hi"))
        text = text_of(self.ui.fragments)
        self.assertIn("⚙ shell", text)
        self.assertIn("✓", text)
        self.assertEqual(self.ui.recent[-1][0], "shell")

    def test_note_marks_errors(self):
        self.ui.note("boom", error=True)
        self.assertIn("✗ boom", text_of(self.ui.fragments))

    def test_status_line_starts_on_its_own_line(self):
        self.ui.token("partial")
        self.ui.status("retrying")
        self.assertIn("\n· retrying", text_of(self.ui.fragments))

    def test_append_ansi_parses_rich_output(self):
        self.ui.append_ansi("\x1b[32mgreen text\x1b[0m\n")
        self.assertIn("green text", text_of(self.ui.fragments))

    def test_scroll_state_starts_following(self):
        self.assertTrue(self.ui.follow)
        self.assertEqual(self.ui.scroll_offset, 0)


class TestScreenApp(unittest.TestCase):
    def make(self, columns: int = 40, rows: int = 12) -> tuple[ScreenApp, ScreenUI]:
        ui = ScreenUI()
        app = ScreenApp(
            ui,
            submit=lambda text: None,
            restart_backend=lambda: None,
            should_exit=lambda: False,
            stats=lambda: {"name": "Hauhau"},
        )
        app._output_size = lambda: (columns, rows)  # type: ignore[method-assign]
        return app, ui

    def test_chat_rows_leave_the_chrome_alone(self):
        app, _ui = self.make(rows=12)
        self.assertEqual(app._chat_rows(), 8)

    def test_chat_rows_never_collapse(self):
        app, _ui = self.make(rows=3)
        self.assertEqual(app._chat_rows(), 3)

    def test_view_shows_the_newest_lines_while_following(self):
        app, ui = self.make(columns=20, rows=8)
        ui.fragments = [("", "\n".join(f"line {i}" for i in range(1, 31)))]
        view = lines_of(app._chat_view())
        self.assertEqual(len(view), 4)
        self.assertEqual(view[-1], "line 30")
        self.assertEqual(view[0], "line 27")

    def test_view_starts_at_the_top_when_the_history_is_short(self):
        app, ui = self.make(columns=20, rows=8)
        ui.fragments = [("", "line 1\nline 2")]
        self.assertEqual(lines_of(app._chat_view()), ["line 1", "line 2"])

    def test_scrolling_up_freezes_the_view(self):
        app, ui = self.make(columns=20, rows=8)
        ui.fragments = [("", "\n".join(f"line {i}" for i in range(1, 31)))]
        app._chat_view()  # measures the transcript
        app._scroll_page(direction=1)
        self.assertFalse(ui.follow)
        view = lines_of(app._chat_view())
        self.assertEqual((view[0], view[-1]), ("line 25", "line 28"))

    def test_scrolling_back_down_resumes_following(self):
        app, ui = self.make(columns=20, rows=8)
        ui.fragments = [("", "\n".join(f"line {i}" for i in range(1, 31)))]
        app._chat_view()
        app._scroll_page(direction=1)
        app._scroll_page(direction=-1)
        self.assertTrue(ui.follow)
        self.assertEqual(ui.scroll_offset, 0)
        self.assertEqual(lines_of(app._chat_view())[-1], "line 30")

    def test_typing_returns_to_the_newest_line(self):
        app, ui = self.make(columns=20, rows=8)
        ui.fragments = [("", "\n".join(f"line {i}" for i in range(1, 31)))]
        app._chat_view()
        app._scroll_page(direction=1)
        app._on_typed(app.input_buffer)
        self.assertTrue(ui.follow)
        self.assertEqual(lines_of(app._chat_view())[-1], "line 30")

    def test_the_input_buffer_accepts_enter(self):
        """prompt_toolkit 3.0.53 defaults Buffer to multiline, where Enter only
        inserts a newline and the accept handler never runs."""
        app, _ui = self.make()
        self.assertFalse(app.input_buffer.multiline())
        self.assertTrue(app.input_buffer.is_returnable)

    def test_busy_input_is_read_only_through_a_filter(self):
        app, _ui = self.make()
        app._set_read_only(True)
        self.assertTrue(app.input_buffer.read_only())
        app._set_read_only(False)
        self.assertFalse(app.input_buffer.read_only())


if __name__ == "__main__":
    unittest.main()
