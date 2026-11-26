from __future__ import annotations

import json
import unittest
from io import StringIO

from prompt_toolkit.document import Document
from rich.console import Console

from hauhau.llm import Usage
from hauhau.tools.base import ToolResult, ok
from hauhau.ui import (
    COMMANDS,
    PROMPT_GLYPH,
    Progress,
    RichUI,
    SlashCompleter,
    StreamRenderer,
    banner,
    data_table,
    error_panel,
    help_panel,
    kv_table,
    nearest_command,
    note,
    result_preview,
    short_name,
    summarize_args,
    toolbar_text,
)


def render(func, *args, **kwargs) -> str:
    out = StringIO()
    console = Console(file=out, width=100, force_terminal=False)
    func(console, *args, **kwargs)
    return out.getvalue()


class TestSummarizeArgs(unittest.TestCase):
    def test_shell_command_is_extracted(self) -> None:
        self.assertEqual(summarize_args("shell", json.dumps({"cmd": "df -h /"})), "df -h /")

    def test_shell_extra_args_are_appended(self) -> None:
        text = summarize_args("shell", json.dumps({"cmd": "ls", "cwd": "/tmp"}))
        self.assertIn("ls", text)
        self.assertIn("cwd=/tmp", text)

    def test_file_tools_show_the_path(self) -> None:
        self.assertEqual(summarize_args("read_file", json.dumps({"path": "/etc/hosts"})), "/etc/hosts")

    def test_multiline_values_are_flattened(self) -> None:
        text = summarize_args("rag_write", json.dumps({"text": "line one\nline two"}))
        self.assertNotIn("\n", text)
        self.assertIn("line one line two", text)

    def test_long_values_are_truncated(self) -> None:
        text = summarize_args("shell", json.dumps({"cmd": "x" * 400}))
        self.assertLessEqual(len(text), 145)

    def test_falsey_extras_are_skipped(self) -> None:
        text = summarize_args("grep", json.dumps({"pattern": "TODO", "ignore_case": False, "all": None}))
        self.assertEqual(text, "TODO")

    def test_malformed_json_falls_back(self) -> None:
        self.assertEqual(summarize_args("shell", "{not json"), "{not json")

    def test_empty_args(self) -> None:
        self.assertEqual(summarize_args("shell", "{}"), "{}")

    def test_unknown_tool_prints_json(self) -> None:
        text = summarize_args("mystery", json.dumps({"alpha": 1}))
        self.assertIn("alpha", text)


class TestResultPreview(unittest.TestCase):
    def test_strips_exit_code_line(self) -> None:
        preview, more = result_preview(ToolResult(True, "exit=0\n109Gi free"))
        self.assertEqual(preview, "109Gi free")
        self.assertEqual(more, 0)

    def test_counts_hidden_lines(self) -> None:
        preview, more = result_preview(ToolResult(True, "a\nb\nc\nd\ne"))
        self.assertEqual(preview, "a")
        self.assertEqual(more, 4)

    def test_multiline_output_stays_on_one_line(self) -> None:
        preview, more = result_preview(ToolResult(True, "/dev/disk 228Gi\n31G /Users/johan/Documents"))
        self.assertNotIn("/ 31G", preview)
        self.assertEqual(more, 1)

    def test_ls_total_line_is_skipped(self) -> None:
        preview, _more = result_preview(ToolResult(True, "total 993272\n-rw-r--r--  1 j vibe.dmg"))
        self.assertIn("vibe.dmg", preview)
        self.assertNotIn("total", preview)

    def test_only_noise_falls_back_to_it(self) -> None:
        preview, _ = result_preview(ToolResult(True, "exit=0"))
        self.assertEqual(preview, "exit=0")

    def test_empty_output(self) -> None:
        self.assertEqual(result_preview(ToolResult(True, "")), ("(no output)", 0))

    def test_exit_only_output_still_shows_it(self) -> None:
        preview, _ = result_preview(ToolResult(False, "exit=3"))
        self.assertIn("exit=3", preview)

    def test_long_lines_are_capped(self) -> None:
        preview, _ = result_preview(ToolResult(True, "y" * 900))
        self.assertLessEqual(len(preview), 400)


class TestShortName(unittest.TestCase):
    def test_short_name_unchanged(self) -> None:
        self.assertEqual(short_name("model.gguf"), "model.gguf")

    def test_long_name_keeps_quantisation_suffix(self) -> None:
        name = "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"
        out = short_name(name)
        self.assertLessEqual(len(out), 34)
        self.assertIn("…", out)
        self.assertTrue(out.endswith("Q4_K_P.gguf"))


class TestStreamRenderer(unittest.TestCase):
    def test_piped_output_is_plain_and_greppable(self) -> None:
        out = StringIO()
        console = Console(file=out, width=100, force_terminal=False)
        renderer = StreamRenderer(console)
        renderer.feed("hello ")
        renderer.feed("world")
        self.assertEqual(renderer.finish(), "hello world")
        self.assertIn("hello world", out.getvalue())
        self.assertNotIn("\x1b[", out.getvalue())

    def test_abort_is_safe_without_output(self) -> None:
        console = Console(file=StringIO(), force_terminal=False)
        renderer = StreamRenderer(console)
        renderer.abort()
        renderer.feed("late")
        renderer.abort()

    def test_markdown_is_rendered_on_a_terminal(self) -> None:
        out = StringIO()
        console = Console(file=out, width=100, force_terminal=True)
        renderer = StreamRenderer(console)
        for chunk in ["**bold** ", "and `code`"]:
            renderer.feed(chunk)
        renderer.finish()
        text = out.getvalue()
        # Markdown markers are consumed by the renderer, not shown raw.
        self.assertIn("bold", text)
        self.assertNotIn("**bold**", text)


class TestRichUI(unittest.TestCase):
    def setUp(self) -> None:
        self.out = StringIO()
        self.console = Console(file=self.out, width=100, force_terminal=False)
        self.ui = RichUI(self.console)

    def text(self) -> str:
        return self.out.getvalue()

    def test_tokens_are_assembled_into_one_answer(self) -> None:
        for chunk in ("The disk ", "is ", "fine."):
            self.ui.token(chunk)
        self.ui.newline()
        self.assertIn("The disk is fine.", self.text())

    def test_tool_call_shows_name_and_summary(self) -> None:
        self.ui.tool_call("shell", json.dumps({"cmd": "df -h /"}))
        text = self.text()
        self.assertIn("shell", text)
        self.assertIn("df -h /", text)
        self.assertNotIn('{"cmd"', text)

    def test_tool_result_uses_status_glyphs(self) -> None:
        self.ui.tool_result("shell", ToolResult(True, "exit=0\nall good"))
        self.ui.tool_result("shell", ToolResult(False, "exit=1\nit broke"))
        text = self.text()
        self.assertIn("✓", text)
        self.assertIn("✗", text)
        self.assertIn("all good", text)
        self.assertIn("it broke", text)

    def test_tool_result_reports_hidden_lines(self) -> None:
        self.ui.tool_result("shell", ToolResult(True, "one\ntwo"))
        self.assertIn("+1 lines", self.text())

    def test_reasoning_is_labelled(self) -> None:
        self.ui.reasoning("considering options")
        self.ui.newline()
        self.assertIn("thinking", self.text())
        self.assertIn("considering options", self.text())

    def test_status_is_prefixed(self) -> None:
        self.ui.status("step budget reached")
        self.assertIn("· step budget reached", self.text())

    def test_turn_end_is_silent_when_not_a_terminal(self) -> None:
        self.ui.turn_end(Usage(completion_tokens=10, tok_per_second=26.0), 2.0)
        self.assertEqual(self.text().strip(), "")

    def test_turn_end_prints_meta_on_a_terminal(self) -> None:
        out = StringIO()
        console = Console(file=out, width=100, force_terminal=True)
        RichUI(console).turn_end(Usage(completion_tokens=42, tok_per_second=26.4), 3.2)
        visible = out.getvalue()
        self.assertIn("3.2s", visible)
        self.assertIn("42 tok", visible)
        self.assertIn("26 tok/s", visible)

    def test_interrupt_resets_streaming_state(self) -> None:
        self.ui.token("partial answer")
        self.ui.interrupt()
        self.ui.token("new answer")
        self.ui.newline()
        self.assertIn("new answer", self.text())


class TestStaticPieces(unittest.TestCase):
    def test_banner_shows_model_and_counts(self) -> None:
        text = render(
            banner,
            name="Hauhau",
            model="Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf",
            ctx=8192,
            threads=4,
            ram=8.0,
            memories=3,
            url="http://127.0.0.1:8080",
        )
        self.assertIn("Hauhau", text)
        self.assertIn("8192 ctx", text)
        self.assertIn("3 memories", text)
        self.assertIn("8 GB RAM", text)
        self.assertIn("Q4_K_P", text)

    def test_banner_singularises_one_memory(self) -> None:
        text = render(
            banner,
            name="H",
            model="m.gguf",
            ctx=1,
            threads=1,
            ram=8.0,
            memories=1,
            url="u",
        )
        self.assertIn("1 memory", text)
        self.assertNotIn("1 memories", text)

    def test_kv_table_pairs_rows(self) -> None:
        text = render(kv_table, "Backend", [("model", "m.gguf"), ("context", "8192 tokens")])
        self.assertIn("Backend", text)
        self.assertIn("model", text)
        self.assertIn("8192 tokens", text)

    def test_data_table_has_headers(self) -> None:
        text = render(data_table, "Tools", ["name", "description"], [["shell", "run bash"]])
        self.assertIn("name", text)
        self.assertIn("shell", text)

    def test_help_panel_lists_every_command(self) -> None:
        text = render(help_panel)
        for command, _description in COMMANDS:
            self.assertIn(command, text)

    def test_error_panel_marks_the_error(self) -> None:
        text = render(error_panel, "cannot reach llama-server")
        self.assertIn("error", text.lower())
        self.assertIn("cannot reach llama-server", text)

    def test_note_prints_verbatim_without_markup(self) -> None:
        text = render(note, "[b]not markup[/b]")
        self.assertIn("[b]not markup[/b]", text)


class TestHelpAndSuggestions(unittest.TestCase):
    def test_help_is_grouped(self) -> None:
        text = render(help_panel)
        for group in ("this session", "working", "memory", "model", "self-tuning"):
            self.assertIn(group, text)
        self.assertIn("/show", text)
        self.assertIn("/status", text)

    def test_help_for_one_command_shows_usage(self) -> None:
        text = render(help_panel, "tune")
        self.assertIn("/tune", text)
        self.assertIn("/tune 6", text)
        self.assertNotIn("/mem", text)

    def test_help_for_an_unknown_command_suggests(self) -> None:
        text = render(help_panel, "propmt")
        self.assertIn("no such command", text)
        self.assertIn("/prompt", text)

    def test_nearest_command(self) -> None:
        self.assertEqual(nearest_command("/modle"), "/model")
        self.assertEqual(nearest_command("savee"), "/save")
        self.assertEqual(nearest_command("/zzzzzz"), "")


class TestProgress(unittest.TestCase):
    def test_piped_output_prints_plain_steps(self) -> None:
        out = StringIO()
        console = Console(file=out, width=100, force_terminal=False)
        with Progress(console, "working…") as progress:
            progress.update("halfway…")
            progress.step("done: 3 of 3")
        text = out.getvalue()
        self.assertIn("working…", text)
        self.assertIn("done: 3 of 3", text)


class TestShowBuffer(unittest.TestCase):
    def test_recent_results_are_newest_first_and_capped(self) -> None:
        ui = RichUI(Console(file=StringIO()))
        for index in range(7):
            ui.tool_result("shell", ToolResult(True, f"line {index}"))
        recent = ui.recent_results()
        self.assertEqual(len(recent), 5, "kept to five results")
        self.assertEqual(recent[0][1].content, "line 6")
        self.assertEqual(recent[-1][1].content, "line 2")

    def test_failures_wrap_instead_of_ellipsizing(self) -> None:
        out = StringIO()
        ui = RichUI(Console(file=out, width=40, force_terminal=False))
        ui.tool_result("shell", ToolResult(False, "long explanation " * 10))
        self.assertIn("long explanation", out.getvalue())


class TestPromptChrome(unittest.TestCase):
    def test_toolbar_summarises_state(self) -> None:
        text = toolbar_text(ctx=8192, memories=3, ram=8.0, shell_mode="auto")
        self.assertIn("8192 ctx", text)
        self.assertIn("3 mem", text)
        self.assertIn("shell auto", text)

    def test_toolbar_handles_empty_memory(self) -> None:
        self.assertIn("no memory", toolbar_text(ctx=1, memories=0, ram=8.0, shell_mode="confirm"))

    def test_completer_completes_slash_commands(self) -> None:
        completer = SlashCompleter()
        items = list(completer.get_completions(Document("/me"), None))
        self.assertEqual([i.text for i in items], ["/mem"])

    def test_completer_lists_all_on_slash(self) -> None:
        items = list(SlashCompleter().get_completions(Document("/"), None))
        self.assertEqual(len(items), len(COMMANDS))

    def test_completer_is_silent_after_arguments(self) -> None:
        self.assertEqual(list(SlashCompleter().get_completions(Document("/mem query"), None)), [])

    def test_argument_completion_uses_the_provider(self) -> None:
        completer = SlashCompleter(lambda text: [("v1", "baseline"), ("v2", "100%")])
        items = list(completer.get_completions(Document("/prompt v"), None))
        self.assertEqual([item.text for item in items], ["v1", "v2"])
        self.assertEqual(items[0].display_meta_text, "baseline")

    def test_argument_completion_looks_only_at_the_last_word(self) -> None:
        completer = SlashCompleter(lambda text: [("evolve", "rewrite the prompt")] if "help" in text else [])
        self.assertEqual([i.text for i in completer.get_completions(Document("/help ev"), None)], ["evolve"])
        self.assertEqual(list(completer.get_completions(Document("/help pr"), None)), [])

    def test_argument_completion_needs_a_provider(self) -> None:
        self.assertEqual(list(SlashCompleter().get_completions(Document("/prompt v"), None)), [])

    def test_completer_ignores_plain_text(self) -> None:
        self.assertEqual(list(SlashCompleter().get_completions(Document("hello"), None)), [])

    def test_esc_enter_inserts_a_newline_instead_of_submitting(self) -> None:
        import tempfile
        from pathlib import Path as _Path

        from prompt_toolkit.keys import Keys

        from hauhau.ui import make_prompt_session

        session = make_prompt_session(_Path(tempfile.mkdtemp()) / "hist", lambda: "tb")
        bound = {tuple(b.keys) for b in session.key_bindings.bindings}
        # prompt_toolkit normalises "enter" to ControlM.
        self.assertIn((Keys.Escape, Keys.ControlM), bound)

    def test_prompt_session_has_completion_and_history(self) -> None:
        import tempfile
        from pathlib import Path as _Path

        from hauhau.ui import make_prompt_session

        session = make_prompt_session(_Path(tempfile.mkdtemp()) / "hist", lambda: "tb")
        self.assertIsInstance(session.completer, SlashCompleter)
        self.assertTrue(session.complete_while_typing)
        self.assertIsNotNone(session.bottom_toolbar)

    def test_prompt_glyph(self) -> None:
        self.assertEqual(PROMPT_GLYPH, "❯")  # noqa: RUF001 - the glyph under test

    def test_brackets_in_model_output_are_never_parsed_as_markup(self) -> None:
        # Whatever the model emits must reach the terminal verbatim.
        out = StringIO()
        console = Console(file=out, width=100, force_terminal=False)
        StreamRenderer(console).feed("[b]literal[/b] <tag>")
        self.assertIn("[b]literal[/b] <tag>", out.getvalue())


class TestTurnLifecycle(unittest.TestCase):
    def test_turn_hooks_are_called_around_a_turn(self) -> None:
        from hauhau.agent import Agent
        from hauhau.config import Config
        from hauhau.llm import Completion
        from hauhau.tools.base import Registry, Tool

        seen: list[str] = []

        class SpyUI(RichUI):
            def turn_start(self, label: str = "") -> None:
                seen.append("start")

            def turn_end(self, usage=None, elapsed: float = 0.0) -> None:
                seen.append(f"end:{usage.completion_tokens if usage else 0}")

        class Client:
            def chat(self, messages, tools=None, **kwargs):
                return Completion(
                    content="done",
                    finish_reason="stop",
                    usage=Usage(completion_tokens=7, tok_per_second=26.0),
                )

        cfg = Config()
        cfg.brain.enabled = False
        registry = Registry([Tool("noop", "no-op.", {"type": "object", "properties": {}}, lambda a: ok("done"))])
        agent = Agent(cfg, Client(), registry, brain=None, ui=SpyUI(Console(file=StringIO())))
        agent.run_turn("hi")
        self.assertEqual(seen, ["start", "end:7"])

    def test_turn_end_runs_even_when_the_model_fails(self) -> None:
        from hauhau.agent import Agent
        from hauhau.config import Config
        from hauhau.llm import LLMError
        from hauhau.tools.base import Registry

        seen: list[str] = []

        class SpyUI(RichUI):
            def turn_end(self, usage=None, elapsed: float = 0.0) -> None:
                seen.append("end")

        class Client:
            def chat(self, *a, **kw):
                raise LLMError("backend down")

        cfg = Config()
        cfg.brain.enabled = False
        agent = Agent(cfg, Client(), Registry([]), brain=None, ui=SpyUI(Console(file=StringIO())))
        with self.assertRaises(LLMError):
            agent.run_turn("hi")
        self.assertEqual(seen, ["end"])

    def test_usage_from_every_step_is_summed(self) -> None:
        from hauhau.agent import Agent
        from hauhau.config import Config
        from hauhau.llm import Completion, ToolCall
        from hauhau.tools.base import Registry, Tool

        usages: list[int] = []

        class SpyUI(RichUI):
            def turn_end(self, usage=None, elapsed: float = 0.0) -> None:
                usages.append(usage.completion_tokens)

        class Client:
            def __init__(self) -> None:
                self.replies = [
                    Completion(
                        tool_calls=[ToolCall(id="1", name="noop", arguments="{}")],
                        finish_reason="tool_calls",
                        usage=Usage(completion_tokens=5),
                    ),
                    Completion(content="done", finish_reason="stop", usage=Usage(completion_tokens=9)),
                ]

            def chat(self, *a, **kw):
                return self.replies.pop(0)

        cfg = Config()
        cfg.brain.enabled = False
        registry = Registry([Tool("noop", "no-op.", {"type": "object", "properties": {}}, lambda a: ok("done"))])
        Agent(cfg, Client(), registry, brain=None, ui=SpyUI(Console(file=StringIO()))).run_turn("hi")
        self.assertEqual(usages, [14])


if __name__ == "__main__":
    unittest.main()
