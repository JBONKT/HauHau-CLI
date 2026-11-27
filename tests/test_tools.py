from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from hauhau.audit import Audit
from hauhau.brain import Brain
from hauhau.config import Config
from hauhau.tools import build_registry
from hauhau.tools.base import Registry, Tool, ok


class ToolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.root / "data")
        self.cfg.tools.workspace = str(self.root)
        self.cfg.tools.shell_timeout = 20.0
        self.audit = Audit(self.cfg.audit_path, "test-session")
        self.registry = build_registry(self.cfg, self.audit, brain=None)
        self.audit_entries: list[dict] = []

    def tearDown(self) -> None:
        self.audit.close()
        self._tmp.cleanup()

    def run_tool(self, name: str, **args):
        tool = self.registry.get(name)
        assert tool is not None, name
        return tool.run(args)


class TestMemoryTools(unittest.TestCase):
    """The agent's own memory tools, with the real brain and real audit log."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.root / "data")
        self.cfg.brain.enabled = True
        self.cfg.brain.embeddings_url = ""
        self.audit = Audit(self.cfg.audit_path, "test-memory")
        self.brain = Brain(self.cfg, self.audit)
        self.registry = build_registry(self.cfg, self.audit, self.brain)

    def tearDown(self) -> None:
        self.brain.close()
        self.audit.close()
        self._tmp.cleanup()

    def run_tool(self, name: str, **args):
        tool = self.registry.get(name)
        assert tool is not None, name
        return tool.run(args)

    def test_rag_write_stores_and_audits(self) -> None:
        # Regression: Audit.log named its first parameter ``kind``, so the tool's
        # own kind= argument crashed every single rag_write call.
        result = self.run_tool(
            "rag_write", text="Deploys use the staging pipeline at ci.example/stage.", kind="procedural"
        )
        self.assertTrue(result.ok, result.content)
        self.assertIn("stored", result.content)
        self.assertEqual(self.brain.stats()["total"], 1)
        entries = self.audit.tail(5)
        self.assertTrue(
            any(e.get("event") == "rag_write" and e.get("memory_kind") == "procedural" for e in entries),
            entries,
        )

    def test_rag_write_rejects_empty_text(self) -> None:
        result = self.run_tool("rag_write", text="   ")
        self.assertFalse(result.ok)
        self.assertIn("text", result.content)

    def test_note_lesson_stores_a_procedure(self) -> None:
        result = self.run_tool(
            "note_lesson", lesson="Kör `llama-server --help` för att lista flaggorna."
        )
        self.assertTrue(result.ok, result.content)
        self.assertIn("learned", result.content)
        hits = self.brain.recall("llama-server flaggor", k=3)
        self.assertTrue(any("--help" in hit.text for hit in hits), [h.text for h in hits])
        self.assertEqual(
            next(hit.kind for hit in hits if "--help" in hit.text), "procedural"
        )

    def test_note_lesson_is_deduplicated(self) -> None:
        first = self.run_tool("note_lesson", lesson="Starta om backend med /model.")
        second = self.run_tool("note_lesson", lesson="Starta om backend med /model.")
        self.assertIn("learned", first.content)
        self.assertIn("already knew", second.content)
        self.assertEqual(self.brain.stats()["total"], 1)

    def test_note_lesson_rejects_an_empty_lesson(self) -> None:
        self.assertFalse(self.run_tool("note_lesson", lesson="  ").ok)

    def test_note_lesson_is_audited(self) -> None:
        self.run_tool("note_lesson", lesson="Sätt n_ctx till 8192 på 8 GB.")
        entries = self.audit.tail(20)
        self.assertTrue(any(e.get("event") == "note_lesson" for e in entries), entries)

    def test_rag_query_finds_what_was_written(self) -> None:
        self.run_tool("rag_write", text="The model lives at ~/modeller/gguf in this project.")
        result = self.run_tool("rag_query", query="where does the model live")
        self.assertTrue(result.ok, result.content)
        self.assertIn("modeller/gguf", result.content)
        self.assertTrue(any(e.get("event") == "rag_query" for e in self.audit.tail(5)))


    def test_rag_write_rejects_an_unknown_kind(self) -> None:
        # The store silently relabelled a bogus kind "semantic", so the reply and
        # the audit log claimed a label that was never stored.
        result = self.run_tool("rag_write", text="Something to keep.", kind="bogus")
        self.assertFalse(result.ok)
        self.assertIn("semantic", result.content)
        self.assertEqual(self.brain.stats()["total"], 0)

    def test_rag_write_keeps_explicit_zero_importance(self) -> None:
        # `importance or 0.5` turned an explicit 0 into the default weight.
        result = self.run_tool("rag_write", text="A throwaway note about nothing.", importance=0)
        self.assertTrue(result.ok, result.content)
        row = self.brain.store.db.execute("SELECT importance FROM memories").fetchone()
        self.assertEqual(row["importance"], 0.0)

    def test_rag_write_rejects_out_of_range_importance(self) -> None:
        result = self.run_tool("rag_write", text="x", importance=9)
        self.assertFalse(result.ok)
        self.assertIn("importance", result.content)

    def test_rag_query_rejects_a_negative_k(self) -> None:
        # SQLite reads LIMIT -1 as "no limit": a negative k dumped the whole brain.
        self.run_tool("rag_write", text="The build uses make.")
        result = self.run_tool("rag_query", query="build", k=-1)
        self.assertFalse(result.ok)
        self.assertIn("'k'", result.content)


class TestSchemaIsCompact(unittest.TestCase):
    """Every tool schema is prompt: it is sent on each step of every turn."""

    def test_empty_values_are_dropped(self) -> None:
        tool = Tool(
            "x",
            "does x",
            {
                "type": "object",
                "properties": {"a": {"type": "string", "description": ""}, "b": {"type": "string"}},
                "required": [],
            },
            lambda args: ok("x"),
        )
        params = tool.schema()["function"]["parameters"]
        self.assertEqual(params["properties"]["a"], {"type": "string"})
        self.assertNotIn("required", params)

    def test_a_multiline_description_is_one_line(self) -> None:
        tool = Tool("x", "first line\n  second line", {"type": "object", "properties": {}}, lambda a: ok("x"))
        self.assertEqual(tool.schema()["function"]["description"], "first line second line")

    def test_the_real_registry_is_under_budget(self) -> None:
        # A guard, not a style rule: the whole reason to trim descriptions is that
        # prefill time on this machine is linear in these characters.
        registry = Registry([
            Tool(name, "d" * 40, {"type": "object", "properties": {}}, lambda a: ok("x"))
            for name in ("a", "b")
        ])
        self.assertLess(len(str(registry.schemas())), 400)


class TestShell(ToolTestCase):
    def test_echo_captures_output_and_exit(self) -> None:
        result = self.run_tool("shell", cmd="echo hello")
        self.assertTrue(result.ok)
        self.assertIn("hello", result.content)
        self.assertIn("exit=0", result.content)

    def test_failing_command_reports_exit_code(self) -> None:
        result = self.run_tool("shell", cmd="exit 3")
        self.assertFalse(result.ok)
        self.assertIn("exit=3", result.content)

    def test_stderr_is_captured(self) -> None:
        result = self.run_tool("shell", cmd="echo oops >&2; exit 1")
        self.assertIn("oops", result.content)

    def test_runs_in_workspace_and_honours_cwd(self) -> None:
        # macOS temp dirs live behind the /var -> /private/var symlink.
        self.assertEqual(
            os.path.realpath(self.run_tool("shell", cmd="pwd").content.strip().splitlines()[1]),
            os.path.realpath(str(self.root)),
        )
        sub = self.root / "sub"
        sub.mkdir()
        result = self.run_tool("shell", cmd="pwd", cwd=str(sub))
        self.assertIn(os.path.realpath(str(sub)), os.path.realpath(result.content.strip()))

    def test_timeout_is_reported(self) -> None:
        self.cfg.tools.shell_timeout = 0.5
        result = self.run_tool("shell", cmd="sleep 5")
        self.assertFalse(result.ok)
        self.assertIn("killed after 0.5s", result.content)
        self.assertIn("timeout", result.content)

    def test_timeout_kills_background_children(self) -> None:
        # Regression: the timeout only killed bash, leaving `sleep` running.
        pidfile = self.root / "child.pid"
        self.cfg.tools.shell_timeout = 0.5
        result = self.run_tool("shell", cmd=f"sleep 30 & echo $! > {pidfile}; wait")
        self.assertFalse(result.ok)
        import os as _os
        import time as _time

        _time.sleep(0.5)
        pid = int(pidfile.read_text().strip())
        with self.assertRaises(ProcessLookupError):
            _os.kill(pid, 0)

    def test_huge_output_is_bounded_while_reading(self) -> None:
        # The read loop buffered every byte and clipped only after exit, so
        # `yes` or a verbose build could allocate gigabytes before returning.
        self.cfg.tools.max_output_chars = 2000
        result = self.run_tool("shell", cmd="yes 'flood' | head -n 400000")
        self.assertTrue(result.ok, result.content[:200])
        self.assertIn("more bytes; the rest was not kept", result.content)
        self.assertLess(len(result.content), 2000 * 4 + 1000)

    def test_interrupt_kills_the_command(self) -> None:
        # The child runs in its own session, so Ctrl-C never reaches it: the
        # interrupt used to leave the whole process group running.
        import os as _os
        import signal
        import threading
        import time as _time

        pidfile = self.root / "interrupted.pid"
        tool = self.registry.get("shell")

        def arm() -> None:
            _time.sleep(0.7)
            signal.raise_signal(signal.SIGINT)

        threading.Thread(target=arm, daemon=True).start()
        with self.assertRaises(KeyboardInterrupt):
            tool.run({"cmd": f"sleep 30 & echo $! > {pidfile}; sleep 30"})
        self.assertTrue(pidfile.is_file(), "the command never started")
        _time.sleep(0.7)
        pid = int(pidfile.read_text().strip())
        with self.assertRaises(ProcessLookupError):
            _os.kill(pid, 0)

    def test_command_is_audited(self) -> None:
        self.run_tool("shell", cmd="echo audited")
        entries = self.audit.tail(5)
        shell_entries = [e for e in entries if e["event"] == "shell"]
        self.assertTrue(shell_entries)
        self.assertEqual(shell_entries[-1]["cmd"], "echo audited")
        self.assertEqual(shell_entries[-1]["session"], "test-session")

    def test_denylist_blocks_before_running(self) -> None:
        self.cfg.tools.denylist = [r"rm\s+-rf\s+/"]
        marker = self.root / "should-not-exist"
        result = self.run_tool("shell", cmd=f"touch {marker} && rm -rf /")
        self.assertFalse(result.ok)
        self.assertIn("denylist", result.content)
        self.assertFalse(marker.exists())
        self.assertTrue(any(e["event"] == "shell.denied" for e in self.audit.tail(5)))

    def test_denylist_ignores_a_quoted_mention(self) -> None:
        # Searching the whole command line refused `echo "rm -rf / is what the
        # docs warn about"` - the pattern only appears inside the message.
        self.cfg.tools.denylist = [r"rm\s+-rf\s+/"]
        result = self.run_tool("shell", cmd='echo "rm -rf / is what the docs warn about"')
        self.assertTrue(result.ok, result.content)
        self.assertIn("docs warn", result.content)

    def test_denylist_still_blocks_a_real_statement(self) -> None:
        self.cfg.tools.denylist = [r"rm\s+-rf\s+/"]
        for cmd in ("rm -rf /", "echo ok && rm -rf /", "printf 'x'; rm -rf /"):
            self.assertFalse(self.run_tool("shell", cmd=cmd).ok, cmd)

    def test_starting_another_agent_is_refused(self) -> None:
        # Observed in a live session: the tool ran `hauhau /config` to print a
        # configuration it could have read as a file, loading a second model.
        for cmd in (
            "hauhau /config",
            "hauhau",
            ".venv/bin/python -m hauhau",
            "cd /tmp && hauhau --resume",
            "uv run hauhau /status",
        ):
            result = self.run_tool("shell", cmd=cmd)
            self.assertFalse(result.ok, cmd)
            self.assertIn("second copy of this agent", result.content, cmd)
        denied = [e for e in self.audit.tail(10) if e["event"] == "shell.denied"]
        self.assertTrue(denied)

    def test_mentioning_the_agent_is_not_starting_it(self) -> None:
        for cmd in ("echo hauhau", "printf 'hauhau\\n' | wc -l", "python -c 'print(1)'"):
            result = self.run_tool("shell", cmd=cmd)
            self.assertTrue(result.ok, f"{cmd}: {result.content}")

    def test_output_of_a_background_child_is_flagged_as_incomplete(self) -> None:
        # The shell exits while the child still holds the pipe; reporting a clean
        # exit=0 over the truncated output looks like a success.
        result = self.run_tool("shell", cmd="(sleep 1; echo late) & echo early", timeout=20)
        self.assertTrue(result.ok, result.content)
        self.assertIn("early", result.content)
        self.assertIn("background", result.content)

    def test_confirm_callback_can_decline(self) -> None:
        self.cfg.tools.shell_confirm = True
        registry = build_registry(self.cfg, self.audit, brain=None, confirm=lambda cmd: False)
        result = registry.get("shell").run({"cmd": "echo nope"})
        self.assertFalse(result.ok)
        self.assertIn("declined", result.content)

    def test_a_background_child_holding_stdout_does_not_block(self) -> None:
        # communicate() waits for every holder of the pipe: a daemon started by
        # the command (or a shell profile) kept the REPL stuck for the whole
        # timeout on its very first `shell` call.
        # `sleep 3 &` inherits the stdout pipe; communicate() would block until it
        # exited. (An earlier version of this test wrote "> /dev/null", which
        # redirects the holder away and made the test unable to fail.)
        started = time.monotonic()
        result = self.run_tool("shell", cmd="sleep 3 & echo done", timeout=10)
        elapsed = time.monotonic() - started
        self.assertTrue(result.ok, result.content)
        self.assertIn("done", result.content)
        self.assertLess(elapsed, 1.5, "the tool waited for an unrelated pipe holder")

    def test_empty_command_rejected(self) -> None:
        self.assertFalse(self.run_tool("shell", cmd="   ").ok)


class TestFiles(ToolTestCase):
    def test_write_then_read(self) -> None:
        path = self.root / "a.txt"
        self.assertTrue(self.run_tool("write_file", path=str(path), content="one\ntwo\n").ok)
        result = self.run_tool("read_file", path=str(path))
        self.assertTrue(result.ok)
        self.assertIn("two", result.content)

    def test_a_relative_path_means_the_workspace(self) -> None:
        # shell/grep run in the workspace and the prompt names it, so read_file
        # with a bare name must not read from the process CWD.
        (self.root / "notes.txt").write_text("workspace copy\n", encoding="utf-8")
        result = self.run_tool("read_file", path="notes.txt")
        self.assertTrue(result.ok, result.content)
        self.assertIn("workspace copy", result.content)

    def test_a_relative_write_lands_in_the_workspace(self) -> None:
        self.assertTrue(self.run_tool("write_file", path="made.txt", content="x").ok)
        self.assertTrue((self.root / "made.txt").exists())

    def test_write_creates_parents(self) -> None:
        path = self.root / "deep" / "nested" / "b.txt"
        self.assertTrue(self.run_tool("write_file", path=str(path), content="x").ok)
        self.assertTrue(path.exists())

    def test_overwrite_snapshots_previous_version(self) -> None:
        path = self.root / "c.txt"
        self.run_tool("write_file", path=str(path), content="original")
        self.run_tool("write_file", path=str(path), content="replaced")
        snaps = list(self.cfg.snapshots_dir.glob("*c.txt"))
        self.assertTrue(snaps)
        self.assertEqual(snaps[0].read_text(), "original")

    def test_read_missing_file_fails(self) -> None:
        self.assertFalse(self.run_tool("read_file", path=str(self.root / "nope.txt")).ok)

    def test_read_offset_and_limit(self) -> None:
        path = self.root / "lines.txt"
        self.run_tool("write_file", path=str(path), content="\n".join(f"line{i}" for i in range(20)))
        result = self.run_tool("read_file", path=str(path), offset=5, limit=3)
        self.assertIn("line5", result.content)
        self.assertIn("line7", result.content)
        self.assertNotIn("line8", result.content)

    def test_read_offset_past_end_is_an_error(self) -> None:
        path = self.root / "short.txt"
        self.run_tool("write_file", path=str(path), content="only\nthree\nlines")
        result = self.run_tool("read_file", path=str(path), offset=99)
        self.assertFalse(result.ok)
        self.assertIn("past the end", result.content)
        self.assertNotIn("empty file", result.content)

    def test_read_is_bounded_by_limit(self) -> None:
        path = self.root / "many.txt"
        self.run_tool("write_file", path=str(path), content="\n".join(f"line{i}" for i in range(5000)))
        result = self.run_tool("read_file", path=str(path), limit=2)
        self.assertIn("line0", result.content)
        self.assertIn("line1", result.content)
        self.assertNotIn("line2", result.content)
        self.assertIn("more lines follow", result.content)

    def test_over_long_line_is_truncated_not_loaded(self) -> None:
        path = self.root / "long.txt"
        self.run_tool("write_file", path=str(path), content="y" * 20000 + "\nshort\n")
        result = self.run_tool("read_file", path=str(path), limit=2)
        self.assertIn("line truncated", result.content)
        self.assertIn("short", result.content)
        self.assertLess(len(result.content), 20000)

    def test_non_integer_offset_is_a_clear_error(self) -> None:
        path = self.root / "x.txt"
        self.run_tool("write_file", path=str(path), content="a\nb")
        result = self.run_tool("read_file", path=str(path), offset="five")
        self.assertFalse(result.ok)
        self.assertIn("must be an integer", result.content)

    def test_edit_refuses_a_file_too_large_to_rewrite(self) -> None:
        from hauhau.tools.files import EDIT_MAX_BYTES

        path = self.root / "huge.log"
        with path.open("wb") as fh:
            fh.seek(EDIT_MAX_BYTES + 1)
            fh.write(b"x")
        result = self.run_tool("edit_file", path=str(path), old="x", new="y")
        self.assertFalse(result.ok)
        self.assertIn("too large", result.content)
        self.assertIn("shell tool", result.content)

    def test_read_empty_file_says_so(self) -> None:
        path = self.root / "empty.txt"
        self.run_tool("write_file", path=str(path), content="")
        self.assertIn("empty file", self.run_tool("read_file", path=str(path)).content)

    def test_edit_replaces_single_occurrence(self) -> None:
        path = self.root / "d.txt"
        self.run_tool("write_file", path=str(path), content="alpha beta gamma")
        result = self.run_tool("edit_file", path=str(path), old="beta", new="BETA")
        self.assertTrue(result.ok)
        self.assertEqual(path.read_text(), "alpha BETA gamma")

    def test_edit_requires_unique_match(self) -> None:
        path = self.root / "e.txt"
        self.run_tool("write_file", path=str(path), content="dup dup dup")
        result = self.run_tool("edit_file", path=str(path), old="dup", new="x")
        self.assertFalse(result.ok)
        self.assertIn("3 times", result.content)

    def test_edit_replace_all(self) -> None:
        path = self.root / "f.txt"
        self.run_tool("write_file", path=str(path), content="dup dup dup")
        result = self.run_tool("edit_file", path=str(path), old="dup", new="x", replace_all=True)
        self.assertTrue(result.ok)
        self.assertEqual(path.read_text(), "x x x")

    def test_edit_missing_anchor_fails(self) -> None:
        path = self.root / "g.txt"
        self.run_tool("write_file", path=str(path), content="abc")
        self.assertFalse(self.run_tool("edit_file", path=str(path), old="zzz", new="y").ok)

    def test_list_dir_marks_directories(self) -> None:
        (self.root / "folder").mkdir()
        (self.root / "file.txt").write_text("x")
        result = self.run_tool("list_dir", path=str(self.root))
        self.assertIn("folder/", result.content)
        self.assertIn("file.txt", result.content)


class TestSearch(ToolTestCase):
    def test_grep_finds_line_numbers(self) -> None:
        (self.root / "code.py").write_text("import os\nNEEDLE = 1\n")
        result = self.run_tool("grep", pattern="NEEDLE", path=str(self.root), glob="*.py")
        self.assertTrue(result.ok)
        self.assertIn("code.py:2", result.content)

    def test_grep_no_match_is_not_an_error(self) -> None:
        (self.root / "code.py").write_text("nothing here\n")
        result = self.run_tool("grep", pattern="absent", path=str(self.root))
        self.assertTrue(result.ok)
        self.assertIn("no matches", result.content)

    def test_grep_ignores_git_dir(self) -> None:
        git_dir = self.root / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("TOKEN\n")
        (self.root / "app.py").write_text("TOKEN\n")
        result = self.run_tool("grep", pattern="TOKEN", path=str(self.root))
        self.assertIn("app.py", result.content)
        self.assertNotIn(".git", result.content)

    def test_find_by_glob(self) -> None:
        (self.root / "target.toml").write_text("x")
        result = self.run_tool("find", pattern="*.toml", path=str(self.root))
        self.assertIn("target.toml", result.content)

class TestToolHardening(ToolTestCase):
    """Reproducers for the tool-layer bugs the audit found.

    Each case either crashed the whole turn, ran a command in the wrong place,
    or silently changed data - so each asserts on *both* the reply and the
    side effect on disk.
    """

    def test_non_utf8_output_is_kept_not_raised(self) -> None:
        # subprocess(...) with text=True raised UnicodeDecodeError after the
        # command had already run, so the work happened with no audit entry.
        result = self.run_tool("shell", cmd="printf \'ok\\xff\\xfe done\\n\'")
        self.assertTrue(result.ok, result.content)
        self.assertIn("ok", result.content)
        self.assertTrue(any(e.get("event") == "shell" for e in self.audit.tail(5)))

    def test_invalid_cwd_fails_instead_of_running_elsewhere(self) -> None:
        marker = self.root / "ran-here"
        result = self.run_tool("shell", cmd=f"touch {marker}", cwd=str(self.root / "missing"))
        self.assertFalse(result.ok)
        self.assertIn("cwd", result.content)
        self.assertFalse(marker.exists())

    def test_infinite_timeout_is_a_clean_failure(self) -> None:
        # float("inf") reached Popen and raised OverflowError, orphaning a child.
        result = self.run_tool("shell", cmd="sleep 30", timeout=float("inf"))
        self.assertFalse(result.ok)
        self.assertIn("timeout", result.content)

    def test_confirmation_without_a_prompt_fails_closed(self) -> None:
        self.cfg.tools.shell_confirm = True
        marker = self.root / "unasked"
        result = self.run_tool("shell", cmd=f"touch {marker}")
        self.assertFalse(result.ok)
        self.assertIn("no prompt", result.content)
        self.assertFalse(marker.exists())

    def test_string_false_does_not_authorise_replace_all(self) -> None:
        path = self.root / "twice.txt"
        self.run_tool("write_file", path=str(path), content="dup dup")
        result = self.run_tool("edit_file", path=str(path), old="dup", new="x", replace_all="false")
        self.assertFalse(result.ok)
        self.assertEqual(path.read_text(), "dup dup")

    def test_non_string_content_is_rejected(self) -> None:
        path = self.root / "n.txt"
        result = self.run_tool("write_file", path=str(path), content={"a": 1})
        self.assertFalse(result.ok)
        self.assertIn("must be a string", result.content)
        self.assertFalse(path.exists())

    def test_edit_preserves_crlf_line_endings(self) -> None:
        # Rewriting with newline translation turned every CRLF file into LF.
        path = self.root / "win.txt"
        path.write_bytes(b"one\r\ntwo\r\nthree\r\n")
        result = self.run_tool("edit_file", path=str(path), old="two", new="TWO")
        self.assertTrue(result.ok, result.content)
        self.assertEqual(path.read_bytes(), b"one\r\nTWO\r\nthree\r\n")

    def test_negative_read_limit_does_not_dump_the_file(self) -> None:
        path = self.root / "big.txt"
        self.run_tool("write_file", path=str(path), content="\n".join(f"l{i}" for i in range(50)))
        result = self.run_tool("read_file", path=str(path), limit=-1)
        self.assertFalse(result.ok)
        self.assertIn("limit", result.content)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores directory modes")
    def test_unreadable_directory_is_a_clean_failure(self) -> None:
        locked = self.root / "locked"
        locked.mkdir()
        os.chmod(locked, 0o000)
        try:
            result = self.run_tool("list_dir", path=str(locked))
        finally:
            # addCleanup would run after tearDown has deleted the tree.
            os.chmod(locked, 0o700)
        self.assertFalse(result.ok)
        self.assertIn("permission", result.content.lower())

    def test_grep_rejects_a_non_numeric_max_results(self) -> None:
        result = self.run_tool("grep", pattern="x", path=str(self.root), max_results="lots")
        self.assertFalse(result.ok)
        self.assertIn("max_results", result.content)

    def test_grep_does_not_treat_the_string_false_as_ignore_case(self) -> None:
        (self.root / "c.py").write_text("NEEDLE\n")
        result = self.run_tool("grep", pattern="needle", path=str(self.root), ignore_case="false")
        self.assertIn("no matches", result.content)

    def test_find_and_grep_agree_about_hidden_trees(self) -> None:
        # grep skipped hidden directories while find walked them, so the same
        # search reported different files depending on the tool.
        hidden = self.root / ".work"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("x")
        (self.root / "visible.txt").write_text("x")
        found = self.run_tool("find", pattern="*.txt", path=str(self.root)).content
        self.assertIn("visible.txt", found)
        self.assertNotIn("secret.txt", found)

    def test_find_caps_max_results(self) -> None:
        for i in range(5):
            (self.root / f"f{i}.txt").write_text("x")
        result = self.run_tool("find", pattern="*.txt", path=str(self.root), max_results=2)
        self.assertLessEqual(len(result.content.splitlines()), 2)


class TestClipText(unittest.TestCase):
    def test_never_exceeds_the_limit(self) -> None:
        from hauhau.util import clip_text

        for limit in (41, 60, 100, 320, 8000):
            out = clip_text("x" * 20000, limit)
            self.assertLessEqual(len(out), limit, f"limit {limit} produced {len(out)}")

    def test_short_text_is_untouched(self) -> None:
        from hauhau.util import clip_text

        self.assertEqual(clip_text("hello", 100), "hello")


class TestRegistryShape(ToolTestCase):
    def test_schemas_are_valid_openai_tools(self) -> None:
        schemas = self.registry.schemas()
        self.assertGreaterEqual(len(schemas), 7)
        for schema in schemas:
            self.assertEqual(schema["type"], "function")
            fn = schema["function"]
            self.assertTrue(fn["name"])
            self.assertTrue(fn["description"])
            self.assertEqual(fn["parameters"]["type"], "object")

    def test_unknown_tool_returns_none(self) -> None:
        self.assertIsNone(self.registry.get("nope"))

    def test_duplicate_registration_rejected(self) -> None:
        from hauhau.tools.base import Tool

        tool = self.registry.get("shell")
        with self.assertRaises(ValueError):
            self.registry.add(Tool(tool.name, "dup", {}, lambda a: None))


class TestShellStatementSplitting(unittest.TestCase):
    """A second line of a pasted script is a statement of its own."""

    def test_a_newline_separates_statements(self) -> None:
        from hauhau.tools.shell import _segments, _self_spawn

        self.assertEqual(_segments("echo hi\nrm -rf /tmp/x"), ["echo hi", "rm -rf /tmp/x"])
        self.assertIsNotNone(_self_spawn("echo hi\nhauhau /config"))

    def test_a_denylisted_second_line_is_caught(self) -> None:
        from hauhau.tools.shell import _denylist_hit

        self.assertEqual(_denylist_hit("echo ok\nrm -rf /", [r"rm\s+-rf"]), r"rm\s+-rf")

    def test_quoted_text_is_still_not_a_statement(self) -> None:
        from hauhau.tools.shell import _denylist_hit

        self.assertIsNone(_denylist_hit('echo "rm -rf / is what the docs warn about"', [r"rm\s+-rf"]))


class TestIntegerArguments(unittest.TestCase):
    def test_a_digit_looking_string_that_int_rejects_is_a_bad_argument(self) -> None:
        from hauhau.tools.args import BadArgument, integer

        for raw in ("--5", "\u00b2"):
            with self.assertRaises(BadArgument) as caught:
                integer({"n": raw}, "n", 1)
            self.assertIn("integer", str(caught.exception))

    def test_a_numeric_string_is_accepted(self) -> None:
        from hauhau.tools.args import integer

        self.assertEqual(integer({"n": " 7 "}, "n", 1), 7)


if __name__ == "__main__":
    unittest.main()
