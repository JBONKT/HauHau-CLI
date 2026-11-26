from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from hauhau.audit import Audit
from hauhau.config import Config
from hauhau.tools import build_registry


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
        self.assertIn("timed out", result.content)

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

    def test_confirm_callback_can_decline(self) -> None:
        self.cfg.tools.shell_confirm = True
        registry = build_registry(self.cfg, self.audit, brain=None, confirm=lambda cmd: False)
        result = registry.get("shell").run({"cmd": "echo nope"})
        self.assertFalse(result.ok)
        self.assertIn("declined", result.content)

    def test_empty_command_rejected(self) -> None:
        self.assertFalse(self.run_tool("shell", cmd="   ").ok)


class TestFiles(ToolTestCase):
    def test_write_then_read(self) -> None:
        path = self.root / "a.txt"
        self.assertTrue(self.run_tool("write_file", path=str(path), content="one\ntwo\n").ok)
        result = self.run_tool("read_file", path=str(path))
        self.assertTrue(result.ok)
        self.assertIn("two", result.content)

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


if __name__ == "__main__":
    unittest.main()
