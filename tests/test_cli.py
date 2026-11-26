from __future__ import annotations

import tempfile
import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

from hauhau.agent import Agent
from hauhau.audit import Audit
from hauhau.brain import Brain
from hauhau.brain.learn import Learner
from hauhau.cli import Repl, RichUI
from hauhau.config import Config
from hauhau.llm import Completion
from hauhau.tools import build_registry


class FakeClient:
    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])

    def chat(self, messages, tools=None, **kwargs) -> Completion:
        return Completion(content=self.replies.pop(0) if self.replies else "[]", finish_reason="stop")


class FakeManager:
    def status(self) -> dict:
        return {
            "url": "http://127.0.0.1:8080",
            "managed": True,
            "running": True,
            "gguf": "model.gguf",
            "n_ctx": 8192,
            "n_threads": 4,
            "n_gpu_layers": 99,
            "log": "/tmp/server.log",
        }

    def stop(self) -> None:
        pass


class ReplTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.root / "data")
        self.cfg.tools.workspace = str(self.root)
        self.out = StringIO()
        self.console = Console(file=self.out, width=120, force_terminal=False)
        self.ui = RichUI(self.console)
        self.audit = Audit(self.cfg.audit_path, "test")
        self.brain = Brain(self.cfg)
        self.client = FakeClient()
        registry = build_registry(self.cfg, self.audit, self.brain)
        self.agent = Agent(self.cfg, self.client, registry, self.brain, self.audit, self.ui)
        self.session_path = self.cfg.sessions_dir / "s.json"
        self.repl = Repl(
            self.cfg, self.client, FakeManager(), self.brain,
            Learner(self.cfg, self.client, self.brain), self.agent,
            self.audit, self.ui, self.session_path,
        )

    def tearDown(self) -> None:
        self.audit.close()
        self.brain.close()
        self._tmp.cleanup()

    def text(self) -> str:
        return self.out.getvalue()


class TestCommands(ReplTestCase):
    def test_help_lists_commands(self) -> None:
        self.repl.handle("/help")
        self.assertIn("/mem", self.text())
        self.assertIn("/learn", self.text())

    def test_unknown_command_says_so(self) -> None:
        self.repl.handle("/nonsense")
        self.assertIn("unknown command", self.text())

    def test_tools_lists_registry(self) -> None:
        self.repl.handle("/tools")
        output = self.text()
        self.assertIn("shell", output)
        self.assertIn("rag_query", output)
        self.assertIn("edit_file", output)

    def test_remember_then_search(self) -> None:
        self.repl.handle("/remember the user's editor is helix")
        self.assertIn("stored", self.text())
        self.repl.handle("/mem helix editor")
        self.assertIn("helix", self.text())

    def test_mem_without_argument_shows_stats(self) -> None:
        self.repl.handle("/mem")
        self.assertIn("total", self.text())

    def test_forget_removes_memory(self) -> None:
        self.repl.handle("/remember a throwaway fact to delete")
        self.repl.handle("/forget 1")
        self.assertIn("forgotten", self.text())

    def test_forget_rejects_bad_id(self) -> None:
        self.repl.handle("/forget notanumber")
        self.assertIn("usage", self.text())

    def test_forget_reports_when_memory_is_off(self) -> None:
        self.repl.brain = None
        self.repl.handle("/forget 1")
        self.assertIn("memory is disabled", self.text())

    def test_context_reports_turns(self) -> None:
        self.agent.messages = [{"role": "user", "content": "hi"}]
        self.repl.handle("/context")
        self.assertIn("turns", self.text())

    def test_save_writes_session_file(self) -> None:
        self.agent.messages = [{"role": "user", "content": "persist me"}]
        self.repl.handle("/save")
        self.assertTrue(self.session_path.exists())
        self.assertIn("persist me", self.session_path.read_text())

    def test_clear_resets_context(self) -> None:
        self.agent.messages = [{"role": "user", "content": "hi"}]
        self.repl.handle("/clear")
        self.assertEqual(self.agent.messages, [])
        self.assertIn("cleared", self.text())

    def test_model_shows_backend_table(self) -> None:
        self.repl.handle("/model")
        self.assertIn("model.gguf", self.text())
        self.assertIn("8192", self.text())

    def test_config_shows_policy(self) -> None:
        self.repl.handle("/config")
        output = self.text()
        self.assertIn("shell confirm", output)
        self.assertIn("False", output)

    def test_audit_tails_log(self) -> None:
        self.audit.log("shell", cmd="echo hi")
        self.repl.handle("/audit 5")
        self.assertIn("echo hi", self.text())

    def test_compact_merges_duplicates(self) -> None:
        import time as _time

        now = _time.time()
        for text in (
            "The model files are in /Users/johan/modeller/gguf.",
            "The model files are in /Users/johan/modeller/gguf right now.",
        ):
            self.brain.store.db.execute(
                "INSERT INTO memories (kind, text, tags, norm_hash, importance, uses, created_at, last_used_at)"
                " VALUES ('semantic', ?, '', ?, 0.5, 0, ?, ?)",
                (text, text, now, now),
            )
        self.brain.store.db.commit()
        self.repl.handle("/compact")
        self.assertIn("merged 1 duplicate", self.text())
        self.assertEqual(self.brain.stats()["total"], 1)

    def test_prune_removes_stale_memories(self) -> None:
        import time as _time

        old_ts = _time.time() - 86400 * 400
        self.brain.store.db.execute(
            "INSERT INTO memories (kind, text, tags, norm_hash, importance, uses, created_at, last_used_at)"
            " VALUES ('episodic', 'a throwaway note', '', 'h', 0.1, 0, ?, ?)",
            (old_ts, old_ts),
        )
        self.brain.store.db.commit()
        self.repl.handle("/prune")
        self.assertIn("pruned 1 stale memories", self.text())
        self.assertEqual(self.brain.stats()["total"], 0)

    def test_prune_keeps_used_memories(self) -> None:
        row_id, _ = self.brain.remember("semantic", "Deploy uses make release.", importance=0.9)
        self.brain.store.db.execute("UPDATE memories SET uses = 3 WHERE id = ?", (row_id,))
        self.brain.store.db.commit()
        self.repl.handle("/prune")
        self.assertIn("pruned 0 stale memories", self.text())
        self.assertEqual(self.brain.stats()["total"], 1)

    def test_prune_rejects_a_bad_weight(self) -> None:
        self.repl.handle("/prune abc")
        self.assertIn("usage: /prune", self.text())

    def test_prune_reports_when_memory_is_off(self) -> None:
        self.repl.brain = None
        self.repl.handle("/prune")
        self.assertIn("memory is disabled", self.text())

    def test_quit_stops_loop(self) -> None:
        self.assertTrue(self.repl.running)
        self.repl.handle("/quit")
        self.assertFalse(self.repl.running)

    def test_learn_with_nothing_durable(self) -> None:
        self.agent.messages = [{"role": "user", "content": "hello there friend"}]
        self.repl.handle("/learn")
        self.assertIn("too thin", self.text())
        self.assertEqual(self.brain.stats()["total"], 0)

    def test_learn_stores_extracted_memories(self) -> None:
        self.client.replies = [
            '[{"kind":"preference","text":"The user prefers helix as their editor.","importance":0.8}]'
        ]
        self.agent.messages = [
            {"role": "user", "content": "I always use helix as my editor and never vim, remember that"},
            {"role": "tool", "name": "shell", "content": "helix --version\nhelix 25.01"},
        ]
        self.repl.handle("/learn")
        self.assertIn("helix", self.text())
        self.assertEqual(self.brain.stats()["total"], 1)


class TestSessionPath(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        from hauhau.config import Config

        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.logging.data_dir = str(Path(self._tmp.name) / "data")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_new_session_gets_a_timestamped_json_path(self) -> None:
        from hauhau.cli import _session_path

        path, warning = _session_path(self.cfg, None)
        self.assertTrue(path.name.endswith(".json"))
        self.assertIsNone(warning)

    def test_last_picks_the_newest_existing_session(self) -> None:
        from hauhau.cli import _session_path

        self.cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
        for name in ("20260101T000000", "20260202T000000"):
            (self.cfg.sessions_dir / f"{name}.json").write_text("{}")
        path, warning = _session_path(self.cfg, "last")
        self.assertEqual(path.stem, "20260202T000000")
        self.assertIsNone(warning)

    def test_last_with_no_sessions_starts_fresh(self) -> None:
        from hauhau.cli import _session_path

        path, warning = _session_path(self.cfg, "last")
        self.assertTrue(path.name.endswith(".json"))
        self.assertIsNone(warning)

    def test_named_session_is_resolved(self) -> None:
        from hauhau.cli import _session_path

        self.cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
        target = self.cfg.sessions_dir / "abc.json"
        target.write_text("{}")
        self.assertEqual(_session_path(self.cfg, "abc"), (target, None))

    def test_name_with_extension_is_accepted(self) -> None:
        from hauhau.cli import _session_path

        self.cfg.sessions_dir.mkdir(parents=True, exist_ok=True)
        target = self.cfg.sessions_dir / "abc.json"
        target.write_text("{}")
        self.assertEqual(_session_path(self.cfg, "abc.json"), (target, None))

    def test_unknown_name_warns_and_does_not_create_a_stray_file(self) -> None:
        # Regression: this used to return <dir>/does-not-exist and then persist to it.
        from hauhau.cli import _session_path

        path, warning = _session_path(self.cfg, "does-not-exist")
        self.assertTrue(path.name.endswith(".json"))
        self.assertIn("no session named", warning)
        self.assertFalse((self.cfg.sessions_dir / "does-not-exist").exists())


class TestUI(ReplTestCase):
    def test_streamed_tokens_reach_the_console(self) -> None:
        self.ui.token("hello ")
        self.ui.token("world")
        self.ui.newline()
        self.assertIn("hello world", self.text())

    def test_tool_result_marks_failure(self) -> None:
        from hauhau.tools.base import fail, ok

        self.ui.tool_call("shell", '{"cmd":"ls"}')
        self.ui.tool_result("shell", ok("exit=0\nfile.txt"))
        self.ui.tool_result("shell", fail("boom"))
        output = self.text()
        self.assertIn("shell", output)
        self.assertIn("file.txt", output)
        self.assertIn("boom", output)


if __name__ == "__main__":
    unittest.main()
