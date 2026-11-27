"""cli.main() end to end with a fake backend: wiring, exit codes, guards.

The unit tests cover each command; this file covers the path that joins them -
the path where every serious bug this project has had actually lived (a leaked
llama-server, a shutdown skipped by a failing learn pass, a config typo that
printed a stack trace).
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

from rich.console import Console

from hauhau import cli
from hauhau.llm import Completion, LLMError, Usage
from hauhau.profiler import HardwareProfile

PROFILE = HardwareProfile(
    os_name="Darwin",
    arch="arm64",
    cpu="Apple M4",
    cores=8,
    ram_gb=8.0,
    gpu="Apple Silicon",
    metal=True,
    disk_free_gb=100.0,
    llama_server="/opt/homebrew/bin/llama-server",
    llama_version="9999",
)


class FakeBackend:
    """Stands in for ServerManager + LlamaClient."""

    behaviour = "ok"
    instances: ClassVar[list[FakeBackend]] = []

    def __init__(self, cfg, audit=None) -> None:
        self.stopped = 0
        self.restarts = 0
        FakeBackend.instances.append(self)

    def ensure(self):
        return self.connection

    def client(self):
        return self.connection

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
        self.stopped += 1

    def restart(self):
        self.restarts += 1
        return self.connection

    class Chat:
        def health(self) -> bool:
            return True

        def chat(self, messages, tools=None, on_token=None, **_kwargs) -> Completion:
            behaviour = FakeBackend.behaviour
            if behaviour == "error":
                raise LLMError("backend died")
            if behaviour == "interrupt":
                raise KeyboardInterrupt
            if on_token:
                on_token("fi")
                on_token("ne")
            return Completion(
                content="fine", finish_reason="stop", usage=Usage(completion_tokens=3, tok_per_second=27.0)
            )

    connection = Chat()


class MainTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        FakeBackend.behaviour = "ok"
        FakeBackend.instances.clear()
        self.out = io.StringIO()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_main(self, argv: list[str], *, data_dir: str = "data", lines: list[str] | None = None):
        scripted = list(lines or [])

        def fake_prompt(*_args, **_kwargs):
            if not scripted:
                raise EOFError
            return scripted.pop(0)

        session = type("Session", (), {"prompt": staticmethod(fake_prompt)})()
        with (
            mock.patch.object(cli, "ServerManager", FakeBackend),
            mock.patch.object(
                cli, "make_console", lambda: Console(file=self.out, width=100, force_terminal=False)
            ),
            mock.patch.object(cli, "make_prompt_session", lambda *a, **k: session),
            mock.patch.object(cli, "probe", lambda: PROFILE),
        ):
            code = cli.main(["--data-dir", str(self.root / data_dir), *argv])
        return code, self.out.getvalue()

    @property
    def stops(self) -> list[int]:
        return [backend.stopped for backend in FakeBackend.instances]


class TestOneShot(MainTestCase):
    def test_answers_and_exits_zero(self) -> None:
        code, out = self.run_main(["--yolo", "hello"])
        self.assertEqual(code, 0)
        self.assertIn("fine", out)
        self.assertEqual(self.stops, [1], "the backend is always stopped")

    def test_backend_failure_exits_nonzero(self) -> None:
        # A script cannot tell a failed run from a successful one if this is 0.
        FakeBackend.behaviour = "error"
        code, out = self.run_main(["--yolo", "hello"])
        self.assertEqual(code, 1)
        self.assertIn("backend died", out)
        self.assertEqual(self.stops, [1])

    def test_empty_prompt_is_refused(self) -> None:
        code, out = self.run_main(["--yolo", "   "])
        self.assertEqual(code, 2)
        self.assertIn("prompt is empty", out)
        self.assertEqual(self.stops, [1])

    def test_learner_crash_does_not_skip_shutdown(self) -> None:
        with mock.patch.object(cli.Learner, "distill", side_effect=RuntimeError("boom")):
            code, out = self.run_main(["--yolo"], lines=["hello there", "/quit"])
        self.assertEqual(code, 0)
        self.assertEqual(self.stops, [1])
        self.assertNotIn("Traceback", out)


class TestStartup(MainTestCase):
    def test_profile_flag_prints_hardware_and_exits(self) -> None:
        code, out = self.run_main(["--profile"])
        self.assertEqual(code, 0)
        self.assertIn("cores", out)

    def test_version_flag_prints_and_exits_without_backend(self) -> None:
        from hauhau import __version__

        stdout = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout):
            code, _out = self.run_main(["--version"])
        self.assertEqual(code, 0)
        self.assertIn(f"hauhau {__version__}", stdout.getvalue())
        self.assertEqual(self.stops, [], "--version must not start llama-server")

    def test_version_short_flag(self) -> None:
        from hauhau import __version__

        stdout = io.StringIO()
        with mock.patch.object(sys, "stdout", stdout):
            code, _out = self.run_main(["-V"])
        self.assertEqual(code, 0)
        self.assertIn(f"hauhau {__version__}", stdout.getvalue())
        self.assertEqual(self.stops, [])

    def test_missing_config_is_an_error_not_a_traceback(self) -> None:
        code, out = self.run_main(["--config", str(self.root / "nope.toml")])
        self.assertEqual(code, 2)
        self.assertIn("not found", out)

    def test_malformed_config_names_the_problem(self) -> None:
        broken = self.root / "broken.toml"
        broken.write_text("[model\nn_ctx = 8192\n")
        code, out = self.run_main(["--config", str(broken), "hi"])
        self.assertEqual(code, 2)
        self.assertIn("not valid TOML", out)
        self.assertNotIn("Traceback", out)

    def test_negative_max_steps_is_clamped(self) -> None:
        code, _out = self.run_main(["--max-steps", "-4", "hi"])
        self.assertEqual(code, 0)


class TestResume(MainTestCase):
    def test_unknown_session_starts_a_new_one(self) -> None:
        code, out = self.run_main(["--resume", "nope", "--yolo"], lines=["/quit"])
        self.assertEqual(code, 0)
        self.assertIn("no session named", out)

    def test_resume_cannot_escape_the_sessions_directory(self) -> None:
        outside = self.root / "notes.json"
        outside.write_text(json.dumps({"messages": [{"role": "user", "content": "secret"}], "summary": ""}))
        code, out = self.run_main(["--resume", "../notes", "--yolo"], lines=["/quit"])
        self.assertEqual(code, 0)
        self.assertIn("refusing --resume", out)
        # The stray file must still hold exactly what it held before.
        self.assertEqual(json.loads(outside.read_text())["messages"][0]["content"], "secret")

    def test_corrupt_session_file_is_ignored(self) -> None:
        sessions = self.root / "data" / "sessions"
        sessions.mkdir(parents=True)
        (sessions / "last.json").write_text("{not json")
        code, out = self.run_main(["--resume", "last", "--yolo"], lines=["/quit"])
        self.assertEqual(code, 0)
        self.assertNotIn("Traceback", out)


class TestInteractive(MainTestCase):
    def test_unknown_command_does_not_end_the_session(self) -> None:
        code, out = self.run_main(["--yolo"], lines=["", "/nonsense", "/help", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("unknown command", out)
        self.assertNotIn("Traceback", out)

    def test_backend_failure_shows_a_panel_and_keeps_going(self) -> None:
        FakeBackend.behaviour = "error"
        code, out = self.run_main(["--yolo"], lines=["do something", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("backend died", out)
        self.assertEqual(self.stops, [1])

    def test_ctrl_c_mid_turn_is_reported_and_survivable(self) -> None:
        FakeBackend.behaviour = "interrupt"
        code, out = self.run_main(["--yolo"], lines=["do something", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("interrupted", out)

    def test_a_command_that_raises_is_contained(self) -> None:
        # Observed: /save raised FileExistsError out of the prompt loop.
        with mock.patch.object(cli.Repl, "cmd_save", side_effect=OSError("disk is full")):
            code, out = self.run_main(["--yolo"], lines=["/save", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("/save failed", out)
        self.assertNotIn("Traceback", out)
        self.assertEqual(self.stops, [1])


if __name__ == "__main__":
    unittest.main()
