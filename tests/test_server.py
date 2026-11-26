from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest import mock

from hauhau.config import Config
from hauhau.llm import LLMError
from hauhau.server import ServerManager


def make_cfg(root: Path) -> Config:
    cfg = Config()
    cfg.logging.data_dir = str(root / "data")
    cfg.model.gguf = str(root / "model.gguf")
    cfg.model.n_ctx = 4096
    cfg.model.n_threads = 3
    cfg.model.n_gpu_layers = 99
    return cfg


class ServerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = make_cfg(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestSpawnArgs(ServerTestCase):
    def test_args_include_the_configured_parameters(self) -> None:
        manager = ServerManager(self.cfg)
        args = manager._spawn_args()
        self.assertEqual(args[0], self.cfg.model.llama_server)
        self.assertIn(str(self.root / "model.gguf"), args)
        self.assertIn("4096", args)
        self.assertIn("99", args)
        self.assertIn("3", args)
        self.assertIn("--port", args)
        self.assertIn("8080", args)

    def test_host_and_port_come_from_the_url(self) -> None:
        self.cfg.model.server_url = "http://0.0.0.0:9111"
        args = ServerManager(self.cfg)._spawn_args()
        self.assertEqual(args[args.index("--host") + 1], "0.0.0.0")
        self.assertEqual(args[args.index("--port") + 1], "9111")

    def test_extra_args_are_appended(self) -> None:
        self.cfg.model.extra_args = ["--no-warmup", "--verbose"]
        args = ServerManager(self.cfg)._spawn_args()
        self.assertEqual(args[-2:], ["--no-warmup", "--verbose"])


class TestIsolation(ServerTestCase):
    def test_child_is_started_in_its_own_session(self) -> None:
        # Regression: the child shared the terminal's process group, so Ctrl-C
        # during a long answer killed the model server for the rest of the session.
        manager = ServerManager(self.cfg)
        with mock.patch.object(subprocess, "Popen") as popen:
            manager._popen(["llama-server", "-m", "x.gguf"], StringIO())
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_stdio_is_not_inherited(self) -> None:
        manager = ServerManager(self.cfg)
        with mock.patch.object(subprocess, "Popen") as popen:
            manager._popen(["llama-server"], StringIO())
        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs["stdout"].__class__.__name__, "StringIO")
        self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)


class FakeClient:
    def __init__(self, healthy: bool) -> None:
        self.healthy = healthy

    def health(self) -> bool:
        return self.healthy


class TestEnsure(ServerTestCase):
    def test_existing_server_is_reused(self) -> None:
        manager = ServerManager(self.cfg)
        client = FakeClient(healthy=True)
        with mock.patch.object(manager, "client", return_value=client), \
                mock.patch.object(manager, "_popen") as popen:
            self.assertIs(manager.ensure(), client)
        popen.assert_not_called()

    def test_refuses_to_spawn_when_manage_server_is_off(self) -> None:
        self.cfg.model.manage_server = False
        manager = ServerManager(self.cfg)
        with (
            mock.patch.object(manager, "client", return_value=FakeClient(healthy=False)),
            self.assertRaises(LLMError) as ctx,
        ):
            manager.ensure()
        self.assertIn("manage_server is off", str(ctx.exception))
        self.assertIn("llama-server", str(ctx.exception), "must show how to start one")

    def test_refuses_when_no_model_is_configured(self) -> None:
        self.cfg.model.gguf = ""
        manager = ServerManager(self.cfg)
        with (
            mock.patch.object(manager, "client", return_value=FakeClient(healthy=False)),
            self.assertRaises(LLMError) as ctx,
        ):
            manager.ensure()
        self.assertIn("no model configured", str(ctx.exception))

    def test_refuses_when_the_model_file_is_missing(self) -> None:
        manager = ServerManager(self.cfg)
        with (
            mock.patch.object(manager, "client", return_value=FakeClient(healthy=False)),
            self.assertRaises(LLMError) as ctx,
        ):
            manager.ensure()
        self.assertIn("model file not found", str(ctx.exception))

    def test_reports_a_child_that_died_during_startup(self) -> None:
        (self.root / "model.gguf").write_text("stub")
        manager = ServerManager(self.cfg)
        dead = mock.Mock()
        dead.poll.return_value = 3
        dead.returncode = 3
        with mock.patch.object(manager, "client", return_value=FakeClient(healthy=False)), \
                mock.patch.object(manager, "_popen", return_value=dead), self.assertRaises(LLMError) as ctx:
            manager.ensure()
        self.assertIn("exited with code 3", str(ctx.exception))
        self.assertIn("server.log", str(ctx.exception))


class TestFailedStartupLeavesNothingBehind(ServerTestCase):
    def _stub_server(self, name: str = "stub-server") -> tuple[Path, Path]:
        """A fake llama-server that records its pid and then blocks."""
        pidfile = self.root / f"{name}.pid"
        script = self.root / name
        script.write_text(f"#!/bin/sh\necho $$ > '{pidfile}'\nexec sleep 30\n")
        script.chmod(0o755)
        return script, pidfile

    def test_failed_startup_kills_the_child_it_spawned(self) -> None:
        # The CLI exits on a startup error, so an unkilled child is an orphan.
        script, pidfile = self._stub_server()
        (self.root / "model.gguf").write_text("stub")
        self.cfg.model.llama_server = str(script)
        self.cfg.model.startup_timeout = 1.0
        manager = ServerManager(self.cfg)
        with mock.patch.object(manager, "client", return_value=FakeClient(healthy=False)), \
                self.assertRaises(LLMError) as ctx:
            manager.ensure()
        self.assertIn("did not become healthy", str(ctx.exception))
        pid = int(pidfile.read_text().strip())
        with self.assertRaises(OSError, msg=f"orphan llama-server survived startup failure (pid {pid})"):
            os.kill(pid, 0)

    def test_stop_closes_the_log_handle(self) -> None:
        script, _ = self._stub_server("stub-server-2")
        (self.root / "model.gguf").write_text("stub")
        self.cfg.model.llama_server = str(script)
        manager = ServerManager(self.cfg)
        # Unhealthy before the spawn (so ensure() actually starts one), healthy after.
        probe = mock.MagicMock()
        probe.health.side_effect = [False, True]
        probe.chat.return_value = "unused"
        with mock.patch.object(manager, "client", return_value=probe):
            manager.ensure()
        handle = manager._log
        self.assertIsNotNone(handle)
        manager.stop()
        self.assertIsNone(manager._log)
        self.assertTrue(handle.closed, "server.log handle leaked")


class TestStop(ServerTestCase):
    def test_stop_terminates_a_running_child(self) -> None:
        manager = ServerManager(self.cfg)
        proc = mock.Mock()
        proc.poll.return_value = None
        manager._proc = proc
        manager.stop()
        proc.terminate.assert_called_once()
        self.assertIsNone(manager._proc)

    def test_stop_escalates_when_the_child_ignores_sigterm(self) -> None:
        manager = ServerManager(self.cfg)
        proc = mock.Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("llama-server", 10)
        manager._proc = proc
        manager.stop()
        proc.kill.assert_called_once()

    def test_stop_is_safe_without_a_child(self) -> None:
        ServerManager(self.cfg).stop()

    def test_status_reports_an_external_server(self) -> None:
        status = ServerManager(self.cfg).status()
        self.assertFalse(status["managed"])
        self.assertFalse(status["running"])
        self.assertEqual(status["url"], self.cfg.model.server_url)


if __name__ == "__main__":
    unittest.main()
