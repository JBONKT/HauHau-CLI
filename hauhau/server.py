"""Ownership of the llama-server child process.

The agent starts the server itself (with profiler-derived parameters) so that
config changes are applied to a process it owns and can restart. If a server is
already listening on the configured URL, it is reused untouched.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import subprocess
import time
import urllib.parse
from pathlib import Path

from .llm import LlamaClient, LLMError


class ServerManager:
    def __init__(self, cfg, audit=None) -> None:
        self.cfg = cfg
        self.audit = audit
        self._proc: subprocess.Popen | None = None
        self._log = None  # kept so stop() can close the server.log handle
        self._log_path = cfg.data_dir / "logs" / "server.log"
        atexit.register(self.stop)

    # --- helpers -------------------------------------------------------
    def client(self) -> LlamaClient:
        return LlamaClient(self.cfg.model.server_url)

    def _host_port(self) -> tuple[str, int]:
        parsed = urllib.parse.urlparse(self.cfg.model.server_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise LLMError(f"invalid port in [model] server_url {self.cfg.model.server_url!r}") from exc
        return parsed.hostname or "127.0.0.1", port or 8080

    def _spawn_args(self) -> list[str]:
        m = self.cfg.model
        host, port = self._host_port()
        args = [
            m.llama_server,
            "-m", os.path.expanduser(m.gguf),
            "-c", str(m.n_ctx),
            "-ngl", str(m.n_gpu_layers),
            "-t", str(m.n_threads),
            "--host", host,
            "--port", str(port),
        ]
        args.extend(m.extra_args)
        return args

    # --- lifecycle -----------------------------------------------------
    def ensure(self) -> LlamaClient:
        client = self.client()
        if client.health():
            return client
        if not self.cfg.model.manage_server:
            raise LLMError(
                f"no llama-server at {self.cfg.model.server_url} and manage_server is off. "
                f"Start one manually: {' '.join(self._spawn_args())}"
            )
        if not self.cfg.model.gguf:
            raise LLMError("no model configured: set [model] gguf in config.toml or $HAUHAU_GGUF")
        if not os.path.isfile(os.path.expanduser(self.cfg.model.gguf)):
            raise LLMError(f"model file not found: {self.cfg.model.gguf}")

        # A previous managed process (e.g. after a failed revive) must not be
        # orphaned: stop() terminates it and closes its log handle first.
        if self._proc is not None:
            self.stop()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        args = self._spawn_args()
        if self.audit:
            self.audit.log("server.start", argv=args)
        log = self._log_path.open("a", encoding="utf-8")
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} starting ===\n")
        log.flush()
        self._log = log
        try:
            self._proc = self._popen(args, log)
        except OSError as exc:
            # A missing llama-server binary must not leak the log fd or escape
            # as a raw FileNotFoundError the CLI does not catch.
            log.close()
            self._log = None
            raise LLMError(
                f"could not start {args[0]!r}: {exc}. Install llama.cpp or set "
                f"[model] llama_server to an absolute path."
            ) from exc

        deadline = time.monotonic() + self.cfg.model.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                # A half-started server must not survive the error path: the CLI
                # exits, so nobody would ever stop the process.
                code = self._proc.returncode
                self.stop()
                raise LLMError(f"llama-server exited with code {code}; see {self._log_path}")
            if client.health():
                return client
            time.sleep(0.5)
        self.stop()
        raise LLMError(
            f"llama-server did not become healthy in {self.cfg.model.startup_timeout}s; see {self._log_path}"
        )

    def _popen(self, args: list[str], log) -> subprocess.Popen:
        """Start the server in its own session.

        Without ``start_new_session`` the child sits in the terminal's foreground
        process group, so Ctrl-C during a long answer killed the model server and
        every later turn failed with "Connection refused".
        """
        return subprocess.Popen(
            args,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    def restart(self) -> LlamaClient:
        self.stop()
        return self.ensure()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._proc.wait(timeout=5)
        self._proc = None
        self._close_log()

    def _close_log(self) -> None:
        """Drop the append handle; one descriptor leaked per start otherwise."""
        if self._log is not None:
            with contextlib.suppress(OSError):
                self._log.close()
            self._log = None

    def status(self) -> dict:
        return {
            "url": self.cfg.model.server_url,
            "managed": self._proc is not None,
            "running": bool(self._proc and self._proc.poll() is None),
            "gguf": os.path.basename(os.path.expanduser(self.cfg.model.gguf)),
            "n_ctx": self.cfg.model.n_ctx,
            "n_threads": self.cfg.model.n_threads,
            "n_gpu_layers": self.cfg.model.n_gpu_layers,
            "log": str(Path(self._log_path)),
        }
