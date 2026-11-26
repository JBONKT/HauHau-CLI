"""Ownership of the llama-server child process.

The agent starts the server itself (with profiler-derived parameters) so that
config changes are applied to a process it owns and can restart. If a server is
already listening on the configured URL, it is reused untouched.
"""

from __future__ import annotations

import atexit
import os
import subprocess
import time
import urllib.parse
from pathlib import Path

from .llm import LLMError, LlamaClient


class ServerManager:
    def __init__(self, cfg, audit=None) -> None:
        self.cfg = cfg
        self.audit = audit
        self._proc: subprocess.Popen | None = None
        self._log_path = cfg.data_dir / "logs" / "server.log"
        atexit.register(self.stop)

    # --- helpers -------------------------------------------------------
    def client(self) -> LlamaClient:
        return LlamaClient(self.cfg.model.server_url)

    def _host_port(self) -> tuple[str, int]:
        parsed = urllib.parse.urlparse(self.cfg.model.server_url)
        return parsed.hostname or "127.0.0.1", parsed.port or 8080

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

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        args = self._spawn_args()
        if self.audit:
            self.audit.log("server.start", argv=args)
        log = self._log_path.open("a", encoding="utf-8")
        log.write(f"\n=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} starting ===\n")
        log.flush()
        self._proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

        deadline = time.monotonic() + self.cfg.model.startup_timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise LLMError(
                    f"llama-server exited with code {self._proc.returncode}; see {self._log_path}"
                )
            if client.health():
                return client
            time.sleep(0.5)
        raise LLMError(f"llama-server did not become healthy in {self.cfg.model.startup_timeout}s; see {self._log_path}")

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
        self._proc = None

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
