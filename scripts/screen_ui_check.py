#!/usr/bin/env python
"""Check the full-screen interface on a real terminal.

Runs the app in a pty against a stub llama-server that speaks SSE, renders the
byte stream in a terminal emulator, and asserts the layout: header row, chrome
pinned to the bottom, one fixed frame, the transcript scrolling inside it.

    .venv/bin/pip install pyte
    .venv/bin/python scripts/screen_ui_check.py

Exits non-zero when a check fails.
"""

from __future__ import annotations

import fcntl
import json
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import pyte
except ImportError:  # pragma: no cover - a manual check, not a unit test
    sys.exit("this check needs pyte: .venv/bin/pip install pyte")

ROWS, COLS = 24, 100
TURNS = 6


class Stub(BaseHTTPRequestHandler):
    """llama-server lookalike: one shell tool call, then a plain answer."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        body = b'{"status":"ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        seen_tool_result = any(m.get("role") == "tool" for m in request.get("messages", []))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()

        if seen_tool_result:
            emit({"choices": [{"index": 0, "delta": {"content": "The shell tool printed stub-tool-ran."}}]})
            emit({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        else:
            emit(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "shell",
                                            "arguments": json.dumps({"cmd": "echo stub-tool-ran"}),
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                }
            )
            emit({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        emit(
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "timings": {"predicted_per_second": 42.0},
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    with tempfile.TemporaryDirectory() as tmp:
        config = os.path.join(tmp, "config.toml")
        with open(config, "w") as handle:
            handle.write(
                f'[model]\nserver_url = "http://127.0.0.1:{port}"\nmanage_server = false\n\n'
                f'[logging]\ndata_dir = "{tmp}/data"\n'
            )

        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        env = dict(os.environ, TERM="xterm-256color", COLUMNS=str(COLS), LINES=str(ROWS))
        proc = subprocess.Popen(
            [sys.executable, "-m", "hauhau", "--config", config],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            close_fds=True,
        )
        os.close(slave)

        screen = pyte.Screen(COLS, ROWS)
        stream = pyte.ByteStream(screen)

        def pump(seconds: float) -> None:
            end = time.time() + seconds
            while time.time() < end:
                ready, _, _ = select.select([master], [], [], 0.2)
                if not ready:
                    continue
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    return
                if not chunk:
                    return
                stream.feed(chunk)

        def snapshot() -> list[str]:
            return [line.rstrip() for line in screen.display]

        def send(text: str) -> None:
            os.write(master, text.encode())

        try:
            pump(4)
            startup = snapshot()
            for turn in range(1, TURNS + 1):
                send(f"turn number {turn}\r")
                pump(3)
            after = snapshot()
            send("\x1b[5~")  # PgUp
            pump(1.5)
            scrolled = snapshot()
            send("x")  # typing returns to the newest line
            pump(1.5)
            back = snapshot()

            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 14, 50, 0, 0))
            os.kill(proc.pid, signal.SIGWINCH)
            screen.resize(14, 50)
            pump(2)
            narrow = snapshot()
            send("\x04")
            pump(2)
        finally:
            proc.kill()
            server.shutdown()

    transcript = "\n".join(after)
    checks = {
        "fixed size, no terminal scroll": len(startup) == ROWS
        and all(len(line) <= COLS for line in startup),
        "header on the first row": "Hauhau" in startup[0] and "ctx" in startup[0],
        "input pinned to the last row": startup[-1].strip().startswith("❯"),  # noqa: RUF001
        "turns stream into the chat": f"turn number {TURNS}" in transcript,
        "tool call and result shown": "⚙ shell" in transcript and "✓ stub-tool-ran" in transcript,
        "old turns leave the fixed frame": transcript.count("turn number") < TURNS + 2,
        "PgUp scrolls inside the frame": scrolled != after and "Hauhau" in scrolled[0],
        "typing returns to the newest line": f"turn number {TURNS}" in "\n".join(back),
        "live resize keeps the frame pinned": len(narrow) == 14
        and "Hauhau" in narrow[0]
        and narrow[-1].strip().startswith("❯"),  # noqa: RUF001 - prompt glyph
        "no traceback": not any("Traceback" in line for line in after + scrolled + back + narrow),
    }
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
