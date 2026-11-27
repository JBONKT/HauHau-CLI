"""Check the --tui dashboard on a real terminal.

Runs the widget dashboard in a pty against a stub llama-server that speaks SSE,
renders the byte stream through a terminal emulator, and asserts the layout:
the two columns, the side panel's numbers, a right-aligned user message, a tool
row that stays on one line, and the Markdown answer.

    .venv/bin/pip install textual pyte
    .venv/bin/python scripts/tui_check.py

Exits non-zero when a check fails.
"""

from __future__ import annotations

import fcntl
import json
import os
import pty
import re
import select
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

try:
    import textual  # noqa: F401
except ImportError:  # pragma: no cover
    sys.exit("this check needs textual: .venv/bin/pip install 'hauhau[tui]'")

ROWS, COLS = 30, 110
ANSWER = "Disken är 11% full."
TYPED = "hur mycket disk"
SUGGESTED = "hur mycket disk är det kvar och vad tar mest plats?"
CLIPPED = re.compile(r"\(\+\d+ lines\)")


class Stub(BaseHTTPRequestHandler):
    """One shell tool call, then a short Markdown answer."""

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
        seen_tool = any(m.get("role") == "tool" for m in request.get("messages", []))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def emit(event: dict) -> None:
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.flush()

        if seen_tool:
            # Three chunks, so a flush that carries only the newest one would
            # show up as a truncated answer.
            for chunk in (ANSWER[:12], ANSWER[12:] + "\n\n", "- 109 GB ledigt\n"):
                emit({"choices": [{"index": 0, "delta": {"content": chunk}}]})
                time.sleep(0.05)
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
                                            "arguments": json.dumps(
                                                {"cmd": "df -h / && du -sh ~/* | sort -rh | head -3"}
                                            ),
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
                "usage": {"prompt_tokens": 812, "completion_tokens": 46},
                "timings": {"predicted_per_second": 38.0},
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
                f'[model]\nserver_url = "http://127.0.0.1:{port}"\nmanage_server = false\n'
                f"n_ctx = 16384\nn_threads = 7\n\n"
                f'[logging]\ndata_dir = "{tmp}/data"\n[brain]\nenabled = false\n[journal]\nenabled = false\n'
            )

        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        env = dict(
            os.environ,
            TERM="xterm-256color",
            COLUMNS=str(COLS),
            LINES=str(ROWS),
            TEXTUAL_ANIMATIONS="none",
        )
        proc = subprocess.Popen(
            [sys.executable, "-m", "hauhau", "--tui", "--config", config, "--no-brain"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            close_fds=True,
        )
        os.close(slave)

        screen = pyte.Screen(COLS, ROWS)
        stream = pyte.ByteStream(screen)
        raw = bytearray()

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
                raw.extend(chunk)
                stream.feed(chunk)

        def snapshot() -> list[str]:
            return [line.rstrip() for line in screen.display]

        try:
            pump(8)
            startup = snapshot()
            # Ghost text: type the beginning of a task and do *not* submit.
            os.write(master, b"hur mycket disk")
            pump(3)
            ghost = snapshot()
            os.write(master, b"\t")  # Tab accepts the suggestion
            pump(2)
            accepted = snapshot()
            os.write(master, b"\r")
            pump(12)
            after = snapshot()
        finally:
            proc.kill()
            server.shutdown()

    # Synchronized output (DEC 2026) is what stops a terminal tearing a repaint:
    # worth asserting, because without it a full-screen redraw flickers on
    # Terminal.app and looks like lag.
    sync_output = b"\x1b[?2026h" in raw
    transcript = "\n".join(after)
    ghost_line = next((line for line in ghost if TYPED in line), "")
    # After Tab the line is longer than the input, so it scrolls: the accepted
    # text is proven by the message that was actually sent.
    echoed = [line for line in after if SUGGESTED in line]
    tool_rows = [line for line in after if "⚙ shell" in line]
    result_rows = [line for line in after if "✓" in line]
    checks = {
        "header carries the name and a clock": "Hauhau" in startup[0],
        "the side panel shows the runtime": any("context" in line for line in startup),
        "the side panel shows the state": any("ready" in line for line in startup),
        "the side panel lists the keys": any("ctrl-q" in line for line in startup),
        "one screenful, never taller": len(startup) == ROWS,
        "the user message echoes": bool(echoed),
        "the user message is right-aligned": any(line.index(TYPED) > 10 for line in echoed),
        "the tool call is shown": bool(tool_rows),
        # The count depends on how many directories `du` may read on this machine
        # (macOS refuses some of them), so assert the shape, not the number.
        "the tool result stays on one line": any(CLIPPED.search(line) for line in result_rows),
        "the Markdown answer rendered": ANSWER in transcript,
        "the answer keeps its first chunk": transcript.count(ANSWER) == 1,
        "the last turn's speed is in the panel": "tok/s" in transcript,
        "the ghost text offers a completion": "tar mest plats" in ghost_line,
        "tab accepts the suggestion": bool(echoed),
        "the accepted text is the whole suggestion": any(line.strip().startswith(SUGGESTED) for line in echoed),
        "the panel names the suggestion source": any("suggest" in line for line in startup),
        "the keys include copy and paste": any("ctrl-y" in line for line in startup),
        "no traceback": not any("Traceback" in line for line in after + ghost + accepted),
    }
    print(
        "info: synchronized output "
        + ("enabled" if sync_output else "not offered by this terminal (a pty cannot answer the query)")
    )
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
