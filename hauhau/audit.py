"""Append-only audit trail for every shell command, write and config change."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


class Audit:
    def __init__(self, path: Path, session_id: str, enabled: bool = True) -> None:
        self.path = path
        self.session_id = session_id
        self.enabled = enabled
        self._lock = threading.Lock()
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            created = not path.exists()
            self._fh = path.open("a", encoding="utf-8")
            if created:
                os.chmod(path, 0o600)

    def log(self, event: str, **fields: Any) -> None:
        """Append one JSON line.

        The parameter is ``event``, not ``kind``, so callers can log their own
        ``kind=`` field: ``rag_write`` passed ``kind=`` straight into a
        parameter of the same name and crashed on every call.
        """
        if not self.enabled:
            return
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session": self.session_id,
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self.enabled and not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Audit:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return not self.enabled or self._fh.closed

    def tail(self, n: int = 10) -> list[dict]:
        if not self.path.exists():
            return []
        # lines[-0:] is the whole file and a negative n slices from the front,
        # so /audit 0 printed the entire log.
        n = max(1, int(n))
        out = []
        for line in self._tail_lines(n):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def _tail_lines(self, n: int) -> list[str]:
        """The last ``n`` lines, read backwards from the end of the file.

        The log only grows, so reading it whole to show twenty lines made
        ``/audit`` slower every week. A UTF-8 character can straddle the window
        edge, which is why the chunk is decoded with ``replace``.
        """
        window = 4096
        with self.path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            while True:
                start = max(0, size - window)
                handle.seek(start)
                lines = handle.read(size - start).decode("utf-8", "replace").splitlines()
                if start > 0 and lines:
                    lines = lines[1:]  # the first line is probably cut in half
                if start == 0 or len(lines) >= n or window >= size:
                    return lines[-n:]
                window *= 4
