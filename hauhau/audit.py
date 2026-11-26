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

    def tail(self, n: int = 10) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
