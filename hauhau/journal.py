"""The journal: everything that happened, logged once, searchable by keyword.

Separate from the memory brain on purpose. A brain holds a few dozen distilled
lessons and decays them; the journal holds every mail, message, call, event and
reminder, and must not decay at all - "what did Dana say in March" is a question
about a record, not about a lesson. Keeping the two apart also means a sync every
15 minutes cannot flood the memories that get injected into the prompt.

Rows are keyed by ``(source, external_id)`` so a repeated sync is idempotent, and
each source keeps a cursor so a run resumes instead of re-reading history.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

SOURCES = ("mail", "messages", "calls", "calendar", "reminders")
KINDS = ("email", "message", "call", "event", "reminder")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT    NOT NULL,
    external_id TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    ts          REAL    NOT NULL,
    title       TEXT    NOT NULL DEFAULT '',
    body        TEXT    NOT NULL DEFAULT '',
    people      TEXT    NOT NULL DEFAULT '',
    meta        TEXT    NOT NULL DEFAULT '{}',
    ingested_at REAL    NOT NULL,
    UNIQUE(source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_people ON events(people);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    title, body, people,
    content='events',
    content_rowid='id',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, title, body, people)
    VALUES (new.id, new.title, new.body, new.people);
END;
CREATE TRIGGER IF NOT EXISTS events_ad AFTER DELETE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, title, body, people)
    VALUES ('delete', old.id, old.title, old.body, old.people);
END;
CREATE TRIGGER IF NOT EXISTS events_au AFTER UPDATE ON events BEGIN
    INSERT INTO events_fts(events_fts, rowid, title, body, people)
    VALUES ('delete', old.id, old.title, old.body, old.people);
    INSERT INTO events_fts(rowid, title, body, people)
    VALUES (new.id, new.title, new.body, new.people);
END;
CREATE TABLE IF NOT EXISTS cursors (
    source     TEXT PRIMARY KEY,
    cursor     TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS source_state (
    source     TEXT PRIMARY KEY,
    ok         INTEGER NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    checked_at REAL NOT NULL
);
"""


def _locked(method):
    """Serialize access: the sync job and an interactive turn can overlap."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class Entry:
    """One thing that happened, as the fetchers hand it over."""

    source: str
    external_id: str
    kind: str
    ts: float
    title: str = ""
    body: str = ""
    people: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; known: {', '.join(SOURCES)}")
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; known: {', '.join(KINDS)}")
        if not self.external_id:
            raise ValueError("external_id is required for idempotent sync")


@dataclass
class Row:
    """A stored event, as search and the brief return it."""

    id: int
    source: str
    kind: str
    ts: float
    title: str
    body: str
    people: str
    meta: dict
    score: float = 0.0

    @property
    def when(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.ts))

    def line(self, width: int = 160) -> str:
        text = (self.title or self.body or "").replace("\n", " ")
        return f"[{self.when}] {self.kind:<8} {text[:width]}"


class Journal:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    # --- writes --------------------------------------------------------
    @_locked
    def add(self, entries: list[Entry]) -> int:
        """Insert what is new. Returns how many rows were actually added."""
        now = time.time()
        added = 0
        for entry in entries:
            try:
                cursor = self.conn.execute(
                    "INSERT OR IGNORE INTO events"
                    " (source, external_id, kind, ts, title, body, people, meta, ingested_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        entry.source,
                        entry.external_id,
                        entry.kind,
                        float(entry.ts),
                        entry.title,
                        entry.body,
                        ",".join(entry.people),
                        json.dumps(entry.meta, ensure_ascii=False),
                        now,
                    ),
                )
            except sqlite3.Error as exc:
                raise RuntimeError(f"journal insert failed: {exc}") from exc
            # INSERT OR IGNORE reports 0 for a row that already existed, which is
            # what makes a repeated sync cost nothing instead of duplicating.
            added += max(0, cursor.rowcount)
        self.conn.commit()
        return added

    @_locked
    def cursor(self, source: str) -> str | None:
        row = self.conn.execute("SELECT cursor FROM cursors WHERE source = ?", (source,)).fetchone()
        return row["cursor"] if row else None

    @_locked
    def set_cursor(self, source: str, cursor: str) -> None:
        self.conn.execute(
            "INSERT INTO cursors (source, cursor, updated_at) VALUES (?, ?, ?)"
            " ON CONFLICT(source) DO UPDATE SET cursor = excluded.cursor,"
            " updated_at = excluded.updated_at",
            (source, str(cursor), time.time()),
        )
        self.conn.commit()

    @_locked
    def set_state(self, source: str, ok: bool, detail: str = "") -> None:
        """Remember how a source behaved, so a status check need not re-read it.

        Probing by syncing is expensive (Mail takes a minute) and leaves the
        permissions untested exactly when nobody is watching; recording the
        outcome of the real run keeps ``--status`` instant and honest.
        """
        self.conn.execute(
            "INSERT INTO source_state (source, ok, detail, checked_at) VALUES (?, ?, ?, ?)"
            " ON CONFLICT(source) DO UPDATE SET ok = excluded.ok,"
            " detail = excluded.detail, checked_at = excluded.checked_at",
            (source, 1 if ok else 0, detail, time.time()),
        )
        self.conn.commit()

    @_locked
    def states(self) -> dict[str, dict]:
        return {
            row["source"]: {
                "ok": bool(row["ok"]),
                "detail": row["detail"],
                "checked_at": row["checked_at"],
            }
            for row in self.conn.execute("SELECT * FROM source_state")
        }

    # --- reads ---------------------------------------------------------
    @_locked
    def search(
        self,
        query: str,
        k: int = 10,
        *,
        sources: list[str] | None = None,
        since: float | None = None,
    ) -> list[Row]:
        """Keyword search, newest-first on ties. Falls back to LIKE on bad syntax."""
        match = _fts_query(query)
        if match:
            params: list[object] = [match]
            sql = (
                "SELECT e.*, bm25(events_fts) AS rank FROM events_fts"
                " JOIN events e ON e.id = events_fts.rowid"
                " WHERE events_fts MATCH ?"
            )
            if sources:
                sql += f" AND e.source IN ({','.join('?' * len(sources))})"
                params.extend(sources)
            if since is not None:
                sql += " AND e.ts >= ?"
                params.append(since)
            sql += " ORDER BY rank LIMIT ?"
            params.append(k)
            try:
                rows = self.conn.execute(sql, params).fetchall()
                return [self._row(r, -float(r["rank"])) for r in rows]
            except sqlite3.OperationalError:
                pass
        return self._like(query, k, sources, since)

    @_locked
    def _like(self, query: str, k: int, sources: list[str] | None, since: float | None) -> list[Row]:
        needle = f"%{query.strip()}%"
        sql = "SELECT * FROM events WHERE (title LIKE ? OR body LIKE ? OR people LIKE ?)"
        params: list[object] = [needle, needle, needle]
        if sources:
            sql += f" AND source IN ({','.join('?' * len(sources))})"
            params.extend(sources)
        if since is not None:
            sql += " AND ts >= ?"
            params.append(since)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(k)
        return [self._row(r) for r in self.conn.execute(sql, params).fetchall()]

    @_locked
    def between(
        self,
        start: float,
        end: float,
        *,
        sources: list[str] | None = None,
        kinds: list[str] | None = None,
        limit: int = 200,
    ) -> list[Row]:
        sql = "SELECT * FROM events WHERE ts >= ? AND ts < ?"
        params: list[object] = [float(start), float(end)]
        if sources:
            sql += f" AND source IN ({','.join('?' * len(sources))})"
            params.extend(sources)
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        sql += " ORDER BY ts LIMIT ?"
        params.append(limit)
        return [self._row(r) for r in self.conn.execute(sql, params).fetchall()]

    @_locked
    def person(self, who: str, k: int = 15) -> list[Row]:
        needle = f"%{who.strip()}%"
        rows = self.conn.execute(
            "SELECT * FROM events WHERE people LIKE ? ORDER BY ts DESC LIMIT ?", (needle, k)
        ).fetchall()
        return [self._row(r) for r in rows]

    @_locked
    def latest(self, before: float | None = None) -> Row | None:
        """The newest event at or before ``before`` (default: now)."""
        cut = time.time() if before is None else before
        row = self.conn.execute(
            "SELECT * FROM events WHERE ts <= ? ORDER BY ts DESC LIMIT 1", (cut,)
        ).fetchone()
        return self._row(row) if row else None

    @_locked
    def stats(self) -> dict[str, object]:
        by_source = {
            row["source"]: row["n"]
            for row in self.conn.execute(
                "SELECT source, COUNT(*) AS n FROM events GROUP BY source ORDER BY source"
            )
        }
        newest = self.conn.execute("SELECT MAX(ts) AS ts FROM events").fetchone()["ts"]
        cursors = {
            row["source"]: row["cursor"]
            for row in self.conn.execute("SELECT source, cursor FROM cursors")
        }
        total = self.conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        return {"total": total, "by_source": by_source, "newest": newest, "cursors": cursors}

    @_locked
    def close(self) -> None:
        self.conn.close()

    # --- plumbing ------------------------------------------------------
    @staticmethod
    def _row(row: sqlite3.Row, score: float = 0.0) -> Row:
        try:
            meta = json.loads(row["meta"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        return Row(
            id=row["id"],
            source=row["source"],
            kind=row["kind"],
            ts=row["ts"],
            title=row["title"],
            body=row["body"],
            people=row["people"],
            meta=meta,
            score=score,
        )


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Quoting every token keeps a stray ``AND``, ``*`` or ``(`` from being parsed
    as syntax; a query with no usable tokens becomes empty so the caller can
    fall back to LIKE instead of raising.
    """
    tokens = [t for t in "".join(c if c.isalnum() or c.isspace() else " " for c in query).split() if t]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"*' for t in tokens)
