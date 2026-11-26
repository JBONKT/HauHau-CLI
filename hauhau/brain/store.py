"""Persistent memory store: sqlite + FTS5 keyword retrieval, optional vectors.

Chosen over ChromaDB/FAISS deliberately: on an 8 GB machine, a torch-based
embedder plus a vector index would compete with the model for the same RAM, and
``all-MiniLM`` style stacks have no wheels for Python 3.15. Keyword retrieval via
FTS5 is dependency-free and survives even when semantic search is unavailable.
"""

from __future__ import annotations

import array
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from ..util import jaccard, norm_hash, normalize, tokens_of

KINDS = ("semantic", "procedural", "episodic", "preference")

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT    NOT NULL,
    text         TEXT    NOT NULL,
    tags         TEXT    NOT NULL DEFAULT '',
    norm_hash    TEXT    NOT NULL,
    importance   REAL    NOT NULL DEFAULT 0.5,
    uses         INTEGER NOT NULL DEFAULT 0,
    created_at   REAL    NOT NULL,
    last_used_at REAL    NOT NULL,
    embedding    BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(norm_hash);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, tags,
    content='memories',
    content_rowid='id',
    tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text, tags) VALUES ('delete', old.id, old.text, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text, tags) VALUES ('delete', old.id, old.text, old.tags);
    INSERT INTO memories_fts(rowid, text, tags) VALUES (new.id, new.text, new.tags);
END;
"""


@dataclass
class Hit:
    id: int
    kind: str
    text: str
    tags: str
    score: float
    created_at: float
    uses: int

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.created_at) / 86400.0)


def _match_expr(query: str) -> str | None:
    """Build a safe FTS5 OR-expression from free text."""
    terms = [t for t in normalize(query).split() if len(t) > 1]
    terms = [t for t in ("".join(c for c in term if c.isalnum() or c == "_") for term in terms) if t]
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))


def _pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryStore:
    def __init__(
        self,
        path: Path,
        half_life_days: float = 30.0,
        dedupe_jaccard: float = 0.9,
        dedupe_containment: float = 0.8,
    ) -> None:
        self.path = path
        self.half_life_days = max(0.1, half_life_days)
        self.dedupe_jaccard = dedupe_jaccard
        self.dedupe_containment = dedupe_containment
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # --- writes --------------------------------------------------------
    def _bump(self, row_id: int, now: float, new_text: str | None = None) -> None:
        """Record a reuse. When ``new_text`` is a richer phrasing, adopt it."""
        current = self.db.execute("SELECT text FROM memories WHERE id = ?", (row_id,)).fetchone()
        adopt = (
            new_text
            and current
            and len(new_text) > len(current["text"])
            and new_text.strip() != current["text"].strip()
        )
        if adopt:
            self.db.execute(
                "UPDATE memories SET uses = uses + 1, last_used_at = ?, text = ?, norm_hash = ?, "
                "importance = MIN(1.0, importance + 0.03) WHERE id = ?",
                (now, new_text, norm_hash(new_text), row_id),
            )
        else:
            self.db.execute(
                "UPDATE memories SET uses = uses + 1, last_used_at = ?, "
                "importance = MIN(1.0, importance + 0.03) WHERE id = ?",
                (now, row_id),
            )
        self.db.commit()

    def write(
        self,
        kind: str,
        text: str,
        tags: str = "",
        importance: float = 0.5,
        embedding: list[float] | None = None,
    ) -> tuple[int, bool]:
        """Store a memory. Returns (id, created); near-duplicates are merged."""
        text = text.strip()
        if not text:
            raise ValueError("cannot store an empty memory")
        kind = kind if kind in KINDS else "semantic"
        now = time.time()
        digest = norm_hash(text)

        row = self.db.execute(
            "SELECT id FROM memories WHERE norm_hash = ? AND kind = ?", (digest, kind)
        ).fetchone()
        if row:
            self._bump(row["id"], now, new_text=text)
            return int(row["id"]), False

        candidates = tokens_of(text)
        for cand in self.db.execute(
            "SELECT id, text FROM memories WHERE kind = ? ORDER BY last_used_at DESC LIMIT 400", (kind,)
        ):
            other = tokens_of(cand["text"])
            if not other:
                continue
            overlap = len(candidates & other)
            # Jaccard catches rewordings of the same length; containment catches
            # the same fact stated with an extra clause ("X is at P" vs
            # "X files are located at P"), which Jaccard scores too low.
            if (
                jaccard(candidates, other) >= self.dedupe_jaccard
                or overlap / min(len(candidates), len(other)) >= self.dedupe_containment
            ):
                self._bump(int(cand["id"]), now, new_text=text)
                return int(cand["id"]), False

        cur = self.db.execute(
            "INSERT INTO memories (kind, text, tags, norm_hash, importance, uses, created_at, last_used_at, embedding)"
            " VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)",
            (kind, text, tags, digest, max(0.0, min(1.0, importance)), now, now,
             _pack(embedding) if embedding else None),
        )
        self.db.commit()
        return int(cur.lastrowid), True

    def forget(self, row_id: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (row_id,))
        self.db.commit()
        return cur.rowcount > 0

    # --- reads ---------------------------------------------------------
    def _decay(self, created_at: float, now: float) -> float:
        age_days = max(0.0, (now - created_at) / 86400.0)
        return math.pow(0.5, age_days / self.half_life_days)

    def search(
        self,
        query: str,
        k: int = 5,
        query_embedding: list[float] | None = None,
        kinds: list[str] | None = None,
    ) -> list[Hit]:
        now = time.time()
        base: dict[int, float] = {}

        expr = _match_expr(query)
        if expr:
            try:
                rows = self.db.execute(
                    "SELECT rowid AS id, bm25(memories_fts) AS rank FROM memories_fts "
                    "WHERE memories_fts MATCH ? ORDER BY rank LIMIT 64",
                    (expr,),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            if rows:
                best = min(r["rank"] for r in rows)
                for r in rows:
                    # bm25() is negative and lower-is-better; flip to a 0..1 scale.
                    base[int(r["id"])] = math.tanh((best - r["rank"]) * 0.5) * 0.5 + 0.5

        if query_embedding:
            for r in self.db.execute("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"):
                sim = _cosine(query_embedding, _unpack(r["embedding"]))
                if sim <= 0:
                    continue
                base[int(r["id"])] = base.get(int(r["id"]), 0.0) * 0.6 + sim * 0.4

        if not base:
            return []

        hits: list[Hit] = []
        placeholders = ",".join("?" * len(base))
        sql = f"SELECT id, kind, text, tags, importance, uses, created_at FROM memories WHERE id IN ({placeholders})"
        for r in self.db.execute(sql, tuple(base)):
            if kinds and r["kind"] not in kinds:
                continue
            decay = self._decay(r["created_at"], now)
            score = base[int(r["id"])] * (0.35 + 0.65 * decay) * (0.75 + 0.5 * r["importance"])
            hits.append(Hit(int(r["id"]), r["kind"], r["text"], r["tags"], score, r["created_at"], r["uses"]))

        hits.sort(key=lambda h: h.score, reverse=True)
        top = hits[:k]
        if top:
            self.db.executemany(
                "UPDATE memories SET uses = uses + 1, last_used_at = ? WHERE id = ?",
                [(now, h.id) for h in top],
            )
            self.db.commit()
        return top

    def recent(self, kind: str | None = None, limit: int = 20) -> list[Hit]:
        sql = "SELECT id, kind, text, tags, importance, uses, created_at, last_used_at FROM memories"
        params: tuple = ()
        if kind:
            sql += " WHERE kind = ?"
            params = (kind,)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params += (limit,)
        return [
            Hit(int(r["id"]), r["kind"], r["text"], r["tags"], r["importance"], r["created_at"], r["uses"])
            for r in self.db.execute(sql, params)
        ]

    def compact(self) -> int:
        """Merge near-duplicate memories written before a dedupe change.

        Keeps the richest phrasing, sums reuse counts and keeps the earliest
        timestamp. Returns how many rows were absorbed.
        """
        absorbed = 0
        for kind in KINDS:
            rows = list(
                self.db.execute(
                    "SELECT id, text, uses, created_at, importance FROM memories WHERE kind = ? "
                    "ORDER BY created_at ASC",
                    (kind,),
                )
            )
            kept: list[dict] = []
            for row in rows:
                tokens = tokens_of(row["text"])
                match = None
                for survivor in kept:
                    other = survivor["tokens"]
                    if not tokens or not other:
                        continue
                    overlap = len(tokens & other)
                    if (
                        jaccard(tokens, other) >= self.dedupe_jaccard
                        or overlap / min(len(tokens), len(other)) >= self.dedupe_containment
                    ):
                        match = survivor
                        break
                if match is None:
                    kept.append(
                        {"id": row["id"], "text": row["text"], "tokens": tokens,
                         "uses": row["uses"], "created_at": row["created_at"],
                         "importance": row["importance"]}
                    )
                    continue
                richest = max(match["text"], row["text"], key=len)
                match["uses"] += row["uses"] + 1
                match["importance"] = max(match["importance"], row["importance"])
                if richest != match["text"]:
                    match["text"] = richest
                    match["tokens"] = tokens_of(richest)
                self.db.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
                absorbed += 1
            for survivor in kept:
                self.db.execute(
                    "UPDATE memories SET text = ?, norm_hash = ?, uses = ?, importance = ? WHERE id = ?",
                    (survivor["text"], norm_hash(survivor["text"]), survivor["uses"],
                     survivor["importance"], survivor["id"]),
                )
        self.db.commit()
        return absorbed

    def prune(self, min_weight: float = 0.02, keep_uses: int = 1) -> int:
        """Drop stale, never-reused memories. Returns the number removed."""
        now = time.time()
        doomed = []
        for r in self.db.execute("SELECT id, created_at, importance, uses FROM memories"):
            if r["uses"] >= keep_uses:
                continue
            weight = self._decay(r["created_at"], now) * r["importance"]
            if weight < min_weight:
                doomed.append(int(r["id"]))
        if doomed:
            self.db.executemany("DELETE FROM memories WHERE id = ?", [(i,) for i in doomed])
            self.db.commit()
        return len(doomed)

    def stats(self) -> dict:
        counts = {row["kind"]: row["n"] for row in self.db.execute(
            "SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind")}
        total = sum(counts.values())
        with_vec = self.db.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE embedding IS NOT NULL").fetchone()["n"]
        return {
            "total": total,
            "by_kind": counts,
            "with_embeddings": with_vec,
            "db": str(self.path),
            "db_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }
