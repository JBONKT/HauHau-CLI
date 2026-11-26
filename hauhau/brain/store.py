"""Persistent memory store: sqlite + FTS5 keyword retrieval, optional vectors.

Chosen over ChromaDB/FAISS deliberately: on an 8 GB machine, a torch-based
embedder plus a vector index would compete with the model for the same RAM, and
``all-MiniLM`` style stacks have no wheels for Python 3.15. Keyword retrieval via
FTS5 is dependency-free and survives even when semantic search is unavailable.
"""

from __future__ import annotations

import array
import math
import re
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
    """Build a safe FTS5 OR-expression from free text.

    Punctuation is *split*, not deleted: FTS5 tokenises ``line-length`` as
    ``line`` + ``length``, so the old ``"linelength"`` pseudo-token could never
    match the row that mentions it. A phrase of the parts matches the real
    tokens, and the parts alone are added for looser recall.
    """
    phrases: list[str] = []
    for raw in normalize(query).split():
        parts = [p for p in re.split(r"[^0-9a-z_]+", raw) if p]
        if not parts:
            continue
        if len(parts) > 1:
            phrases.append(" ".join(parts))
        phrases.extend(part for part in parts if len(part) > 1)
    if not phrases:
        return None
    return " OR ".join(f'"{p}"' for p in dict.fromkeys(phrases))


# Tokens too common to be worth indexing: every row shares them, so bucketing by
# them degenerates to comparing every pair (which took ~18s on 2000 rows).
_COMMON_TOKENS = frozenset({
    "uses", "using", "with", "this", "that", "from", "into", "your", "have", "were",
    "will", "must", "should", "does", "then", "than", "them", "they", "there", "here",
    "when", "where", "which", "while", "what", "also", "only", "just", "some", "such",
    "same", "very", "more", "most", "much", "many", "each", "every", "other", "about",
    "after", "before", "during", "under", "over", "between", "through", "against",
    "being", "been", "both", "onto", "upon", "located", "purposes",
})


# With a concrete path in common, a little less lexical overlap is enough: the
# distiller writes "The release checklist is in docs/release.md and linting uses
# ruff…" as one row and "The project utilizes a release checklist located in
# docs/release.md" as another, scoring 0.75 containment against a 0.80 threshold.
SHARED_ARTIFACT_CONTAINMENT = 0.6


def _numbers(text: str) -> frozenset[str]:
    """Numeric tokens carry meaning: ports, versions, counts, step ids."""
    return frozenset(tok for tok in tokens_of(text) if tok.isdigit())


# `\b/` was meant to anchor the path but a word boundary cannot sit between a
# space and "/", so absolute paths never matched and every artifact-aware rule
# below was dead. A negative lookbehind is the anchor that works.
_ARTIFACT = re.compile(
    r"(?:~|(?<![\w])/(?:Users|opt|etc|usr|var|srv|home|private|Applications)\b)[\w./~-]*"
    r"|\b[\w-]+\.(?:py|toml|json|md|txt|gguf|sh|zsh|bash|db|ya?ml|cfg|ini|env)\b"
)


def _artifacts(text: str) -> frozenset[str]:
    """Concrete files, paths and URLs a memory points at."""
    return frozenset(match.group(0).lower() for match in _ARTIFACT.finditer(text))


def _numbers_agree(left: frozenset[str], right: frozenset[str]) -> bool:
    """Do two number sets permit a merge?

    Only a *conflict* blocks one: "port 5432" against "port 5433" is two facts,
    but "docs/release.md, line-length 120" against "docs/release.md" is the same
    fact with less detail. The sets are precomputed because the string version
    re-tokenised both sides on every pair - 1,000,000 regex calls, ~17s per
    compact() of 2000 rows.
    """
    return not left or not right or left == right


# A negated and a non-negated sentence are two facts, not one: the old dedupe
# merged "always runs the tests with -x" into "never runs the tests with -x" and
# kept the longer text, so the store asserted the opposite of what was said.
_NEGATIONS = frozenset({
    "not", "no", "never", "none", "nothing", "without", "avoid", "instead", "cannot",
    "cant", "dont", "doesnt", "isnt", "arent", "wont", "shouldnt", "mustnt", "disable",
    "disabled", "off",
})


# Swapping one of these for another does not change the fact: the distiller
# writes "the checklist is in docs/release.md" and "…lives in docs/release.md"
# in the same pass, and those must still merge.
_FUNCTION_WORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "lives", "lived", "located",
    "sits", "resides", "uses", "use", "runs", "run", "goes", "go", "has", "have",
    "the", "a", "an", "in", "on", "at", "for", "to", "of", "with", "and", "that",
})


def _mergeable(
    new_text: str,
    new_tokens: set[str],
    old_text: str,
    old_tokens: set[str],
    *,
    shared_artifact: bool,
    jaccard_min: float,
    containment_min: float,
) -> bool:
    """Should the incoming memory be merged into an existing one?

    Similarity alone is not enough. Two guards keep the store honest:

    * a negation on one side only means the texts disagree, whatever their
      word overlap;
    * containment is measured against the *shorter* text, so a one-word memory
      ("Ruff") scores 1.0 against any sentence that uses the word. Merging
      those deleted the incoming fact, so a large length mismatch is kept apart.
    """
    if not new_tokens or not old_tokens:
        return False
    if bool(new_tokens & _NEGATIONS) != bool(old_tokens & _NEGATIONS):
        return False
    # One word swapped for another is a different fact, however similar the
    # rest of the sentence: "deploys go through the staging pipeline" and
    # "... the production pipeline" scored 0.83 containment, and merging them
    # kept one value while reporting the other as stored.
    swapped = (new_tokens ^ old_tokens)
    if len(new_tokens - old_tokens) == 1 and len(old_tokens - new_tokens) == 1 and not swapped <= _FUNCTION_WORDS:
        return False
    if jaccard(new_tokens, old_tokens) >= jaccard_min:
        return True
    overlap = len(new_tokens & old_tokens)
    containment = overlap / min(len(new_tokens), len(old_tokens))
    if shared_artifact and containment >= SHARED_ARTIFACT_CONTAINMENT:
        return True
    if containment < containment_min:
        return False
    if len(new_text) <= len(old_text):
        short_text, short_tokens, long_text, long_tokens = new_text, new_tokens, old_text, old_tokens
    else:
        short_text, short_tokens, long_text, long_tokens = old_text, old_tokens, new_text, new_tokens
    if len(short_text) >= 0.5 * len(long_text):
        return True
    # A much shorter memory is only a repetition when the long one says
    # everything it says; "Ruff" is the subject of a sentence, not a repeat of it.
    return len(short_tokens) >= 3 and short_tokens <= long_tokens


def _merge_tags(old: str, new: str, limit: int = 400) -> str:
    """Union of two tag lists, trimmed at a tag boundary.

    A merge used to drop the incoming tags, and the first version of this
    function then cut the joined string at 400 characters - which wrote half a
    tag ("...tag056,t") into the row.
    """
    parts = [p.strip() for p in (old or "").split(",") if p.strip()]
    for part in (new or "").split(","):
        part = part.strip()
        if part and part not in parts:
            parts.append(part)
    kept: list[str] = []
    length = 0
    for part in parts:
        added = len(part) + (1 if kept else 0)
        if length + added > limit:
            break
        kept.append(part)
        length += added
    return ",".join(kept)


def _pack(vec: list[float]) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    return list(arr)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
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
        try:
            self.db.executescript(SCHEMA)
            # Rebuild the index *before* migrating: the cleanup UPDATEs below fire
            # the FTS triggers, and a delete against an index that was never built
            # for these rows made SQLite report "database disk image is malformed".
            self.db.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            self.db.commit()
            self._migrate()
        except sqlite3.OperationalError:
            # A restored backup on a read-only mount can still be searched; the
            # old unconditional rebuild made memory impossible there.
            pass

    def _migrate(self) -> None:
        """Bring an older ``brain.db`` up to the current schema.

        ``CREATE TABLE IF NOT EXISTS`` is a no-op on a mismatch, so an existing
        file kept its old columns and the first query failed - outside the
        "memory must never block startup" guard in the CLI, i.e. a traceback.
        """
        cols = {row["name"] for row in self.db.execute("PRAGMA table_info(memories)")}
        if not cols:
            return
        wanted = {
            "tags": "TEXT NOT NULL DEFAULT ''",
            "norm_hash": "TEXT NOT NULL DEFAULT ''",
            "importance": "REAL NOT NULL DEFAULT 0.5",
            "uses": "INTEGER NOT NULL DEFAULT 0",
            "created_at": "REAL NOT NULL DEFAULT 0",
            "last_used_at": "REAL NOT NULL DEFAULT 0",
            "embedding": "BLOB",
        }
        for name, decl in wanted.items():
            if name not in cols:
                self.db.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
        now = time.time()
        # Five full-table UPDATEs on every open would rewrite a large brain.db
        # for nothing; one probe decides whether any row needs repair.
        damaged = self.db.execute(
            "SELECT 1 FROM memories WHERE norm_hash IS NULL OR importance IS NULL"
            " OR uses IS NULL OR created_at > ? OR last_used_at IS NULL"
            " OR last_used_at = 0 OR last_used_at > ? LIMIT 1",
            (now, now),
        ).fetchone()
        if not damaged:
            self.db.commit()
            return
        self.db.execute("UPDATE memories SET norm_hash = '' WHERE norm_hash IS NULL")
        self.db.execute("UPDATE memories SET importance = 0.5 WHERE importance IS NULL")
        self.db.execute("UPDATE memories SET uses = 0 WHERE uses IS NULL")
        self.db.execute("UPDATE memories SET created_at = ? WHERE created_at > ?", (now, now))
        self.db.execute(
            "UPDATE memories SET last_used_at = created_at WHERE last_used_at IS NULL OR last_used_at = 0"
        )
        # A timestamp from the future (clock skew, a database copied from another
        # machine) pinned decay at 1.0 forever, so the row could never be pruned.
        self.db.execute("UPDATE memories SET last_used_at = ? WHERE last_used_at > ?", (now, now))
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # --- writes --------------------------------------------------------
    def _bump(self, row_id: int, now: float, new_text: str | None = None, new_tags: str = "") -> None:
        """Record a reuse. When ``new_text`` is a richer phrasing, adopt it."""
        current = self.db.execute("SELECT text, tags FROM memories WHERE id = ?", (row_id,)).fetchone()
        adopt = (
            new_text
            and current
            and len(new_text) > len(current["text"])
            and new_text.strip() != current["text"].strip()
        )
        tags = _merge_tags(current["tags"] if current else "", new_tags)
        if adopt:
            self.db.execute(
                "UPDATE memories SET uses = uses + 1, last_used_at = ?, text = ?, norm_hash = ?, "
                "tags = ?, importance = MIN(1.0, importance + 0.03) WHERE id = ?",
                (now, new_text, norm_hash(new_text), tags, row_id),
            )
        else:
            self.db.execute(
                "UPDATE memories SET uses = uses + 1, last_used_at = ?, tags = ?, "
                "importance = MIN(1.0, importance + 0.03) WHERE id = ?",
                (now, tags, row_id),
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

        # Dedupe ignores the kind label: the distiller writes the same fact as
        # both "semantic" and "procedural" in one pass, and two rows saying the
        # same thing crowd each other out of a top-k prompt.
        row = self.db.execute("SELECT id FROM memories WHERE norm_hash = ?", (digest,)).fetchone()
        if row:
            self._bump(row["id"], now, new_text=text, new_tags=tags)
            return int(row["id"]), False

        candidates = tokens_of(text)
        new_numbers = _numbers(text)
        new_artifacts = _artifacts(text)
        # Non-Latin text has no [a-z0-9_] token: dividing by zero here crashed
        # every write after the first for a user writing in Russian or Chinese.
        if candidates:
            for cand in self.db.execute(
                "SELECT id, text FROM memories ORDER BY last_used_at DESC LIMIT 400"
            ):
                other = tokens_of(cand["text"])
                if not other:
                    continue
                # Jaccard catches rewordings of the same length; containment
                # catches the same fact with an extra clause. Both are overridden
                # by the numbers: "port 5432" and "port 5433" differ.
                if not _numbers_agree(new_numbers, _numbers(cand["text"])):
                    continue
                if _mergeable(
                    text, candidates, cand["text"], other,
                    shared_artifact=bool(new_artifacts & _artifacts(cand["text"])),
                    jaccard_min=self.dedupe_jaccard,
                    containment_min=self.dedupe_containment,
                ):
                    self._bump(int(cand["id"]), now, new_text=text, new_tags=tags)
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
    def _decay(self, created_at: float, last_used_at: float, now: float) -> float:
        """Freshness halves every ``half_life_days``, counted from reinforcement.

        A fact re-confirmed today is fresh again: decay from ``created_at``
        alone would sink a memory that is still being used, which is the
        opposite of what a decay model is for. ``last_used_at`` is the newer
        of the two whenever the memory was merged or deliberately recalled.
        """
        reference = min(max(created_at or 0.0, last_used_at or 0.0), now)
        age_days = max(0.0, (now - reference) / 86400.0)
        return math.pow(0.5, age_days / self.half_life_days)

    def search(
        self,
        query: str,
        k: int = 5,
        query_embedding: list[float] | None = None,
        kinds: list[str] | None = None,
        record_use: bool = True,
    ) -> list[Hit]:
        """Rank memories for ``query``.

        ``record_use`` exists because the agent injects a memory block into
        every prompt: counting that as a "use" made ``uses`` meaningless and
        immunised every memory against ``prune`` after a single session. Only
        deliberate recalls (``/mem``, ``rag_query``) and write-merges count.
        """
        if k <= 0:
            return []
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
                # bm25() is negative and lower-is-better. Spread the results
                # across 0.5..1.0 so the best hit scores 1.0 instead of the 0.5
                # that a symmetric tanh put it at.
                best = min(r["rank"] for r in rows)
                spread = max(1e-6, max(r["rank"] for r in rows) - best)
                for r in rows:
                    base[int(r["id"])] = 1.0 - 0.5 * ((r["rank"] - best) / spread)

        if query_embedding:
            for r in self.db.execute("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"):
                try:
                    vector = _unpack(r["embedding"])
                except (ValueError, TypeError):
                    # A truncated or non-float32 blob (older writer, interrupted
                    # write) crashed every search once embeddings were enabled.
                    continue
                sim = _cosine(query_embedding, vector)
                if sim <= 0:
                    continue
                base[int(r["id"])] = base.get(int(r["id"]), 0.0) * 0.6 + sim * 0.4

        if not base:
            return []

        hits: list[Hit] = []
        placeholders = ",".join("?" * len(base))
        sql = (
            "SELECT id, kind, text, tags, importance, uses, created_at, last_used_at "
            f"FROM memories WHERE id IN ({placeholders})"
        )
        for r in self.db.execute(sql, tuple(base)):
            if kinds and r["kind"] not in kinds:
                continue
            decay = self._decay(r["created_at"], r["last_used_at"], now)
            # Both factors are <= 1.0, so the score stays within 0..1 and /mem
            # never shows a "perfect" match as 1.23. Importance nudges the rank
            # by at most 20% - clamped here too, because rows written before the
            # clamp existed (or by hand) are not.
            importance = r["importance"] if isinstance(r["importance"], (int, float)) else 0.5
            importance = max(0.0, min(1.0, float(importance)))
            score = base[int(r["id"])] * (0.35 + 0.65 * decay) * (0.8 + 0.2 * importance)
            hits.append(Hit(int(r["id"]), r["kind"], r["text"], r["tags"], score, r["created_at"], r["uses"]))

        hits.sort(key=lambda h: h.score, reverse=True)
        top = hits[:k]
        if top and record_use:
            try:
                self.db.executemany(
                    "UPDATE memories SET uses = uses + 1, last_used_at = ? WHERE id = ?",
                    [(now, h.id) for h in top],
                )
                self.db.commit()
            except sqlite3.OperationalError:
                # A database on a read-only mount is still worth reading; the
                # recall itself succeeded, only the usage counter could not be
                # written. Losing the counter beats losing the memory.
                self.db.rollback()
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
        rows = list(
            self.db.execute(
                "SELECT id, text, tags, uses, created_at, importance FROM memories ORDER BY created_at ASC"
            )
        )
        kept: list[dict] = []
        # Two texts that share no token can never pass either similarity test,
        # so index the survivors by token instead of comparing every pair:
        # the naive version took 16s on 2000 rows.
        index: dict[str, list[dict]] = {}
        for row in rows:
            tokens = tokens_of(row["text"])
            # Bucket on every distinctive token, not just the >=4-character ones:
            # "Use uv for the build" / "Use uv for the tests" share "uv" but no
            # long token, so compact() missed a pair that write() merges.
            keys = {tok for tok in tokens if tok not in _COMMON_TOKENS}
            seen: set[int] = set()
            candidates: list[dict] = []
            for token in keys:
                # No cap: a window made compact() permanently blind to exactly
                # the duplicates it exists to remove (write() merged them, so the
                # second pass never ran).
                for survivor in index.get(token, ()):
                    if id(survivor) not in seen:
                        seen.add(id(survivor))
                        candidates.append(survivor)
            if not keys:
                # Nothing distinctive to look up; compare the long way.
                candidates = list(kept)
            match = None
            nums = _numbers(row["text"])
            arts = _artifacts(row["text"])
            for survivor in candidates:
                if not _numbers_agree(nums, survivor["nums"]):
                    continue
                if _mergeable(
                    row["text"], tokens, survivor["text"], survivor["tokens"],
                    shared_artifact=bool(arts & survivor["arts"]),
                    jaccard_min=self.dedupe_jaccard,
                    containment_min=self.dedupe_containment,
                ):
                    match = survivor
                    break
            if match is None:
                entry = {
                    "id": row["id"], "text": row["text"], "tags": row["tags"] or "",
                    "tokens": tokens, "nums": nums,
                    "arts": arts, "uses": row["uses"], "created_at": row["created_at"],
                    "importance": row["importance"], "dirty": False,
                }
                kept.append(entry)
                for token in keys:
                    index.setdefault(token, []).append(entry)
                continue
            richest = max(match["text"], row["text"], key=len)
            match["uses"] += row["uses"] + 1
            match["tags"] = _merge_tags(match.get("tags", ""), row["tags"] or "")
            match["importance"] = max(match["importance"], row["importance"])
            match["dirty"] = True
            if richest != match["text"]:
                match["text"] = richest
                match["tokens"] = tokens_of(richest)
                match["nums"] = _numbers(richest)
                match["arts"] = _artifacts(richest)
                for token in (tok for tok in match["tokens"] if tok not in _COMMON_TOKENS):
                    bucket = index.setdefault(token, [])
                    if match not in bucket:
                        bucket.append(match)
            self.db.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
            absorbed += 1
        for survivor in kept:
            # Only touched rows are rewritten: every UPDATE fires the FTS
            # trigger, and rewriting all 2000 unchanged rows took ~17s.
            if not survivor["dirty"]:
                continue
            self.db.execute(
                "UPDATE memories SET text = ?, norm_hash = ?, tags = ?, uses = ?, importance = ? WHERE id = ?",
                (survivor["text"], norm_hash(survivor["text"]), survivor["tags"], survivor["uses"],
                 survivor["importance"], survivor["id"]),
            )
        self.db.commit()
        return absorbed

    def prune(self, min_weight: float = 0.02, keep_uses: int = 1) -> int:
        """Drop stale, never-reused memories. Returns the number removed."""
        now = time.time()
        doomed = []
        for r in self.db.execute("SELECT id, created_at, last_used_at, importance, uses FROM memories"):
            if r["uses"] >= keep_uses:
                continue
            weight = self._decay(r["created_at"], r["last_used_at"], now) * r["importance"]
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
