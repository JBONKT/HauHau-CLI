from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from hauhau.brain.store import MemoryStore


class StoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "brain.db"
        self.store = MemoryStore(self.db, half_life_days=30.0, dedupe_jaccard=0.9)

    def tearDown(self) -> None:
        self.store.close()
        self._tmp.cleanup()


class TestWrite(StoreTestCase):
    def test_write_returns_new_id(self) -> None:
        row_id, created = self.store.write("semantic", "The user prefers uv over pip for Python projects.")
        self.assertTrue(created)
        self.assertGreater(row_id, 0)

    def test_exact_duplicate_merges_instead_of_inserting(self) -> None:
        first, _ = self.store.write("semantic", "The build uses make test-all.")
        second, created = self.store.write("semantic", "The   build uses make test-all.")
        self.assertFalse(created)
        self.assertEqual(first, second)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_near_duplicate_merges(self) -> None:
        self.store.write("preference", "The user wants short answers with no preamble in this terminal.")
        _, created = self.store.write(
            "preference", "The user wants short answers with no preamble in this terminal session."
        )
        self.assertFalse(created)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_near_duplicate_merges_across_kinds(self) -> None:
        # Observed live: one learn pass wrote the same fact as "semantic" and as
        # "procedural". The kind is a label; the sentence is the memory.
        first, _ = self.store.write("semantic", "The release checklist is in docs/release.md.")
        second, created = self.store.write("procedural", "The release checklist lives in docs/release.md.")
        self.assertFalse(created)
        self.assertEqual(first, second)
        row = self.store.db.execute("SELECT kind FROM memories").fetchone()
        self.assertEqual(row["kind"], "semantic", "the first label wins")

    def test_different_fact_in_another_kind_is_kept(self) -> None:
        self.store.write("semantic", "Deploys happen from the main branch.")
        _, created = self.store.write(
            "procedural", "Run pytest -x before a deploy so a failure stops the push."
        )
        self.assertTrue(created)
        self.assertEqual(self.store.stats()["total"], 2)

    def test_containment_catches_rephrased_fact(self) -> None:
        # Observed live: the same path fact stored twice because Jaccard (0.61)
        # fell under the threshold while containment (0.80) would have caught it.
        first, _ = self.store.write(
            "semantic", "The directory for GGUF files is located at /Users/johan/modeller/gguf."
        )
        second, created = self.store.write(
            "semantic", "The GGUF model files are located at /Users/johan/modeller/gguf."
        )
        self.assertFalse(created, "rephrasing of a known fact must merge")
        self.assertEqual(first, second)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_different_numbers_are_different_facts(self) -> None:
        # "port 5432" and "port 5433" are 0.89 contained in each other, but a port
        # number is not prose: merging them would lose a real distinction.
        self.store.write("semantic", "The staging database listens on port 5432.")
        _, created = self.store.write("semantic", "The staging database listens on port 5433.")
        self.assertTrue(created, "different ports must not merge")
        self.assertEqual(self.store.stats()["total"], 2)

    def test_shared_file_merges_at_lower_containment(self) -> None:
        # Observed live: one learn pass wrote both of these (0.70 containment,
        # under the 0.80 threshold) and they filled two of the five prompt slots.
        self.store.write(
            "procedural", "The release checklist is in docs/release.md and linting uses ruff with a line-length of 120."
        )
        _, created = self.store.write(
            "semantic", "The project utilizes a release checklist located in docs/release.md."
        )
        self.assertFalse(created)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_shared_file_does_not_merge_different_facts(self) -> None:
        # A number present on one side only is detail, not contradiction...
        self.store.write("semantic", "The deploy script bin/deploy.sh takes about 2.5 minutes to run.")
        _, created = self.store.write("semantic", "The deploy script bin/deploy.sh takes a while to run.")
        self.assertFalse(created, "same file, number only on one side")
        # ...but two different branches are two facts.
        with tempfile.TemporaryDirectory() as other_dir:
            other = MemoryStore(Path(other_dir) / "other.db", half_life_days=30.0, dedupe_jaccard=0.9)
            other.write("semantic", "Deploys come from the staging-branch pipeline on merges.")
            _, created = other.write("semantic", "Deploys come from the production-branch pipeline on tags.")
            other.close()
        self.assertTrue(created)

    def test_identical_numbers_still_merge(self) -> None:
        # Same fact plus an extra clause: containment clears the threshold and the
        # numbers agree, so this must merge.
        self.store.write("semantic", "The deploy script is bin/deploy.sh on port 5432.")
        _, created = self.store.write(
            "semantic", "The deploy script is bin/deploy.sh on port 5432 with --force."
        )
        self.assertFalse(created)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_merge_adopts_the_richer_phrasing(self) -> None:
        self.store.write("preference", "Uses uv for python.")
        richer = "Uses uv for all python projects on this machine."
        row_id, created = self.store.write("preference", richer)
        self.assertFalse(created)
        row = self.store.db.execute("SELECT text, uses FROM memories WHERE id = ?", (row_id,)).fetchone()
        self.assertEqual(row["text"], richer)
        self.assertEqual(row["uses"], 1)

    def test_merge_keeps_the_shorter_text_when_it_is_already_richest(self) -> None:
        longer = "The user always runs the test suite with the -x flag on this machine."
        first, _ = self.store.write("preference", longer)
        _, created = self.store.write("preference", "The user always runs the test suite with -x flag.")
        self.assertFalse(created)
        row = self.store.db.execute("SELECT text FROM memories WHERE id = ?", (first,)).fetchone()
        self.assertEqual(row["text"], longer)

    def test_empty_text_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.store.write("semantic", "   ")


class TestSearch(StoreTestCase):
    def test_keyword_hit(self) -> None:
        self.store.write("semantic", "The GPU is an Apple M1 with 8 GB unified memory.")
        self.store.write("preference", "The user dislikes trailing summaries.")
        hits = self.store.search("which GPU does the machine have", k=5)
        self.assertTrue(hits)
        self.assertIn("M1", hits[0].text)

    def test_best_match_scores_near_one(self) -> None:
        # Regression: a symmetric tanh mapped the best hit to 0.5, so /mem showed
        # a perfect match as 0.50.
        self.store.write("semantic", "Ruff is the linter and mypy runs separately.")
        self.store.write("semantic", "The release branch is cut on Fridays.")
        hits = self.store.search("ruff linter", k=2)
        self.assertTrue(hits)
        self.assertGreater(hits[0].score, 0.85, f"best hit scored {hits[0].score:.2f}")
        self.assertIn("Ruff", hits[0].text)

    def test_scores_never_exceed_one(self) -> None:
        # Regression: the importance boost could push a score to 1.23, which read
        # as a bug in /mem. Max importance plus maximum recency must still be <= 1.
        row_id, _ = self.store.write("semantic", "Everything about the deploy pipeline.", importance=1.0)
        self.store.db.execute("UPDATE memories SET uses = 20 WHERE id = ?", (row_id,))
        self.store.db.commit()
        hits = self.store.search("deploy pipeline", k=1)
        self.assertTrue(hits)
        self.assertLessEqual(hits[0].score, 1.0)

    def test_scores_are_monotonic_with_rank(self) -> None:
        self.store.write("semantic", "deploy script one")
        self.store.write("semantic", "deploy script two")
        hits = self.store.search("deploy script", k=5)
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_match_returns_empty(self) -> None:
        self.store.write("semantic", "Postgres runs on port 5432.")
        self.assertEqual(self.store.search("zzzqqq"), [])

    def test_recall_bumps_use_counter(self) -> None:
        row_id, _ = self.store.write("procedural", "Run pytest with the -x flag to stop on first failure.")
        self.store.search("pytest", k=1)
        recent = self.store.recent(limit=1)[0]
        self.assertEqual(recent.id, row_id)
        self.assertEqual(recent.uses, 1)

    def test_decay_favours_recent_memory(self) -> None:
        old_id, _ = self.store.write("semantic", "the deploy script lives in bin/deploy")
        self.store.db.execute("UPDATE memories SET created_at = ? WHERE id = ?", (time.time() - 86400 * 300, old_id))
        self.store.db.commit()
        new_id, _ = self.store.write("semantic", "the deploy script moved to scripts/deploy")
        hits = self.store.search("deploy script", k=5)
        self.assertGreaterEqual(len(hits), 2)
        self.assertEqual(hits[0].id, new_id)

    def test_kind_filter(self) -> None:
        self.store.write("preference", "User likes tabs over spaces in Go files.")
        self.store.write("semantic", "The Go project uses tabs for indentation.")
        hits = self.store.search("tabs spaces indentation", k=5, kinds=["preference"])
        self.assertTrue(all(h.kind == "preference" for h in hits))


class TestVectors(StoreTestCase):
    """Hybrid retrieval: the keyword half and the vector half, separately."""

    def test_embedding_is_stored(self) -> None:
        self.store.write("semantic", "Vector fact.", embedding=[1.0, 0.0, 0.0])
        self.assertEqual(self.store.stats()["with_embeddings"], 1)

    def test_vector_only_match_is_found(self) -> None:
        # No shared keywords with the query: only the vector can match this.
        self.store.write("semantic", "Quarterly revenue grew.", embedding=[1.0, 0.0, 0.0])
        self.assertEqual(self.store.search("zzz unrelated keywords"), [])
        hits = self.store.search("zzz unrelated keywords", query_embedding=[1.0, 0.0, 0.0])
        self.assertEqual(len(hits), 1)
        self.assertIn("revenue", hits[0].text)

    def test_orthogonal_vector_does_not_match(self) -> None:
        self.store.write("semantic", "Quarterly revenue grew.", embedding=[1.0, 0.0, 0.0])
        self.assertEqual(self.store.search("zzz", query_embedding=[0.0, 1.0, 0.0]), [])

    def test_hybrid_prefers_keyword_and_vector_agreement(self) -> None:
        self.store.write("semantic", "Deploy with make release.", embedding=[1.0, 0.0, 0.0])
        self.store.write("semantic", "Deploy with scripts/release.sh.", embedding=[0.0, 1.0, 0.0])
        hits = self.store.search("make release", query_embedding=[1.0, 0.0, 0.0])
        self.assertEqual(hits[0].text, "Deploy with make release.")

    def test_mismatched_dimensions_are_not_scored(self) -> None:
        self.store.write("semantic", "Vector fact.", embedding=[1.0, 0.0, 0.0])
        self.assertEqual(self.store.search("zzz", query_embedding=[1.0, 0.0]), [])

    def test_memories_without_vectors_still_match_by_keyword(self) -> None:
        self.store.write("semantic", "Plain keyword memory.")
        hits = self.store.search("plain keyword", query_embedding=[1.0, 0.0, 0.0])
        self.assertEqual(len(hits), 1)


class TestCompaction(StoreTestCase):
    def test_merges_rows_written_before_dedupe_existed(self) -> None:
        # Simulate a legacy store: bypass write() dedupe with raw inserts.
        import time as _time

        now = _time.time()
        for text in (
            "The directory for GGUF files is located at /Users/johan/modeller/gguf.",
            "The GGUF model files are located at /Users/johan/modeller/gguf.",
        ):
            self.store.db.execute(
                "INSERT INTO memories (kind, text, tags, norm_hash, importance, uses, created_at, last_used_at)"
                " VALUES ('semantic', ?, '', ?, 0.5, 2, ?, ?)",
                (text, text, now, now),
            )
        self.store.db.commit()
        self.assertEqual(self.store.stats()["total"], 2)

        self.assertEqual(self.store.compact(), 1)
        self.assertEqual(self.store.stats()["total"], 1)
        row = self.store.db.execute("SELECT text, uses FROM memories").fetchone()
        self.assertEqual(row["uses"], 5, "reuse counts are summed plus the merge itself")
        self.assertIn("modeller/gguf", row["text"])

    def test_leaves_distinct_memories_alone(self) -> None:
        self.store.write("semantic", "The linter is ruff, run with make lint.")
        self.store.write("semantic", "Deploys go out from the release branch on Fridays.")
        self.assertEqual(self.store.compact(), 0)
        self.assertEqual(self.store.stats()["total"], 2)

    def test_is_idempotent(self) -> None:
        self.store.write("semantic", "The database lives in ~/.hauhau/brain/brain.db.")
        self.store.write("semantic", "The database lives in ~/.hauhau/brain/brain.db on this machine.")
        first = self.store.compact()
        self.assertEqual(self.store.compact(), 0)
        self.assertLessEqual(first, 1)
        self.assertEqual(self.store.stats()["total"], 1)


class TestMaintenance(StoreTestCase):
    def test_prune_drops_stale_unused(self) -> None:
        row_id, _ = self.store.write("episodic", "Session about a throwaway temp file.", importance=0.1)
        # An untouched memory has always had last_used_at == created_at; backdate both.
        old = time.time() - 86400 * 400
        self.store.db.execute(
            "UPDATE memories SET created_at = ?, last_used_at = ? WHERE id = ?", (old, old, row_id)
        )
        self.store.db.commit()
        self.assertEqual(self.store.prune(min_weight=0.05), 1)
        self.assertEqual(self.store.stats()["total"], 0)

    def test_reinforced_memory_outranks_a_stale_one(self) -> None:
        """Decay counts from the last reinforcement, not from creation."""
        old_id, _ = self.store.write(
            "semantic", "Deploys go through the staging-docker pipeline at ci.example.", importance=0.8
        )
        self.store.write("semantic", "The weather widget uses a greyscale palette.", importance=0.8)
        long_ago = time.time() - 86400 * 400
        self.store.db.execute(
            "UPDATE memories SET created_at = ?, last_used_at = ? WHERE id = ?",
            (long_ago, time.time(), old_id),
        )
        self.store.db.commit()
        hits = self.store.search("staging-docker pipeline deploy", k=1)
        self.assertEqual(hits[0].id, old_id)
        # A 400-day-old memory re-confirmed today must not score as a dead one.
        self.assertGreater(hits[0].score, 0.5)

    def test_prune_keeps_used_memory(self) -> None:
        row_id, _ = self.store.write("semantic", "Never delete the .env file.", importance=0.9)
        self.store.db.execute("UPDATE memories SET uses = 5, created_at = ? WHERE id = ?",
                              (time.time() - 86400 * 400, row_id))
        self.store.db.commit()
        self.assertEqual(self.store.prune(min_weight=0.5), 0)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_forget(self) -> None:
        row_id, _ = self.store.write("semantic", "Temporary fact.")
        self.assertTrue(self.store.forget(row_id))
        self.assertFalse(self.store.forget(row_id))

    def test_stats_counts_by_kind(self) -> None:
        self.store.write("semantic", "Fact one about the repo.")
        self.store.write("procedural", "How to run the linter.")
        self.store.write("preference", "Prefer short answers.")
        stats = self.store.stats()
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["by_kind"]["semantic"], 1)
        self.assertEqual(stats["with_embeddings"], 0)

    def test_persists_across_reopen(self) -> None:
        self.store.write("semantic", "The CI runs on GitHub Actions.")
        self.store.close()
        reopened = MemoryStore(self.db)
        try:
            hits = reopened.search("GitHub Actions CI", k=3)
            self.assertTrue(hits)
        finally:
            reopened.close()
        self.store = MemoryStore(self.db)  # for tearDown


class TestStoreHardening(StoreTestCase):
    """Reproducers for the memory defects found by the adversarial review."""

    def test_non_latin_memory_is_stored(self) -> None:
        # tokens_of() only matches [a-z0-9_]+, so a Cyrillic or CJK memory had no
        # tokens at all and the containment divide-by-zero killed the write.
        self.store.write("semantic", "The deploy script is bin/deploy.sh")
        for text in (
            "Пользователь предпочитает краткие ответы",
            "用户偏好简短回答",
            "!!!",
            "→ ←",
        ):
            row_id, created = self.store.write("preference", text)
            self.assertTrue(created, text)
            self.assertGreater(row_id, 0)

    def test_negated_fact_is_not_merged_into_its_opposite(self) -> None:
        self.store.write("preference", "The user always runs the tests with the -x flag.")
        self.store.write("preference", "The user never runs the tests with the -x flag on this machine.")
        self.assertEqual(self.store.stats()["total"], 2)
        texts = [row["text"] for row in self.store.db.execute("SELECT text FROM memories")]
        self.assertTrue(any("always" in text for text in texts))
        self.assertTrue(any("never" in text for text in texts))

    def test_a_one_word_fact_is_not_swallowed(self) -> None:
        # Containment is measured against the shorter text, so "Ruff" scored 1.0
        # against a whole sentence that mentioned ruff and was silently dropped.
        self.store.write("procedural", "Run ruff check . before every commit.")
        _, created = self.store.write("semantic", "Ruff")
        self.assertTrue(created)
        self.assertEqual(self.store.stats()["total"], 2)

    def test_merge_keeps_the_incoming_tags(self) -> None:
        self.store.write("semantic", "Uses uv for every python project here.", tags="python,packaging")
        self.store.write("semantic", "Uses uv for every python project on this machine.", tags="tooling,venv")
        tags = self.store.db.execute("SELECT tags FROM memories").fetchone()["tags"]
        for expected in ("python", "packaging", "tooling", "venv"):
            self.assertIn(expected, tags)

    def test_identifier_punctuation_is_searchable(self) -> None:
        # FTS5 splits "line-length" into two tokens; the old query stripped the
        # dash and searched for "linelength", which can never match.
        self.store.write("semantic", "Ruff is configured with line-length 120 in pyproject.toml.")
        for query in ("line-length", "pyproject.toml", "line length"):
            self.assertTrue(self.store.search(query), query)

    def test_corrupt_embedding_blob_does_not_break_search(self) -> None:
        self.store.write("semantic", "Quarterly revenue grew.", embedding=[1.0, 0.0, 0.0])
        self.store.db.execute("UPDATE memories SET embedding = ?", (b"\x00\x00\x80\x3f\x01",))
        self.store.db.commit()
        hits = self.store.search("revenue", k=2, query_embedding=[1.0, 0.0, 0.0])
        self.assertTrue(hits)

    def test_search_with_non_positive_k_returns_nothing(self) -> None:
        self.store.write("semantic", "Uses uv for python.")
        for k in (0, -1):
            self.assertEqual(self.store.search("uv", k=k), [])

    def test_out_of_range_importance_cannot_exceed_a_score_of_one(self) -> None:
        row_id, _ = self.store.write("semantic", "Everything about the deploy pipeline.", importance=1.0)
        self.store.db.execute("UPDATE memories SET importance = 2.5 WHERE id = ?", (row_id,))
        self.store.db.commit()
        self.assertLessEqual(self.store.search("deploy pipeline", k=1, record_use=False)[0].score, 1.0)

    def test_compact_finds_a_pair_that_write_would_merge(self) -> None:
        # These two share "uv", "uses", "every", "python" and "project" but no
        # >=4-character token distinct enough for the old bucketing, which is why
        # /compact reported nothing while write() merged the pair on sight.
        now = time.time()
        for digest, text in (
            ("a", "Uses uv for every python project here."),
            ("b", "Uses uv for every python project on this machine."),
        ):
            self.store.db.execute(
                "INSERT INTO memories (kind, text, tags, norm_hash, importance, uses, created_at, last_used_at)"
                " VALUES ('semantic', ?, '', ?, 0.5, 0, ?, ?)",
                (text, digest, now, now),
            )
        self.store.db.commit()
        self.assertEqual(self.store.compact(), 1)
        self.assertEqual(self.store.stats()["total"], 1)

    def test_a_legacy_schema_with_nulls_is_migrated(self) -> None:
        # A pre-embedding table has nullable columns. The migration UPDATEs fired
        # the FTS triggers *before* the index was rebuilt, and SQLite reported
        # "database disk image is malformed" - a startup crash.
        import sqlite3

        legacy = Path(self._tmp.name) / "nullable.db"
        db = sqlite3.connect(legacy)
        db.executescript(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT,"
            " text TEXT, tags TEXT, norm_hash TEXT, importance REAL, uses INTEGER,"
            " created_at REAL, last_used_at REAL);"
        )
        db.execute(
            "INSERT INTO memories (kind, text, norm_hash, created_at) VALUES ('semantic', 'old null fact', 'h', ?)",
            (time.time(),),
        )
        db.commit()
        db.close()
        store = MemoryStore(legacy)
        try:
            self.assertEqual(store.stats()["total"], 1)
            self.assertEqual([hit.text for hit in store.search("null fact")], ["old null fact"])
        finally:
            store.close()

    def test_a_read_only_database_can_still_be_searched(self) -> None:
        # search(record_use=True) writes the usage counter; on a read-only mount
        # that raised out of every recall and disabled memory entirely.
        path = Path(self._tmp.name) / "readonly.db"
        store = MemoryStore(path)
        store.write("semantic", "A fact worth reading back.")
        store.close()
        os.chmod(path, 0o444)
        try:
            store = MemoryStore(path)
            try:
                self.assertEqual([hit.text for hit in store.search("fact")], ["A fact worth reading back."])
                self.assertEqual(store.stats()["total"], 1)
            finally:
                store.close()
        finally:
            os.chmod(path, 0o644)

    def test_one_word_substitution_is_a_different_fact(self) -> None:
        # 0.83 containment and no negation either way: staging and production are
        # two facts, and the old rule merged them and kept one value.
        self.store.write("semantic", "Deploys go through the staging pipeline.")
        self.store.write("semantic", "Deploys go through the production pipeline.")
        self.assertEqual(self.store.stats()["total"], 2)

    def test_a_reworded_copula_still_merges(self) -> None:
        # "is in" vs "lives in" is the same fact; the substitution guard must not
        # block it, or one learn pass stores the same sentence twice.
        first, _ = self.store.write("semantic", "The release checklist is in docs/release.md.")
        second, created = self.store.write("procedural", "The release checklist lives in docs/release.md.")
        self.assertFalse(created)
        self.assertEqual(first, second)

    def test_an_absolute_path_counts_as_an_artifact(self) -> None:
        # The pattern was anchored with \b before "/", which cannot match after a
        # space, so artifact-aware merging had never worked for absolute paths.
        from hauhau.brain.store import _artifacts

        self.assertEqual(
            _artifacts("The models are at /Users/johan/modeller/gguf."),
            frozenset({"/users/johan/modeller/gguf."}),
        )

    def test_compact_keeps_the_tags_of_both_rows(self) -> None:
        now = time.time()
        for text, tags in (
            ("Uses uv for every python project here.", "python,packaging"),
            ("Uses uv for every python project on this machine.", "tooling,venv"),
        ):
            self.store.db.execute(
                "INSERT INTO memories (kind, text, tags, norm_hash, importance, uses, created_at, last_used_at)"
                " VALUES ('semantic', ?, ?, ?, 0.5, 0, ?, ?)",
                (text, tags, text, now, now),
            )
        self.store.db.commit()
        self.assertEqual(self.store.compact(), 1)
        tags = self.store.db.execute("SELECT tags FROM memories").fetchone()["tags"]
        for expected in ("python", "packaging", "tooling", "venv"):
            self.assertIn(expected, tags)

    def test_compact_finds_a_duplicate_beyond_a_bucket_window(self) -> None:
        # A capped bucket made compact() permanently blind to the duplicates it
        # exists to remove: write() merged them, the second pass never saw them.
        now = time.time()
        rows = [("semantic", "deploy 42 staging", "", "h0")]
        rows += [("semantic", f"deploy 42 staging {i}", "", f"h{i + 1}") for i in range(70)]
        for kind, text, tags, digest in rows:
            self.store.db.execute(
                "INSERT INTO memories (kind, text, tags, norm_hash, importance, uses, created_at, last_used_at)"
                " VALUES (?, ?, ?, ?, 0.5, 0, ?, ?)",
                (kind, text, tags, digest, now, now),
            )
        self.store.db.commit()
        self.assertGreaterEqual(self.store.compact(), 1)
        self.assertLess(self.store.stats()["total"], len(rows))

    def test_merge_tags_trims_at_a_tag_boundary(self) -> None:
        from hauhau.brain.store import _merge_tags

        old = ",".join(f"tag{i:03d}" for i in range(60))
        merged = _merge_tags(old, "extra,tooling", limit=60)
        self.assertLessEqual(len(merged), 60)
        self.assertTrue(merged.split(",")[-1].startswith("tag"), merged)

    def test_legacy_database_is_migrated_on_open(self) -> None:
        import sqlite3

        legacy = Path(self._tmp.name) / "legacy.db"
        db = sqlite3.connect(legacy)
        db.executescript(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,"
            " text TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '', norm_hash TEXT NOT NULL,"
            " importance REAL NOT NULL DEFAULT 0.5, uses INTEGER NOT NULL DEFAULT 0,"
            " created_at REAL NOT NULL, last_used_at REAL NOT NULL);"
        )
        db.execute(
            "INSERT INTO memories (kind, text, norm_hash, created_at, last_used_at)"
            " VALUES ('semantic', 'old fact', 'h', ?, ?)",
            (time.time(), time.time()),
        )
        db.commit()
        db.close()
        store = MemoryStore(legacy)
        try:
            self.assertEqual(store.stats()["total"], 1)
            self.assertEqual([hit.text for hit in store.search("old fact")], ["old fact"])
        finally:
            store.close()

    def test_future_timestamp_is_clamped_on_open(self) -> None:
        row_id, _ = self.store.write("episodic", "Throwaway note about a temp file.", importance=0.05)
        future = time.time() + 86400 * 3650
        self.store.db.execute(
            "UPDATE memories SET created_at = ?, last_used_at = ? WHERE id = ?", (future, future, row_id)
        )
        self.store.db.commit()
        self.store.close()
        store = MemoryStore(self.db)
        try:
            created = store.db.execute("SELECT created_at FROM memories WHERE id = ?", (row_id,)).fetchone()[0]
            self.assertLessEqual(created, time.time())
        finally:
            store.close()
        self.store = MemoryStore(self.db)
    def test_a_table_from_before_the_extra_columns_is_repaired(self) -> None:
        # The CREATE INDEX inside SCHEMA raised "no such column" on this shape and
        # aborted the script before the migration could add the columns, so the
        # store never recovered.
        import sqlite3

        legacy = Path(self._tmp.name) / "ancient.db"
        db = sqlite3.connect(legacy)
        db.executescript(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,"
            " text TEXT NOT NULL, created_at REAL NOT NULL, last_used_at REAL NOT NULL);"
        )
        db.execute(
            "INSERT INTO memories (kind, text, created_at, last_used_at) VALUES ('semantic', 'ancient fact', ?, ?)",
            (time.time(), time.time()),
        )
        db.commit()
        db.close()

        store = MemoryStore(legacy)
        try:
            self.assertEqual(store.stats()["total"], 1)
            self.assertEqual([hit.text for hit in store.search("ancient fact")], ["ancient fact"])
            row_id, created = store.write("semantic", "a brand new fact about disks", tags="disk")
            self.assertTrue(created)
            self.assertGreater(row_id, 0)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
