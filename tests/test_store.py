from __future__ import annotations

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

    def test_different_kind_is_not_merged(self) -> None:
        self.store.write("semantic", "Deploys happen from the main branch.")
        _, created = self.store.write("procedural", "Deploys happen from the main branch.")
        self.assertTrue(created)

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
        self.store.db.execute("UPDATE memories SET created_at = ? WHERE id = ?", (time.time() - 86400 * 400, row_id))
        self.store.db.commit()
        self.assertEqual(self.store.prune(min_weight=0.05), 1)
        self.assertEqual(self.store.stats()["total"], 0)

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


if __name__ == "__main__":
    unittest.main()
