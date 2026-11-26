from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hauhau.brain import Brain
from hauhau.config import Config


class TestBrain(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg = Config()
        self.cfg.logging.data_dir = str(Path(self._tmp.name) / "data")
        self.brain = Brain(self.cfg)

    def tearDown(self) -> None:
        self.brain.close()
        self._tmp.cleanup()


class TestRememberRecall(TestBrain):
    def test_remember_then_recall(self) -> None:
        row_id, created = self.brain.remember("semantic", "The model is a Gemma 4 E2B GGUF at Q4_K_P.")
        self.assertTrue(created)
        hits = self.brain.recall("which quantization does the model use")
        self.assertTrue(hits)
        self.assertEqual(hits[0].id, row_id)

    def test_context_block_formats_memories(self) -> None:
        self.brain.remember("preference", "The user wants answers shorter than three sentences.")
        block = self.brain.context_block("how should I answer")
        self.assertIn("[preference]", block)
        self.assertIn("three sentences", block)
        self.assertTrue(block.startswith("What you remember"))

    def test_context_block_empty_when_nothing_matches(self) -> None:
        self.assertEqual(self.brain.context_block("zzzznothing"), "")

    def test_context_block_respects_char_budget(self) -> None:
        self.cfg.brain.inject_max_chars = 120
        for i in range(10):
            self.brain.remember("semantic", f"Durable fact number {i} about the local machine setup here.")
        block = self.brain.context_block("durable fact machine setup")
        self.assertLessEqual(len(block), 200)

    def test_injection_does_not_count_as_a_use(self) -> None:
        """Automatic injection must not inflate ``uses`` (it disabled /prune)."""
        row_id, _ = self.brain.remember("semantic", "The postgres port here is 5432 on localhost.", importance=0.7)
        for _ in range(10):
            self.brain.context_block("which postgres port do I use")
        row = self.brain.store.db.execute("SELECT uses FROM memories WHERE id = ?", (row_id,)).fetchone()
        self.assertEqual(row["uses"], 0, "prompt injection is not a deliberate reuse")
        # ...but an explicit recall is, and it is what keeps a memory alive.
        self.brain.recall("which postgres port do I use")
        row = self.brain.store.db.execute("SELECT uses FROM memories WHERE id = ?", (row_id,)).fetchone()
        self.assertEqual(row["uses"], 1)

    def test_context_block_never_splits_a_memory(self) -> None:
        """Half a sentence reads as a fact; skip it and say so instead."""
        self.cfg.brain.inject_max_chars = 320
        for i in range(6):
            self.brain.remember(
                "semantic", f"Fact {i} about the build server: the artifact cache is at /srv/cache/{i}.", importance=0.6
            )
        block = self.brain.context_block("build server artifact cache")
        self.assertLessEqual(len(block), 320)
        body = block.split("\n", 1)[1]
        for line in body.splitlines():
            if line.startswith("- (") or not line.startswith("- ["):
                continue
            self.assertTrue(line.endswith(")"), f"truncated memory: {line!r}")
            self.assertIn("/srv/cache/", line)

    def test_context_block_does_not_exceed_a_tiny_budget(self) -> None:
        # A header plus one clipped memory was over budget *and* shipped half a
        # sentence; with no room for a whole memory the block must be empty.
        self.cfg.brain.inject_max_chars = 100
        self.brain.remember(
            "procedural",
            "The release checklist lives in docs/release.md and starts with a branch freeze.",
            importance=0.9,
        )
        block = self.brain.context_block("release checklist")
        self.assertLessEqual(len(block), 100)
        self.assertNotIn("…", block)
        self.assertNotIn("The release c", block)

    def test_context_block_holds_its_budget_at_the_configured_minimum(self) -> None:
        self.cfg.brain.inject_max_chars = 71  # header length: nothing can fit
        self.brain.remember("semantic", "A durable fact about this machine.")
        self.assertEqual(self.brain.context_block("durable fact"), "")

    def test_disabled_brain_injects_nothing(self) -> None:
        self.cfg.brain.enabled = False
        self.assertEqual(self.brain.context_block("anything"), "")

    def test_stats_reports_keyword_mode(self) -> None:
        stats = self.brain.stats()
        self.assertFalse(stats["embeddings_enabled"])
        self.assertEqual(stats["with_embeddings"], 0)


class TestEmbedderDisabled(TestBrain):
    def test_no_url_means_no_vectors(self) -> None:
        self.assertFalse(self.brain.embedder.enabled)
        self.assertIsNone(self.brain.embedder.embed("text"))

    def test_unreachable_endpoint_disables_after_first_error(self) -> None:
        self.cfg.brain.embeddings_url = "http://127.0.0.1:9/v1/embeddings"
        brain = Brain(self.cfg)
        try:
            self.assertIsNone(brain.embedder.embed("text"))
            self.assertFalse(brain.embedder.enabled, "one failure disables vectors for the session")
            self.assertTrue(brain.embedder.last_error)
        finally:
            brain.close()

    def test_keyword_recall_still_works_without_vectors(self) -> None:
        self.brain.remember("procedural", "Start llama-server with -ngl 99 for full Metal offload.")
        self.assertTrue(self.brain.recall("metal offload"))


if __name__ == "__main__":
    unittest.main()
