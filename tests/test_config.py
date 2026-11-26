from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from hauhau import config as config_mod
from hauhau.config import Config


class ConfigTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._env = os.environ.pop("HAUHAU_GGUF", None)

    def tearDown(self) -> None:
        if self._env is not None:
            os.environ["HAUHAU_GGUF"] = self._env
        self._tmp.cleanup()


class TestDefaults(ConfigTestCase):
    def test_defaults_are_usable(self) -> None:
        cfg = Config()
        self.assertEqual(cfg.model.n_ctx, 8192)
        self.assertFalse(cfg.tools.shell_confirm)
        self.assertEqual(cfg.tools.denylist, [])
        self.assertTrue(cfg.brain.enabled)

    def test_paths_derive_from_data_dir(self) -> None:
        cfg = Config()
        cfg.logging.data_dir = str(self.tmp)
        self.assertEqual(cfg.db_path, self.tmp / "brain" / "brain.db")
        self.assertEqual(cfg.audit_path, self.tmp / "logs" / "audit.jsonl")


class TestRoundTrip(ConfigTestCase):
    def test_save_then_load_preserves_values(self) -> None:
        cfg = Config()
        cfg.model.gguf = "/models/x.gguf"
        cfg.model.n_ctx = 16384
        cfg.agent.name = "Test"
        cfg.agent.personality = "line one\nline two"
        cfg.tools.denylist = ["rm -rf /", "mkfs"]
        cfg.generation.temperature = 0.35
        cfg.brain.enabled = False
        path = self.tmp / "config.toml"
        config_mod.save(cfg, path)

        loaded, warnings = config_mod.load(path)
        self.assertEqual(warnings, [])
        self.assertEqual(loaded.model.gguf, "/models/x.gguf")
        self.assertEqual(loaded.model.n_ctx, 16384)
        self.assertEqual(loaded.agent.personality, "line one\nline two")
        self.assertEqual(loaded.tools.denylist, ["rm -rf /", "mkfs"])
        self.assertAlmostEqual(loaded.generation.temperature, 0.35)
        self.assertFalse(loaded.brain.enabled)

    def test_unknown_key_warns_instead_of_crashing(self) -> None:
        path = self.tmp / "config.toml"
        path.write_text("[model]\nn_ctx = 4096\nnonsense = 1\n\n[mystery]\nx = 1\n", encoding="utf-8")
        loaded, warnings = config_mod.load(path)
        self.assertEqual(loaded.model.n_ctx, 4096)
        self.assertTrue(any("nonsense" in w for w in warnings))
        self.assertTrue(any("mystery" in w for w in warnings))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            config_mod.load(self.tmp / "nope.toml")


class TestGgufDiscovery(ConfigTestCase):
    def test_env_override_wins(self) -> None:
        os.environ["HAUHAU_GGUF"] = "/tmp/custom.gguf"
        self.assertEqual(config_mod.find_gguf(), "/tmp/custom.gguf")

    def test_empty_when_nothing_found(self) -> None:
        # Point the search at an empty dir by overriding the module tuple.
        original = config_mod.GGUF_SEARCH_DIRS
        config_mod.GGUF_SEARCH_DIRS = (str(self.tmp),)
        try:
            self.assertEqual(config_mod.find_gguf(), "")
        finally:
            config_mod.GGUF_SEARCH_DIRS = original


if __name__ == "__main__":
    unittest.main()
