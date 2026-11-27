"""The model catalog: discovery, name matching and per-model settings."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from hauhau import models
from hauhau.config import Config


def fake_gguf(path: Path, size: int = 1024) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))
    return path


class CatalogTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.models_dir = self.tmp / "gguf"
        self._env = os.environ.pop("HAUHAU_GGUF", None)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.tmp / "data")
        self.cfg.model.gguf = str(fake_gguf(self.models_dir / "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf"))
        fake_gguf(self.models_dir / "LFM3-1.7B-Q4_K_M.gguf", size=2048)
        # The catalog scans the configured model's directory plus the defaults;
        # point the defaults at an empty directory so only our files are seen.
        self._patch = unittest.mock.patch.object(
            models, "GGUF_SEARCH_DIRS", (str(self.models_dir),)
        )
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        if self._env is not None:
            os.environ["HAUHAU_GGUF"] = self._env
        self._tmp.cleanup()

    def test_display_name_keeps_the_version_number_whole(self):
        # Splitting on "." produced "LFM3-1-7B" in the header.
        self.assertEqual(models.display_name("LFM3-1.7B-Q4_K_M"), "LFM3-1.7B")

    def test_display_name_drops_quantisation_and_noise(self):
        self.assertEqual(
            models.display_name("Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q4_K_P"),
            "Gemma-4-E2B-Uncensored",
        )
        # The header chip has less room, so it stops at the model itself.
        self.assertEqual(
            models.display_name("Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q4_K_P", width=18),
            "Gemma-4-E2B",
        )

    def test_catalog_finds_every_file_and_marks_the_active_one(self):
        entries = models.catalog(self.cfg)
        names = {entry.name for entry in entries}
        self.assertEqual(names, {"Gemma-4-E2B-Uncensored", "LFM3-1.7B"})
        active = [entry for entry in entries if entry.active]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].name, "Gemma-4-E2B-Uncensored")

    def test_catalog_reports_sizes(self):
        entries = {entry.name: entry for entry in models.catalog(self.cfg)}
        self.assertGreater(entries["LFM3-1.7B"].size, entries["Gemma-4-E2B-Uncensored"].size)

    def test_resolve_by_number(self):
        entries = models.catalog(self.cfg)
        chosen = models.resolve(self.cfg, "2")
        self.assertEqual(chosen.path, entries[1].path)

    def test_resolve_by_name(self):
        self.assertEqual(models.resolve(self.cfg, "lfm3-1.7b").name, "LFM3-1.7B")

    def test_resolve_by_file_name(self):
        self.assertEqual(
            models.resolve(self.cfg, "LFM3-1.7B-Q4_K_M.gguf").name, "LFM3-1.7B"
        )

    def test_resolve_by_unique_fragment(self):
        self.assertEqual(models.resolve(self.cfg, "uncensored").name, "Gemma-4-E2B-Uncensored")

    def test_an_ambiguous_fragment_lists_the_candidates(self):
        with self.assertRaises(models.ModelError) as caught:
            models.resolve(self.cfg, "gguf")
        self.assertIn("matches", str(caught.exception))

    def test_an_unknown_name_is_an_error_not_a_crash(self):
        with self.assertRaises(models.ModelError):
            models.resolve(self.cfg, "qwen")

    def test_an_out_of_range_number_says_how_many_there_are(self):
        with self.assertRaises(models.ModelError) as caught:
            models.resolve(self.cfg, "9")
        self.assertIn("2 models", str(caught.exception))

    def test_an_empty_catalog_is_reported(self):
        empty = Config()
        empty.logging.data_dir = str(self.tmp / "empty")
        with (
            unittest.mock.patch.object(models, "GGUF_SEARCH_DIRS", (str(self.tmp / "nothing"),)),
            self.assertRaises(models.ModelError),
        ):
            models.resolve(empty, "1")


class SwitchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.cfg = Config()
        self.cfg.logging.data_dir = str(self.tmp / "data")
        self.first = fake_gguf(self.tmp / "gemma.gguf", size=64)
        self.second = fake_gguf(self.tmp / "lfm3.gguf", size=64)
        self.cfg.model.gguf = str(self.first)
        self.cfg.model.n_ctx = 16384
        self.entry = models.Entry(path=self.second.resolve(), name="lfm3", size=64, active=False)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_switch_points_the_config_at_the_new_file(self):
        models.switch(self.cfg, self.entry)
        self.assertEqual(self.cfg.model.gguf, str(self.second.resolve()))

    def test_switch_remembers_the_context_of_the_model_it_left(self):
        models.switch(self.cfg, self.entry)
        saved = models.remembered(self.cfg)
        self.assertEqual(saved[str(self.first.resolve())]["n_ctx"], 16384)

    def test_switch_restores_the_context_of_the_new_model(self):
        models.switch(self.cfg, self.entry)  # gemma keeps its 16384
        self.cfg.model.n_ctx = 32768  # the small model can afford more
        self.gemma = models.Entry(path=self.first.resolve(), name="gemma", size=64, active=False)
        models.switch(self.cfg, self.gemma)
        self.assertEqual(self.cfg.model.n_ctx, 16384)
        models.switch(self.cfg, self.entry)
        self.assertEqual(self.cfg.model.n_ctx, 32768)

    def test_switch_honours_the_context_floor(self):
        self.cfg.model.n_ctx = 2048
        models.switch(self.cfg, self.entry)
        models.switch(self.cfg, models.Entry(path=self.first.resolve(), name="gemma", size=64, active=False))
        # The remembered 2048 is below the floor the agent needs to run at all.
        models.switch(self.cfg, self.entry, ctx_floor=8192)
        self.assertEqual(self.cfg.model.n_ctx, 8192)

    def test_switch_remembers_the_server_flags(self):
        self.cfg.model.extra_args = ["-lm", "dio"]
        models.switch(self.cfg, self.entry)
        self.cfg.model.extra_args = []
        models.switch(self.cfg, models.Entry(path=self.first.resolve(), name="gemma", size=64, active=False))
        self.assertEqual(self.cfg.model.extra_args, ["-lm", "dio"])

    def test_a_corrupt_state_file_is_ignored(self):
        path = Path(self.cfg.data_dir) / "models.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(models.remembered(self.cfg), {})
        models.switch(self.cfg, self.entry)
        self.assertEqual(self.cfg.model.gguf, str(self.second.resolve()))

    def test_the_state_file_is_written_atomically_enough_to_be_readable(self):
        models.remember(self.cfg, self.first)
        data = json.loads((Path(self.cfg.data_dir) / "models.json").read_text(encoding="utf-8"))
        self.assertIn(str(self.first.resolve()), data)


if __name__ == "__main__":
    unittest.main()
