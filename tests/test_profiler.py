from __future__ import annotations

import unittest

from hauhau.config import Config
from hauhau.profiler import (
    HardwareProfile,
    apply_recommendations,
    default_tune_trials,
    merge_extra_args,
    recommend,
)


def profile(**kwargs) -> HardwareProfile:
    base = {
        "os_name": "Darwin",
        "arch": "arm64",
        "cpu": "Apple M1",
        "cores": 8,
        "ram_gb": 8.0,
        "gpu": "Apple Silicon (Metal)",
        "metal": True,
        "disk_free_gb": 100.0,
        "llama_server": "/opt/homebrew/bin/llama-server",
        "llama_version": "12359",
        "notes": [],
    }
    base.update(kwargs)
    return HardwareProfile(**base)


class TestRecommend(unittest.TestCase):
    def test_apple_silicon_full_offload(self) -> None:
        rec = recommend(profile())
        self.assertEqual(rec["n_gpu_layers"][0], 99)
        self.assertEqual(rec["n_ctx"][0], 8192)
        self.assertLessEqual(rec["n_threads"][0], 6)
        self.assertEqual(rec["extra_args"][0][:2], ["-np", "1"])
        self.assertIn("-fa", rec["extra_args"][0])

    def test_large_model_on_8gb_streams_weights(self) -> None:
        rec = recommend(profile(ram_gb=8.0), model_bytes=int(3.2 * 1024**3))
        self.assertEqual(rec["extra_args"][0][:2], ["-lm", "none"])
        self.assertIn("-np", rec["extra_args"][0])

    def test_small_ram_keeps_context_small(self) -> None:
        self.assertEqual(recommend(profile(ram_gb=8.0))["n_ctx"][0], 8192)
        self.assertEqual(recommend(profile(ram_gb=16.0))["n_ctx"][0], 16384)
        self.assertEqual(recommend(profile(ram_gb=64.0))["n_ctx"][0], 32768)

    def test_cpu_only_machine_offloads_nothing(self) -> None:
        rec = recommend(profile(metal=False, arch="x86_64", cores=4))
        self.assertEqual(rec["n_gpu_layers"][0], 0)
        self.assertEqual(rec["n_threads"][0], 3)

    def test_oversized_model_warns(self) -> None:
        rec = recommend(profile(ram_gb=8.0), model_bytes=int(6.5 * 1024**3))
        self.assertIn("warn_quant", rec)
        self.assertTrue(rec["warn_quant"][0])

    def test_model_fitting_budget_does_not_warn(self) -> None:
        rec = recommend(profile(ram_gb=8.0), model_bytes=int(3.2 * 1024**3))
        self.assertNotIn("warn_quant", rec)


class TestApply(unittest.TestCase):
    def test_unknown_ram_gets_the_conservative_context(self) -> None:
        # A machine whose RAM cannot be read is not a licence to assume 32k.
        unknown = HardwareProfile(
            os_name="Linux", arch="x86_64", cpu="?", cores=4, ram_gb=0.0, gpu="none",
            metal=False, disk_free_gb=0.0, llama_server="", llama_version="",
        )
        self.assertEqual(recommend(unknown)["n_ctx"][0], 8192)
        self.assertIn("could not be detected", recommend(unknown)["n_ctx"][1])

    def test_values_are_written_and_changes_described(self) -> None:
        cfg = Config()
        cfg.model.n_ctx = 2048
        cfg.model.n_threads = 1
        changes = apply_recommendations(cfg, profile())
        self.assertEqual(cfg.model.n_ctx, 8192)
        self.assertEqual(cfg.model.n_threads, 4)
        self.assertTrue(any("n_ctx" in c for c in changes))

    def test_already_correct_values_report_nothing(self) -> None:
        cfg = Config()
        cfg.model.n_ctx = 8192
        cfg.model.n_threads = 4
        cfg.model.n_gpu_layers = 99
        cfg.model.extra_args = ["-np", "1", "-fa", "on"]
        self.assertEqual(apply_recommendations(cfg, profile()), [])

    def test_merge_extra_args_keeps_user_tokens(self) -> None:
        merged = merge_extra_args(["-lm", "none", "--host", "127.0.0.1"], ["-np", "1", "-fa", "on"])
        self.assertEqual(merged[:2], ["-lm", "none"])
        self.assertIn("-np", merged)
        self.assertIn("1", merged)
        self.assertIn("-fa", merged)
        self.assertIn("--host", merged)

    def test_default_tune_trials_scale_with_ram(self) -> None:
        self.assertEqual(default_tune_trials(profile(ram_gb=8.0)), 12)
        self.assertEqual(default_tune_trials(profile(ram_gb=16.0)), 8)
        self.assertEqual(default_tune_trials(profile(ram_gb=32.0)), 6)


if __name__ == "__main__":
    unittest.main()
