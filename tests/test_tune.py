from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hauhau.config import Config
from hauhau.evolve.tune import (
    MIN_CTX,
    TUNABLE,
    Candidate,
    candidates,
    ctx_floor,
    get_value,
    hill_climb,
    log_tail,
    record,
    rejected_tokens,
    set_value,
)
from hauhau.llm import Completion, LLMError, Usage


class TestTunableGuard(unittest.TestCase):
    def test_set_value_refuses_anything_not_tunable(self) -> None:
        cfg = Config()
        cfg.tools.shell_confirm = True
        with self.assertRaises(ValueError) as ctx:
            set_value(cfg, "tools.shell_confirm", False)
        self.assertIn("not tunable", str(ctx.exception))
        self.assertTrue(cfg.tools.shell_confirm, "safety policy must be untouched")

    def test_safety_settings_are_not_in_the_allowlist(self) -> None:
        for forbidden in ("tools.shell_confirm", "tools.denylist", "logging.audit", "tools.workspace"):
            self.assertNotIn(forbidden, TUNABLE)

    def test_round_trip_of_a_tunable_value(self) -> None:
        cfg = Config()
        set_value(cfg, "model.n_threads", 6)
        self.assertEqual(get_value(cfg, "model.n_threads"), 6)
        self.assertEqual(cfg.model.n_threads, 6)

    def test_bad_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_value(Config(), "n_threads")


class TestCandidates(unittest.TestCase):
    def test_current_values_are_excluded(self) -> None:
        cfg = Config()
        cfg.model.n_threads = 4
        options = candidates(cfg, cores=8)
        threads = [c.value for c in options if c.key == "model.n_threads"]
        self.assertNotIn(4, threads)
        self.assertTrue(threads)
        self.assertTrue(all(c.key in TUNABLE for c in options))

    def test_small_machines_never_get_zero_threads(self) -> None:
        cfg = Config()
        options = candidates(cfg, cores=1)
        self.assertTrue(all(c.value >= 2 for c in options if c.key == "model.n_threads"))


class TestHillClimb(unittest.TestCase):
    def test_adopts_the_faster_candidate(self) -> None:
        speeds = {1: 20.0, 2: 30.0, 4: 25.0}
        options = [Candidate("model.n_threads", v) for v in speeds]
        result = hill_climb(10.0, options, lambda c: speeds[c.value], max_trials=3)
        self.assertEqual(result.best, 30.0)
        self.assertEqual([c.value for c in result.adopted], [2])
        self.assertTrue(any(t.adopted and t.candidate == "n_threads=2" for t in result.trials))

    def test_no_improvement_adopts_nothing(self) -> None:
        options = [Candidate("model.n_threads", 2), Candidate("model.n_threads", 8)]
        result = hill_climb(30.0, options, lambda c: 30.0, max_trials=4)
        self.assertEqual(result.adopted, [])
        self.assertFalse(result.improved)

    def test_budget_stops_the_search(self) -> None:
        calls: list[int] = []
        options = [Candidate("model.n_threads", v) for v in (2, 3, 5, 6)]

        def measure(c: Candidate) -> float:
            calls.append(c.value)
            return 100.0

        result = hill_climb(10.0, options, measure, max_trials=2)
        self.assertEqual(len(calls), 2)
        self.assertIn("budget", result.stopped)
        self.assertEqual(len(result.trials), 2)

    def test_zero_budget_does_nothing(self) -> None:
        result = hill_climb(10.0, [Candidate("model.n_threads", 2)], lambda c: 99.0, max_trials=0)
        self.assertEqual(result.trials, [])
        self.assertEqual(result.stopped, "trial budget is zero")

    def test_a_failing_measure_stops_without_adopting(self) -> None:
        def measure(_candidate):
            raise LLMError("server died")

        result = hill_climb(10.0, [Candidate("model.n_threads", 2)], measure, max_trials=3)
        self.assertEqual(result.adopted, [])
        self.assertIn("server died", result.stopped)

    def test_margin_ignores_noise(self) -> None:
        options = [Candidate("model.n_threads", 2)]
        # +0.5% is inside the 2% margin: not an improvement worth a restart.
        result = hill_climb(30.0, options, lambda c: 30.15, max_trials=1)
        self.assertEqual(result.adopted, [])

    def test_on_trial_sees_each_trial(self) -> None:
        seen: list[str] = []
        options = [Candidate("model.n_threads", 2), Candidate("model.n_threads", 6)]
        hill_climb(10.0, options, lambda c: 11.0, max_trials=2, on_trial=lambda t: seen.append(t.candidate))
        self.assertEqual(seen, ["n_threads=2", "n_threads=6"])


class TestContextFloor(unittest.TestCase):
    """A smaller KV cache always benchmarks faster, so it needs a floor."""

    def test_the_floor_is_twice_the_conversation_budget(self) -> None:
        cfg = Config()
        cfg.agent.context_budget_tokens = 6000
        self.assertEqual(ctx_floor(cfg), 12000)

    def test_a_tiny_budget_still_gets_the_minimum(self) -> None:
        cfg = Config()
        cfg.agent.context_budget_tokens = 200
        self.assertEqual(ctx_floor(cfg), MIN_CTX)

    def test_candidates_never_offer_a_context_below_the_floor(self) -> None:
        cfg = Config()
        cfg.agent.context_budget_tokens = 5000
        offered = [c.value for c in candidates(cfg, 8, keys=("model.n_ctx",), ram_gb=16)]
        self.assertTrue(offered, "some context sizes must remain offerable")
        self.assertTrue(all(value >= ctx_floor(cfg) for value in offered), offered)
        self.assertNotIn(4096, offered)

    def test_a_big_model_leaves_less_room_for_the_kv_cache(self) -> None:
        from hauhau.evolve.tune import _ctx_options

        small = [value for value, _why in _ctx_options(16, 1.0)]
        large = [value for value, _why in _ctx_options(16, 14.0)]
        self.assertTrue(max(small) > max(large), (small, large))

    def test_the_model_size_comes_from_the_configured_file(self) -> None:
        from hauhau.evolve.tune import _model_gb

        cfg = Config()
        cfg.model.gguf = "/nonexistent/model.gguf"
        self.assertEqual(_model_gb(cfg), 0.0)

    def test_adopting_below_the_floor_is_refused(self) -> None:
        cfg = Config()
        with self.assertRaises(ValueError) as caught:
            set_value(cfg, "model.n_ctx", 4096)
        self.assertIn("at least", str(caught.exception))
        self.assertNotEqual(cfg.model.n_ctx, 4096)

    def test_a_value_the_user_already_had_can_be_restored(self) -> None:
        cfg = Config()
        cfg.model.n_ctx = 4096
        set_value(cfg, "model.n_ctx", 16384)
        set_value(cfg, "model.n_ctx", 4096, allow_below_floor=True)
        self.assertEqual(cfg.model.n_ctx, 4096)


class TestRejectedFlags(unittest.TestCase):
    """A flag the server declines must not win a trial on borrowed speed."""

    REAL_LINE = (
        "0.01.616.975 W srv    load_model: cache_reuse is not supported by this "
        "context, it will be disabled\n"
    )

    def test_the_disabled_flag_is_named(self) -> None:
        refused = rejected_tokens(self.REAL_LINE, ["--cache-reuse", "-lm"])
        self.assertEqual(refused, ["--cache-reuse"])

    def test_a_flag_that_worked_is_not_named(self) -> None:
        log = "0.01.674.370 I srv  server_run_i: model loaded\n"
        self.assertEqual(rejected_tokens(log, ["-fa", "-lm"]), [])

    def test_an_unknown_argument_is_named(self) -> None:
        log = "error: unknown argument: --bogus\n"
        self.assertEqual(rejected_tokens(log, ["--bogus"]), ["--bogus"])

    def test_case_does_not_matter(self) -> None:
        log = "W srv load_model: CACHE_REUSE is not supported\n"
        self.assertEqual(rejected_tokens(log, ["--cache-reuse"]), ["--cache-reuse"])

    def test_log_tail_reads_only_what_is_new(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "server.log"
            path.write_text("old line\n", encoding="utf-8")
            cut = path.stat().st_size
            with path.open("a", encoding="utf-8") as fh:
                fh.write(self.REAL_LINE)
            fresh = log_tail(path, cut)
            self.assertNotIn("old line", fresh)
            self.assertIn("cache_reuse", fresh)

    def test_a_missing_log_reads_as_empty(self) -> None:
        self.assertEqual(log_tail(Path("/nonexistent/server.log")), "")


class TestRecord(unittest.TestCase):
    def test_appends_one_json_line_with_the_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evolve" / "tuning.jsonl"
            result = hill_climb(
                10.0, [Candidate("model.n_threads", 2)], lambda c: 20.0, max_trials=1
            )
            record(path, result, adopted_config="/tmp/config.toml")
            record(path, result, adopted_config="/tmp/config.toml")
            lines = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        entry = lines[0]
        self.assertEqual(entry["event"], "tune")
        self.assertEqual(entry["adopted"], ["n_threads=2"])
        self.assertEqual(entry["best_tok_s"], 20.0)
        self.assertEqual(entry["trials"][0]["candidate"], "n_threads=2")


class BenchmarkClient:
    def __init__(self, rates: list[float], error: Exception | None = None) -> None:
        self.rates = rates
        self.error = error
        self.calls = 0

    def chat(self, messages, tools=None, **kwargs) -> Completion:
        self.calls += 1
        if self.error:
            raise self.error
        rate = self.rates.pop(0) if self.rates else 0.0
        return Completion(content="1,2,3", usage=Usage(completion_tokens=32, tok_per_second=rate))


class TestBenchmark(unittest.TestCase):
    def test_warmup_call_then_median(self) -> None:
        from hauhau.evolve.tune import benchmark

        client = BenchmarkClient([0.0, 20.0, 30.0])
        self.assertEqual(benchmark(client, reps=2), 25.0)
        self.assertEqual(client.calls, 3, "one warmup plus two measured calls")

    def test_missing_timings_is_an_error(self) -> None:
        from hauhau.evolve.tune import benchmark

        with self.assertRaises(LLMError) as ctx:
            benchmark(BenchmarkClient([0.0, 0.0]), reps=1)
        self.assertIn("no timings", str(ctx.exception))


class TestExpandedTunables(unittest.TestCase):
    def test_the_allowlist_covers_model_flags_and_prompt_size(self) -> None:
        self.assertEqual(
            TUNABLE,
            ("model.n_threads", "model.n_ctx", "model.n_gpu_layers", "model.extra_args", "brain.top_k"),
        )

    def test_ranges_are_enforced(self) -> None:
        cfg = Config()
        for key, bad in (
            ("model.n_ctx", 10),
            ("model.n_ctx", 1 << 30),
            ("model.n_threads", 0),
            ("model.n_gpu_layers", 5000),
            ("brain.top_k", 0),
            ("brain.top_k", True),
        ):
            with self.assertRaises(ValueError, msg=f"{key}={bad}"):
                set_value(cfg, key, bad)

    def test_extra_args_rejects_unregistered_flags(self) -> None:
        cfg = Config()
        cfg.model.extra_args = ["--my-own-flag"]
        with self.assertRaises(ValueError):
            set_value(cfg, "model.extra_args", ["--my-own-flag", "--host", "0.0.0.0"])
        # A registered bundle, plus the user's own token, is fine.
        set_value(cfg, "model.extra_args", ["--my-own-flag", "-fa", "on"])
        self.assertEqual(cfg.model.extra_args, ["--my-own-flag", "-fa", "on"])

    def test_gpu_layers_are_only_offered_with_a_gpu(self) -> None:
        cfg = Config()
        with_gpu = candidates(cfg, cores=8, has_gpu=True)
        without = candidates(cfg, cores=8, has_gpu=False)
        self.assertTrue(any(c.key == "model.n_gpu_layers" for c in with_gpu))
        self.assertFalse(any(c.key == "model.n_gpu_layers" for c in without))

    def test_ssd_streaming_is_offered_on_ram_tight_machines_only(self) -> None:
        cfg = Config()
        tight = [c for c in candidates(cfg, cores=8, ram_gb=8.0) if c.key == "model.extra_args"]
        roomy = [c for c in candidates(cfg, cores=8, ram_gb=64.0) if c.key == "model.extra_args"]
        self.assertTrue(any("-lm" in c.value for c in tight), "SSD streaming on an 8 GB box")
        self.assertFalse(any("-lm" in c.value for c in roomy))
        self.assertTrue(any("-cram" in c.value and "16384" in c.value for c in roomy))

    def test_bundle_families_are_alternatives_not_stacks(self) -> None:
        cfg = Config()
        cfg.model.extra_args = ["-fa", "on"]
        options = [c for c in candidates(cfg, cores=8) if c.key == "model.extra_args"]
        for candidate in options:
            value = candidate.value
            self.assertNotIn("-fa", value[:-2] if value[-2:] in ("on", "off") else value,
                             f"two flash-attention flags stacked: {value}")
            self.assertLessEqual(value.count("-fa"), 1)

    def test_top_k_candidates_exclude_the_current_value(self) -> None:
        cfg = Config()
        cfg.brain.top_k = 8
        values = [c.value for c in candidates(cfg, cores=8) if c.key == "brain.top_k"]
        self.assertNotIn(8, values)
        self.assertTrue(values)

    def test_metric_is_per_key(self) -> None:
        from hauhau.evolve.tune import metric_for

        self.assertEqual(metric_for("model.n_ctx"), "gen")
        self.assertEqual(metric_for("model.extra_args"), "gen")
        self.assertEqual(metric_for("brain.top_k"), "prompt")

    def test_prompt_benchmark_uses_prompt_timings(self) -> None:
        from hauhau.evolve.tune import benchmark_prompt

        class PromptClient:
            def __init__(self, rates):
                self.rates = list(rates)
                self.calls = 0

            def chat(self, messages, tools=None, **kwargs):
                self.calls += 1
                rate = self.rates.pop(0) if self.rates else 0.0
                return Completion(content="ok", usage=Usage(prompt_tokens=1500, prompt_per_second=rate))

        client = PromptClient([0.0, 100.0, 200.0])
        self.assertEqual(benchmark_prompt(client, reps=2), 150.0)
        self.assertEqual(client.calls, 3, "one warmup plus two measured calls")
        with self.assertRaises(LLMError):
            benchmark_prompt(PromptClient([0.0, 0.0]), reps=1)




class TestExtraArgCandidates(unittest.TestCase):
    def test_a_flag_keeps_its_value_out_of_the_candidate(self) -> None:
        from hauhau.evolve.tune import _extra_arg_candidates

        cfg = Config()
        cfg.model.extra_args = ["-ub", "512", "--no-warmup"]
        candidates = _extra_arg_candidates(cfg, seen=[list(cfg.model.extra_args)], ram_gb=8.0)
        self.assertTrue(candidates, "there is something to try")
        for candidate in candidates:
            self.assertNotIn("512", candidate.value, "a bare value is not an argument")
            self.assertIn("--no-warmup", candidate.value, "the user's own flags are kept")


if __name__ == "__main__":
    unittest.main()
