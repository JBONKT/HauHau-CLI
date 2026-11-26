from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hauhau.config import Config
from hauhau.evolve.tune import (
    TUNABLE,
    Candidate,
    candidates,
    get_value,
    hill_climb,
    record,
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


if __name__ == "__main__":
    unittest.main()
