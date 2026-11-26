"""Greedy hill-climb over the config values that can actually change speed.

What is *not* here: quantization, GPU offload and context-size-vs-quality
trade-offs that need multi-gigabyte downloads. Those are one-shot profiler
recommendations. What is here is `n_threads` and `n_ctx`, the two knobs that
change inference speed on a machine that already has its model and its Metal
offload.

Every trial costs a server restart (~15s) plus a fixed benchmark (~10s), so the
climb is bounded by `max_trials` and stops at the first sign of no improvement.
The scoring number is generation speed from llama.cpp's own timings, measured
with temperature 0 and a fixed prompt - not conversational latency, which
depends on what the user asked.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm import LLMError

# The tuner may only touch these. Anything else - shell confirmation, the
# denylist, the audit switch, the workspace - is safety policy, and a tuner that
# can edit its own policy is not a tuner.
TUNABLE: tuple[str, ...] = ("model.n_threads", "model.n_ctx")

BENCH_PROMPT = "Write the numbers from 1 to 30, separated by commas. No other words."


@dataclass(frozen=True)
class Candidate:
    key: str
    value: Any
    reason: str = ""

    def __str__(self) -> str:
        return f"{self.key.replace('model.', '')}={self.value}"


@dataclass
class Trial:
    candidate: str
    score: float
    seconds: float
    adopted: bool = False


@dataclass
class TuneResult:
    baseline: float
    best: float
    trials: list[Trial] = field(default_factory=list)
    adopted: list[Candidate] = field(default_factory=list)
    stopped: str = ""

    @property
    def improved(self) -> bool:
        return bool(self.adopted)


def _split(key: str) -> tuple[str, str]:
    section, _, name = key.partition(".")
    if not section or not name:
        raise ValueError(f"not a config key: {key!r}")
    return section, name


def get_value(cfg, key: str) -> Any:
    section, name = _split(key)
    return getattr(getattr(cfg, section), name)


def set_value(cfg, key: str, value: Any) -> None:
    """Set a config value, but only one the tuner is allowed to touch."""
    if key not in TUNABLE:
        raise ValueError(f"{key} is not tunable; allowed: {', '.join(TUNABLE)}")
    section, name = _split(key)
    setattr(getattr(cfg, section), name, value)


def candidates(cfg, cores: int, *, keys: Iterable[str] = TUNABLE) -> list[Candidate]:
    """Alternatives worth trying on this machine, current values excluded."""
    options: dict[str, list[tuple[Any, str]]] = {
        "model.n_threads": [
            (max(2, cores // 2), "half the cores: leaves room for the client"),
            (max(2, cores - 1), "all but one core"),
            (max(2, min(cores, 8)), "capped at 8 threads"),
        ],
        "model.n_ctx": [
            (4096, "smaller KV cache, more RAM for the OS"),
            (8192, "8k context"),
            (16384, "16k context, if RAM allows"),
        ],
    }
    out: list[Candidate] = []
    for key in keys:
        current = get_value(cfg, key)
        seen: set[Any] = {current}
        for value, reason in options.get(key, []):
            if value in seen:
                continue
            seen.add(value)
            out.append(Candidate(key, value, reason))
    return out


def benchmark(client, reps: int = 2, max_tokens: int = 96) -> float:
    """Median generation speed of a fixed prompt, in tokens/second."""
    # One throwaway call so the timing does not include the first-token warmup.
    client.chat(
        [{"role": "user", "content": "Reply with: ok"}],
        None,
        think=False,
        temperature=0.0,
        max_tokens=8,
    )
    scores: list[float] = []
    for _ in range(max(1, reps)):
        completion = client.chat(
            [{"role": "user", "content": BENCH_PROMPT}],
            None,
            think=False,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        rate = completion.usage.tok_per_second
        if rate:
            scores.append(rate)
    if not scores:
        raise LLMError("llama.cpp returned no timings, so there is nothing to compare")
    return statistics.median(scores)


def hill_climb(
    baseline: float,
    options: list[Candidate],
    measure: Callable[[Candidate], float],
    *,
    margin: float = 0.02,
    max_trials: int = 4,
    on_trial: Callable[[Trial], None] | None = None,
) -> TuneResult:
    """Try candidates in order; adopt a value only if it beats the best by `margin`.

    `measure` is injected: the CLI restarts the server and benchmarks, the tests
    pass a function, so the search logic is testable without a model.
    """
    result = TuneResult(baseline=baseline, best=baseline)
    if max_trials <= 0:
        result.stopped = "trial budget is zero"
        return result
    by_key: dict[str, list[Candidate]] = {}
    for option in options:
        by_key.setdefault(option.key, []).append(option)

    for group in by_key.values():
        best: Candidate | None = None
        best_index: int | None = None
        for option in group:
            if len(result.trials) >= max_trials:
                result.stopped = f"trial budget ({max_trials}) reached"
                break
            started = time.monotonic()
            try:
                score = float(measure(option))
            except LLMError as exc:
                result.trials.append(Trial(str(option), 0.0, time.monotonic() - started))
                result.stopped = f"{option} failed: {exc}"
                break
            trial = Trial(str(option), score, time.monotonic() - started)
            result.trials.append(trial)
            if on_trial is not None:
                on_trial(trial)
            if score > result.best + margin * max(1.0, abs(result.best)):
                result.best = score
                best = option
                # Remember *which* trial won: two candidates can print the same
                # label, and the old lookup marked the first match as adopted.
                best_index = len(result.trials) - 1
        if best is not None and best_index is not None:
            result.trials[best_index].adopted = True
            result.adopted.append(best)
    return result


def record(path: Path, result: TuneResult, *, adopted_config: str) -> None:
    """Append the outcome to the evolution log (JSONL, one line per run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event": "tune",
        "baseline_tok_s": round(result.baseline, 2),
        "best_tok_s": round(result.best, 2),
        "adopted": [str(c) for c in result.adopted],
        "config": adopted_config,
        "trials": [
            {"candidate": t.candidate, "tok_s": round(t.score, 2), "seconds": round(t.seconds, 1),
             "adopted": t.adopted}
            for t in result.trials
        ],
        "stopped": result.stopped,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
