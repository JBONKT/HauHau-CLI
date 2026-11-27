"""Greedy hill-climb over the config values that can actually change speed.

Tuned here: `n_threads`, `n_ctx`, GPU layer offload, flash attention and how
many memories are injected per turn. Quantization and model downloads are not
tunable - those are one-shot profiler decisions.

Each key is climbed separately with its own baseline and its own metric:
`model.*` needs a server restart and is scored on generation tok/s, while
`brain.*` only changes the prompt and is scored on prompt tok/s. Mixing the two
in one number would let a fast prompt "win" against a fast decode.

Every model-side trial costs a server restart (~15s) plus a fixed benchmark
(~10s), so the climb is bounded by `max_trials` and stops at the first sign of
no improvement.
"""

from __future__ import annotations

import json
import os
import re
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
TUNABLE: tuple[str, ...] = (
    "model.n_threads",
    "model.n_ctx",
    "model.n_gpu_layers",
    "model.extra_args",
    "brain.top_k",
)

# Hard limits per key, so a candidate can never push a value into a range the
# server would refuse or the machine could not survive.
RANGES: dict[str, tuple[Any, Any]] = {
    "model.n_threads": (1, 256),
    "model.n_ctx": (256, 1 << 18),
    "model.n_gpu_layers": (-128, 999),
    "brain.top_k": (1, 50),
}

@dataclass(frozen=True)
class FlagBundle:
    """One named llama-server flag the tuner may try.

    ``family`` groups mutually exclusive options ("-lm" load mode, "-fa" flash
    attention, ...): a candidate replaces the current bundle of that family
    instead of stacking on top of it. ``when`` decides whether the flag is worth
    offering on this machine at all.
    """

    name: str
    family: str
    tokens: tuple[str, ...]
    reason: str
    when: Callable[[float, bool], bool] = lambda ram_gb, has_gpu: True


# Verified against `llama-server --help` of build 12359: every flag here exists
# in this build, and none of them changes what the model outputs - only how fast
# and how much memory it uses. Anything else belongs in the user's own config.
FLAG_BUNDLES: tuple[FlagBundle, ...] = (
    FlagBundle("flash-attn-on", "-fa", ("-fa", "on"), "flash attention on: faster prompt processing"),
    FlagBundle("flash-attn-off", "-fa", ("-fa", "off"), "flash attention off: this build defaults to auto"),
    FlagBundle(
        "ssd-stream", "-lm", ("-lm", "none"),
        "stream the model from SSD instead of memory-mapping it: less RAM pressure, slower load",
        lambda ram_gb, has_gpu: 0 < ram_gb <= 16,
    ),
    FlagBundle(
        "direct-io", "-lm", ("-lm", "dio"),
        "DirectIO model loading: bypasses the page cache on SSDs",
        lambda ram_gb, has_gpu: 0 < ram_gb <= 16,
    ),
    FlagBundle(
        "kv-cache-q8", "-ctk", ("-ctk", "q8_0", "-ctv", "q8_0"),
        "quantized KV cache (q8_0): half the cache RAM, usually faster prompt processing",
    ),
    FlagBundle(
        "cache-reuse", "--cache-reuse", ("--cache-reuse", "256"),
        "reuse KV chunks across turns (prompt caching + KV shifting) instead of recomputing them",
    ),
    FlagBundle(
        "auto-compact", "--context-shift", ("--context-shift",),
        "auto-compact a full context window (shift out the oldest tokens) instead of failing",
    ),
    FlagBundle(
        "cache-ram-small", "-cram", ("-cram", "2048"),
        "cap the KV cache at 2 GiB: leaves RAM for the OS",
        lambda ram_gb, has_gpu: 0 < ram_gb <= 8,
    ),
    FlagBundle(
        "cache-ram-large", "-cram", ("-cram", "16384"),
        "cap the KV cache at 16 GiB: more room for long contexts",
        lambda ram_gb, has_gpu: ram_gb >= 24,
    ),
    FlagBundle(
        "ubatch-1024", "-ub", ("-ub", "1024"),
        "larger physical batch: faster prompt processing on machines with headroom",
        lambda ram_gb, has_gpu: ram_gb >= 16,
    ),
    FlagBundle(
        "batch-4096", "-b", ("-b", "4096"),
        "larger logical batch",
        lambda ram_gb, has_gpu: False,  # rarely helps; kept off by default
    ),
)

_ALL_BUNDLE_TOKENS = {token for bundle in FLAG_BUNDLES for token in bundle.tokens}

BENCH_PROMPT = "Write the numbers from 1 to 30, separated by commas. No other words."
LONG_BENCH_PROMPT = "The quick brown fox jumps over the lazy dog. " * 200


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


def set_value(cfg, key: str, value: Any, *, allow_below_floor: bool = False) -> None:
    """Set a config value, but only one the tuner is allowed to touch.

    ``allow_below_floor`` exists for restoring a value the user already had: the
    floor governs what the tuner may *adopt*, never what it may put back.
    """
    if key not in TUNABLE:
        raise ValueError(f"{key} is not tunable; allowed: {', '.join(TUNABLE)}")
    if key in RANGES:
        low, high = RANGES[key]
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"{key} must be an integer in {low}..{high}, got {value!r}")
    if key == "model.n_ctx" and not allow_below_floor:
        floor = ctx_floor(cfg)
        if value < floor:
            raise ValueError(
                f"model.n_ctx must be at least {floor}: a smaller context cannot hold a "
                f"turn, so it measures faster and then rejects the next request"
            )
    if key == "model.extra_args":
        _check_extra_args(cfg, value)
    section, name = _split(key)
    setattr(getattr(cfg, section), name, value)


def _check_extra_args(cfg, value: Any) -> None:
    """extra_args may only contain registered bundles plus the user's own tokens.

    The tuner writes this list itself, so a candidate must not be able to smuggle
    in an arbitrary server flag (`--host`, a different model path, ...). Tokens
    the user already had configured always stay allowed; everything else has to
    belong to a registered FlagBundle.
    """
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("model.extra_args must be a list of strings")
    current = cfg.model.extra_args
    own = {item for item in current if item not in _ALL_BUNDLE_TOKENS}
    for item in value:
        if item not in _ALL_BUNDLE_TOKENS and item not in own:
            raise ValueError(f"the tuner may not add {item!r} to model.extra_args")


def metric_for(key: str) -> str:
    """Which benchmark number decides this key: "gen" or "prompt"."""
    return "gen" if key.startswith("model.") else "prompt"


def _model_gb(cfg) -> float:
    """Size of the configured .gguf, or 0 when it cannot be measured."""
    path = getattr(getattr(cfg, "model", None), "gguf", "") or ""
    try:
        return os.path.getsize(os.path.expanduser(path)) / (1024**3)
    except OSError:
        return 0.0


def _ctx_options(ram_gb: float, model_gb: float = 0.0) -> list[tuple[int, str]]:
    """Context sizes worth trying, scaled by the RAM the KV cache can use.

    The weights are resident in the same memory as the KV cache, so the tier is
    chosen from what is left after the model, not from total RAM: on an 8 GB
    machine a 1 GB model leaves room to try 32k where a 3.2 GB one does not.
    """
    ram_gb = max(0.0, ram_gb - model_gb)
    if ram_gb <= 4:
        return [(2048, "tiny RAM: smallest KV cache"), (4096, "4k context")]
    if ram_gb <= 8:
        return [(4096, "smaller KV cache, more RAM for the OS"), (8192, "8k context"), (16384, "16k if it fits")]
    if ram_gb <= 16:
        return [(4096, "smaller KV cache"), (8192, "8k context"), (16384, "16k context"), (32768, "32k context")]
    if ram_gb <= 32:
        return [(8192, "8k context"), (16384, "16k context"), (32768, "32k context"), (65536, "64k context")]
    return [(16384, "16k context"), (32768, "32k context"), (65536, "64k context"), (131072, "128k context")]


MIN_CTX = 8192


def ctx_floor(cfg) -> int:
    """The smallest context worth adopting, in tokens.

    ``model.*`` trials are scored on tok/s alone, so a smaller KV cache always
    looks better - it won one sweep here and adopted 4096 while the session was
    already sending 5.6k-token prompts, which rejected every following request.
    A turn has to fit the conversation budget *and* the tool output that follows
    it, so the floor is twice the budget, never below ``MIN_CTX``.
    """
    budget = int(getattr(cfg.agent, "context_budget_tokens", 5000) or 0)
    return max(MIN_CTX, 2 * budget)


# What llama-server prints when it accepts a flag and then declines to use it,
# or does not know the flag at all. A trial that triggers one of these measured
# nothing, so it must not win on speed.
UNSUPPORTED_MARKERS: tuple[str, ...] = (
    "not supported",
    "unsupported",
    "will be disabled",
    "unknown argument",
    "invalid argument",
    "unrecognized",
)


def rejected_tokens(log_tail: str, tokens: Iterable[str]) -> list[str]:
    """Which of ``tokens`` the server log says it cannot honour.

    llama.cpp names a feature in its warnings rather than repeating the flag -
    `--cache-reuse` is reported as "cache_reuse is not supported by this
    context" - so a flag is matched by its normalised feature name on word
    boundaries, never as a bare substring ("fa" would otherwise hit "failed").
    """
    refused: list[str] = []
    lines = [line.lower() for line in log_tail.splitlines()]
    for token in tokens:
        name = token.lstrip("-").replace("-", "_").lower()
        if not name:
            continue
        pattern = re.compile(rf"(?<![\w]){re.escape(name)}(?![\w])")
        for line in lines:
            if any(marker in line for marker in UNSUPPORTED_MARKERS) and pattern.search(line):
                if token not in refused:
                    refused.append(token)
                break
    return refused


def log_tail(path: Path, offset: int = 0, limit: int = 200_000) -> str:
    """The part of a log written since ``offset`` (0 for the whole tail)."""
    try:
        size = path.stat().st_size
        start = max(0, offset or max(0, size - limit))
        with path.open("rb") as fh:
            fh.seek(start)
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def candidates(
    cfg,
    cores: int,
    *,
    keys: Iterable[str] = TUNABLE,
    ram_gb: float = 0.0,
    has_gpu: bool = False,
) -> list[Candidate]:
    """Alternatives worth trying on this machine, current values excluded."""
    options: dict[str, list[tuple[Any, str]]] = {
        "model.n_threads": [
            (max(2, cores // 2), "half the cores: leaves room for the client"),
            (max(2, cores - 1), "all but one core"),
            (max(2, min(cores, 8)), "capped at 8 threads"),
            (max(2, cores), "every core"),
        ],
        "model.n_ctx": [
            (value, why)
            for value, why in _ctx_options(ram_gb, _model_gb(cfg))
            if value >= ctx_floor(cfg)
        ],
        "brain.top_k": [
            (3, "fewer memories in the prompt = faster prompt processing"),
            (8, "a wider memory block"),
            (12, "the widest useful memory block"),
        ],
    }
    if has_gpu:
        # Negative ngl means "all but the last |n| layers" in llama.cpp.
        options["model.n_gpu_layers"] = [
            (99, "offload everything to the GPU"),
            (0, "CPU only: frees unified memory for the OS"),
            (-8, "all but the last 8 layers: smaller resident model"),
        ]
    out: list[Candidate] = []
    for key in keys:
        current = get_value(cfg, key)
        # A list, not a set: extra_args is a list and lists are unhashable.
        seen: list[Any] = [current]
        for value, reason in options.get(key, []):
            if value in seen:
                continue
            seen.append(value)
            out.append(Candidate(key, value, reason))
        if key == "model.extra_args":
            out.extend(_extra_arg_candidates(cfg, seen, ram_gb, has_gpu))
    return out


def _extra_arg_candidates(cfg, seen: list, ram_gb: float = 0.0, has_gpu: bool = False) -> list[Candidate]:
    """Flag bundles worth trying, on top of whatever the user already passes."""
    current = list(cfg.model.extra_args)
    # The user's own tokens are never bundle tokens; keep them untouched. A
    # bundle token is dropped together with its value, or "-ub 512" lost its flag
    # and the candidate carried a bare "512" that set_value happily accepted.
    base: list[str] = []
    index = 0
    while index < len(current):
        token = current[index]
        index += 1
        if token not in _ALL_BUNDLE_TOKENS:
            base.append(token)
            continue
        if index < len(current) and not current[index].startswith("-"):
            index += 1  # the value of the token just dropped
    # Drop any bundle family already present so candidates are alternatives,
    # not stacks: `-fa on -fa off` would be a broken command line.
    present_families = {
        bundle.family for bundle in FLAG_BUNDLES if any(tok in current for tok in bundle.tokens)
    }
    out: list[Candidate] = []
    for bundle in FLAG_BUNDLES:
        if not bundle.when(ram_gb, has_gpu):
            continue
        if bundle.family in present_families:
            continue
        value = base + list(bundle.tokens)
        if value in seen or value == current:
            continue
        seen.append(value)
        out.append(Candidate("model.extra_args", value, f"{bundle.name}: {bundle.reason}"))
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


def benchmark_prompt(client, reps: int = 2, max_tokens: int = 8) -> float:
    """Median prompt-processing speed (tokens/s) on a long fixed prompt.

    Keys under ``brain.*`` only change how big the prompt is, so generation speed
    would not move; this is the number that does.
    """
    client.chat([{"role": "user", "content": "Reply with: ok"}], None, think=False, temperature=0.0, max_tokens=4)
    scores: list[float] = []
    for _ in range(max(1, reps)):
        completion = client.chat(
            [{"role": "user", "content": LONG_BENCH_PROMPT}],
            None,
            think=False,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        rate = completion.usage.prompt_per_second
        if rate:
            scores.append(rate)
    if not scores:
        raise LLMError("llama.cpp returned no prompt timings, so there is nothing to compare")
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

    stop_all = False
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
                stop_all = True
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
        if stop_all:
            # The backend just failed; every later group would restart it and
            # fail the same way, so end the whole search here.
            break
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
