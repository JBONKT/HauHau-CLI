"""Auto-tuning and prompt self-evolution (the "auto" half of the project)."""

from __future__ import annotations

from .prompts import Evolution, PromptStore, Version, evolve_prompt, validate
from .suite import DEFAULT_TASKS, SuiteResult, Task, available, run_suite
from .tune import (
    TUNABLE,
    Candidate,
    FlagBundle,
    Trial,
    TuneResult,
    benchmark,
    benchmark_prompt,
    candidates,
    hill_climb,
    metric_for,
    record,
)

__all__ = [
    "DEFAULT_TASKS",
    "TUNABLE",
    "Candidate",
    "Evolution",
    "FlagBundle",
    "PromptStore",
    "SuiteResult",
    "Task",
    "Trial",
    "TuneResult",
    "Version",
    "available",
    "benchmark",
    "benchmark_prompt",
    "candidates",
    "evolve_prompt",
    "hill_climb",
    "metric_for",
    "record",
    "run_suite",
    "validate",
]
