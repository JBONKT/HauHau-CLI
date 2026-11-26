"""Auto-tuning and prompt self-evolution (the "auto" half of the project)."""

from __future__ import annotations

from .prompts import Evolution, PromptStore, Version, evolve_prompt, validate
from .suite import DEFAULT_TASKS, SuiteResult, Task, available, run_suite
from .tune import TUNABLE, Candidate, Trial, TuneResult, benchmark, candidates, hill_climb, record

__all__ = [
    "DEFAULT_TASKS",
    "TUNABLE",
    "Candidate",
    "Evolution",
    "PromptStore",
    "SuiteResult",
    "Task",
    "Trial",
    "TuneResult",
    "Version",
    "available",
    "benchmark",
    "candidates",
    "evolve_prompt",
    "hill_climb",
    "record",
    "run_suite",
    "validate",
]
