"""Tool registry assembly."""

from __future__ import annotations

from . import files, memory, search, shell, ui
from . import journal as journal_tools
from . import skills as skill_tools
from .base import Registry, Tool, ToolResult, fail, ok

__all__ = ["Registry", "Tool", "ToolResult", "build_registry", "fail", "ok"]


def build_registry(cfg, audit, brain=None, confirm=None, journal=None, skills=None) -> Registry:
    tools: list[Tool] = []
    tools += shell.build(cfg, audit, confirm)
    tools += files.build(cfg, audit)
    tools += search.build(cfg, audit)
    tools += ui.build(cfg, audit, skills)
    if brain is not None:
        tools += memory.build(cfg, audit, brain)
    if journal is not None and cfg.journal.enabled:
        tools += journal_tools.build(cfg, audit, journal)
    if skills:
        tools += skill_tools.build(cfg, audit, skills)
    return Registry(tools)
