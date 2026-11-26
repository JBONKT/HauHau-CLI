"""Tool registry assembly."""

from __future__ import annotations

from . import files, memory, search, shell
from .base import Registry, Tool, ToolResult, fail, ok

__all__ = ["Registry", "Tool", "ToolResult", "build_registry", "fail", "ok"]


def build_registry(cfg, audit, brain=None, confirm=None) -> Registry:
    tools: list[Tool] = []
    tools += shell.build(cfg, audit, confirm)
    tools += files.build(cfg, audit)
    tools += search.build(cfg, audit)
    if brain is not None:
        tools += memory.build(cfg, audit, brain)
    return Registry(tools)
