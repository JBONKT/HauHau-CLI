"""Tool abstraction and registry.

Tools are described once as JSON-schema function definitions and handed to the
model through llama.cpp's native ``tools`` parameter - no ReAct text protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..util import clip_text


@dataclass
class ToolResult:
    ok: bool
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    def clipped(self, limit: int) -> str:
        return clip_text(self.content, limit)


def ok(content: str, **meta: Any) -> ToolResult:
    return ToolResult(True, content, dict(meta))


def fail(content: str, **meta: Any) -> ToolResult:
    return ToolResult(False, content, dict(meta))


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], ToolResult]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        return self.fn(args)


class Registry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.add(tool)

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def describe(self) -> str:
        return "\n".join(f"{t.name}: {t.description.splitlines()[0]}" for t in self._tools.values())
