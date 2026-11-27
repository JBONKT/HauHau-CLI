"""Tool abstraction and registry.

Tools are described once as JSON-schema function definitions and handed to the
model through llama.cpp's native ``tools`` parameter - no ReAct text protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..util import clip_text
from .args import BadArgument


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


def _compact(value: Any) -> Any:
    """Drop empty strings, lists and dicts from a schema, recursively."""
    if isinstance(value, dict):
        cleaned = {key: _compact(item) for key, item in value.items()}
        return {key: item for key, item in cleaned.items() if item not in ("", [], {})}
    if isinstance(value, list):
        return [_compact(item) for item in value]
    return value


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[[dict[str, Any]], ToolResult]

    def schema(self) -> dict[str, Any]:
        """The JSON-schema definition sent to the model on every request.

        Every tool schema is prompt: on this machine a step costs milliseconds of
        prefill per token, and `"description": ""` on a property that has none is
        pure waste. Empty values are dropped and the text is collapsed to one
        space-separated line, which is what the model reads anyway.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": " ".join(self.description.split()),
                "parameters": _compact(self.parameters),
            },
        }

    def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            return self.fn(args)
        except BadArgument as exc:
            # An unusable argument is a normal failure with a fixable message,
            # not a crash: one place for every tool.
            return fail(f"{self.name}: {exc}")


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
