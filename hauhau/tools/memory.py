"""Memory tools exposed to the model."""

from __future__ import annotations

from ..util import clip_text
from .base import Tool, ToolResult, fail, ok


def build(cfg, audit, brain) -> list[Tool]:
    def rag_write(args: dict) -> ToolResult:
        text = (args.get("text") or "").strip()
        if not text:
            return fail("rag_write: 'text' is required")
        kind = str(args.get("kind") or "semantic")
        tags = args.get("tags") or ""
        if isinstance(tags, list):
            tags = ",".join(str(t) for t in tags)
        try:
            row_id, created = brain.remember(kind, text, str(tags), float(args.get("importance") or 0.5))
        except ValueError as exc:
            return fail(f"rag_write: {exc}")
        audit.log("rag_write", id=row_id, memory_kind=kind, created=created)
        return ok(f"{'stored' if created else 'merged into existing memory'} #{row_id} [{kind}]")

    def rag_query(args: dict) -> ToolResult:
        query = (args.get("query") or "").strip()
        if not query:
            return fail("rag_query: 'query' is required")
        k = int(args.get("k") or cfg.brain.top_k)
        hits = brain.recall(query, k=k)
        if not hits:
            return ok(f"nothing remembered about {query!r}")
        lines = [f"#{h.id} [{h.kind}] score={h.score:.2f} {h.text}" for h in hits]
        audit.log("rag_query", query=query, hits=len(hits))
        return ok(clip_text("\n".join(lines), cfg.tools.max_output_chars))

    return [
        Tool(
            name="rag_write",
            description=(
                "Store something worth remembering permanently: a user preference, a fact about the "
                "machine or project, or a procedure that worked. Deduplicated automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "the memory, one self-contained sentence"},
                    "kind": {
                        "type": "string",
                        "enum": ["semantic", "procedural", "episodic", "preference"],
                        "description": "fact / how-to / what happened / user preference",
                    },
                    "tags": {"type": "string", "description": "comma-separated tags"},
                    "importance": {"type": "number", "description": "0.0-1.0"},
                },
                "required": ["text"],
            },
            fn=rag_write,
        ),
        Tool(
            name="rag_query",
            description="Search your persistent memory for relevant past facts, preferences and procedures.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "number of results"},
                },
                "required": ["query"],
            },
            fn=rag_query,
        ),
    ]
