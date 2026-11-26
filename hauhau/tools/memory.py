"""Memory tools exposed to the model."""

from __future__ import annotations

from ..brain.store import KINDS
from ..util import clip_text
from .args import integer, number, text
from .base import Tool, ToolResult, fail, ok


def build(cfg, audit, brain) -> list[Tool]:
    def rag_write(args: dict) -> ToolResult:
        body = text(args, "text", required=True).strip()
        if not body:
            return fail("rag_write: 'text' must not be empty")
        kind = text(args, "kind", "semantic").strip().lower() or "semantic"
        # The store silently relabels an unknown kind "semantic", so the audit
        # log and the reply claimed a label that was never stored.
        if kind not in KINDS:
            return fail(f"rag_write: kind must be one of {', '.join(KINDS)} (got {kind!r})")
        tags = args.get("tags") or ""
        if isinstance(tags, (list, tuple)):
            tags = ",".join(str(t) for t in tags)
        tags = text({"tags": tags}, "tags").strip()
        # `or 0.5` turned an explicit 0 into 0.5: the least important memory
        # was stored with default weight.
        importance = number(args, "importance", 0.5, minimum=0.0, maximum=1.0)
        try:
            row_id, created = brain.remember(kind, body, tags, importance)
        except ValueError as exc:
            return fail(f"rag_write: {exc}")
        audit.log("rag_write", id=row_id, memory_kind=kind, created=created)
        return ok(f"{'stored' if created else 'merged into existing memory'} #{row_id} [{kind}]")

    def rag_query(args: dict) -> ToolResult:
        query = text(args, "query", required=True).strip()
        if not query:
            return fail("rag_query: 'query' must not be empty")
        # LIMIT -1 in SQLite means "no limit": a negative k dumped the whole brain.
        k = integer(args, "k", cfg.brain.top_k, minimum=1, maximum=50)
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
