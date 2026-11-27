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

    def note_lesson(args: dict) -> ToolResult:
        """Write a how-to the moment it is learned, not at the end of the session.

        The distiller runs on exit and summarises a whole conversation; this is the
        in-turn half. When the model works out something it did not know - the flag
        that made the command work, where the config lives - recording it now means
        the next occurrence is a rag_query away instead of a second investigation.
        """
        if brain is None:
            return fail("note_lesson: persistent memory is off (--no-brain or [brain] enabled = false)")
        body = text(args, "lesson", required=True).strip()
        if not body:
            return fail("note_lesson: 'lesson' must not be empty")
        tags = text({"tags": args.get("tags") or "lesson"}, "tags").strip()
        try:
            row_id, created = brain.remember("procedural", body, tags, 0.75)
        except ValueError as exc:
            return fail(f"note_lesson: {exc}")
        audit.log("note_lesson", id=row_id, created=created, text=body[:200])
        return ok(f"{'learned' if created else 'already knew'} #{row_id}: {body[:80]}")

    return [
        Tool(
            name="note_lesson",
            description=(
                "Record a procedure the moment you work it out: the command that finally "
                "succeeded, the file that held the answer. Use it after looking something up or "
                "trying twice, so the next attempt costs one rag_query."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "lesson": {
                        "type": "string",
                        "description": "one self-contained sentence, imperative: how to do it",
                    },
                    "tags": {"type": "string", "description": "comma-separated tags"},
                },
                "required": ["lesson"],
            },
            fn=note_lesson,
        ),
        Tool(
            name="rag_write",
            description=(
                "Store a durable fact: a user preference, something about this machine or project, "
                "or a procedure that worked. Deduplicated automatically."
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
            description="Search your memory for past facts, preferences and procedures.",
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
