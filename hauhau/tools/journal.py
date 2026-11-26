"""Tools for the second-brain journal: what happened, and what is coming.

The brain tools answer "what did I learn"; these answer "what was said", "when is
it", "what did Dana send". Kept separate so a sync of 300 mails cannot be
confused with the few dozen distilled memories injected into the prompt.
"""

from __future__ import annotations

import time

from ..util import clip_text
from .args import integer, text
from .base import Tool, ToolResult, fail, ok


def _line(row, width: int = 200) -> str:
    who = f" ({row.people})" if row.people else ""
    return f"[{row.when}] {row.kind}#{row.id}{who} {row.title or row.body}"[:width]


def build(cfg, audit, journal) -> list[Tool]:
    def journal_search(args: dict) -> ToolResult:
        query = text(args, "query", required=True).strip()
        if not query:
            return fail("journal_search: 'query' must not be empty")
        k = integer(args, "k", 10, minimum=1, maximum=50)
        source = text(args, "source").strip()
        sources = [source] if source else None
        rows = journal.search(query, k=k, sources=sources)
        audit.log("journal_search", query=query, k=k, source=source or None, hits=len(rows))
        if not rows:
            return ok(f"nothing in the journal matches {query!r}")
        return ok(clip_text("\n".join(_line(r) for r in rows), cfg.tools.max_output_chars))

    def agenda(args: dict) -> ToolResult:
        days = integer(args, "days", 7, minimum=1, maximum=90)
        now = time.time()
        end = now + days * 86400
        events = journal.between(now - 3600, end, sources=["calendar"], limit=120)
        due = journal.between(now - 30 * 86400, end, sources=["reminders"], limit=120)
        lines = [f"next {days} days:"]
        lines.extend([_line(r) for r in events] or ["  no calendar events"])
        pending = [r for r in due if not r.meta.get("completed")]
        if pending:
            lines.append("reminders:")
            lines += [_line(r) for r in pending]
        audit.log("agenda", days=days, events=len(events), reminders=len(pending))
        return ok(clip_text("\n".join(lines), cfg.tools.max_output_chars))

    def person(args: dict) -> ToolResult:
        who = text(args, "who", required=True).strip()
        if not who:
            return fail("person: 'who' must not be empty")
        k = integer(args, "k", 15, minimum=1, maximum=50)
        rows = journal.person(who, k=k)
        audit.log("journal_person", who=who, hits=len(rows))
        if not rows:
            return ok(f"nothing in the journal involves {who!r}")
        return ok(clip_text("\n".join(_line(r) for r in rows), cfg.tools.max_output_chars))

    def brief(_args: dict) -> ToolResult:
        from ..brief import compose

        summary = compose(journal, cfg)
        audit.log("brief", lines=len(summary.splitlines()))
        return ok(clip_text(summary, cfg.tools.max_output_chars * 2))

    return [
        Tool(
            name="journal_search",
            description=(
                "Search the record of what actually happened - mails, messages, calls, calendar "
                "events and reminders. Use it for what was said or sent; use rag_query for lessons "
                "learned, and read_file for the contents of a file."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "words to look for"},
                    "k": {"type": "integer", "description": "how many results"},
                    "source": {
                        "type": "string",
                        "enum": ["mail", "messages", "calls", "calendar", "reminders"],
                    },
                },
                "required": ["query"],
            },
            fn=journal_search,
        ),
        Tool(
            name="agenda",
            description="Upcoming calendar events and open reminders for the next few days.",
            parameters={
                "type": "object",
                "properties": {"days": {"type": "integer", "description": "how far ahead, default 7"}},
            },
            fn=agenda,
        ),
        Tool(
            name="person",
            description="Everything in the journal involving one person, by name, email or number.",
            parameters={
                "type": "object",
                "properties": {
                    "who": {"type": "string"},
                    "k": {"type": "integer", "description": "how many results"},
                },
                "required": ["who"],
            },
            fn=person,
        ),
        Tool(
            name="brief",
            description="Today's digest: what came in since midnight, what is due, what is next.",
            parameters={"type": "object", "properties": {}},
            fn=brief,
        ),
    ]
