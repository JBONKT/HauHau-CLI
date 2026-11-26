"""The digest: what arrived, what is due, what is next.

Written to be read in ten seconds on a terminal, and to be short enough that a
07:30 launchd job can push it as a notification. It reads the journal only - no
model call - so it is instant and cannot fail on a busy backend.
"""

from __future__ import annotations

import time

from .journal import Journal

DAY = 86400.0

SRC_ORDER = ("mail", "messages", "calls", "calendar", "reminders")
SRC_LABEL = {
    "mail": ("mail", "mails"),
    "messages": ("message", "messages"),
    "calls": ("call", "calls"),
    "calendar": ("event", "events"),
    "reminders": ("reminder", "reminders"),
}


def midnight(now: float | None = None) -> float:
    """Local midnight for the day containing ``now``."""
    stamp = time.localtime(time.time() if now is None else now)
    return time.mktime((stamp.tm_year, stamp.tm_mon, stamp.tm_mday, 0, 0, 0, 0, 0, -1))


def counts(rows) -> dict[str, int]:
    tally: dict[str, int] = {}
    for row in rows:
        tally[row.source] = tally.get(row.source, 0) + 1
    return tally


def _short(row, width: int = 92) -> str:
    who = row.people.split(",")[0] if row.people else ""
    when = time.strftime("%H:%M", time.localtime(row.ts))
    label = f"{who}: " if who else ""
    body = (row.title or row.body or "").replace("\n", " ").strip()
    return f"  {when} {label}{body}"[:width]


def compose(journal: Journal, cfg, *, now: float | None = None, lookahead_days: int = 2) -> str:
    """The brief for today, as plain text."""
    now = time.time() if now is None else now
    start = midnight(now)
    today = journal.between(start, now + 1, limit=400)
    tally = counts(today)

    lines: list[str] = [f"Brief for {time.strftime('%a %d %b %Y', time.localtime(now))}"]

    if today:
        summary = ", ".join(
            f"{tally[s]} {SRC_LABEL[s][0] if tally[s] == 1 else SRC_LABEL[s][1]}"
            for s in SRC_ORDER
            if tally.get(s)
        )
        lines.append(f"since midnight: {summary}")
        for source in ("mail", "messages", "calls"):
            rows = [r for r in today if r.source == source]
            if not rows:
                continue
            lines.append(f"{SRC_LABEL[source][1]}:")
            lines.extend(_short(r) for r in rows[-4:])
    else:
        lines.append("since midnight: nothing new")

    due = [
        r
        for r in journal.between(now - 30 * DAY, now + lookahead_days * DAY, sources=["reminders"])
        if not r.meta.get("completed")
    ]
    if due:
        lines.append("reminders:")
        lines.extend(f"  {r.when[:10]} {r.title}" for r in due[:8])

    upcoming = journal.between(now, now + lookahead_days * DAY, sources=["calendar"], limit=50)
    if upcoming:
        lines.append("next:")
        lines.extend(f"  {r.when} {r.title}" for r in upcoming[:8])

    stats = journal.stats()
    if stats.get("total"):
        last = journal.latest(now)
        if last is not None:
            age_hours = (now - last.ts) / 3600
            lines.append(f"journal: {stats['total']} events, last one {age_hours:.1f}h ago")
        else:
            lines.append(f"journal: {stats['total']} events, none before now")
    else:
        lines.append("journal: empty - run a sync")
    return "\n".join(lines)
