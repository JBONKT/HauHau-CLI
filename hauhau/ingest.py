"""Reading the Mac's own records into the journal.

Five sources, three mechanisms:

* **AppleScript** for Mail, Calendar and Reminders. No credentials, but the app
  must be installed and macOS asks for Automation permission once per pair.
* **sqlite, read-only** for iMessage/SMS (``~/Library/Messages/chat.db``) and call
  history (``CallHistoryDB``). Both live behind Full Disk Access: without it the
  open fails with "authorization denied" and the source reports a permission
  problem rather than pretending there was nothing to read.
* **Nothing else**. There is no network call here, and no account password.

Every fetcher returns ``(entries, cursor)``. The cursor is stored so the next run
resumes instead of re-reading a year of history, and inserting is idempotent
anyway, so a repeated fetch costs nothing.
"""

from __future__ import annotations

import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .journal import Entry

# Apple's reference date, 2001-01-01, used by chat.db and CallHistoryDB.
APPLE_EPOCH = 978307200.0

MESSAGES_DB = Path("~/Library/Messages/chat.db")
CALLS_DB = Path("~/Library/Application Support/CallHistoryDB/CallHistory.storedata")


class SourceError(RuntimeError):
    """A source could not be read, with the reason the user needs to fix."""


@dataclass
class Fetch:
    entries: list[Entry]
    cursor: str | None
    note: str = ""


# --- AppleScript plumbing ---------------------------------------------
def run_applescript(script: str, timeout: float = 60.0) -> str:
    """Run AppleScript and return its stdout, raising :class:`SourceError`."""
    try:
        done = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # not macOS
        raise SourceError("osascript is not available on this platform") from exc
    except subprocess.TimeoutExpired as exc:
        raise SourceError(f"AppleScript timed out after {timeout:.0f}s") from exc
    if done.returncode != 0:
        message = (done.stderr or done.stdout or "").strip().splitlines()
        raise SourceError(message[-1] if message else f"osascript exited {done.returncode}")
    return done.stdout


def _ensure_app(name: str) -> None:
    """Start an app in the background, so AppleScript can talk to it.

    A quit app answers -600 ("the application isn't running"), and
    ``tell application X to launch`` failed with the same -600 on this machine,
    so the app is opened hidden with ``open -gja`` instead: no window takes
    focus, which matters for a sync that runs every 15 minutes.
    """
    try:
        subprocess.run(["open", "-gja", name], capture_output=True, timeout=30, check=False)
        time.sleep(1.0)
    except (OSError, subprocess.SubprocessError):
        pass


def rows_of(output: str, fields: int) -> list[list[str]]:
    """Parse ``field TAB field … NEWLINE`` rows, dropping malformed ones.

    AppleScript output is the only interface these apps have, so a row that
    arrives with the wrong number of fields is skipped rather than crashing the
    whole sync - one odd mail must not cost the other 300.
    """
    parsed: list[list[str]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < fields:
            continue
        parsed.append(parts[:fields])
    return parsed


def iso_to_epoch(text: str) -> float:
    """AppleScript's ``«class isot»`` string to epoch seconds.

    A trailing ``Z`` or a numeric offset means the value is absolute, so it is
    parsed timezone-aware; a naive value is local time (AppleScript's default).
    """
    from datetime import datetime

    cleaned = text.strip()
    if not cleaned:
        return time.time()
    try:
        # Python 3.11+ fromisoformat accepts both "Z" and "+02:00"/"+0200".
        return datetime.fromisoformat(cleaned).timestamp()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(cleaned, fmt))
        except ValueError:
            continue
    return time.time()


# --- sources -----------------------------------------------------------
def _script(*lines: str) -> str:
    """Join AppleScript lines, so no single source line has to be enormous."""
    return "\n".join(lines)


MAIL_SCRIPT = _script(
    "set daysBack to {days}",
    "set cutoff to (current date) - daysBack * days",
    'tell application "Mail"',
    "    set out to {{}}",
    "    set n to 0",
    "    repeat with m in (messages of inbox whose date received >= cutoff)",
    "        if n >= {limit} then exit repeat",
    "        set theRow to (id of m as string) & tab & (sender of m) & tab"
    ' & (subject of m) & tab & ((date received of m) as \u00abclass isot\u00bb as string)',
    "        set end of out to theRow",
    "        set n to n + 1",
    "    end repeat",
    "end tell",
    "set AppleScript's text item delimiters to linefeed",
    "return out as string",
)


def fetch_mail(days: int = 7, limit: int = 300) -> Fetch:
    """Recent INBOX mail. Bodies are not read: one AppleScript call per message
    is slow enough to stall the sync, and the subject carries the signal.

    The timeout is generous because the first call of the day also has to launch
    Mail.app, which took longer than a minute on a cold start.
    """
    _ensure_app("Mail")
    output = run_applescript(MAIL_SCRIPT.format(days=int(days), limit=int(limit)), timeout=180.0)
    entries: list[Entry] = []
    for uid, sender, subject, when in rows_of(output, 4):
        entries.append(
            Entry(
                source="mail",
                external_id=f"mail:{uid}",
                kind="email",
                ts=iso_to_epoch(when),
                title=subject.strip(),
                body="",
                people=[sender.strip()] if sender.strip() else [],
                meta={"mailbox": "inbox"},
            )
        )
    entries.sort(key=lambda e: e.ts)
    newest = f"mail:{entries[-1].ts:.0f}" if entries else None
    return Fetch(entries, newest)


CALENDAR_SCRIPT = _script(
    "set startDate to (current date) - {back} * days",
    "set endDate to (current date) + {ahead} * days",
    'tell application "Calendar"',
    "    set out to {{}}",
    "    repeat with cal in calendars",
    "        repeat with ev in (every event of cal whose start date >= startDate"
    " and start date <= endDate)",
    "            set theRow to (summary of ev) & tab"
    ' & ((start date of ev) as \u00abclass isot\u00bb as string)'
    " & tab & (name of cal) & tab & (uid of ev)",
    "            set end of out to theRow",
    "        end repeat",
    "    end repeat",
    "end tell",
    "set AppleScript's text item delimiters to linefeed",
    "return out as string",
)


def fetch_calendar(days_back: int = 1, days_ahead: int = 14) -> Fetch:
    _ensure_app("Calendar")
    output = run_applescript(
        CALENDAR_SCRIPT.format(back=int(days_back), ahead=int(days_ahead)), timeout=120.0
    )
    entries: list[Entry] = []
    for summary, start, cal, uid in rows_of(output, 4):
        entries.append(
            Entry(
                source="calendar",
                external_id=f"cal:{uid.strip()}",
                kind="event",
                ts=iso_to_epoch(start),
                title=summary.strip(),
                body="",
                people=[],
                meta={"calendar": cal.strip()},
            )
        )
    entries.sort(key=lambda e: e.ts)
    newest = f"cal:{entries[-1].ts:.0f}" if entries else None
    return Fetch(entries, newest)


REMINDERS_SCRIPT = _script(
    'tell application "Reminders"',
    "    set out to {}",
    "    repeat with l in lists",
    "        set listName to name of l",
    "        repeat with r in (every reminder of l whose completed is false)",
    '            set dueText to ""',
    "            try",
    '                set dueText to ((due date of r) as \u00abclass isot\u00bb as string)',
    "            end try",
    # A reminder reached through "every reminder of every list" does not know its
    # own list, so the list name is carried in from the outer loop.
    "            set theRow to (name of r) & tab & dueText & tab"
    " & listName & tab & (id of r)",
    "            set end of out to theRow",
    "        end repeat",
    "    end repeat",
    "end tell",
    "set AppleScript's text item delimiters to linefeed",
    "return out as string",
)


def fetch_reminders() -> Fetch:
    _ensure_app("Reminders")
    output = run_applescript(REMINDERS_SCRIPT, timeout=120.0)
    entries: list[Entry] = []
    now = time.time()
    for name, due, listname, uid in rows_of(output, 4):
        entries.append(
            Entry(
                source="reminders",
                external_id=f"rem:{uid.strip()}",
                kind="reminder",
                # An undated reminder belongs at "now" so the brief still shows it.
                ts=iso_to_epoch(due) if due.strip() else now,
                title=name.strip(),
                body="",
                people=[],
                meta={"list": listname.strip(), "due": bool(due.strip())},
            )
        )
    return Fetch(entries, None)


def _open_ro(path: Path) -> sqlite3.Connection:
    """Open a system database read-only, with the reason when macOS refuses."""
    if not path.exists():
        raise SourceError(f"not found: {path}")
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=10.0)
        conn.execute("SELECT 1")
        return conn
    except sqlite3.Error as exc:
        raise SourceError(
            f"cannot read {path}: {exc}. Grant Full Disk Access to the Python "
            f"interpreter that runs Hauhau (System Settings > Privacy & Security > "
            f"Full Disk Access), then sync again."
        ) from exc


MESSAGES_SQL = """
SELECT m.ROWID, m.date, m.is_from_me, m.text, h.id
FROM message m
LEFT JOIN handle h ON m.handle_id = h.ROWID
WHERE m.date > ?
ORDER BY m.date DESC
LIMIT ?
"""


def fetch_messages(since: float | None = None, limit: int = 500) -> Fetch:
    """iMessage and SMS, from a read-only handle on chat.db.

    The newest ``limit`` rows are read (DESC) and then reversed, so the first
    sync after a cursor wipe gets the latest messages, not the oldest ones
    from years back.
    """
    conn = _open_ro(MESSAGES_DB.expanduser())
    try:
        cutoff = 0 if since is None else int((since - APPLE_EPOCH) * 1_000_000_000)
        entries: list[Entry] = []
        rows = conn.execute(MESSAGES_SQL, (cutoff, limit)).fetchall()
        for rowid, date, is_from_me, text, handle in reversed(rows):
            ts = (date or 0) / 1_000_000_000 + APPLE_EPOCH
            body = (text or "").strip()
            if not body:
                # Some messages keep their text in attributedBody only.
                body = "(attachment or rich text)"
            who = (handle or "unknown").strip()
            entries.append(
                Entry(
                    source="messages",
                    external_id=f"msg:{rowid}",
                    kind="message",
                    ts=ts,
                    title=body[:120],
                    body=body,
                    people=[who],
                    meta={"from_me": bool(is_from_me)},
                )
            )
    finally:
        conn.close()
    newest = f"msg:{max((e.ts for e in entries), default=time.time()):.0f}" if entries else None
    return Fetch(entries, newest)


CALLS_SQL = """
SELECT Z_PK, ZDATE, ZDURATION, ZADDRESS, ZORIGINATED, ZCALLTYPE
FROM Z_CALLRECORD
WHERE ZDATE > ?
ORDER BY ZDATE DESC
LIMIT ?
"""


def fetch_calls(since: float | None = None, limit: int = 300) -> Fetch:
    conn = _open_ro(CALLS_DB.expanduser())
    try:
        cutoff = 0.0 if since is None else (since - APPLE_EPOCH)
        entries: list[Entry] = []
        rows = conn.execute(CALLS_SQL, (cutoff, limit)).fetchall()
        for pk, date, duration, address, originated, call_type in reversed(rows):
            ts = (date or 0) + APPLE_EPOCH
            outgoing = bool(originated)
            entries.append(
                Entry(
                    source="calls",
                    external_id=f"call:{pk}",
                    kind="call",
                    ts=ts,
                    title=f"{'outgoing' if outgoing else 'incoming'} call"
                    f"{f' ({int(duration)}s)' if duration else ''}",
                    body="",
                    people=[(address or "unknown").strip()],
                    meta={"outgoing": outgoing, "seconds": float(duration or 0), "type": call_type},
                )
            )
    finally:
        conn.close()
    newest = f"call:{max((e.ts for e in entries), default=time.time()):.0f}" if entries else None
    return Fetch(entries, newest)


FETCHERS = {
    "mail": fetch_mail,
    "calendar": fetch_calendar,
    "reminders": fetch_reminders,
    "messages": fetch_messages,
    "calls": fetch_calls,
}


def sync(
    journal,
    sources: list[str] | None = None,
    *,
    journal_cfg=None,
    progress=None,
) -> dict[str, object]:
    """Refresh every source. Returns per-source counts and any error text.

    Cursor handling is per source: messages and calls take the newest timestamp
    they have seen so the next run only asks for the delta; mail and calendar
    re-read their whole window (their fetchers take a day range, not a cursor)
    and stay correct because inserts are idempotent. ``journal_cfg`` is the
    ``[journal]`` config section; its limits and windows apply here. A source
    that raises is recorded and skipped - one missing permission must not stop
    the other four.
    """

    def cfg_value(key: str, default: int) -> int:
        return int(getattr(journal_cfg, key, default)) if journal_cfg is not None else default

    report: dict[str, object] = {}
    for name in sources or list(FETCHERS):
        if progress is not None:
            progress(name)
        try:
            fetcher = FETCHERS[name]
            cursor = journal.cursor(name)
            since = _cursor_epoch(cursor)
            kwargs: dict[str, object] = {}
            if name == "messages":
                kwargs["limit"] = cfg_value("messages_limit", 500)
                if since is not None:
                    kwargs["since"] = since
            elif name == "calls":
                kwargs["limit"] = cfg_value("calls_limit", 300)
                if since is not None:
                    kwargs["since"] = since
            elif name == "mail":
                kwargs["days"] = cfg_value("mail_days", 7)
            elif name == "calendar":
                kwargs["days_back"] = cfg_value("calendar_days_back", 1)
                kwargs["days_ahead"] = cfg_value("calendar_days_ahead", 14)
            fetched = fetcher(**kwargs)
            added = journal.add(fetched.entries)
            if fetched.cursor:
                journal.set_cursor(name, fetched.cursor)
            journal.set_state(name, True)
            report[name] = {"fetched": len(fetched.entries), "added": added}
        except SourceError as exc:
            journal.set_state(name, False, str(exc))
            report[name] = {"error": str(exc)}
        except Exception as exc:  # a broken source must not stop the sync
            journal.set_state(name, False, f"{exc.__class__.__name__}: {exc}")
            report[name] = {"error": f"{exc.__class__.__name__}: {exc}"}
    return report


def _cursor_epoch(cursor: str | None) -> float | None:
    """The epoch a stored ``prefix:<seconds>`` cursor encodes."""
    if not cursor or ":" not in cursor:
        return None
    try:
        return float(cursor.rsplit(":", 1)[1])
    except ValueError:
        return None
