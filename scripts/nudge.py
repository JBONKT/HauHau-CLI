#!/usr/bin/env python
"""Push the running app onward when it has gone quiet.

Hauhau works in turns: it answers, then waits for the next line. Left alone it
sits at the prompt forever. This watchdog notices a session that has produced
nothing for a while (no audit event and no server output) and types one line
into the terminal tab the app is attached to, so the model continues on its own.

    python scripts/nudge.py                 # push if idle
    python scripts/nudge.py --dry-run       # report what it would do
    python scripts/nudge.py --idle-seconds 300 --max-per-hour 6

Guards, because a stuck model must not be fed lines forever:

* nothing happens while the app or llama-server is still producing output;
* a push waits at least ``--min-gap`` seconds after the previous one;
* at most ``--max-per-hour`` pushes in any rolling hour;
* every push is appended to ``~/.hauhau/logs/nudge.jsonl`` and printed once.

Delivery is macOS-specific: it uses Terminal's ``do script ... in tab`` to type
into the tab whose tty the app owns, which needs Automation permission for
Terminal (System Settings → Privacy & Security → Automation). On Linux the
gateway is writing to ``/proc/<pid>/fd/0``; that path is not implemented here,
so the script exits 3 rather than pretending.

Exit codes: 0 pushed or healthy, 1 no app running, 2 refused by a guard,
3 could not reach the app's terminal.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DATA_DIR = Path(os.environ.get("HAUHAU_DATA", "~/.hauhau")).expanduser()
AUDIT = DATA_DIR / "logs" / "audit.jsonl"
SERVER_LOG = DATA_DIR / "logs" / "server.log"
NUDGE_LOG = DATA_DIR / "logs" / "nudge.jsonl"


def app_pid() -> int | None:
    """The pid of the interactive app, ignoring this script and its shell."""
    pgrep = shutil.which("pgrep")
    if pgrep is None:
        return None
    out = subprocess.run(
        [pgrep, "-f", "[Pp]ython[0-9.]* -m hauhau$"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    for candidate in out:
        if candidate.isdigit() and int(candidate) != os.getpid():
            return int(candidate)
    return None


def app_tty(pid: int) -> str | None:
    out = subprocess.run(
        ["ps", "-o", "tty=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not out or out in {"??", "-"}:
        return None
    return out if out.startswith("/dev/") else f"/dev/{out}"


def last_activity(paths: list[Path]) -> float:
    """The newest mtime among the files the app writes while it works."""
    stamps = [p.stat().st_mtime for p in paths if p.exists()]
    return max(stamps) if stamps else 0.0


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def guard(records: list[dict], now: float, min_gap: float, max_per_hour: int) -> str | None:
    """Why this push is refused, or None to allow it."""
    stamps = sorted(r.get("ts", 0.0) for r in records)
    if stamps and now - stamps[-1] < min_gap:
        return f"the last push was {now - stamps[-1]:.0f}s ago (min gap {min_gap:.0f}s)"
    recent = [t for t in stamps if now - t < 3600]
    if len(recent) >= max_per_hour:
        return f"{len(recent)} pushes in the last hour (cap {max_per_hour})"
    return None


def applescript_for(tty: str, text: str) -> str:
    """Type ``text`` into the Terminal tab attached to ``tty``."""
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    return (
        'tell application "Terminal"\n'
        "  repeat with w from 1 to (count of windows)\n"
        "    repeat with i from 1 to (count of tabs of window w)\n"
        f'      if (tty of tab i of window w) is "{tty}" then\n'
        f'        do script "{safe}" in tab i of window w\n'
        "        return \"pushed\"\n"
        "      end if\n"
        "    end repeat\n"
        "  end repeat\n"
        '  return "no tab"\n'
        "end tell"
    )


def deliver(tty: str, text: str) -> str:
    osascript = shutil.which("osascript")
    if osascript is None:
        return "osascript is not available on this platform"
    done = subprocess.run(
        [osascript, "-e", applescript_for(tty, text)], capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        return f"osascript failed: {done.stderr.strip() or done.stdout.strip()}"
    if done.stdout.strip() != "pushed":
        return f"no Terminal tab is attached to {tty}"
    return ""


def health(url: str) -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(f"{url}/health", timeout=3) as reply:
            return json.loads(reply.read() or b"{}").get("status") == "ok"
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="push the app onward when it goes quiet")
    parser.add_argument("--idle-seconds", type=float, default=180.0)
    parser.add_argument("--min-gap", type=float, default=240.0)
    parser.add_argument("--max-per-hour", type=int, default=8)
    parser.add_argument("--text", default="continue")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    pid = app_pid()
    if pid is None:
        print("no hauhau app is running")
        return 1
    if not health(args.url):
        print(f"app {pid} is running but the model server is not answering")
        return 1

    now = time.time()
    quiet = now - last_activity([SERVER_LOG, AUDIT])
    if quiet < args.idle_seconds:
        print(f"busy: output {quiet:.0f}s ago (idle threshold {args.idle_seconds:.0f}s)")
        return 0

    refusal = guard(read_log(NUDGE_LOG), now, args.min_gap, args.max_per_hour)
    if refusal is not None:
        print(f"quiet {quiet:.0f}s but refused: {refusal}")
        return 2

    tty = app_tty(pid)
    if tty is None:
        print(f"app {pid} has no terminal attached")
        return 3

    if args.dry_run:
        print(f"would push {args.text!r} to {tty} after {quiet:.0f}s quiet")
        return 0

    problem = deliver(tty, args.text)
    if problem:
        print(problem)
        return 3

    NUDGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with NUDGE_LOG.open("a", encoding="utf-8") as out:
        out.write(
            json.dumps(
                {"ts": now, "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                 "pid": pid, "tty": tty, "text": args.text, "quiet_seconds": round(quiet)}
            )
            + "\n"
        )
    print(f"pushed {args.text!r} to {tty} after {quiet:.0f}s quiet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
