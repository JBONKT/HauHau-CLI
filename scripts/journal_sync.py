#!/usr/bin/env python
"""Background journal sync and morning brief, as a launchd agent.

    .venv/bin/python scripts/journal_sync.py --once       # sync now
    .venv/bin/python scripts/journal_sync.py --brief      # print today's digest
    .venv/bin/python scripts/journal_sync.py --install    # every 15 min + 07:30 brief
    .venv/bin/python scripts/journal_sync.py --status
    .venv/bin/python scripts/journal_sync.py --uninstall

Two agents are written to ``~/Library/LaunchAgents``:

* ``com.hauhau.sync``  - every 900 s, silent, logs to ``~/.hauhau/logs/sync.log``
* ``com.hauhau.brief`` - at 07:30, prints the digest and shows a notification

Both run without a terminal, so macOS permissions must be granted to the
interpreter they use, not to Terminal: Full Disk Access for chat.db and call
history, Automation for Mail/Calendar/Reminders. ``--status`` reports which of
those currently work, because a silent "permission denied" every 15 minutes is
worse than no job at all.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hauhau import config as config_mod  # noqa: E402
from hauhau.brief import compose  # noqa: E402
from hauhau.ingest import FETCHERS, sync  # noqa: E402
from hauhau.journal import Journal  # noqa: E402

LABEL_SYNC = "com.hauhau.sync"
LABEL_BRIEF = "com.hauhau.brief"


def paths() -> tuple[Path, Path, Path]:
    cfg, _warnings = config_mod.load(ROOT / "config.toml")
    return cfg.journal_path, cfg.data_dir / "logs", cfg  # type: ignore[return-value]


def plist(label: str, program: list[str], *, interval: int | None, calendar: tuple[int, int] | None,
          out: Path) -> str:
    args = "".join(f"        <string>{part}</string>\n" for part in program)
    schedule = ""
    if interval:
        schedule += f"    <key>StartInterval</key>\n    <integer>{interval}</integer>\n"
    if calendar:
        hour, minute = calendar
        schedule += (
            "    <key>StartCalendarInterval</key>\n    <dict>\n"
            f"        <key>Hour</key><integer>{hour}</integer>\n"
            f"        <key>Minute</key><integer>{minute}</integer>\n"
            "    </dict>\n"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n<dict>\n'
        f"    <key>Label</key>\n    <string>{label}</string>\n"
        f"    <key>ProgramArguments</key>\n    <array>\n{args}    </array>\n"
        f"{schedule}"
        "    <key>RunAtLoad</key>\n    <false/>\n"
        f"    <key>StandardOutPath</key>\n    <string>{out}</string>\n"
        f"    <key>StandardErrorPath</key>\n    <string>{out}</string>\n"
        "    <key>ProcessType</key>\n    <string>Background</string>\n"
        "</dict>\n</plist>\n"
    )


def agents_dir() -> Path:
    return Path(os.path.expanduser("~/Library/LaunchAgents"))


def install() -> int:
    journal_path, logs, _cfg = paths()
    logs.mkdir(parents=True, exist_ok=True)
    script = str(Path(__file__).resolve())
    python = sys.executable
    agents_dir().mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            LABEL_SYNC,
            [python, script, "--once"],
            {"interval": 900, "calendar": None, "out": logs / "sync.log"},
        ),
        (
            LABEL_BRIEF,
            [python, script, "--brief", "--notify"],
            {"interval": None, "calendar": (7, 30), "out": logs / "brief.log"},
        ),
    ]
    for label, program, options in jobs:
        target = agents_dir() / f"{label}.plist"
        target.write_text(plist(label, program, **options), encoding="utf-8")
        subprocess.run(["launchctl", "unload", str(target)], capture_output=True, check=False)
        done = subprocess.run(["launchctl", "load", str(target)], capture_output=True, text=True,
                              check=False)
        state = "loaded" if done.returncode == 0 else f"failed: {done.stderr.strip()}"
        print(f"{label}: {state} ({target})")
    print(f"journal: {journal_path}")
    print("logs:    " + str(logs))
    print("Grant Full Disk Access to: " + python)
    return 0


def uninstall() -> int:
    for label in (LABEL_SYNC, LABEL_BRIEF):
        target = agents_dir() / f"{label}.plist"
        if target.exists():
            subprocess.run(["launchctl", "unload", str(target)], capture_output=True, check=False)
            target.unlink()
            print(f"{label}: removed")
        else:
            print(f"{label}: not installed")
    return 0


def status() -> int:
    journal_path, _logs, _cfg = paths()
    journal = Journal(journal_path)
    try:
        info = journal.stats()
        print(f"journal {journal_path}: {info['total']} events")
        for source, count in dict(info["by_source"]).items():
            print(f"  {source:<10} {count:>6}")
        for label in (LABEL_SYNC, LABEL_BRIEF):
            target = agents_dir() / f"{label}.plist"
            print(f"{label}: {'installed' if target.exists() else 'not installed'}")
        # Recorded by the last real sync: probing here would take a minute
        # (Mail has to launch) and would leave the permissions untested when
        # nobody is watching.
        states = journal.states()
        print("sources (from the last sync — run --once to re-test):")
        for name in FETCHERS:
            state = states.get(name)
            if state is None:
                print(f"  {name:<10} never synced")
            elif state["ok"]:
                print(f"  {name:<10} ok")
            else:
                print(f"  {name:<10} FAILS: {state['detail'][:120]}")
    finally:
        journal.close()
    return 0


def notify(text: str) -> None:
    """A notification without extra dependencies."""
    title = "Hauhau brief"
    body = text.splitlines()[0] if text else title
    script = f'display notification "{body}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="sync the journal and print the brief")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--once", action="store_true", help="sync every source now")
    group.add_argument("--brief", action="store_true", help="print today's digest")
    group.add_argument("--install", action="store_true", help="install the launchd agents")
    group.add_argument("--uninstall", action="store_true", help="remove the launchd agents")
    group.add_argument("--status", action="store_true", help="what the journal holds, and what it can read")
    parser.add_argument("--notify", action="store_true", help="also show a notification")
    args = parser.parse_args(argv)

    if args.install:
        return install()
    if args.uninstall:
        return uninstall()
    if args.status:
        return status()

    journal_path, _logs, cfg = paths()
    journal = Journal(journal_path)
    try:
        if args.once:
            report = sync(journal, progress=lambda name: print(f"syncing {name}…", file=sys.stderr))
            for name, result in report.items():
                if "error" in result:
                    print(f"{name}: FAILED {result['error']}")
                else:
                    print(f"{name}: {result['added']} new of {result['fetched']} seen")
            return 0
        digest = compose(journal, cfg)
        print(digest)
        if args.notify:
            notify(digest)
        return 0
    finally:
        journal.close()


if __name__ == "__main__":
    sys.exit(main())
