"""Shell execution with full terminal access, an audit trail, and optional gating.

The default policy for this build is auto-run everything (``shell_confirm``
off, empty denylist), per the project owner's choice. The denylist mechanism
stays available and is checked *before* execution when populated.
"""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable

from ..util import clip_text
from .base import Tool, ToolResult, fail


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the command *and its children*.

    ``subprocess.run(timeout=...)`` only kills the direct child, so anything a
    command backgrounded survived the timeout: a measured ``sleep 120 &`` was
    still running after the tool reported the timeout. The command therefore
    runs in its own session and the whole group is signalled.
    """
    with contextlib.suppress(OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    with contextlib.suppress(OSError):
        proc.kill()


def _denylist_hit(cmd: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        try:
            if re.search(pattern, cmd):
                return pattern
        except re.error:
            if pattern in cmd:
                return pattern
    return None


def build(
    cfg,
    audit,
    confirm: Callable[[str], bool] | None = None,
) -> list[Tool]:
    """Return the shell tool bound to this config/audit pair."""

    def shell(args: dict) -> ToolResult:
        cmd = (args.get("cmd") or "").strip()
        if not cmd:
            return fail("shell: 'cmd' is required")
        cwd = os.path.expanduser(str(args["cwd"])) if args.get("cwd") else str(cfg.workspace)
        timeout = float(args.get("timeout") or cfg.tools.shell_timeout)

        hit = _denylist_hit(cmd, cfg.tools.denylist)
        if hit:
            audit.log("shell.denied", cmd=cmd, pattern=hit)
            return fail(
                f"shell: NOT RUN - the command was blocked by denylist pattern {hit!r}."
            )
        if cfg.tools.shell_confirm and confirm is not None and not confirm(cmd):
            audit.log("shell.declined", cmd=cmd)
            return fail(
                "shell: NOT RUN - the user declined this command. Nothing executed; "
                "do not report any output."
            )

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                cwd=cwd if os.path.isdir(cwd) else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            audit.log("shell.error", cmd=cmd, error=str(exc))
            return fail(f"shell: could not run command: {exc}")

        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.communicate()
            audit.log("shell.timeout", cmd=cmd, timeout=timeout)
            return fail(
                f"shell: the command was killed after {timeout:g}s (timeout). Its partial output "
                "was discarded, and any children it started were killed too."
            )

        duration = time.monotonic() - started
        audit.log("shell", cmd=cmd, cwd=cwd, exit=proc.returncode, duration=round(duration, 3))
        body = clip_text((output or "").rstrip(), cfg.tools.max_output_chars)
        text = f"exit={proc.returncode}\n{body}" if body else f"exit={proc.returncode}\n(no output)"
        return ToolResult(proc.returncode == 0, text, {"exit": proc.returncode, "duration": duration})

    return [
        Tool(
            name="shell",
            description=(
                "Run a bash command on the user's machine and return stdout+stderr with the exit code. "
                "Use for inspecting files, running tests, git, package managers, and any CLI work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "bash command line to execute"},
                    "cwd": {"type": "string", "description": "working directory (default: workspace)"},
                    "timeout": {"type": "number", "description": "seconds before the command is killed"},
                },
                "required": ["cmd"],
            },
            fn=shell,
        )
    ]
