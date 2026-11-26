"""Shell execution with full terminal access, an audit trail, and optional gating.

The default policy for this build is auto-run everything (``shell_confirm``
off, empty denylist), per the project owner's choice. The denylist mechanism
stays available and is checked *before* execution when populated.
"""

from __future__ import annotations

import contextlib
import os
import re
import select
import signal
import subprocess
import time
from collections.abc import Callable

from ..util import clip_text
from .args import number, text
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


def _normalise(cmd: str) -> str:
    """Flatten the easy ways of writing the same command differently.

    A raw regex over the raw command missed ``RM -rf /``, ``rm \\<newline>-rf``
    and ``r''m -rf``. Quoting, line continuation, case and repeated spaces are
    normalised before matching. Substitution (``X=rm; $X -rf``) still gets
    through: this list is a speed bump, and ``shell_confirm`` is the real gate.
    """
    flat = cmd.replace("\\\n", " ")
    flat = re.sub(r"['\"]", "", flat)
    return re.sub(r"\s+", " ", flat).strip().lower()


def _denylist_hit(cmd: str, patterns: list[str]) -> str | None:
    normalised = _normalise(cmd)
    for pattern in patterns:
        try:
            if re.search(pattern, normalised, re.IGNORECASE):
                return pattern
        except re.error:
            if pattern.lower() in normalised:
                return pattern
    return None


def build(
    cfg,
    audit,
    confirm: Callable[[str], bool] | None = None,
) -> list[Tool]:
    """Return the shell tool bound to this config/audit pair."""

    def shell(args: dict) -> ToolResult:
        cmd = text(args, "cmd", required=True).strip()
        if not cmd:
            return fail("shell: 'cmd' must not be empty")
        asked_for_cwd = text(args, "cwd").strip()
        if asked_for_cwd:
            cwd = os.path.expanduser(asked_for_cwd)
            if not os.path.isdir(cwd):
                # Falling back to the agent's own directory ran "rm -rf build"
                # somewhere the caller never named, and reported exit=0.
                return fail(f"shell: cwd is not a directory: {cwd}")
        else:
            cwd = str(cfg.workspace)
        timeout = number(args, "timeout", cfg.tools.shell_timeout, minimum=0.1, maximum=3600.0)

        hit = _denylist_hit(cmd, cfg.tools.denylist)
        if hit:
            audit.log("shell.denied", cmd=cmd, pattern=hit)
            return fail(
                f"shell: NOT RUN - the command was blocked by denylist pattern {hit!r}."
            )
        if cfg.tools.shell_confirm:
            if confirm is None:
                # Asking is the configured behaviour; with nobody to ask, the
                # only safe answer is no.
                audit.log("shell.denied", cmd=cmd, why="confirmation required but none available")
                return fail(
                    "shell: NOT RUN - this session requires confirmation, but no prompt is "
                    "available to ask. Nothing executed; do not report any output."
                )
            if not confirm(cmd):
                audit.log("shell.declined", cmd=cmd)
                return fail(
                    "shell: NOT RUN - the user declined this command. Nothing executed; "
                    "do not report any output."
                )

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                ["bash", "-lc", cmd],
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            audit.log("shell.error", cmd=cmd, error=str(exc))
            return fail(f"shell: could not run command: {exc}")

        # Read until the child is gone, not until the pipe reaches EOF.
        # communicate() waits for *every* holder of the write end, so a daemon
        # started by the command (or by the shell profile) inherited our stdout
        # and the call blocked for the full timeout - the interactive REPL hung
        # on its first `shell` call while a piped run of the same command was fine.
        chunks: list[bytes] = []
        deadline = started + timeout
        timed_out = False
        stream = proc.stdout
        if stream is None:  # stdout=PIPE guarantees one; keep mypy honest
            _kill_tree(proc)
            return fail("shell: could not capture the command's output")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([stream], [], [], min(0.2, remaining))
            if ready:
                data = os.read(stream.fileno(), 65536)
                if not data:
                    break
                chunks.append(data)
            elif proc.poll() is not None:
                # The command finished and nothing is buffered: stop here rather
                # than waiting for an unrelated process to close the pipe.
                break
        if timed_out:
            _kill_tree(proc)
            with contextlib.suppress(OSError):
                proc.wait(timeout=2)
            audit.log("shell.timeout", cmd=cmd, timeout=timeout)
            return fail(
                f"shell: the command was killed after {timeout:g}s (timeout). Its partial output "
                "was discarded, and any children it started were killed too."
            )
        with contextlib.suppress(OSError):
            proc.wait(timeout=2)

        duration = time.monotonic() - started
        output = b"".join(chunks).decode("utf-8", "replace")
        audit.log("shell", cmd=cmd, cwd=cwd, exit=proc.returncode, duration=round(duration, 3))
        body = clip_text(output.rstrip(), cfg.tools.max_output_chars)
        result = f"exit={proc.returncode}\n{body}" if body else f"exit={proc.returncode}\n(no output)"
        return ToolResult(proc.returncode == 0, result, {"exit": proc.returncode, "duration": duration})

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
