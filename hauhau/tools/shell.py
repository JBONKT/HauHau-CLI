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


def _segments(cmd: str) -> list[str]:
    """The statements a command runs, in order, after normalisation."""
    return [segment.strip() for segment in re.split(r"&&|\|\||;|\|", _normalise(cmd)) if segment.strip()]


def _denylist_hit(cmd: str, patterns: list[str]) -> str | None:
    """Match a pattern at the start of a statement, not anywhere in the line.

    Searching the whole command refused ``echo "rm -rf / is what the docs warn
    about"`` - the pattern only appears inside a quoted message, and the command
    that would run is ``echo``. Each ``;``/``&&``/``||``/``|``-separated segment
    is therefore matched from its start.
    """
    for pattern in patterns:
        for segment in _segments(cmd):
            try:
                if re.match(pattern, segment, re.IGNORECASE):
                    return pattern
            except re.error:
                if segment.startswith(pattern.lower()):
                    return pattern
    return None


# Starting the agent from inside itself loads a second model, a second brain and
# a second audit writer, and can recurse. Observed twice in one session: the tool
# ran `hauhau /config` to print a configuration it could have read as a file.
SELF_SPAWN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:\S*/)?hauhau(?:\s|$)"),
    re.compile(r"^(?:\S*/)?python[\d.]*\s+-m\s+hauhau(?:\s|$)"),
    re.compile(r"^(?:\S*/)?uv\s+run\s+(?:\S*/)?hauhau(?:\s|$)"),
)


def _self_spawn(cmd: str) -> str | None:
    """The statement that would launch another agent, if there is one."""
    for segment in _segments(cmd):
        for pattern in SELF_SPAWN_PATTERNS:
            if pattern.match(segment):
                return segment
    return None


def _refusal(cfg, audit, cmd: str, confirm) -> ToolResult | None:
    """Why this command must not run, or None. Never runs anything itself."""
    hit = _denylist_hit(cmd, cfg.tools.denylist)
    if hit:
        audit.log("shell.denied", cmd=cmd, pattern=hit)
        return fail(f"shell: NOT RUN - the command was blocked by denylist pattern {hit!r}.")
    spawned = _self_spawn(cmd)
    if spawned:
        audit.log("shell.denied", cmd=cmd, why="would start another agent")
        return fail(
            f"shell: NOT RUN - {spawned!r} starts a second copy of this agent, which "
            "loads another model and can recurse. Read config.toml with read_file, or "
            "ask for the slash command in the conversation instead."
        )
    if not cfg.tools.shell_confirm:
        return None
    if confirm is None:
        # Asking is the configured behaviour; with nobody to ask, the only safe
        # answer is no.
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
    return None


def _collect(proc, deadline: float, limit: int) -> tuple[bytes | None, bool, bool, int]:
    """Read the child's output until it exits or the deadline passes.

    Returns (output, timed_out, background, dropped). ``output`` is None when the
    pipe could not be read at all. The loop stops appending at ``limit`` bytes:
    buffering everything a chatty command writes meant gigabytes in memory before
    the single clip that happens after the process exits.
    """
    chunks: list[bytes] = []
    kept = dropped = 0
    timed_out = background = False
    stream = proc.stdout
    if stream is None:  # stdout=PIPE guarantees one; keep mypy honest
        _kill_tree(proc)
        return None, False, False, 0
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # The deadline can pass in the same breath as the exit.
                timed_out = proc.poll() is None
                break
            ready, _, _ = select.select([stream], [], [], min(0.2, remaining))
            if not ready:
                if proc.poll() is not None:
                    # The command finished and nothing is buffered: stop here rather
                    # than waiting for an unrelated process to close the pipe.
                    background = True
                    break
                continue
            data = os.read(stream.fileno(), 65536)
            if not data:
                break
            room = limit - kept
            if room >= len(data):
                chunks.append(data)
                kept += len(data)
            elif room > 0:
                chunks.append(data[:room])
                kept += room
                dropped += len(data) - room
            else:
                dropped += len(data)
    except BaseException:
        # The child runs in its own session, so the terminal's Ctrl-C never
        # reaches it: without this, an interrupted build or `sleep 600` kept
        # running after the user stopped the turn. Reap it too, or the Popen
        # object is collected while its child is still alive.
        _kill_tree(proc)
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            proc.wait(timeout=2)
        raise
    finally:
        # Drained or killed, the descriptor goes back now; leaving it to the GC
        # leaked one fd per shell call (ResourceWarning: unclosed file).
        stream.close()
    return b"".join(chunks), timed_out, background, dropped


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

        refusal = _refusal(cfg, audit, cmd, confirm)
        if refusal is not None:
            return refusal

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
        # Timed from the spawn: `bash -lc` startup is not the caller's budget.
        started = time.monotonic()

        # Read until the child is gone, not until the pipe reaches EOF.
        # communicate() waits for *every* holder of the write end, so a daemon
        # started by the command (or by the shell profile) inherited our stdout
        # and the call blocked for the full timeout - the interactive REPL hung
        # on its first `shell` call while a piped run of the same command was fine.
        output, timed_out, background, dropped = _collect(
            proc, started + timeout, max(4096, cfg.tools.max_output_chars * 4)
        )
        if output is None:
            return fail("shell: could not capture the command's output")
        if timed_out:
            _kill_tree(proc)
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                proc.wait(timeout=2)
            audit.log("shell.timeout", cmd=cmd, timeout=timeout)
            return fail(
                f"shell: the command was killed after {timeout:g}s (timeout). Its partial output "
                "was discarded, and any children it started were killed too."
            )
        # The loop can stop before the process does (it closed stdout). Waiting on
        # the process itself is not an OSError if it overruns: TimeoutExpired is a
        # SubprocessError, and it used to escape the tool as a crash.
        try:
            proc.wait(timeout=max(0.1, started + timeout - time.monotonic()))
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                proc.wait(timeout=2)

        duration = time.monotonic() - started
        text_out = output.decode("utf-8", "replace")
        audit.log("shell", cmd=cmd, cwd=cwd, exit=proc.returncode, duration=round(duration, 3))
        body = clip_text(text_out.rstrip(), cfg.tools.max_output_chars)
        if dropped:
            body += f"\n(the command wrote {dropped} more bytes; the rest was not kept)"
        result = f"exit={proc.returncode}\n{body}" if body else f"exit={proc.returncode}\n(no output)"
        if background:
            # The shell exited while a background process may still be writing.
            # Reporting exit=0 over truncated output looks like a clean success.
            result += (
                "\n(note: the shell exited before the output pipe closed; a background "
                "process may still be writing, and its output is not in this result)"
            )
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
