"""File tools. Every mutation snapshots the previous version first."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from ..util import clip_text
from .args import flag, integer, text
from .base import Tool, ToolResult, fail, ok

# A single very long line is truncated rather than loaded whole, and files too
# big to rewrite in memory are refused by edit_file.
MAX_LINE_CHARS = 4000
EDIT_MAX_BYTES = 32 * 1024 * 1024
# A read with no limit used to load the whole file (397 MB for a 107 MB log to
# return 8 KB), so "no limit" now means a page, not everything.
DEFAULT_READ_LINES = 2000


def _resolve(path: str) -> Path:
    return Path(os.path.expanduser(path)).expanduser()


def _snapshot(cfg, audit, path: Path) -> None:
    if not path.is_file():
        return
    dest_dir = cfg.snapshots_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    dest = dest_dir / f"{stamp}_{path.name}"
    # exists()-then-copy let two writers pick the same name and clobber each
    # other's pre-image; O_EXCL reserves the name first.
    reserved: Path | None = None
    for counter in range(1000):
        dest = dest_dir / (f"{stamp}_{path.name}" if counter == 0 else f"{stamp}-{counter}_{path.name}")
        try:
            handle = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(handle)
        reserved = dest
        break
    if reserved is None:
        # Every name was taken in the same second; skipping beats clobbering
        # another writer's pre-image with copy2.
        audit.log("snapshot_skipped", path=str(path), reason="1000 name collisions in one second")
        return
    shutil.copy2(path, reserved)
    audit.log("snapshot", path=str(path), copy=str(reserved))


def _read_window(path: Path, offset: int, limit: int) -> tuple[list[str], bool]:
    """Read at most `limit` lines from `offset` without loading the file.

    `read_text()` pulled the whole file into memory: asking for 3 lines of a
    286 MB log cost 903 MB of resident memory. This walks the file with a bound
    on both line length and line count.
    """
    window: list[str] = []
    has_more = False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        index = 0
        while True:
            line = fh.readline(MAX_LINE_CHARS + 1)
            if not line:
                break
            if len(line) > MAX_LINE_CHARS and not line.endswith("\n"):
                # Over-long line: keep the head and discard the remainder.
                while True:
                    chunk = fh.readline(65536)
                    if not chunk or chunk.endswith("\n"):
                        break
                line = line[:MAX_LINE_CHARS] + " …[line truncated]\n"
            if index < offset:
                index += 1
                continue
            if limit and len(window) >= limit:
                has_more = True
                break
            window.append(line.rstrip("\n"))
            index += 1
    return window, has_more


def build(cfg, audit) -> list[Tool]:
    max_chars = cfg.tools.max_output_chars

    def read_file(args: dict) -> ToolResult:
        path = _resolve(text(args, "path", required=True))
        if not path.is_file():
            return fail(f"read_file: not a file: {path}")
        offset = integer(args, "offset", 0, minimum=0)
        limit = integer(args, "limit", 0, minimum=0)
        if limit <= 0:
            # "no limit", -1 and 0 all mean "a page", never "the whole file".
            limit = DEFAULT_READ_LINES
        try:
            window, has_more = _read_window(path, offset, limit)
        except OSError as exc:
            return fail(f"read_file: {exc}")
        if not window:
            if offset:
                return fail(f"read_file: offset {offset} is past the end of the file")
            return ok("(empty file)")
        body = "\n".join(window)
        if not body:
            # A file holding just a newline used to come back as an empty string,
            # indistinguishable from a failed read.
            body = "(one empty line)"
        if has_more:
            body += "\n...[more lines follow; use offset/limit]..."
        audit.log("read_file", path=str(path), lines=len(window))
        return ok(clip_text(body, max_chars))

    def write_file(args: dict) -> ToolResult:
        path = _resolve(text(args, "path", required=True))
        content = text(args, "content", required=True)
        existed = path.is_file()
        _snapshot(cfg, audit, path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return fail(f"write_file: {exc}")
        written = len(content.encode("utf-8"))
        audit.log("write_file", path=str(path), bytes=written, existed=existed)
        # Bytes, not characters: "日本語" is 3 characters and 9 bytes, and the
        # report used to say 3 while the file on disk was 9 bytes larger.
        return ok(f"wrote {written} bytes to {path}" + ("" if existed else " (new file)"))

    def edit_file(args: dict) -> ToolResult:
        path = _resolve(text(args, "path", required=True))
        old = text(args, "old" if "old" in args else "old_string", required=True)
        new = text(args, "new" if "new" in args else "new_string", required=True)
        if not old:
            return fail("edit_file: 'old' must not be empty")
        replace_all = flag(args, "replace_all")
        if not path.is_file():
            return fail(f"edit_file: not a file: {path}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            return fail(f"edit_file: {exc}")
        if size > EDIT_MAX_BYTES:
            return fail(
                f"edit_file: {path} is {size // 1024 // 1024} MB, too large to rewrite in memory; "
                "use the shell tool (sed, perl, or a script) instead"
            )
        try:
            # newline="" keeps CRLF files byte-identical apart from the edit;
            # universal newlines used to turn every line ending in the file into
            # "\n", which is a whole-file diff for a one-word change.
            with path.open("r", encoding="utf-8", newline="") as handle:
                current = handle.read()
        except UnicodeDecodeError:
            return fail(f"edit_file: {path} is not UTF-8 text; use the shell tool (sed, perl) instead")
        except OSError as exc:
            return fail(f"edit_file: {exc}")
        target = old
        if target not in current and "\r\n" in current:
            # match_file shows CRLF as "\n", so accept an anchor written that way
            target = old.replace("\r\n", "\n").replace("\n", "\r\n")
        count = current.count(target)
        if count == 0:
            return fail("edit_file: 'old' not found; read the file first and match it exactly")
        if count > 1 and not replace_all:
            return fail(f"edit_file: 'old' occurs {count} times; pass replace_all=true or use a unique anchor")
        _snapshot(cfg, audit, path)
        replacement = new
        if target is not old and "\r\n" in current:
            replacement = new.replace("\r\n", "\n").replace("\n", "\r\n")
        updated = current.replace(target, replacement) if replace_all else current.replace(target, replacement, 1)
        try:
            path.write_text(updated, encoding="utf-8", newline="")
        except OSError as exc:
            return fail(f"edit_file: {exc}")
        audit.log("edit_file", path=str(path), replacements=count)
        return ok(f"replaced {count if replace_all else 1} occurrence(s) in {path}")

    def list_dir(args: dict) -> ToolResult:
        path = _resolve(text(args, "path", "."))
        if not path.is_dir():
            return fail(f"list_dir: not a directory: {path}")
        show_all = flag(args, "all")
        try:
            listing = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except OSError as exc:
            # iterdir() itself raises on an unreadable directory; guarding only
            # entry.stat() was not enough.
            return fail(f"list_dir: {exc}")
        entries = []
        for entry in listing:
            if entry.name.startswith(".") and not show_all:
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            entries.append(f"{entry.name}/" if entry.is_dir() else f"{entry.name}  {size}b")
        audit.log("list_dir", path=str(path), count=len(entries))
        return ok(clip_text("\n".join(entries[:400]) or "(empty)", max_chars))

    return [
        Tool(
            name="read_file",
            description="Read a UTF-8 text file. Use offset/limit for large files.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "0-based first line"},
                    "limit": {"type": "integer", "description": "max lines to return"},
                },
                "required": ["path"],
            },
            fn=read_file,
        ),
        Tool(
            name="write_file",
            description="Write a file, creating parents. Overwrites; the old version is snapshotted.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            fn=write_file,
        ),
        Tool(
            name="edit_file",
            description=(
                "Replace an exact string in a file. 'old' must match exactly once unless replace_all is set."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string", "description": "exact text to find"},
                    "new": {"type": "string", "description": "replacement text"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old", "new"],
            },
            fn=edit_file,
        ),
        Tool(
            name="list_dir",
            description="List the entries of a directory with sizes.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}, "all": {"type": "boolean"}},
                "required": ["path"],
            },
            fn=list_dir,
        ),
    ]
