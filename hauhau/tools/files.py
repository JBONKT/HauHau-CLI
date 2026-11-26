"""File tools. Every mutation snapshots the previous version first."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from ..util import clip_text
from .base import Tool, ToolResult, fail, ok

# A single very long line is truncated rather than loaded whole, and files too
# big to rewrite in memory are refused by edit_file.
MAX_LINE_CHARS = 4000
EDIT_MAX_BYTES = 32 * 1024 * 1024


def _resolve(path: str) -> Path:
    return Path(os.path.expanduser(path)).expanduser()


def _snapshot(cfg, audit, path: Path) -> None:
    if not path.is_file():
        return
    dest_dir = cfg.snapshots_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    dest = dest_dir / f"{stamp}_{path.name}"
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{stamp}-{counter}_{path.name}"
        counter += 1
    shutil.copy2(path, dest)
    audit.log("snapshot", path=str(path), copy=str(dest))


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
        path = _resolve(str(args.get("path") or ""))
        if not path.is_file():
            return fail(f"read_file: not a file: {path}")
        try:
            offset = max(0, int(args.get("offset") or 0))
            limit = max(0, int(args.get("limit") or 0))
        except (TypeError, ValueError):
            return fail("read_file: offset and limit must be integers")
        try:
            window, has_more = _read_window(path, offset, limit)
        except OSError as exc:
            return fail(f"read_file: {exc}")
        if not window:
            if offset:
                return fail(f"read_file: offset {offset} is past the end of the file")
            return ok("(empty file)")
        body = "\n".join(window)
        if has_more:
            body += "\n...[more lines follow; use offset/limit]..."
        audit.log("read_file", path=str(path), lines=len(window))
        return ok(clip_text(body, max_chars))

    def write_file(args: dict) -> ToolResult:
        path = _resolve(str(args.get("path") or ""))
        content = args.get("content")
        if content is None:
            return fail("write_file: 'content' is required")
        existed = path.is_file()
        _snapshot(cfg, audit, path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
        except OSError as exc:
            return fail(f"write_file: {exc}")
        audit.log("write_file", path=str(path), bytes=len(str(content)), existed=existed)
        return ok(f"wrote {len(str(content))} bytes to {path}" + ("" if existed else " (new file)"))

    def edit_file(args: dict) -> ToolResult:
        path = _resolve(str(args.get("path") or ""))
        old = args.get("old") if "old" in args else args.get("old_string")
        new = args.get("new") if "new" in args else args.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            return fail("edit_file: 'old' and 'new' strings are required")
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
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return fail(f"edit_file: {exc}")
        count = text.count(old)
        if count == 0:
            return fail("edit_file: 'old' not found; read the file first and match it exactly")
        if count > 1 and not args.get("replace_all"):
            return fail(f"edit_file: 'old' occurs {count} times; pass replace_all=true or use a unique anchor")
        _snapshot(cfg, audit, path)
        updated = text.replace(old, new) if args.get("replace_all") else text.replace(old, new, 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return fail(f"edit_file: {exc}")
        audit.log("edit_file", path=str(path), replacements=count)
        return ok(f"replaced {count if args.get('replace_all') else 1} occurrence(s) in {path}")

    def list_dir(args: dict) -> ToolResult:
        path = _resolve(str(args.get("path") or "."))
        if not path.is_dir():
            return fail(f"list_dir: not a directory: {path}")
        entries = []
        for entry in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name)):
            if entry.name.startswith(".") and not args.get("all"):
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
