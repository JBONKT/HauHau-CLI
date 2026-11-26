"""Repository-wide search: regex grep and glob find, implemented without deps."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from ..util import clip_text
from .args import flag, integer, text
from .base import Tool, ToolResult, fail, ok

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", "dist", "build", ".tox"}
MAX_FILE_BYTES = 2 * 1024 * 1024


def build(cfg, audit) -> list[Tool]:
    max_chars = cfg.tools.max_output_chars

    def grep(args: dict) -> ToolResult:
        pattern = text(args, "pattern", required=True)
        if not pattern:
            return fail("grep: 'pattern' must not be empty")
        root = Path(os.path.expanduser(text(args, "path", str(cfg.workspace))))
        if not root.is_dir():
            return fail(f"grep: not a directory: {root}")
        flags = re.IGNORECASE if flag(args, "ignore_case") else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return fail(f"grep: bad regex: {exc}")
        glob = text(args, "glob", "*")
        max_results = integer(args, "max_results", 60, minimum=1, maximum=500)
        hits: list[str] = []
        scanned = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                if not fnmatch.fnmatch(name, glob):
                    continue
                fpath = Path(dirpath) / name
                try:
                    if fpath.stat().st_size > MAX_FILE_BYTES:
                        continue
                    with fpath.open("r", encoding="utf-8", errors="strict") as fh:
                        scanned += 1
                        for lineno, line in enumerate(fh, 1):
                            if regex.search(line):
                                rel = fpath.relative_to(root)
                                hits.append(f"{rel}:{lineno}: {line.rstrip()[:300]}")
                                if len(hits) >= max_results:
                                    break
                except (OSError, UnicodeDecodeError):
                    continue
                if len(hits) >= max_results:
                    break
            if len(hits) >= max_results:
                break
        audit.log("grep", pattern=str(pattern), root=str(root), hits=len(hits), scanned=scanned)
        if not hits:
            return ok(f"no matches for {pattern!r} under {root} ({scanned} files scanned)")
        return ok(clip_text("\n".join(hits), max_chars), hits=len(hits))

    def find(args: dict) -> ToolResult:
        pattern = text(args, "pattern", "*") or "*"
        root = Path(os.path.expanduser(text(args, "path", str(cfg.workspace))))
        if not root.is_dir():
            return fail(f"find: not a directory: {root}")
        max_results = integer(args, "max_results", 200, minimum=1, maximum=2000)
        found: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for name in filenames + dirnames:
                if fnmatch.fnmatch(name, pattern):
                    found.append(str((Path(dirpath) / name).relative_to(root)))
                    if len(found) >= max_results:
                        break
            if len(found) >= max_results:
                break
        audit.log("find", pattern=pattern, root=str(root), hits=len(found))
        if not found:
            return ok(f"no files matching {pattern!r} under {root}")
        return ok(clip_text("\n".join(found), max_chars), hits=len(found))

    return [
        Tool(
            name="grep",
            description="Regex search across a directory tree, returning file:line matches.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regex"},
                    "path": {"type": "string", "description": "root directory (default: workspace)"},
                    "glob": {"type": "string", "description": "filename filter, e.g. '*.py'"},
                    "ignore_case": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
            fn=grep,
        ),
        Tool(
            name="find",
            description="Find files or directories by glob pattern under a root.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "glob, e.g. '*.toml'"},
                    "path": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
            fn=find,
        ),
    ]
