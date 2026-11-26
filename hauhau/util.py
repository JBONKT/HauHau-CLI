"""Small shared helpers: text clipping, token estimation, memory normalisation."""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9_]+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Deliberately cheap: ~4 chars per token."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: Iterable[dict]) -> int:
    total = 0
    for msg in messages:
        total += estimate_tokens(str(msg.get("content") or ""))
        for call in msg.get("tool_calls") or []:
            total += estimate_tokens(str(call))
        total += 4  # per-message overhead
    return total


def clip_text(text: str, limit: int) -> str:
    """Keep head and tail when a tool output exceeds the limit.

    The truncation marker counts against the limit: the old version trimmed to
    ``limit`` and then appended ~28 characters, so every clipped tool result
    was over budget.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    marker_room = 40  # "[NNNNN chars truncated]" plus the newlines around it
    if limit <= marker_room:
        return text[:limit]
    budget = limit - marker_room
    head = int(budget * 0.7)
    tail = budget - head
    removed = len(text) - head - tail
    marker = f"\n...[{removed} chars truncated]...\n"
    overflow = head + tail + len(marker) - limit
    if overflow > 0:
        tail = max(0, tail - overflow)
        removed = len(text) - head - tail
        marker = f"\n...[{removed} chars truncated]...\n"
    return text[:head] + marker + (text[-tail:] if tail else "")


def snapshot_file(path: Path, dest_dir: Path) -> Path | None:
    """Copy ``path`` into ``dest_dir`` with a timestamp; return the copy.

    Used before the agent rewrites its own config or prompt: a version file is
    the rollback, but the config needs a plain copy because it is a single file.
    """
    if not path.is_file():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    dest = dest_dir / f"{stamp}_{path.name}"
    counter = 1
    while dest.exists():
        dest = dest_dir / f"{stamp}-{counter}_{path.name}"
        counter += 1
    shutil.copy2(path, dest)
    return dest


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def norm_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def tokens_of(text: str) -> set[str]:
    return set(_WORD.findall(normalize(text)))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"
