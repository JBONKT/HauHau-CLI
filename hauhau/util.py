"""Small shared helpers: text clipping, token estimation, memory normalisation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable

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
    """Keep head and tail when a tool output exceeds the limit."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    removed = len(text) - head - tail
    return f"{text[:head]}\n...[{removed} chars truncated]...\n{text[-tail:]}"


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
