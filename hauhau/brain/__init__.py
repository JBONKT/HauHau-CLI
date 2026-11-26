"""High-level brain: remember/recall plus the prompt block injection."""

from __future__ import annotations

import time
from pathlib import Path

from ..util import clip_text
from .embed import Embedder
from .store import Hit, MemoryStore


class Brain:
    def __init__(self, cfg, audit=None) -> None:
        self.cfg = cfg
        self.audit = audit
        b = cfg.brain
        self.store = MemoryStore(
            Path(cfg.db_path), b.half_life_days, b.dedupe_jaccard, b.dedupe_containment
        )
        self.embedder = Embedder(b.embeddings_url, b.embeddings_model) if b.embeddings_url else Embedder("")

    def remember(
        self,
        kind: str,
        text: str,
        tags: str = "",
        importance: float = 0.5,
    ) -> tuple[int, bool]:
        vector = self.embedder.embed(text) if self.embedder.enabled else None
        return self.store.write(kind, text, tags, importance, vector)

    def recall(self, query: str, k: int | None = None, kinds: list[str] | None = None) -> list[Hit]:
        k = k or self.cfg.brain.top_k
        vector = self.embedder.embed(query) if self.embedder.enabled else None
        return self.store.search(query, k, vector, kinds)

    def context_block(self, query: str) -> str:
        """Format relevant memories for the system prompt, or '' when empty."""
        if not self.cfg.brain.enabled:
            return ""
        hits = self.recall(query)
        if not hits:
            return ""
        lines = ["What you remember from earlier sessions (may be incomplete or stale):"]
        budget = self.cfg.brain.inject_max_chars
        for hit in hits:
            when = time.strftime("%Y-%m-%d", time.localtime(hit.created_at))
            lines.append(f"- [{hit.kind}] {hit.text} ({when}, used {hit.uses}x)")
        return clip_text("\n".join(lines), budget)

    def stats(self) -> dict:
        stats = self.store.stats()
        stats["embeddings_enabled"] = bool(self.embedder.enabled)
        if self.embedder.last_error:
            stats["embeddings_error"] = self.embedder.last_error
        return stats

    def close(self) -> None:
        self.store.close()
