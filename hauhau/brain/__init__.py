"""High-level brain: remember/recall plus the prompt block injection."""

from __future__ import annotations

import time
from pathlib import Path

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

    def recall(
        self, query: str, k: int | None = None, kinds: list[str] | None = None, record: bool = True
    ) -> list[Hit]:
        k = k or self.cfg.brain.top_k
        vector = self.embedder.embed(query) if self.embedder.enabled else None
        return self.store.search(query, k, vector, kinds, record_use=record)

    def context_block(self, query: str) -> str:
        """Format relevant memories for the system prompt, or '' when empty.

        Memories are whole units of meaning, so the block is assembled memory by
        memory until the prompt budget is spent. Clipping the joined string cut
        sentences mid-word ("/srv/... che lives at") and presented the fragments
        as facts. Automatic injection does not count as a use: only deliberate
        recalls do, otherwise ``prune`` can never forget anything.
        """
        if not self.cfg.brain.enabled:
            return ""
        hits = self.recall(query, record=False)
        if not hits:
            return ""
        budget = max(0, int(self.cfg.brain.inject_max_chars))
        lines = ["What you remember from earlier sessions (may be incomplete or stale):"]
        used = len(lines[0]) + 1
        if budget <= used + 40:
            # A header with no room for a single memory is prompt noise.
            return ""
        withheld = 0
        for hit in hits:
            when = time.strftime("%Y-%m-%d", time.localtime(hit.created_at))
            suffix = f" ({when}{f', used {hit.uses}x' if hit.uses else ''})"
            line = f"- [{hit.kind}] {hit.text}{suffix}"
            # A memory is a whole fact: skip the ones that do not fit and say how
            # many were withheld. Clipping shipped half a sentence as a fact.
            if used + len(line) + 1 > budget:
                withheld += 1
                continue
            lines.append(line)
            used += len(line) + 1
        if len(lines) == 1:
            return ""
        if withheld:
            note_line = f"- ({withheld} more memories withheld: prompt budget spent)"
            if used + len(note_line) <= budget:
                lines.append(note_line)
        return "\n".join(lines)

    def stats(self) -> dict:
        stats = self.store.stats()
        stats["embeddings_enabled"] = bool(self.embedder.enabled)
        if self.embedder.last_error:
            stats["embeddings_error"] = self.embedder.last_error
        return stats

    def close(self) -> None:
        self.store.close()
