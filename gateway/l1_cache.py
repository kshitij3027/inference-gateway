"""In-process LRU cache for semantic cache entries (L1 tier).

Stores response embeddings and JSON in an OrderedDict with LRU eviction
and TTL-based invalidation. Designed for asyncio single-threaded use.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@dataclass
class L1Entry:
    """A single L1 cache entry."""

    scope_key: str
    embedding: list[float]
    response_json: str
    created_at: float
    hit_count: int = 0


class L1Cache:
    """In-process LRU cache with TTL for semantic cache entries.

    Entries are scoped by (model, system_hash) and looked up via cosine
    similarity against stored embeddings — matching the L2 (Redis) tier behavior.

    Uses OrderedDict for O(1) LRU operations. No locks needed under asyncio.
    """

    def __init__(self, max_entries: int = 500, ttl_seconds: float = 3600.0) -> None:
        self._entries: OrderedDict[str, L1Entry] = OrderedDict()
        self._scope_index: dict[str, set[str]] = {}
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def lookup(
        self,
        scope_key: str,
        query_embedding: list[float],
        threshold: float,
    ) -> tuple[str | None, float | None]:
        """Find best matching entry in scope above similarity threshold.

        Returns (response_json, similarity) or (None, None).
        Touches matched entry for LRU freshness.
        """
        entry_ids = self._scope_index.get(scope_key, set())
        if not entry_ids:
            self._misses += 1
            return None, None

        best_response: str | None = None
        best_similarity: float = -1.0
        best_entry_id: str | None = None
        expired: list[str] = []

        for eid in list(entry_ids):
            entry = self._entries.get(eid)
            if entry is None:
                expired.append(eid)
                continue
            if self._is_expired(entry):
                expired.append(eid)
                continue

            sim = cosine_similarity(query_embedding, entry.embedding)
            if sim >= threshold and sim > best_similarity:
                best_similarity = sim
                best_response = entry.response_json
                best_entry_id = eid

        # Clean up expired entries
        for eid in expired:
            self._evict(eid)

        if best_response is not None and best_entry_id is not None:
            self._entries[best_entry_id].hit_count += 1
            self._entries.move_to_end(best_entry_id)
            self._hits += 1
            return best_response, round(best_similarity, 4)

        self._misses += 1
        return None, None

    def store(
        self,
        entry_id: str,
        scope_key: str,
        embedding: list[float],
        response_json: str,
    ) -> None:
        """Store an entry, evicting oldest if at capacity."""
        if entry_id in self._entries:
            self._entries.move_to_end(entry_id)
            return

        while len(self._entries) >= self._max_entries:
            self._evict_oldest()

        self._entries[entry_id] = L1Entry(
            scope_key=scope_key,
            embedding=embedding,
            response_json=response_json,
            created_at=time.monotonic(),
        )
        self._scope_index.setdefault(scope_key, set()).add(entry_id)

    def flush(self) -> int:
        """Remove all entries. Returns count deleted."""
        count = len(self._entries)
        self._entries.clear()
        self._scope_index.clear()
        self._hits = 0
        self._misses = 0
        return count

    def stats(self) -> dict:
        """Return L1 cache statistics."""
        return {
            "l1_entries": len(self._entries),
            "l1_max_entries": self._max_entries,
            "l1_hits": self._hits,
            "l1_misses": self._misses,
        }

    def _is_expired(self, entry: L1Entry) -> bool:
        return time.monotonic() - entry.created_at > self._ttl

    def _evict(self, entry_id: str) -> None:
        """Remove a specific entry by ID."""
        entry = self._entries.pop(entry_id, None)
        if entry is not None:
            scope_entries = self._scope_index.get(entry.scope_key)
            if scope_entries is not None:
                scope_entries.discard(entry_id)
                if not scope_entries:
                    del self._scope_index[entry.scope_key]

    def _evict_oldest(self) -> None:
        """Remove the least-recently-used entry."""
        if self._entries:
            entry_id, entry = self._entries.popitem(last=False)
            scope_entries = self._scope_index.get(entry.scope_key)
            if scope_entries is not None:
                scope_entries.discard(entry_id)
                if not scope_entries:
                    del self._scope_index[entry.scope_key]
