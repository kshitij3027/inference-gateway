"""Rolling-window latency tracker for backend routing decisions.

Records per-(backend, model) latency observations and computes P95
over a configurable sliding window (default 60 seconds).
"""

from __future__ import annotations

import time
from collections import deque


class LatencyTracker:
    """Track request latencies per (backend, model) with a rolling window.

    Uses time.monotonic() for clock-immune timing and deque for O(1)
    append/popleft. Pruning happens lazily on access.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window = window_seconds
        self._observations: dict[tuple[str, str], deque[tuple[float, float]]] = {}

    def record(self, backend: str, model: str, latency_ms: float) -> None:
        """Record a latency observation."""
        key = (backend, model)
        if key not in self._observations:
            self._observations[key] = deque()
        self._observations[key].append((time.monotonic(), latency_ms))
        self._prune(key)

    def p95(self, backend: str, model: str) -> float | None:
        """Return P95 latency in ms, or None if no data."""
        key = (backend, model)
        self._prune(key)
        obs = self._observations.get(key)
        if not obs:
            return None
        latencies = sorted(v for _, v in obs)
        idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        return latencies[idx]

    def get_all_p95(self, model: str) -> dict[str, float]:
        """Return {backend_name: p95_ms} for all backends with data for this model."""
        result: dict[str, float] = {}
        for (b, m) in list(self._observations.keys()):
            if m == model:
                p95 = self.p95(b, m)
                if p95 is not None:
                    result[b] = p95
        return result

    def snapshot(self) -> dict:
        """Return admin-friendly snapshot of all tracked data."""
        result: dict[str, dict[str, dict]] = {}
        for (backend, model) in list(self._observations.keys()):
            self._prune((backend, model))
            obs = self._observations.get((backend, model))
            if not obs:
                continue
            if model not in result:
                result[model] = {}
            latencies = [v for _, v in obs]
            result[model][backend] = {
                "count": len(latencies),
                "p95_ms": self.p95(backend, model),
                "min_ms": min(latencies) if latencies else None,
                "max_ms": max(latencies) if latencies else None,
            }
        return result

    def _prune(self, key: tuple[str, str]) -> None:
        """Remove observations older than the window."""
        obs = self._observations.get(key)
        if not obs:
            return
        cutoff = time.monotonic() - self._window
        while obs and obs[0][0] <= cutoff:
            obs.popleft()
