"""Routing strategy protocol and implementations.

Strategies determine which backend handles a given request for a model.
"""

from __future__ import annotations

from typing import Protocol

from gateway.latency_tracker import LatencyTracker
from gateway.routing import ConsistentHashRing


class RoutingStrategy(Protocol):
    """Protocol for backend selection strategies."""

    def select(
        self,
        candidates: list[str],
        exclude: frozenset[str] = frozenset(),
        routing_key: str | None = None,
    ) -> str | None:
        """Select a backend name from candidates.

        Args:
            candidates: Backend names serving this model.
            exclude: Backend names to skip (circuit-broken).
            routing_key: Optional key for deterministic routing.

        Returns:
            Selected backend name, or None if all excluded/empty.
        """
        ...


class ConsistentHashStrategy:
    """Wraps ConsistentHashRing to implement RoutingStrategy."""

    def __init__(self, ring: ConsistentHashRing) -> None:
        self._ring = ring

    def select(
        self,
        candidates: list[str],
        exclude: frozenset[str] = frozenset(),
        routing_key: str | None = None,
    ) -> str | None:
        if routing_key is not None:
            return self._ring.get_node(routing_key, exclude=exclude)
        # Fallback: first non-excluded candidate
        for name in candidates:
            if name not in exclude:
                return name
        return None


class LatencyAwareStrategy:
    """Route to the backend with the lowest P95 latency.

    Falls back to first non-excluded candidate when no latency data exists (cold start).
    """

    def __init__(self, tracker: LatencyTracker, model: str) -> None:
        self._tracker = tracker
        self._model = model

    def select(
        self,
        candidates: list[str],
        exclude: frozenset[str] = frozenset(),
        routing_key: str | None = None,
    ) -> str | None:
        eligible = [c for c in candidates if c not in exclude]
        if not eligible:
            return None
        p95_map = self._tracker.get_all_p95(self._model)
        # Sort by P95 ascending; backends without data go last (inf); tie-break by name
        eligible.sort(key=lambda b: (p95_map.get(b, float("inf")), b))
        return eligible[0]


class CostAwareStrategy:
    """Route to the cheapest healthy backend.

    Backends without cost data are treated as infinitely expensive.
    """

    def __init__(self, costs: dict[str, float]) -> None:
        self._costs = costs

    def select(
        self,
        candidates: list[str],
        exclude: frozenset[str] = frozenset(),
        routing_key: str | None = None,
    ) -> str | None:
        eligible = [c for c in candidates if c not in exclude]
        if not eligible:
            return None
        eligible.sort(key=lambda b: (self._costs.get(b, float("inf")), b))
        return eligible[0]
