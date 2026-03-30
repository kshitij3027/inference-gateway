from gateway.latency_tracker import LatencyTracker
from gateway.routing import ConsistentHashRing
from gateway.strategies import ConsistentHashStrategy, CostAwareStrategy, LatencyAwareStrategy


class TestConsistentHashStrategy:
    def _make_strategy(self) -> tuple[ConsistentHashStrategy, ConsistentHashRing]:
        ring = ConsistentHashRing([("a", 1), ("b", 1), ("c", 1)])
        return ConsistentHashStrategy(ring), ring

    def test_deterministic(self):
        """Same routing_key always returns the same backend."""
        strategy, _ = self._make_strategy()
        results = [
            strategy.select(["a", "b", "c"], routing_key="stable-key")
            for _ in range(100)
        ]
        assert len(set(results)) == 1

    def test_different_keys_distribute(self):
        """Different routing keys hit multiple backends."""
        strategy, _ = self._make_strategy()
        nodes_hit = {
            strategy.select(["a", "b", "c"], routing_key=f"key-{i}")
            for i in range(100)
        }
        assert len(nodes_hit) >= 2

    def test_exclude_skips_backend(self):
        """Excluded backend is never returned."""
        strategy, _ = self._make_strategy()
        for i in range(50):
            result = strategy.select(
                ["a", "b", "c"],
                exclude=frozenset({"a"}),
                routing_key=f"key-{i}",
            )
            assert result != "a"

    def test_exclude_all_returns_none(self):
        """All candidates excluded returns None."""
        strategy, _ = self._make_strategy()
        result = strategy.select(
            ["a", "b", "c"],
            exclude=frozenset({"a", "b", "c"}),
            routing_key="any-key",
        )
        assert result is None

    def test_no_routing_key_returns_first_candidate(self):
        """Without routing_key, returns first non-excluded candidate."""
        strategy, _ = self._make_strategy()
        result = strategy.select(["b", "a", "c"])
        assert result == "b"

    def test_no_routing_key_with_exclude(self):
        """Without routing_key, skips excluded and returns next candidate."""
        strategy, _ = self._make_strategy()
        result = strategy.select(["a", "b", "c"], exclude=frozenset({"a"}))
        assert result == "b"

    def test_empty_candidates(self):
        """Empty candidates list returns None."""
        strategy, _ = self._make_strategy()
        result = strategy.select([])
        assert result is None

    def test_matches_ring_behavior(self):
        """Strategy select produces identical results to ring get_node."""
        strategy, ring = self._make_strategy()
        for i in range(200):
            key = f"verify-{i}"
            ring_result = ring.get_node(key)
            strategy_result = strategy.select(
                ["a", "b", "c"], routing_key=key
            )
            assert strategy_result == ring_result, (
                f"key={key}: ring={ring_result}, strategy={strategy_result}"
            )

        # Also verify with exclusions
        for i in range(50):
            key = f"exclude-verify-{i}"
            exclude = frozenset({"a"})
            ring_result = ring.get_node(key, exclude=exclude)
            strategy_result = strategy.select(
                ["a", "b", "c"], exclude=exclude, routing_key=key
            )
            assert strategy_result == ring_result, (
                f"key={key}: ring={ring_result}, strategy={strategy_result}"
            )


class TestLatencyAwareStrategy:
    def _make_tracker(self, latencies: dict[str, float], model: str = "gpt-4") -> LatencyTracker:
        """Create a tracker with recorded observations.

        Records 20 identical observations per backend so P95 is deterministic.
        """
        tracker = LatencyTracker(window_seconds=300.0)
        for backend, latency_ms in latencies.items():
            for _ in range(20):
                tracker.record(backend, model, latency_ms)
        return tracker

    def test_routes_to_lowest_p95(self):
        """Selects backend with the lowest P95 latency."""
        tracker = self._make_tracker({"backend-a": 100.0, "backend-b": 50.0, "backend-c": 200.0})
        strategy = LatencyAwareStrategy(tracker, model="gpt-4")
        result = strategy.select(["backend-a", "backend-b", "backend-c"])
        assert result == "backend-b"

    def test_cold_start_returns_first_candidate(self):
        """Empty tracker returns first candidate (all tied at inf, alphabetical)."""
        tracker = LatencyTracker(window_seconds=300.0)
        strategy = LatencyAwareStrategy(tracker, model="gpt-4")
        result = strategy.select(["a", "b", "c"])
        assert result == "a"

    def test_excludes_lowest_p95_backend(self):
        """Excluded lowest-P95 backend is skipped; returns next lowest."""
        tracker = self._make_tracker({"backend-a": 100.0, "backend-b": 50.0, "backend-c": 200.0})
        strategy = LatencyAwareStrategy(tracker, model="gpt-4")
        result = strategy.select(
            ["backend-a", "backend-b", "backend-c"],
            exclude=frozenset({"backend-b"}),
        )
        assert result == "backend-a"

    def test_all_excluded_returns_none(self):
        """All candidates excluded returns None."""
        tracker = self._make_tracker({"a": 100.0, "b": 50.0})
        strategy = LatencyAwareStrategy(tracker, model="gpt-4")
        result = strategy.select(["a", "b"], exclude=frozenset({"a", "b"}))
        assert result is None

    def test_tie_breaking_by_name(self):
        """Backends with equal P95 are tie-broken alphabetically."""
        tracker = self._make_tracker({"backend-a": 100.0, "backend-b": 100.0})
        strategy = LatencyAwareStrategy(tracker, model="gpt-4")
        result = strategy.select(["backend-b", "backend-a"])
        assert result == "backend-a"


class TestCostAwareStrategy:
    def test_routes_to_cheapest(self):
        """Selects the backend with the lowest cost."""
        strategy = CostAwareStrategy(costs={"a": 0.03, "b": 0.01, "c": 0.05})
        result = strategy.select(["a", "b", "c"])
        assert result == "b"

    def test_excludes_cheapest(self):
        """Cheapest backend excluded; returns next cheapest."""
        strategy = CostAwareStrategy(costs={"a": 0.03, "b": 0.01, "c": 0.05})
        result = strategy.select(["a", "b", "c"], exclude=frozenset({"b"}))
        assert result == "a"

    def test_missing_cost_treated_as_infinity(self):
        """Backend without a cost entry is sorted last."""
        strategy = CostAwareStrategy(costs={"a": 0.03, "b": 0.01})
        result = strategy.select(["a", "b", "c"])
        assert result == "b"
        # Verify "c" (no cost) would only be picked if others excluded
        result_only_c = strategy.select(["a", "b", "c"], exclude=frozenset({"a", "b"}))
        assert result_only_c == "c"

    def test_all_excluded_returns_none(self):
        """All candidates excluded returns None."""
        strategy = CostAwareStrategy(costs={"a": 0.03, "b": 0.01})
        result = strategy.select(["a", "b"], exclude=frozenset({"a", "b"}))
        assert result is None

    def test_tie_breaking_by_name(self):
        """Backends with equal cost are tie-broken alphabetically."""
        strategy = CostAwareStrategy(costs={"a": 0.01, "b": 0.01})
        result = strategy.select(["b", "a"])
        assert result == "a"
