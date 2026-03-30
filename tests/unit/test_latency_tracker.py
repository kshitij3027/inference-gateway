from unittest.mock import patch

from gateway.latency_tracker import LatencyTracker


class TestLatencyTracker:
    def test_record_and_p95_single(self):
        tracker = LatencyTracker()
        tracker.record("openai", "gpt-4", 150.0)
        assert tracker.p95("openai", "gpt-4") == 150.0

    def test_p95_correct_percentile(self):
        tracker = LatencyTracker()
        for i in range(1, 21):
            tracker.record("openai", "gpt-4", float(i))
        # 20 values: [1..20], index = min(int(20*0.95), 19) = min(19, 19) = 19
        # sorted[19] = 20
        assert tracker.p95("openai", "gpt-4") == 20.0

    def test_p95_unknown_returns_none(self):
        tracker = LatencyTracker()
        assert tracker.p95("openai", "gpt-4") is None

    @patch("gateway.latency_tracker.time")
    def test_window_expiry(self, mock_time):
        mock_time.monotonic.return_value = 100.0
        tracker = LatencyTracker(window_seconds=60.0)

        tracker.record("openai", "gpt-4", 150.0)
        tracker.record("openai", "gpt-4", 200.0)
        assert tracker.p95("openai", "gpt-4") is not None

        # Advance past window
        mock_time.monotonic.return_value = 161.0
        assert tracker.p95("openai", "gpt-4") is None

    def test_get_all_p95(self):
        tracker = LatencyTracker()
        tracker.record("openai", "gpt-4", 100.0)
        tracker.record("anthropic", "gpt-4", 200.0)
        tracker.record("ollama", "gpt-4", 300.0)

        result = tracker.get_all_p95("gpt-4")
        assert len(result) == 3
        assert result["openai"] == 100.0
        assert result["anthropic"] == 200.0
        assert result["ollama"] == 300.0

    def test_get_all_p95_filters_by_model(self):
        tracker = LatencyTracker()
        tracker.record("openai", "gpt-4", 100.0)
        tracker.record("openai", "gpt-3.5", 50.0)
        tracker.record("anthropic", "gpt-4", 200.0)

        result = tracker.get_all_p95("gpt-4")
        assert len(result) == 2
        assert "openai" in result
        assert "anthropic" in result

        result_35 = tracker.get_all_p95("gpt-3.5")
        assert len(result_35) == 1
        assert "openai" in result_35

    def test_empty_tracker(self):
        tracker = LatencyTracker()
        assert tracker.get_all_p95("gpt-4") == {}

    def test_snapshot(self):
        tracker = LatencyTracker()
        tracker.record("openai", "gpt-4", 100.0)
        tracker.record("openai", "gpt-4", 200.0)
        tracker.record("anthropic", "gpt-4", 150.0)

        snap = tracker.snapshot()
        assert "gpt-4" in snap
        assert "openai" in snap["gpt-4"]
        assert "anthropic" in snap["gpt-4"]

        openai_snap = snap["gpt-4"]["openai"]
        assert openai_snap["count"] == 2
        assert openai_snap["p95_ms"] is not None
        assert openai_snap["min_ms"] == 100.0
        assert openai_snap["max_ms"] == 200.0
