"""Unit tests for the CostTracker."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from gateway.cost_tracker import CostTracker


class MockBackend:
    """Mock BackendConfig for testing."""

    def __init__(
        self,
        cost_per_1k_tokens=None,
        input_cost_per_1k_tokens=None,
        output_cost_per_1k_tokens=None,
    ):
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.input_cost_per_1k_tokens = input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = output_cost_per_1k_tokens


class TestCalculateCost:
    def test_with_input_output_prices(self):
        backend = MockBackend(input_cost_per_1k_tokens=0.01, output_cost_per_1k_tokens=0.03)
        cost = CostTracker.calculate_cost(backend, prompt_tokens=1000, completion_tokens=500)
        # (1000 * 0.01 + 500 * 0.03) / 1000 = 0.01 + 0.015 = 0.025
        assert cost == pytest.approx(0.025)

    def test_fallback_to_single_price(self):
        backend = MockBackend(cost_per_1k_tokens=0.05)
        cost = CostTracker.calculate_cost(backend, prompt_tokens=2000, completion_tokens=1000)
        # (2000 * 0.05 + 1000 * 0.05) / 1000 = 0.1 + 0.05 = 0.15
        assert cost == pytest.approx(0.15)

    def test_input_override_with_fallback_output(self):
        backend = MockBackend(cost_per_1k_tokens=0.05, input_cost_per_1k_tokens=0.01)
        cost = CostTracker.calculate_cost(backend, prompt_tokens=1000, completion_tokens=1000)
        # input uses override: (1000 * 0.01) / 1000 = 0.01
        # output uses fallback: (1000 * 0.05) / 1000 = 0.05
        assert cost == pytest.approx(0.06)

    def test_no_prices_returns_zero(self):
        backend = MockBackend()
        cost = CostTracker.calculate_cost(backend, prompt_tokens=1000, completion_tokens=500)
        assert cost == 0.0

    def test_zero_tokens(self):
        backend = MockBackend(cost_per_1k_tokens=0.05)
        cost = CostTracker.calculate_cost(backend, prompt_tokens=0, completion_tokens=0)
        assert cost == 0.0


class TestRecordCost:
    @pytest.mark.asyncio
    async def test_increments_redis(self):
        redis = AsyncMock()
        redis.incrbyfloat = AsyncMock(return_value=0.025)
        redis.expire = AsyncMock()
        tracker = CostTracker(redis)

        with patch("gateway.cost_tracker.ESTIMATED_COST", create=True):
            total = await tracker.record_cost("t1", "gpt-4", 0.025)

        assert total == 0.025
        redis.incrbyfloat.assert_called_once()
        call_args = redis.incrbyfloat.call_args
        assert call_args[0][0].startswith("cost:t1:")
        assert call_args[0][1] == 0.025

    @pytest.mark.asyncio
    async def test_sets_ttl(self):
        redis = AsyncMock()
        redis.incrbyfloat = AsyncMock(return_value=0.05)
        redis.expire = AsyncMock()
        tracker = CostTracker(redis)

        with patch("gateway.cost_tracker.ESTIMATED_COST", create=True):
            await tracker.record_cost("t1", "gpt-4", 0.025)

        redis.expire.assert_called_once()
        assert redis.expire.call_args[0][1] == 90000

    @pytest.mark.asyncio
    async def test_zero_cost_skipped(self):
        redis = AsyncMock()
        tracker = CostTracker(redis)
        total = await tracker.record_cost("t1", "gpt-4", 0.0)
        assert total == 0.0
        redis.incrbyfloat.assert_not_called()

    @pytest.mark.asyncio
    async def test_redis_error_returns_zero(self):
        redis = AsyncMock()
        redis.incrbyfloat = AsyncMock(side_effect=Exception("connection lost"))
        tracker = CostTracker(redis)
        total = await tracker.record_cost("t1", "gpt-4", 0.025)
        assert total == 0.0


class TestGetDailyCost:
    @pytest.mark.asyncio
    async def test_returns_float(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value="0.15")
        tracker = CostTracker(redis)
        cost = await tracker.get_daily_cost("t1", "2025-01-15")
        assert cost == 0.15

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_data(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value=None)
        tracker = CostTracker(redis)
        cost = await tracker.get_daily_cost("t1", "2025-01-15")
        assert cost == 0.0


class TestGetCostSummary:
    @pytest.mark.asyncio
    async def test_queries_multiple_days(self):
        redis = AsyncMock()
        redis.get = AsyncMock(return_value="0.10")
        tracker = CostTracker(redis)
        summary = await tracker.get_cost_summary("t1", days=3)
        assert summary["tenant_id"] == "t1"
        assert len(summary["costs_by_date"]) == 3
        assert summary["today"] == 0.10
        assert redis.get.call_count == 3
