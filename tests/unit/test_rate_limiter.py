from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.rate_limiter import RateLimiter


def _make_mock_redis(script_return=None):
    """Create a mock Redis client for testing."""
    mock_redis = AsyncMock()

    # Mock register_script to return an async callable
    mock_script = AsyncMock(return_value=script_return or [1, 0])
    mock_redis.register_script = MagicMock(return_value=mock_script)

    return mock_redis, mock_script


class TestCheckRateLimit:
    async def test_allows_under_rps_limit(self):
        redis, script = _make_mock_redis([1, 3])  # allowed, count=3
        rl = RateLimiter(redis)
        allowed, info = await rl.check_rate_limit("t1", "req-1", rps_limit=10, rpm_limit=None)
        assert allowed is True
        assert info is None

    async def test_denies_at_rps_limit(self):
        redis, script = _make_mock_redis([0, 10])  # denied, count=10
        rl = RateLimiter(redis)
        allowed, info = await rl.check_rate_limit("t1", "req-1", rps_limit=10, rpm_limit=None)
        assert allowed is False
        assert info["limit_type"] == "rps"
        assert info["limit"] == 10
        assert info["current"] == 10
        assert info["retry_after"] == 1.0

    async def test_rps_passes_rpm_denies(self):
        redis, script = _make_mock_redis()
        rl = RateLimiter(redis)

        # First call (RPS) allows, second call (RPM) denies
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [1, 5]  # RPS allowed
            return [0, 60]  # RPM denied

        script.side_effect = side_effect
        allowed, info = await rl.check_rate_limit("t1", "req-1", rps_limit=10, rpm_limit=60)
        assert allowed is False
        assert info["limit_type"] == "rpm"
        assert info["retry_after"] == 60.0

    async def test_rps_checked_before_rpm(self):
        redis, script = _make_mock_redis([0, 10])  # RPS denied
        rl = RateLimiter(redis)
        allowed, info = await rl.check_rate_limit("t1", "req-1", rps_limit=10, rpm_limit=60)
        assert info["limit_type"] == "rps"  # RPS denied first
        assert script.call_count == 1  # RPM not even checked

    async def test_no_limits_always_allows(self):
        redis, script = _make_mock_redis()
        rl = RateLimiter(redis)
        allowed, info = await rl.check_rate_limit(
            "t1", "req-1", rps_limit=None, rpm_limit=None
        )
        assert allowed is True
        assert info is None
        assert script.call_count == 0  # No Redis calls

    async def test_both_limits_pass(self):
        redis, script = _make_mock_redis([1, 5])  # Both allow
        rl = RateLimiter(redis)
        allowed, info = await rl.check_rate_limit("t1", "req-1", rps_limit=10, rpm_limit=60)
        assert allowed is True
        assert script.call_count == 2  # Both checked


class TestTokenBudget:
    async def test_under_budget_allows(self):
        redis, _ = _make_mock_redis()
        redis.get = AsyncMock(return_value="5000")
        rl = RateLimiter(redis)
        allowed, info = await rl.check_token_budget("t1", budget=100000)
        assert allowed is True

    async def test_exceeded_budget_denies(self):
        redis, _ = _make_mock_redis()
        redis.get = AsyncMock(return_value="100001")
        rl = RateLimiter(redis)
        allowed, info = await rl.check_token_budget("t1", budget=100000)
        assert allowed is False
        assert info["limit_type"] == "token_budget_daily"
        assert info["current"] == 100001

    async def test_none_budget_always_allows(self):
        redis, _ = _make_mock_redis()
        rl = RateLimiter(redis)
        allowed, info = await rl.check_token_budget("t1", budget=None)
        assert allowed is True
        redis.get.assert_not_called()

    async def test_no_existing_tokens_allows(self):
        redis, _ = _make_mock_redis()
        redis.get = AsyncMock(return_value=None)  # No key exists
        rl = RateLimiter(redis)
        allowed, info = await rl.check_token_budget("t1", budget=100000)
        assert allowed is True


class TestRecordTokens:
    async def test_increments_counter(self):
        redis, _ = _make_mock_redis()
        redis.incrby = AsyncMock(return_value=8500)
        redis.expire = AsyncMock()
        rl = RateLimiter(redis)
        result = await rl.record_tokens("t1", 500)
        assert result == 8500
        redis.incrby.assert_called_once()
        redis.expire.assert_called_once()


class TestGetRemaining:
    async def test_rps_remaining(self):
        redis, _ = _make_mock_redis()
        redis.zremrangebyscore = AsyncMock()
        redis.zcard = AsyncMock(return_value=3)
        rl = RateLimiter(redis)
        remaining = await rl.get_remaining("t1", rps_limit=10, rpm_limit=None)
        assert remaining["rps"] == 7

    async def test_rpm_remaining(self):
        redis, _ = _make_mock_redis()
        redis.zremrangebyscore = AsyncMock()
        redis.zcard = AsyncMock(return_value=50)
        rl = RateLimiter(redis)
        remaining = await rl.get_remaining("t1", rps_limit=None, rpm_limit=60)
        assert remaining["rpm"] == 10

    async def test_no_limits_empty_dict(self):
        redis, _ = _make_mock_redis()
        rl = RateLimiter(redis)
        remaining = await rl.get_remaining("t1", rps_limit=None, rpm_limit=None)
        assert remaining == {}

    async def test_at_limit_returns_zero(self):
        redis, _ = _make_mock_redis()
        redis.zremrangebyscore = AsyncMock()
        redis.zcard = AsyncMock(return_value=10)
        rl = RateLimiter(redis)
        remaining = await rl.get_remaining("t1", rps_limit=10, rpm_limit=None)
        assert remaining["rps"] == 0
