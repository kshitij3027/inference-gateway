from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.chaos import ChaosConfig, ChaosHttpClient


def _make_mock_client():
    """Create a mock httpx.AsyncClient with sensible defaults."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post = AsyncMock(
        return_value=httpx.Response(200, request=httpx.Request("POST", "http://test"))
    )
    client.headers = httpx.Headers({})
    return client


class TestChaosConfig:
    def test_defaults(self):
        config = ChaosConfig()
        assert config.error_rate == 0.10
        assert config.timeout_rate == 0.05
        assert config.latency_rate == 0.30
        assert config.latency_min_ms == 50.0
        assert config.latency_max_ms == 2000.0
        assert config.seed is None

    def test_custom_values(self):
        config = ChaosConfig(error_rate=0.5, timeout_rate=0.2, seed=42)
        assert config.error_rate == 0.5
        assert config.timeout_rate == 0.2
        assert config.seed == 42


class TestChaosHttpClientPost:
    async def test_passthrough_when_disabled(self):
        """All rates at 0 -- calls pass through unchanged."""
        mock = _make_mock_client()
        config = ChaosConfig(error_rate=0, timeout_rate=0, latency_rate=0)
        chaos = ChaosHttpClient(mock, config)

        result = await chaos.post("http://test/api", json={"key": "val"})
        assert result.status_code == 200
        mock.post.assert_called_once_with("http://test/api", json={"key": "val"})

    async def test_error_injection(self):
        """error_rate=1.0 -- always raises HTTPStatusError(500)."""
        mock = _make_mock_client()
        config = ChaosConfig(error_rate=1.0, timeout_rate=0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(mock, config)

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await chaos.post("http://test/api")
        assert exc_info.value.response.status_code == 500
        mock.post.assert_not_called()

    async def test_timeout_injection(self):
        """timeout_rate=1.0 -- always raises ReadTimeout."""
        mock = _make_mock_client()
        config = ChaosConfig(timeout_rate=1.0, error_rate=0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(mock, config)

        with pytest.raises(httpx.ReadTimeout):
            await chaos.post("http://test/api")
        mock.post.assert_not_called()

    async def test_latency_injection(self):
        """latency_rate=1.0 -- call succeeds but with added delay."""
        mock = _make_mock_client()
        config = ChaosConfig(
            latency_rate=1.0,
            error_rate=0,
            timeout_rate=0,
            latency_min_ms=100,
            latency_max_ms=100,
            seed=1,
        )
        chaos = ChaosHttpClient(mock, config)

        start = time.perf_counter()
        result = await chaos.post("http://test/api")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result.status_code == 200
        assert elapsed_ms >= 90  # Allow small tolerance
        mock.post.assert_called_once()

    async def test_timeout_priority_over_error(self):
        """When both timeout_rate and error_rate are 1.0, timeout wins (checked first)."""
        mock = _make_mock_client()
        config = ChaosConfig(timeout_rate=1.0, error_rate=1.0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(mock, config)

        with pytest.raises(httpx.ReadTimeout):
            await chaos.post("http://test/api")


class TestChaosHttpClientStream:
    async def test_stream_passthrough(self):
        """stream() delegates to real client when rates are 0."""
        mock = _make_mock_client()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock.stream = MagicMock(return_value=mock_ctx)

        config = ChaosConfig(error_rate=0, timeout_rate=0, latency_rate=0)
        chaos = ChaosHttpClient(mock, config)

        ctx = chaos.stream("POST", "http://test/api", json={"k": "v"})
        async with ctx as response:
            assert response is not None
        mock.stream.assert_called_once_with("POST", "http://test/api", json={"k": "v"})

    async def test_stream_error_injection(self):
        """stream() raises HTTPStatusError on __aenter__ when error_rate=1.0."""
        mock = _make_mock_client()
        config = ChaosConfig(error_rate=1.0, timeout_rate=0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(mock, config)

        ctx = chaos.stream("POST", "http://test/api")
        with pytest.raises(httpx.HTTPStatusError):
            async with ctx:
                pass

    async def test_stream_timeout_injection(self):
        """stream() raises ReadTimeout on __aenter__ when timeout_rate=1.0."""
        mock = _make_mock_client()
        config = ChaosConfig(timeout_rate=1.0, error_rate=0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(mock, config)

        ctx = chaos.stream("POST", "http://test/api")
        with pytest.raises(httpx.ReadTimeout):
            async with ctx:
                pass

    async def test_stream_aexit_skipped_on_injection(self):
        """When injection fires, __aexit__ handles None _real_ctx gracefully."""
        mock = _make_mock_client()
        config = ChaosConfig(timeout_rate=1.0, error_rate=0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(mock, config)

        ctx = chaos.stream("POST", "http://test/api")
        with pytest.raises(httpx.ReadTimeout):
            async with ctx:
                pass
        # No error during __aexit__ -- _real_ctx was never set


class TestChaosHttpClientDelegation:
    async def test_getattr_delegation(self):
        """Attribute access delegates to real client."""
        mock = _make_mock_client()
        config = ChaosConfig(error_rate=0, timeout_rate=0, latency_rate=0)
        chaos = ChaosHttpClient(mock, config)

        result = chaos.headers
        assert result is not None


class TestChaosHttpClientDeterminism:
    async def test_seed_determinism(self):
        """Same seed produces the same injection sequence."""
        results1 = []
        results2 = []

        for seed_results in [results1, results2]:
            mock = _make_mock_client()
            config = ChaosConfig(
                error_rate=0.5,
                timeout_rate=0.2,
                latency_rate=0.3,
                latency_min_ms=10,
                latency_max_ms=10,
                seed=42,
            )
            chaos = ChaosHttpClient(mock, config)

            for _ in range(20):
                injection = chaos._roll_injection("http://test")
                seed_results.append(injection)

        assert results1 == results2

    async def test_different_seeds_differ(self):
        """Different seeds produce different injection sequences."""
        results = []
        for seed in [1, 2]:
            mock = _make_mock_client()
            config = ChaosConfig(
                error_rate=0.3,
                timeout_rate=0.1,
                latency_rate=0.3,
                seed=seed,
            )
            chaos = ChaosHttpClient(mock, config)
            run = [chaos._roll_injection("http://test") for _ in range(20)]
            results.append(run)

        assert results[0] != results[1]


class TestRollInjection:
    def test_no_injection_when_all_rates_zero(self):
        config = ChaosConfig(error_rate=0, timeout_rate=0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(AsyncMock(), config)

        for _ in range(100):
            assert chaos._roll_injection("http://test") is None

    def test_always_timeout_when_rate_one(self):
        config = ChaosConfig(timeout_rate=1.0, error_rate=0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(AsyncMock(), config)

        for _ in range(10):
            assert chaos._roll_injection("http://test") == "timeout"

    def test_always_error_when_rate_one(self):
        config = ChaosConfig(timeout_rate=0, error_rate=1.0, latency_rate=0, seed=1)
        chaos = ChaosHttpClient(AsyncMock(), config)

        for _ in range(10):
            assert chaos._roll_injection("http://test") == "error"

    def test_always_latency_when_rate_one(self):
        config = ChaosConfig(timeout_rate=0, error_rate=0, latency_rate=1.0, seed=1)
        chaos = ChaosHttpClient(AsyncMock(), config)

        for _ in range(10):
            result = chaos._roll_injection("http://test")
            assert isinstance(result, float)

    def test_latency_within_bounds(self):
        config = ChaosConfig(
            timeout_rate=0,
            error_rate=0,
            latency_rate=1.0,
            latency_min_ms=100.0,
            latency_max_ms=500.0,
            seed=42,
        )
        chaos = ChaosHttpClient(AsyncMock(), config)

        for _ in range(50):
            delay = chaos._roll_injection("http://test")
            assert isinstance(delay, float)
            assert 100.0 <= delay <= 500.0
