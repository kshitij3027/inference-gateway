"""Chaos injection for backend HTTP calls.

When CHAOS_ENABLED=true, wraps the shared httpx.AsyncClient to randomly
inject latency, 5xx errors, or timeouts into backend calls.  This exercises
the retry loop, circuit breakers, and failover routing.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class ChaosConfig:
    """Configuration for chaos injection rates and parameters."""

    error_rate: float = 0.10  # Probability of 5xx error injection
    timeout_rate: float = 0.05  # Probability of timeout injection
    latency_rate: float = 0.30  # Probability of latency injection
    latency_min_ms: float = 50.0  # Minimum added latency
    latency_max_ms: float = 2000.0  # Maximum added latency
    seed: int | None = None  # Optional seed for reproducible tests


class ChaosHttpClient:
    """Wraps httpx.AsyncClient to inject chaos into backend calls.

    Intercepts post() and stream() methods.  All other attribute access
    is delegated to the real client via __getattr__.
    """

    def __init__(self, client: httpx.AsyncClient, config: ChaosConfig) -> None:
        self._client = client
        self._config = config
        self._rng = random.Random(config.seed)

    async def post(self, url, **kwargs) -> httpx.Response:
        """Wrap post() with chaos injection."""
        injection = self._roll_injection(str(url))

        if injection == "timeout":
            raise httpx.ReadTimeout("Chaos: simulated timeout")

        if injection == "error":
            mock_request = httpx.Request("POST", url)
            mock_response = httpx.Response(500, request=mock_request)
            raise httpx.HTTPStatusError(
                "Chaos: simulated 500",
                request=mock_request,
                response=mock_response,
            )

        if isinstance(injection, float):  # latency delay in ms
            await asyncio.sleep(injection / 1000.0)

        return await self._client.post(url, **kwargs)

    def stream(self, method: str, url, **kwargs):
        """Wrap stream() with chaos injection."""
        return _ChaosStreamContext(self, method, url, kwargs)

    def _roll_injection(self, url: str) -> str | float | None:
        """Determine what chaos to inject.

        Returns:
            "timeout" -- raise ReadTimeout
            "error"   -- raise HTTPStatusError(500)
            float     -- latency to add in ms (then proceed with real call)
            None      -- no injection
        """
        # Timeout check (highest severity, checked first)
        if self._rng.random() < self._config.timeout_rate:
            logger.warning("chaos_injection", injection_type="timeout", url=url)
            return "timeout"

        # Error check
        if self._rng.random() < self._config.error_rate:
            logger.warning("chaos_injection", injection_type="error_5xx", url=url)
            return "error"

        # Latency check (can co-exist with a successful call)
        if self._rng.random() < self._config.latency_rate:
            delay_ms = self._rng.uniform(
                self._config.latency_min_ms, self._config.latency_max_ms
            )
            logger.warning(
                "chaos_injection",
                injection_type="latency",
                url=url,
                delay_ms=round(delay_ms, 1),
            )
            return delay_ms

        return None

    def __getattr__(self, name):
        """Delegate all other attributes to the real client."""
        return getattr(self._client, name)


class _ChaosStreamContext:
    """Async context manager wrapping client.stream() with chaos."""

    def __init__(
        self,
        chaos_client: ChaosHttpClient,
        method: str,
        url,
        kwargs: dict,
    ) -> None:
        self._chaos_client = chaos_client
        self._method = method
        self._url = url
        self._kwargs = kwargs
        self._real_ctx = None

    async def __aenter__(self):
        injection = self._chaos_client._roll_injection(str(self._url))

        if injection == "timeout":
            raise httpx.ReadTimeout("Chaos: simulated timeout")

        if injection == "error":
            mock_request = httpx.Request(self._method, self._url)
            mock_response = httpx.Response(500, request=mock_request)
            raise httpx.HTTPStatusError(
                "Chaos: simulated 500",
                request=mock_request,
                response=mock_response,
            )

        if isinstance(injection, float):
            await asyncio.sleep(injection / 1000.0)

        self._real_ctx = self._chaos_client._client.stream(
            self._method, self._url, **self._kwargs
        )
        return await self._real_ctx.__aenter__()

    async def __aexit__(self, *args):
        if self._real_ctx is not None:
            return await self._real_ctx.__aexit__(*args)
