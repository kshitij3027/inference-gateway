from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from gateway.routes.chat import _execute_hedge


def _mock_backend(name):
    """Create a mock BackendConfig-like object."""
    backend = MagicMock()
    backend.name = name
    backend.provider = "openai"
    return backend


def _mock_response(content="Hello!"):
    """Create a mock ChatCompletionResponse-like object."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 5
    resp.usage.completion_tokens = 3
    resp.usage.total_tokens = 8
    return resp


class TestExecuteHedge:
    async def test_returns_first_response(self):
        """Both backends succeed -- returns the faster one."""
        b1 = _mock_backend("fast")
        b2 = _mock_backend("slow")
        resp1 = _mock_response("fast response")

        async def translator(client, backend, request):
            if backend.name == "fast":
                return resp1
            await asyncio.sleep(10)
            return _mock_response("slow response")

        result, winner, loser, duration_ms = await _execute_hedge(
            http_client=MagicMock(),
            backend1=b1,
            backend2=b2,
            chat_request=MagicMock(),
            translator=translator,
        )

        assert winner.name == "fast"
        assert loser.name == "slow"
        assert result == resp1
        assert duration_ms >= 0

    async def test_winner_identified_correctly(self):
        """When backend2 finishes first, it should be the winner."""
        b1 = _mock_backend("slow")
        b2 = _mock_backend("fast")
        resp = _mock_response()

        async def translator(client, backend, request):
            if backend.name == "fast":
                return resp
            await asyncio.sleep(10)
            return resp

        result, winner, loser, duration_ms = await _execute_hedge(
            http_client=MagicMock(),
            backend1=b1,
            backend2=b2,
            chat_request=MagicMock(),
            translator=translator,
        )

        assert winner.name == "fast"
        assert loser.name == "slow"

    async def test_both_fail_raises(self):
        """When both tasks fail, the winner's exception propagates."""
        b1 = _mock_backend("b1")
        b2 = _mock_backend("b2")

        async def failing_translator(client, backend, request):
            raise RuntimeError(f"{backend.name} failed")

        with pytest.raises(RuntimeError):
            await _execute_hedge(
                http_client=MagicMock(),
                backend1=b1,
                backend2=b2,
                chat_request=MagicMock(),
                translator=failing_translator,
            )

    async def test_duration_is_positive(self):
        """Duration should be a positive number."""
        b1 = _mock_backend("b1")
        b2 = _mock_backend("b2")
        resp = _mock_response()

        async def translator(client, backend, request):
            await asyncio.sleep(0.01)  # Small delay
            return resp

        result, winner, loser, duration_ms = await _execute_hedge(
            http_client=MagicMock(),
            backend1=b1,
            backend2=b2,
            chat_request=MagicMock(),
            translator=translator,
        )

        assert duration_ms > 0
