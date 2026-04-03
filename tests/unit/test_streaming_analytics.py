"""Unit tests for _wrap_stream_with_analytics streaming wrapper."""

from __future__ import annotations

import json
import time

import pytest
from prometheus_client import REGISTRY

from gateway.routes.chat import _wrap_stream_with_analytics


def _make_chunk(content=None, role=None, finish_reason=None):
    """Build an SSE chunk string in OpenAI format."""
    delta = {}
    if role:
        delta["role"] = role
    if content:
        delta["content"] = content
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "test-model",
        "choices": [choice],
    }
    return f"data: {json.dumps(data)}\n\n"


class TestStreamingAnalytics:

    async def test_ttft_metric_recorded(self):
        """TTFT histogram should be observed after first content token."""
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(finish_reason="stop"),
            "data: [DONE]\n\n",
        ]

        async def gen():
            for c in chunks:
                yield c

        start = time.perf_counter()
        async for _ in _wrap_stream_with_analytics(gen(), "ttft-model", "ttft-be", start):
            pass

        val = REGISTRY.get_sample_value(
            "gateway_ttft_seconds_count",
            {"model": "ttft-model", "backend": "ttft-be"},
        )
        assert val is not None and val >= 1

    async def test_sse_comment_after_first_content(self):
        """SSE comment with TTFT should appear right after first content chunk."""
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(content="Hi"),
            _make_chunk(finish_reason="stop"),
            "data: [DONE]\n\n",
        ]

        async def gen():
            for c in chunks:
                yield c

        start = time.perf_counter()
        collected = []
        async for chunk in _wrap_stream_with_analytics(gen(), "sse-model", "sse-be", start):
            collected.append(chunk)

        # Find the SSE comment
        comments = [c for c in collected if c.startswith(": ttft_ms=")]
        assert len(comments) == 1

        # Should be right after the first content chunk
        content_idx = next(
            i for i, c in enumerate(collected) if "content" in c and "Hi" in c
        )
        assert collected[content_idx + 1].startswith(": ttft_ms=")

        # TTFT value should be a positive float
        ttft_val = float(comments[0].split("=")[1].strip())
        assert ttft_val >= 0

    async def test_itl_between_content_tokens(self):
        """ITL should be recorded between consecutive content tokens."""
        chunks = [
            _make_chunk(content="a"),
            _make_chunk(content="b"),
            _make_chunk(content="c"),
            "data: [DONE]\n\n",
        ]

        async def gen():
            for c in chunks:
                yield c

        start = time.perf_counter()
        async for _ in _wrap_stream_with_analytics(gen(), "itl-model", "itl-be", start):
            pass

        # ITL should have 2 observations (between a->b and b->c)
        val = REGISTRY.get_sample_value(
            "gateway_itl_seconds_count",
            {"model": "itl-model", "backend": "itl-be"},
        )
        assert val is not None and val >= 2

    async def test_generation_duration_at_done(self):
        """Generation duration should be recorded when [DONE] arrives."""
        chunks = [
            _make_chunk(content="first"),
            _make_chunk(content="last"),
            "data: [DONE]\n\n",
        ]

        async def gen():
            for c in chunks:
                yield c

        start = time.perf_counter()
        async for _ in _wrap_stream_with_analytics(gen(), "dur-model", "dur-be", start):
            pass

        val = REGISTRY.get_sample_value(
            "gateway_generation_duration_seconds_count",
            {"model": "dur-model", "backend": "dur-be"},
        )
        assert val is not None and val >= 1

    async def test_no_content_no_metrics(self):
        """If stream has no content tokens, TTFT should not be recorded."""
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(finish_reason="stop"),
            "data: [DONE]\n\n",
        ]

        async def gen():
            for c in chunks:
                yield c

        start = time.perf_counter()
        async for _ in _wrap_stream_with_analytics(gen(), "empty-model", "empty-be", start):
            pass

        val = REGISTRY.get_sample_value(
            "gateway_ttft_seconds_count",
            {"model": "empty-model", "backend": "empty-be"},
        )
        assert val is None or val == 0

    async def test_original_chunks_preserved(self):
        """All original chunks should still be yielded."""
        chunks = [
            _make_chunk(role="assistant"),
            _make_chunk(content="Hello"),
            _make_chunk(finish_reason="stop"),
            "data: [DONE]\n\n",
        ]

        async def gen():
            for c in chunks:
                yield c

        start = time.perf_counter()
        collected = []
        async for chunk in _wrap_stream_with_analytics(gen(), "pres-model", "pres-be", start):
            collected.append(chunk)

        # All original chunks should be present
        for orig in chunks:
            assert orig in collected

        # Plus one SSE comment (only addition)
        extra = [c for c in collected if c not in chunks]
        assert len(extra) == 1
        assert extra[0].startswith(": ttft_ms=")

    async def test_single_content_token(self):
        """Single content token: TTFT recorded, no ITL."""
        chunks = [
            _make_chunk(content="only"),
            "data: [DONE]\n\n",
        ]

        async def gen():
            for c in chunks:
                yield c

        start = time.perf_counter()
        async for _ in _wrap_stream_with_analytics(gen(), "single-model", "single-be", start):
            pass

        ttft = REGISTRY.get_sample_value(
            "gateway_ttft_seconds_count",
            {"model": "single-model", "backend": "single-be"},
        )
        assert ttft is not None and ttft >= 1

        # No ITL for single token
        itl = REGISTRY.get_sample_value(
            "gateway_itl_seconds_count",
            {"model": "single-model", "backend": "single-be"},
        )
        assert itl is None or itl == 0
