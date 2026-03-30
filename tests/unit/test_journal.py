"""Unit tests for request journal."""

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.journal import RequestJournal
from gateway.models import ChatMessage


class TestComputePromptHash:
    def test_deterministic(self):
        msgs = [ChatMessage(role="user", content="hello")]
        h1 = RequestJournal.compute_prompt_hash(msgs)
        h2 = RequestJournal.compute_prompt_hash(msgs)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_inputs(self):
        m1 = [ChatMessage(role="user", content="hello")]
        m2 = [ChatMessage(role="user", content="goodbye")]
        assert RequestJournal.compute_prompt_hash(m1) != RequestJournal.compute_prompt_hash(m2)

    def test_ignores_system_messages(self):
        m1 = [ChatMessage(role="system", content="sys"), ChatMessage(role="user", content="hello")]
        m2 = [ChatMessage(role="user", content="hello")]
        assert RequestJournal.compute_prompt_hash(m1) == RequestJournal.compute_prompt_hash(m2)


class TestRecordRequest:
    async def test_calls_xadd_and_sadd(self):
        redis_mock = AsyncMock()
        journal = RequestJournal(redis_mock, max_len=100000)
        await journal.record_request("req-1", "tenant-a", "gpt-4", "abc123", 1000.0)
        redis_mock.xadd.assert_called_once()
        call_args = redis_mock.xadd.call_args
        assert call_args[0][0] == RequestJournal.STREAM_KEY
        fields = call_args[0][1]
        assert fields["request_id"] == "req-1"
        assert fields["phase"] == "request"
        redis_mock.sadd.assert_called_once_with(RequestJournal.INFLIGHT_KEY, "req-1")

    async def test_exception_swallowed(self):
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(side_effect=Exception("Redis down"))
        journal = RequestJournal(redis_mock)
        # Should not raise
        await journal.record_request("req-1", "t", "m", "h", 1.0)

    async def test_uses_approximate_maxlen(self):
        redis_mock = AsyncMock()
        journal = RequestJournal(redis_mock, max_len=50000)
        await journal.record_request("req-1", "t", "m", "h", 1.0)
        call_kwargs = redis_mock.xadd.call_args
        assert call_kwargs[1]["maxlen"] == 50000
        assert call_kwargs[1]["approximate"] is True


class TestRecordCompletion:
    async def test_calls_xadd_and_srem(self):
        redis_mock = AsyncMock()
        journal = RequestJournal(redis_mock)
        await journal.record_completion("req-1", 200, 150.5, "backend-1", False, 10, 20)
        redis_mock.xadd.assert_called_once()
        fields = redis_mock.xadd.call_args[0][1]
        assert fields["request_id"] == "req-1"
        assert fields["status"] == "200"
        assert fields["phase"] == "completion"
        assert fields["cache_hit"] == "False"
        redis_mock.srem.assert_called_once_with(RequestJournal.INFLIGHT_KEY, "req-1")

    async def test_exception_swallowed(self):
        redis_mock = AsyncMock()
        redis_mock.xadd = AsyncMock(side_effect=Exception("Redis down"))
        journal = RequestJournal(redis_mock)
        await journal.record_completion("req-1", 200, 100.0, "b", False, 0, 0)


class TestGetStats:
    async def test_returns_correct_structure(self):
        redis_mock = AsyncMock()
        redis_mock.xlen = AsyncMock(return_value=42)
        redis_mock.scard = AsyncMock(return_value=3)
        redis_mock.xinfo_stream = AsyncMock(return_value={
            "first-entry": ("1000000000000-0", {}),
            "last-entry": ("1000000060000-0", {}),
        })
        journal = RequestJournal(redis_mock)
        stats = await journal.get_stats()
        assert stats["total"] == 42
        assert stats["inflight"] == 3
        assert stats["entries_per_min"] > 0

    async def test_empty_stream(self):
        redis_mock = AsyncMock()
        redis_mock.xlen = AsyncMock(return_value=0)
        redis_mock.scard = AsyncMock(return_value=0)
        journal = RequestJournal(redis_mock)
        stats = await journal.get_stats()
        assert stats["total"] == 0
        assert stats["entries_per_min"] == 0.0


class TestQuery:
    async def test_returns_grouped_entries(self):
        redis_mock = AsyncMock()
        redis_mock.xrevrange = AsyncMock(return_value=[
            ("1-0", {"request_id": "req-1", "tenant_id": "t-a", "phase": "completion", "status": "200"}),
            ("0-0", {"request_id": "req-1", "tenant_id": "t-a", "phase": "request", "model": "gpt-4"}),
        ])
        journal = RequestJournal(redis_mock)
        results = await journal.query(last=10)
        assert len(results) == 1
        assert results[0]["request_id"] == "req-1"
        assert results[0]["model"] == "gpt-4"
        assert results[0]["status"] == "200"

    async def test_filters_by_tenant(self):
        redis_mock = AsyncMock()
        redis_mock.xrevrange = AsyncMock(return_value=[
            ("2-0", {"request_id": "req-2", "tenant_id": "t-b", "phase": "request"}),
            ("1-0", {"request_id": "req-1", "tenant_id": "t-a", "phase": "request"}),
        ])
        journal = RequestJournal(redis_mock)
        results = await journal.query(tenant_id="t-a", last=10)
        assert len(results) == 1
        assert results[0]["tenant_id"] == "t-a"

    async def test_limits_results(self):
        redis_mock = AsyncMock()
        redis_mock.xrevrange = AsyncMock(return_value=[
            (f"{i}-0", {"request_id": f"req-{i}", "tenant_id": "t", "phase": "request"})
            for i in range(10)
        ])
        journal = RequestJournal(redis_mock)
        results = await journal.query(last=3)
        assert len(results) == 3
