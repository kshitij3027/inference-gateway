"""Integration test for distributed tracing span tree.

Verifies that a single chat completion request produces the expected
span hierarchy with correct names, attributes, and parent-child relationships.

Docker E2E: After deploying with `docker compose up --build -d`, run
`bash scripts/test-tracing-e2e.sh` to verify spans reach Jaeger.
"""
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from gateway.main import app
from gateway.models import ChatCompletionResponse, ChatMessageResponse, Choice, Usage


def _reset_tracer_provider():
    """Reset the global tracer provider so tests can set their own."""
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    trace._TRACER_PROVIDER = None


def _mock_response():
    return ChatCompletionResponse(
        model="tinyllama",
        choices=[Choice(message=ChatMessageResponse(content="Hello!"))],
        usage=Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


class TestTracingSpanTree:
    """Verify complete span tree for a non-streaming chat completion."""

    async def test_full_span_tree(self, test_env):
        """A successful request produces all pipeline spans with correct attributes."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        _reset_tracer_provider()
        trace.set_tracer_provider(provider)

        try:
            async with app.router.lifespan_context(app):
                async with AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://test"
                ) as client:
                    with patch.dict(
                        "gateway.routes.chat.TRANSLATORS",
                        {"ollama": AsyncMock(return_value=_mock_response())},
                    ):
                        resp = await client.post(
                            "/v1/chat/completions",
                            json={"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}]},
                            headers={"Authorization": "Bearer test-alpha-key"},
                        )
            assert resp.status_code == 200

            spans = exporter.get_finished_spans()
            span_names = {s.name for s in spans}

            # All expected pipeline spans must be present
            expected = {
                "gateway.auth",
                "gateway.rate_limit",
                "gateway.cache.lookup",
                "gateway.journal.write",
                "gateway.router",
                "gateway.queue.wait",
                "gateway.translator.request",
                "gateway.circuit_breaker",
                "gateway.cache.store",
            }
            missing = expected - span_names
            assert not missing, f"Missing spans: {missing}"

            # Verify key attributes
            auth = [s for s in spans if s.name == "gateway.auth"][0]
            assert auth.attributes["tenant.id"] == "tenant-alpha"

            rl = [s for s in spans if s.name == "gateway.rate_limit"][0]
            assert rl.attributes["rate_limit.allowed"] is True

            cache = [s for s in spans if s.name == "gateway.cache.lookup"][0]
            assert cache.attributes["cache.hit"] is False

            router = [s for s in spans if s.name == "gateway.router"][0]
            assert router.attributes.get("route.backend") is not None

            trans = [s for s in spans if s.name == "gateway.translator.request"][0]
            assert trans.attributes["translator.streaming"] is False

            cb = [s for s in spans if s.name == "gateway.circuit_breaker"][0]
            assert cb.attributes["cb.outcome"] == "success"

            # Journal has both request and completion phases
            journal_spans = [s for s in spans if s.name == "gateway.journal.write"]
            phases = {s.attributes.get("journal.phase") for s in journal_spans}
            assert "request" in phases
            assert "completion" in phases
        finally:
            provider.shutdown()
            _reset_tracer_provider()
