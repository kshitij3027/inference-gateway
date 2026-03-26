import httpx
import pytest

from gateway.backends.openai import (
    chat_completion,
    translate_request,
    translate_response,
)
from gateway.config import BackendConfig
from gateway.models import ChatCompletionRequest, ChatMessage


def _make_backend(monkeypatch) -> BackendConfig:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    return BackendConfig(
        name="mock-openai",
        provider="openai",
        base_url="http://mock:3000",
        api_key_env="OPENAI_API_KEY",
        models=["gpt-4o-mini"],
    )


def _make_request(**kwargs) -> ChatCompletionRequest:
    defaults = {
        "model": "gpt-4o-mini",
        "messages": [ChatMessage(role="user", content="Hello")],
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


MOCK_OPENAI_RESPONSE = {
    "id": "chatcmpl-backend123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 4,
        "total_tokens": 12,
    },
}


class TestTranslateRequest:
    def test_passthrough_preserves_fields(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request(temperature=0.7, max_tokens=100)
        body, headers = translate_request(req, backend)
        assert body["model"] == "gpt-4o-mini"
        assert body["temperature"] == 0.7
        assert body["max_tokens"] == 100
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_stream_forced_false(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request(stream=True)
        body, _ = translate_request(req, backend)
        assert body["stream"] is False

    def test_none_fields_excluded(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request()  # temperature=None, max_tokens=None
        body, _ = translate_request(req, backend)
        assert "temperature" not in body
        assert "max_tokens" not in body

    def test_auth_header_from_env(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request()
        _, headers = translate_request(req, backend)
        assert headers["Authorization"] == "Bearer test-openai-key"

    def test_no_auth_header_when_no_key(self, monkeypatch):
        backend = BackendConfig(
            name="mock-openai",
            provider="openai",
            base_url="http://mock:3000",
            models=["gpt-4o-mini"],
        )
        req = _make_request()
        _, headers = translate_request(req, backend)
        assert "Authorization" not in headers


class TestTranslateResponse:
    def test_basic_response(self):
        result = translate_response(MOCK_OPENAI_RESPONSE, model="gpt-4o-mini")
        assert result.choices[0].message.content == "Hello there!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 4
        assert result.usage.total_tokens == 12
        assert result.id.startswith("chatcmpl-")
        # ID should be gateway-generated, NOT the backend's ID
        assert result.id != "chatcmpl-backend123"

    def test_finish_reason_preserved(self):
        data = {**MOCK_OPENAI_RESPONSE}
        data["choices"] = [{**data["choices"][0], "finish_reason": "length"}]
        result = translate_response(data, model="gpt-4o-mini")
        assert result.choices[0].finish_reason == "length"

    def test_missing_usage_defaults_to_zero(self):
        data = {**MOCK_OPENAI_RESPONSE}
        del data["usage"]
        result = translate_response(data, model="gpt-4o-mini")
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0

    def test_invalid_structure_raises(self):
        with pytest.raises(ValueError):
            translate_response({"invalid": "data"}, model="gpt-4o-mini")


class TestChatCompletion:
    async def test_happy_path(self, monkeypatch):
        backend = _make_backend(monkeypatch)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/chat/completions"
            assert "Bearer test-openai-key" in request.headers.get("authorization", "")
            return httpx.Response(200, json=MOCK_OPENAI_RESPONSE)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = _make_request()
        result = await chat_completion(client, backend, req)
        assert result.model == "gpt-4o-mini"
        assert result.choices[0].message.content == "Hello there!"
        assert result.usage.prompt_tokens == 8
        await client.aclose()

    async def test_timeout_returns_504(self, monkeypatch):
        backend = _make_backend(monkeypatch)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("timed out")

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = _make_request()
        with pytest.raises(Exception) as exc_info:
            await chat_completion(client, backend, req)
        assert exc_info.value.status_code == 504
        await client.aclose()

    async def test_backend_error_propagates_status(self, monkeypatch):
        backend = _make_backend(monkeypatch)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Rate limited")

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = _make_request()
        with pytest.raises(Exception) as exc_info:
            await chat_completion(client, backend, req)
        assert exc_info.value.status_code == 429
        await client.aclose()

    async def test_malformed_response_returns_502(self, monkeypatch):
        backend = _make_backend(monkeypatch)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"not": "valid"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = _make_request()
        with pytest.raises(Exception) as exc_info:
            await chat_completion(client, backend, req)
        assert exc_info.value.status_code == 502
        await client.aclose()
