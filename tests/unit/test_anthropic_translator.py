import httpx
import pytest

from gateway.backends.anthropic import (
    STOP_REASON_MAP,
    chat_completion,
    translate_request,
    translate_response,
)
from gateway.config import BackendConfig
from gateway.models import ChatCompletionRequest, ChatMessage


def _make_backend(monkeypatch) -> BackendConfig:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    return BackendConfig(
        name="mock-anthropic",
        provider="anthropic",
        base_url="http://mock:3000/anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        models=["claude-sonnet-4-20250514"],
    )


def _make_request(**kwargs) -> ChatCompletionRequest:
    defaults = {
        "model": "claude-sonnet-4-20250514",
        "messages": [ChatMessage(role="user", content="Hello")],
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


MOCK_ANTHROPIC_RESPONSE = {
    "id": "msg_test123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello there!"}],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 5},
}


class TestTranslateRequest:
    def test_basic_request(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request()
        body, headers = translate_request(req, backend)
        assert body["model"] == "claude-sonnet-4-20250514"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"][0]["type"] == "text"
        assert body["messages"][0]["content"][0]["text"] == "Hello"
        assert body["max_tokens"] == 4096  # default
        assert "system" not in body
        assert headers["x-api-key"] == "test-anthropic-key"
        assert headers["anthropic-version"] == "2023-06-01"

    def test_system_message_extracted(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request(
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ]
        )
        body, _ = translate_request(req, backend)
        assert body["system"] == "You are helpful"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_multiple_system_messages_concatenated(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request(
            messages=[
                ChatMessage(role="system", content="Be concise"),
                ChatMessage(role="system", content="Be helpful"),
                ChatMessage(role="user", content="Hello"),
            ]
        )
        body, _ = translate_request(req, backend)
        assert body["system"] == "Be concise\n\nBe helpful"
        assert len(body["messages"]) == 1

    def test_max_tokens_from_request(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request(max_tokens=100)
        body, _ = translate_request(req, backend)
        assert body["max_tokens"] == 100

    def test_stop_string_becomes_list(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request(stop="END")
        body, _ = translate_request(req, backend)
        assert body["stop_sequences"] == ["END"]

    def test_stop_list_passed_through(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request(stop=["END", "STOP"])
        body, _ = translate_request(req, backend)
        assert body["stop_sequences"] == ["END", "STOP"]

    def test_optional_params(self, monkeypatch):
        backend = _make_backend(monkeypatch)
        req = _make_request(temperature=0.7, top_p=0.9)
        body, _ = translate_request(req, backend)
        assert body["temperature"] == 0.7
        assert body["top_p"] == 0.9


class TestTranslateResponse:
    def test_basic_response(self):
        result = translate_response(MOCK_ANTHROPIC_RESPONSE, model="claude-sonnet-4-20250514")
        assert result.choices[0].message.content == "Hello there!"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.id.startswith("chatcmpl-")
        assert result.object == "chat.completion"

    def test_stop_reason_mapping(self):
        for anthropic_reason, openai_reason in STOP_REASON_MAP.items():
            data = {**MOCK_ANTHROPIC_RESPONSE, "stop_reason": anthropic_reason}
            result = translate_response(data, model="claude-sonnet-4-20250514")
            assert result.choices[0].finish_reason == openai_reason

    def test_multiple_content_blocks_concatenated(self):
        data = {
            **MOCK_ANTHROPIC_RESPONSE,
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world!"},
            ],
        }
        result = translate_response(data, model="claude-sonnet-4-20250514")
        assert result.choices[0].message.content == "Hello world!"

    def test_unknown_stop_reason_defaults_to_stop(self):
        data = {**MOCK_ANTHROPIC_RESPONSE, "stop_reason": "unknown_reason"}
        result = translate_response(data, model="claude-sonnet-4-20250514")
        assert result.choices[0].finish_reason == "stop"


class TestChatCompletion:
    async def test_happy_path(self, monkeypatch):
        backend = _make_backend(monkeypatch)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/anthropic/v1/messages"
            assert request.headers["x-api-key"] == "test-anthropic-key"
            assert request.headers["anthropic-version"] == "2023-06-01"
            return httpx.Response(200, json=MOCK_ANTHROPIC_RESPONSE)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = _make_request()
        result = await chat_completion(client, backend, req)
        assert result.model == "claude-sonnet-4-20250514"
        assert result.choices[0].message.content == "Hello there!"
        assert result.usage.prompt_tokens == 10
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
            return httpx.Response(500, text="Internal Server Error")

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = _make_request()
        with pytest.raises(Exception) as exc_info:
            await chat_completion(client, backend, req)
        assert exc_info.value.status_code == 500
        await client.aclose()

    async def test_malformed_response_returns_502(self, monkeypatch):
        backend = _make_backend(monkeypatch)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"invalid": "data"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = _make_request()
        with pytest.raises(Exception) as exc_info:
            await chat_completion(client, backend, req)
        assert exc_info.value.status_code == 502
        await client.aclose()
