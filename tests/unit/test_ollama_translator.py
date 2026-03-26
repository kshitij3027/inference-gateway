import httpx
import pytest

from gateway.backends.ollama import (
    chat_completion,
    translate_request,
    translate_response,
)
from gateway.models import (
    ChatCompletionRequest,
    ChatMessage,
    OllamaResponse,
)


class TestTranslateRequest:
    def test_minimal_request(self):
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        result = translate_request(req)
        assert result.model == "tinyllama"
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"
        assert result.stream is False
        assert result.options is None

    def test_request_with_options(self):
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )
        result = translate_request(req)
        assert result.options == {
            "temperature": 0.7,
            "num_predict": 100,
            "top_p": 0.9,
        }

    def test_system_message_passes_through(self):
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[
                ChatMessage(role="system", content="You are helpful"),
                ChatMessage(role="user", content="Hello"),
            ],
        )
        result = translate_request(req)
        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are helpful"


class TestTranslateResponse:
    def test_full_response(self):
        ollama_resp = OllamaResponse(
            model="tinyllama",
            created_at="2024-01-01T00:00:00Z",
            message={"role": "assistant", "content": "Hello there!"},
            done=True,
            prompt_eval_count=10,
            eval_count=5,
        )
        result = translate_response(ollama_resp, model="tinyllama")
        assert result.model == "tinyllama"
        assert result.choices[0].message.content == "Hello there!"
        assert result.choices[0].message.role == "assistant"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.usage.total_tokens == 15
        assert result.id.startswith("chatcmpl-")
        assert result.object == "chat.completion"

    def test_missing_token_counts_default_to_zero(self):
        ollama_resp = OllamaResponse(
            model="tinyllama",
            created_at="2024-01-01T00:00:00Z",
            message={"role": "assistant", "content": "Hi"},
            done=True,
        )
        result = translate_response(ollama_resp, model="tinyllama")
        assert result.usage.prompt_tokens == 0
        assert result.usage.completion_tokens == 0
        assert result.usage.total_tokens == 0


class TestChatCompletion:
    @pytest.fixture
    def mock_ollama_response(self):
        return {
            "model": "tinyllama",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
            "prompt_eval_count": 8,
            "eval_count": 3,
        }

    async def test_happy_path(self, mock_ollama_response):
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=mock_ollama_response)

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        result = await chat_completion(client, "http://ollama:11434", req)
        assert result.model == "tinyllama"
        assert result.choices[0].message.content == "Hello!"
        assert result.usage.prompt_tokens == 8
        assert result.usage.completion_tokens == 3
        await client.aclose()

    async def test_timeout_returns_504(self):
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("Connection timed out")

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with pytest.raises(Exception) as exc_info:
            await chat_completion(client, "http://ollama:11434", req)
        assert exc_info.value.status_code == 504
        await client.aclose()

    async def test_backend_error_propagates_status(self):
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with pytest.raises(Exception) as exc_info:
            await chat_completion(client, "http://ollama:11434", req)
        assert exc_info.value.status_code == 500
        await client.aclose()

    async def test_malformed_response_returns_502(self):
        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"invalid": "data"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
        )

        with pytest.raises(Exception) as exc_info:
            await chat_completion(client, "http://ollama:11434", req)
        assert exc_info.value.status_code == 502
        await client.aclose()
