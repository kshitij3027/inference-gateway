import pytest
from pydantic import ValidationError

from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageResponse,
    Choice,
    Usage,
)


class TestChatMessage:
    def test_valid_user_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_valid_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful")
        assert msg.role == "system"

    def test_valid_assistant_message(self):
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="invalid", content="test")


class TestChatCompletionRequest:
    def test_minimal_valid_request(self):
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.model == "tinyllama"
        assert len(req.messages) == 1
        assert req.temperature is None
        assert req.stream is False

    def test_request_with_all_optional_fields(self):
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            stop=["END"],
        )
        assert req.temperature == 0.7
        assert req.max_tokens == 100
        assert req.top_p == 0.9
        assert req.stop == ["END"]

    def test_missing_model_rejected(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hello")]
            )

    def test_missing_messages_rejected(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="tinyllama")

    def test_empty_messages_rejected(self):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model="tinyllama", messages=[])

    def test_extra_fields_tolerated(self):
        req = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hello")],
            custom_field="custom_value",
        )
        assert req.model == "tinyllama"


class TestChatCompletionResponse:
    def test_response_round_trip(self):
        resp = ChatCompletionResponse(
            model="tinyllama",
            choices=[
                Choice(message=ChatMessageResponse(content="Hello!"))
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        data = resp.model_dump()
        assert data["object"] == "chat.completion"
        assert data["model"] == "tinyllama"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["id"].startswith("chatcmpl-")
        assert isinstance(data["created"], int)

    def test_usage_total_equals_sum(self):
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
