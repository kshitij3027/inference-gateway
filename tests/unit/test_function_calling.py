"""Unit tests for function calling passthrough."""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageResponse,
    Choice,
    Usage,
)


class TestChatMessageToolSupport:
    def test_tool_role_accepted(self):
        msg = ChatMessage(role="tool", content="result", tool_call_id="call_123")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"

    def test_content_optional(self):
        msg = ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        )
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_standard_message_still_works(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.content == "Hello"


class TestChatMessageResponseToolCalls:
    def test_tool_calls_in_response(self):
        resp = ChatMessageResponse(
            content=None,
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "fn", "arguments": "{}"},
                }
            ],
        )
        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1

    def test_response_without_tool_calls(self):
        resp = ChatMessageResponse(content="Hello")
        assert resp.tool_calls is None


class TestOpenAIToolCallsResponse:
    def test_tool_calls_extracted(self):
        from gateway.backends.openai import translate_response

        data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = translate_response(data, "gpt-4")
        assert result.choices[0].message.tool_calls is not None
        assert (
            result.choices[0].message.tool_calls[0]["function"]["name"] == "get_weather"
        )
        assert result.choices[0].finish_reason == "tool_calls"

    def test_normal_response_no_tool_calls(self):
        from gateway.backends.openai import translate_response

        data = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8,
            },
        }
        result = translate_response(data, "gpt-4")
        assert result.choices[0].message.tool_calls is None
        assert result.choices[0].message.content == "Hi"


class TestAnthropicToolsTranslation:
    def test_tools_request_translation(self):
        from gateway.backends.anthropic import translate_request
        from gateway.config import BackendConfig

        backend = BackendConfig(
            name="test",
            provider="anthropic",
            base_url="http://test:3000",
            models=["claude"],
            api_key_env="TEST_KEY",
        )
        request = ChatCompletionRequest(
            model="claude",
            messages=[ChatMessage(role="user", content="What's the weather?")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        body, headers = translate_request(request, backend)
        assert "tools" in body
        assert body["tools"][0]["name"] == "get_weather"
        assert "input_schema" in body["tools"][0]

    def test_tool_use_response_translation(self):
        from gateway.backends.anthropic import translate_response

        data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
            "model": "claude-3",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = translate_response(data, "claude-3")
        assert result.choices[0].message.tool_calls is not None
        tc = result.choices[0].message.tool_calls[0]
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "NYC"}
        assert result.choices[0].finish_reason == "tool_calls"


class TestOllamaToolsRejection:
    @pytest.mark.asyncio
    async def test_rejects_tools_non_streaming(self):
        from gateway.backends.ollama import chat_completion
        from gateway.config import BackendConfig

        backend = BackendConfig(
            name="test",
            provider="ollama",
            base_url="http://test:11434",
            models=["tinyllama"],
        )
        request = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[{"type": "function", "function": {"name": "fn"}}],
        )
        client = AsyncMock()
        with pytest.raises(HTTPException) as exc_info:
            await chat_completion(client, backend, request)
        assert exc_info.value.status_code == 400
        assert "tools" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_rejects_tools_streaming(self):
        from gateway.backends.ollama import stream_chat_completion
        from gateway.config import BackendConfig

        backend = BackendConfig(
            name="test",
            provider="ollama",
            base_url="http://test:11434",
            models=["tinyllama"],
        )
        request = ChatCompletionRequest(
            model="tinyllama",
            messages=[ChatMessage(role="user", content="Hi")],
            tools=[{"type": "function", "function": {"name": "fn"}}],
        )
        client = AsyncMock()
        with pytest.raises(HTTPException) as exc_info:
            async for _ in stream_chat_completion(client, backend, request):
                pass
        assert exc_info.value.status_code == 400
