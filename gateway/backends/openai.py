import json
import os
import time
import uuid
from collections.abc import AsyncGenerator

import httpx
import structlog
from fastapi import HTTPException

from gateway.config import BackendConfig
from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    Choice,
    Usage,
)

logger = structlog.get_logger()


def translate_request(
    request: ChatCompletionRequest, backend: BackendConfig
) -> tuple[dict, dict]:
    """Convert OpenAI request for forwarding to an OpenAI-compatible backend."""
    body = request.model_dump(exclude_none=True)
    body["stream"] = False  # No streaming in Phase 3

    # Build headers
    api_key = os.environ.get(backend.api_key_env or "", "")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers["Content-Type"] = "application/json"

    return body, headers


def translate_response(data: dict, model: str) -> ChatCompletionResponse:
    """Convert OpenAI-compatible backend response to gateway response.

    Replaces the backend's ID with a gateway-generated one for consistency.
    """
    try:
        choices = data["choices"]
        first_choice = choices[0]
        message = first_choice["message"]
        content = message.get("content")
        finish_reason = first_choice.get("finish_reason", "stop")
        tool_calls = message.get("tool_calls")
    except (KeyError, IndexError) as e:
        raise ValueError(f"Invalid OpenAI response structure: {e}")

    usage = data.get("usage", {})

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                message=ChatMessageResponse(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            ),
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )


async def chat_completion(
    client: httpx.AsyncClient,
    backend: BackendConfig,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Send a chat completion request to an OpenAI-compatible backend."""
    body, headers = translate_request(request, backend)

    start = time.perf_counter()
    try:
        response = await client.post(
            f"{backend.base_url}/v1/chat/completions",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        logger.error("openai_timeout", base_url=backend.base_url, model=request.model)
        raise HTTPException(status_code=504, detail="Backend request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(
            "openai_error",
            base_url=backend.base_url,
            model=request.model,
            status_code=e.response.status_code,
        )
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Backend error: {e.response.text}",
        )

    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    try:
        result = translate_response(response.json(), model=request.model)
    except (ValueError, KeyError) as e:
        logger.error("openai_invalid_response", error=str(e))
        raise HTTPException(status_code=502, detail="Invalid backend response")

    logger.info(
        "openai_completion",
        model=request.model,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        duration_ms=duration_ms,
    )

    return result


async def stream_chat_completion(
    client: httpx.AsyncClient,
    backend: BackendConfig,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Stream chat completion from OpenAI-compatible backend. Near-passthrough with ID replacement."""
    body, headers = translate_request(request, backend)
    body["stream"] = True  # Override to enable streaming

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    try:
        async with client.stream(
            "POST",
            f"{backend.base_url}/v1/chat/completions",
            json=body,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Strip "data: " prefix
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    # Replace backend ID with gateway-generated ID
                    data["id"] = chunk_id
                    data["created"] = created
                    yield f"data: {json.dumps(data)}\n\n"
                except json.JSONDecodeError:
                    continue

    except httpx.HTTPStatusError as e:
        logger.error("openai_stream_error", status_code=e.response.status_code)
        error_data = {
            "error": {
                "message": f"Backend error: {e.response.status_code}",
                "type": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        logger.error("openai_stream_error", error=str(e))
        error_data = {"error": {"message": str(e), "type": "stream_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"

    yield "data: [DONE]\n\n"
