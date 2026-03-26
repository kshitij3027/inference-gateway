import os
import time

import httpx
import structlog
from fastapi import HTTPException
from pydantic import ValidationError

from gateway.config import BackendConfig
from gateway.models import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    Choice,
    Usage,
)

logger = structlog.get_logger()

STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def translate_request(
    request: ChatCompletionRequest, backend: BackendConfig
) -> tuple[dict, dict]:
    """Convert OpenAI chat completion request to Anthropic Messages format."""
    # Extract system messages (Anthropic requires system as top-level field)
    system_parts = []
    messages = []
    for msg in request.messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            messages.append(
                AnthropicMessage(
                    role=msg.role,
                    content=[AnthropicContentBlock(text=msg.content)],
                ).model_dump()
            )

    system = "\n\n".join(system_parts) if system_parts else None

    # Build request body
    body: dict = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens or 4096,
    }
    if system:
        body["system"] = system
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        if isinstance(request.stop, str):
            body["stop_sequences"] = [request.stop]
        else:
            body["stop_sequences"] = request.stop

    # Build headers
    api_key = os.environ.get(backend.api_key_env or "", "")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    return body, headers


def translate_response(data: dict, model: str) -> ChatCompletionResponse:
    """Convert Anthropic Messages response to OpenAI chat completion format."""
    anthropic_resp = AnthropicResponse.model_validate(data)

    # Extract text from content blocks
    text_parts = []
    for block in anthropic_resp.content:
        if block.type == "text":
            text_parts.append(block.text)
    content = "".join(text_parts)

    # Map stop_reason to finish_reason
    finish_reason = STOP_REASON_MAP.get(anthropic_resp.stop_reason or "", "stop")

    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                message=ChatMessageResponse(content=content),
                finish_reason=finish_reason,
            ),
        ],
        usage=Usage(
            prompt_tokens=anthropic_resp.usage.input_tokens,
            completion_tokens=anthropic_resp.usage.output_tokens,
            total_tokens=anthropic_resp.usage.input_tokens + anthropic_resp.usage.output_tokens,
        ),
    )


async def chat_completion(
    client: httpx.AsyncClient,
    backend: BackendConfig,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Send a chat completion request to Anthropic and return OpenAI-format response."""
    body, headers = translate_request(request, backend)

    start = time.perf_counter()
    try:
        response = await client.post(
            f"{backend.base_url}/v1/messages",
            json=body,
            headers=headers,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        logger.error("anthropic_timeout", base_url=backend.base_url, model=request.model)
        raise HTTPException(status_code=504, detail="Backend request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(
            "anthropic_error",
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
    except (ValidationError, ValueError, KeyError) as e:
        logger.error("anthropic_invalid_response", error=str(e))
        raise HTTPException(status_code=502, detail="Invalid backend response")

    logger.info(
        "anthropic_completion",
        model=request.model,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        duration_ms=duration_ms,
    )

    return result
