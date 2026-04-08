import json
import os
import time
import uuid
from collections.abc import AsyncGenerator

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
            system_parts.append(msg.content or "")
        elif msg.role == "tool":
            # Tool result messages → Anthropic tool_result content block in user message
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": msg.tool_call_id or "", "content": msg.content or ""}],
            })
        elif msg.role == "assistant" and msg.tool_calls:
            # Assistant messages with tool_use calls
            content_blocks: list[dict] = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": json.loads(tc.get("function", {}).get("arguments", "{}")) if isinstance(tc.get("function", {}).get("arguments"), str) else tc.get("function", {}).get("arguments", {}),
                    })
            messages.append({"role": "assistant", "content": content_blocks})
        else:
            messages.append(
                AnthropicMessage(
                    role=msg.role,
                    content=[AnthropicContentBlock(text=msg.content or "")],
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

    # Function calling: translate tools from OpenAI to Anthropic format
    tools = request.model_extra.get("tools") if request.model_extra else None
    if tools:
        anthropic_tools = []
        for tool in tools:
            if isinstance(tool, dict) and tool.get("type") == "function":
                fn = tool["function"]
                anthropic_tools.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                })
        if anthropic_tools:
            body["tools"] = anthropic_tools

        tool_choice = request.model_extra.get("tool_choice") if request.model_extra else None
        if tool_choice:
            if isinstance(tool_choice, str):
                body["tool_choice"] = {"type": tool_choice}
            elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                body["tool_choice"] = {"type": "tool", "name": tool_choice["function"]["name"]}

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

    # Extract text and tool_use from content blocks
    text_parts = []
    tool_calls = []
    for block in anthropic_resp.content:
        if block.type == "text" and block.text:
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id or "",
                "type": "function",
                "function": {
                    "name": block.name or "",
                    "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input or ""),
                },
            })

    content = "".join(text_parts) if text_parts else None

    # Map stop_reason to finish_reason
    finish_reason = STOP_REASON_MAP.get(anthropic_resp.stop_reason or "", "stop")

    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                message=ChatMessageResponse(
                    content=content,
                    tool_calls=tool_calls if tool_calls else None,
                ),
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


async def stream_chat_completion(
    client: httpx.AsyncClient,
    backend: BackendConfig,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Stream chat completion from Anthropic, translating SSE events to OpenAI SSE format."""
    from gateway.models import ChatCompletionChunk, ChunkChoice, ChunkDelta

    body, headers = translate_request(request, backend)
    body["stream"] = True

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    # First chunk: announce role
    first_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[ChunkChoice(delta=ChunkDelta(role="assistant"))],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    finish_reason = "stop"

    try:
        async with client.stream(
            "POST",
            f"{backend.base_url}/v1/messages",
            json=body,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue

                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type", "")

                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            chunk = ChatCompletionChunk(
                                id=chunk_id,
                                created=created,
                                model=request.model,
                                choices=[ChunkChoice(delta=ChunkDelta(content=text))],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"

                elif event_type == "message_delta":
                    stop_reason = data.get("delta", {}).get("stop_reason", "")
                    if stop_reason:
                        finish_reason = STOP_REASON_MAP.get(stop_reason, "stop")

                elif event_type == "message_stop":
                    break

    except httpx.HTTPStatusError as e:
        logger.error("anthropic_stream_error", status_code=e.response.status_code)
        error_data = {"error": {"message": f"Backend error: {e.response.status_code}", "type": "stream_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        logger.error("anthropic_stream_error", error=str(e))
        error_data = {"error": {"message": str(e), "type": "stream_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"

    # Final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[ChunkChoice(delta=ChunkDelta(), finish_reason=finish_reason)],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
