import json
import time
import uuid
from collections.abc import AsyncGenerator

import httpx
import structlog
from fastapi import HTTPException
from pydantic import ValidationError

from gateway.config import BackendConfig
from gateway.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    Choice,
    ChunkChoice,
    ChunkDelta,
    OllamaMessage,
    OllamaRequest,
    OllamaResponse,
    Usage,
)

logger = structlog.get_logger()


def translate_request(request: ChatCompletionRequest) -> OllamaRequest:
    """Convert OpenAI chat completion request to Ollama format."""
    messages = [
        OllamaMessage(role=msg.role, content=msg.content) for msg in request.messages
    ]

    options = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.top_p is not None:
        options["top_p"] = request.top_p

    return OllamaRequest(
        model=request.model,
        messages=messages,
        stream=False,
        options=options if options else None,
    )


def translate_response(ollama_response: OllamaResponse, model: str) -> ChatCompletionResponse:
    """Convert Ollama response to OpenAI chat completion format."""
    prompt_tokens = ollama_response.prompt_eval_count or 0
    completion_tokens = ollama_response.eval_count or 0

    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                message=ChatMessageResponse(
                    content=ollama_response.message.content,
                ),
            ),
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def chat_completion(
    client: httpx.AsyncClient,
    backend: BackendConfig,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Send a chat completion request to Ollama and return OpenAI-format response."""
    ollama_request = translate_request(request)

    start = time.perf_counter()
    try:
        response = await client.post(
            f"{backend.base_url}/api/chat",
            json=ollama_request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        logger.error("ollama_timeout", base_url=backend.base_url, model=request.model)
        raise HTTPException(status_code=504, detail="Backend request timed out")
    except httpx.HTTPStatusError as e:
        logger.error(
            "ollama_error",
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
        ollama_response = OllamaResponse.model_validate(response.json())
    except (ValidationError, ValueError) as e:
        logger.error("ollama_invalid_response", error=str(e))
        raise HTTPException(status_code=502, detail="Invalid backend response")

    result = translate_response(ollama_response, model=request.model)

    logger.info(
        "ollama_completion",
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
    """Stream chat completion from Ollama, translating NDJSON to OpenAI SSE format."""
    ollama_request = translate_request(request)
    ollama_request.stream = True

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

    try:
        async with client.stream(
            "POST",
            f"{backend.base_url}/api/chat",
            json=ollama_request.model_dump(exclude_none=True),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue

                data = json.loads(line)
                content = data.get("message", {}).get("content", "")
                done = data.get("done", False)

                if content:
                    chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[ChunkChoice(delta=ChunkDelta(content=content))],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                if done:
                    final_chunk = ChatCompletionChunk(
                        id=chunk_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChunkChoice(delta=ChunkDelta(), finish_reason="stop")
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    break

    except httpx.HTTPStatusError as e:
        logger.error("ollama_stream_error", status_code=e.response.status_code)
        error_data = {
            "error": {
                "message": f"Backend error: {e.response.status_code}",
                "type": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
    except Exception as e:
        logger.error("ollama_stream_error", error=str(e))
        error_data = {
            "error": {
                "message": str(e),
                "type": "stream_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"

    yield "data: [DONE]\n\n"
