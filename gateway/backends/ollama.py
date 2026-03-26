import time

import httpx
import structlog
from fastapi import HTTPException
from pydantic import ValidationError

from gateway.config import BackendConfig
from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    Choice,
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
