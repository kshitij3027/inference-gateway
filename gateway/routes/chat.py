import os
import time

import structlog
from fastapi import APIRouter, Request

from gateway.backends import ollama
from gateway.models import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()
logger = structlog.get_logger()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    chat_request: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse:
    logger.info(
        "chat_request_received",
        model=chat_request.model,
        message_count=len(chat_request.messages),
    )

    start = time.perf_counter()
    result = await ollama.chat_completion(
        client=request.app.state.http_client,
        base_url=OLLAMA_BASE_URL,
        request=chat_request,
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "chat_request_completed",
        model=chat_request.model,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        duration_ms=duration_ms,
    )

    return result
