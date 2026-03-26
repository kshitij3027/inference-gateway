import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from gateway.auth import get_current_tenant
from gateway.backends import ollama
from gateway.config import TenantConfig
from gateway.models import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()
logger = structlog.get_logger()


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    chat_request: ChatCompletionRequest,
    request: Request,
    tenant: TenantConfig = Depends(get_current_tenant),
) -> ChatCompletionResponse:
    # Check model access
    if "*" not in tenant.allowed_models and chat_request.model not in tenant.allowed_models:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{chat_request.model}' not allowed for tenant '{tenant.id}'",
        )

    # Find backend for model
    registry = request.app.state.registry
    backend = registry.find_backend_for_model(chat_request.model)
    if backend is None:
        raise HTTPException(
            status_code=404,
            detail=f"No backend available for model: {chat_request.model}",
        )

    logger.info(
        "chat_request_received",
        model=chat_request.model,
        tenant_id=tenant.id,
        backend=backend.name,
        message_count=len(chat_request.messages),
    )

    start = time.perf_counter()
    result = await ollama.chat_completion(
        client=request.app.state.http_client,
        base_url=backend.base_url,
        request=chat_request,
    )
    duration_ms = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "chat_request_completed",
        model=chat_request.model,
        tenant_id=tenant.id,
        backend=backend.name,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        duration_ms=duration_ms,
    )

    return result
