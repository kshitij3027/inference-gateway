import time

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from gateway.auth import get_current_tenant
from gateway.backends import anthropic as anthropic_backend
from gateway.backends import ollama
from gateway.backends import openai as openai_backend
from gateway.config import TenantConfig
from gateway.models import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()
logger = structlog.get_logger()

TRANSLATORS = {
    "ollama": ollama.chat_completion,
    "openai": openai_backend.chat_completion,
    "anthropic": anthropic_backend.chat_completion,
}

STREAM_TRANSLATORS = {
    "ollama": ollama.stream_chat_completion,
    "openai": openai_backend.stream_chat_completion,
    "anthropic": anthropic_backend.stream_chat_completion,
}


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
    routing_key = f"{tenant.id}:{chat_request.model}"

    # Streaming path — single-attempt lookup (failover added later)
    if chat_request.stream:
        backend = registry.find_backend_for_model(
            chat_request.model, routing_key=routing_key
        )
        if backend is None:
            raise HTTPException(
                status_code=404,
                detail=f"No backend available for model: {chat_request.model}",
            )

        request.state.backend_name = backend.name

        stream_translator = STREAM_TRANSLATORS.get(backend.provider)
        if stream_translator is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider for streaming: {backend.provider}",
            )

        logger.info(
            "chat_request_received",
            model=chat_request.model,
            tenant_id=tenant.id,
            backend=backend.name,
            provider=backend.provider,
            streaming=True,
            message_count=len(chat_request.messages),
        )

        return StreamingResponse(
            stream_translator(
                client=request.app.state.http_client,
                backend=backend,
                request=chat_request,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming path with failover retry loop
    cb_registry = request.app.state.circuit_breakers
    exclude = cb_registry.get_open_backends()
    last_error: HTTPException | None = None

    for attempt in range(3):
        backend = registry.find_backend_for_model(
            chat_request.model, routing_key=routing_key, exclude=exclude
        )
        if backend is None:
            break

        request.state.backend_name = backend.name

        translator = TRANSLATORS.get(backend.provider)
        if translator is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported provider: {backend.provider}",
            )

        logger.info(
            "chat_request_received",
            model=chat_request.model,
            tenant_id=tenant.id,
            backend=backend.name,
            provider=backend.provider,
            message_count=len(chat_request.messages),
        )

        try:
            start = time.perf_counter()
            result = await translator(
                client=request.app.state.http_client,
                backend=backend,
                request=chat_request,
            )
            duration_ms = round((time.perf_counter() - start) * 1000, 2)

            cb = cb_registry.get(backend.name)
            if cb:
                cb.record_success()

            logger.info(
                "chat_request_completed",
                model=chat_request.model,
                tenant_id=tenant.id,
                backend=backend.name,
                provider=backend.provider,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                duration_ms=duration_ms,
            )

            return result

        except HTTPException as e:
            if e.status_code >= 500:
                cb = cb_registry.get(backend.name)
                if cb:
                    cb.record_failure()
                exclude = exclude | {backend.name}
                last_error = e
                logger.warning(
                    "backend_failed_trying_next",
                    backend=backend.name,
                    status_code=e.status_code,
                    attempt=attempt + 1,
                )
                continue
            raise  # 4xx errors are not backend failures

        except httpx.ConnectError:
            cb = cb_registry.get(backend.name)
            if cb:
                cb.record_failure()
            exclude = exclude | {backend.name}
            last_error = HTTPException(
                status_code=502,
                detail=f"Backend {backend.name} connection refused",
            )
            logger.warning(
                "backend_connect_failed",
                backend=backend.name,
                attempt=attempt + 1,
            )
            continue

    # All attempts exhausted or no backend available
    if last_error:
        raise last_error
    raise HTTPException(
        status_code=404,
        detail=f"No backend available for model: {chat_request.model}",
    )
