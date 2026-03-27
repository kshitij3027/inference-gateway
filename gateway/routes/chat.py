import time
import uuid

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


async def _wrap_stream_with_circuit_breaker(gen, circuit_breaker):
    """Wrap a streaming generator to record circuit breaker outcomes."""
    error_detected = False
    try:
        async for chunk in gen:
            if not error_detected and '"stream_error"' in chunk:
                error_detected = True
            yield chunk
    except Exception:
        error_detected = True
        raise
    finally:
        if circuit_breaker:
            if error_detected:
                circuit_breaker.record_failure()
            else:
                circuit_breaker.record_success()


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
    # Rate limit check (after auth, before routing)
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    rate_limiter = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is not None:
        try:
            allowed, deny_info = await rate_limiter.check_rate_limit(
                tenant.id, request_id, tenant.rate_limit_rps, tenant.rate_limit_rpm
            )
            if not allowed:
                logger.warning("rate_limit_exceeded", tenant_id=tenant.id, **deny_info)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "rate_limit_exceeded",
                        "type": deny_info["limit_type"],
                        "limit": deny_info["limit"],
                        "current": deny_info["current"],
                        "retry_after": deny_info["retry_after"],
                    },
                    headers={"Retry-After": str(int(deny_info["retry_after"]))},
                )

            # Token budget pre-check
            budget_ok, budget_info = await rate_limiter.check_token_budget(
                tenant.id, tenant.token_budget_daily
            )
            if not budget_ok:
                logger.warning("token_budget_exceeded", tenant_id=tenant.id, **budget_info)
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "token_budget_exceeded",
                        "type": "token_budget_daily",
                        "limit": budget_info["limit"],
                        "current": budget_info["current"],
                        "retry_after": budget_info["retry_after"],
                    },
                    headers={"Retry-After": str(int(budget_info["retry_after"]))},
                )

            # Store remaining counts for response headers
            request.state.rate_limit_remaining = await rate_limiter.get_remaining(
                tenant.id, tenant.rate_limit_rps, tenant.rate_limit_rpm
            )
        except HTTPException:
            raise  # Re-raise 429s
        except Exception as e:
            logger.warning("rate_limit_check_failed", error=str(e))
            # Graceful degradation: allow request if Redis fails

    # Check model access
    if "*" not in tenant.allowed_models and chat_request.model not in tenant.allowed_models:
        raise HTTPException(
            status_code=403,
            detail=f"Model '{chat_request.model}' not allowed for tenant '{tenant.id}'",
        )

    # Find backend for model
    registry = request.app.state.registry
    routing_key = f"{tenant.id}:{chat_request.model}"

    # Streaming path with circuit breaker wrapper
    if chat_request.stream:
        cb_registry = request.app.state.circuit_breakers
        exclude = cb_registry.get_open_backends()
        backend = registry.find_backend_for_model(
            chat_request.model, routing_key=routing_key, exclude=exclude
        )
        if backend is None:
            raise HTTPException(
                status_code=503,
                detail="All backends unavailable for this model",
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

        cb = cb_registry.get(backend.name)
        raw_gen = stream_translator(
            client=request.app.state.http_client,
            backend=backend,
            request=chat_request,
        )

        return StreamingResponse(
            _wrap_stream_with_circuit_breaker(raw_gen, cb),
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

            # Record token usage for daily budget
            if tenant.token_budget_daily and rate_limiter:
                try:
                    await rate_limiter.record_tokens(tenant.id, result.usage.total_tokens)
                except Exception as e:
                    logger.warning("token_recording_failed", error=str(e))

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
