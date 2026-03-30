import json
import time
import uuid

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from gateway.auth import get_current_tenant
from gateway.journal import RequestJournal
from gateway.priority_queue import QueueFullError, QueueTimeoutError
from gateway.token_counting import count_tokens
from gateway.backends import anthropic as anthropic_backend
from gateway.backends import ollama
from gateway.backends import openai as openai_backend
from gateway.config import TenantConfig
from gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageResponse,
    Choice,
    Usage,
)
from gateway.observability.metrics import (
    ACTIVE_REQUESTS,
    CACHE_OPERATIONS,
    QUEUE_DEPTH,
    RATE_LIMIT_REJECTIONS,
    TOKENS_CONSUMED,
)

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


async def _tee_stream_for_cache(gen, semantic_cache, model, messages, tenant_id, cache_isolation):
    """Wrap a streaming generator to buffer chunks and cache the assembled response."""
    buffered_content = []

    async for chunk in gen:
        yield chunk
        # Parse SSE chunk to extract content
        if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
            try:
                data = json.loads(chunk[6:])
                choices = data.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        buffered_content.append(content)
            except (json.JSONDecodeError, IndexError, KeyError):
                pass

    # After stream completes, store the assembled response in cache
    if buffered_content and semantic_cache is not None:
        full_content = "".join(buffered_content)
        assembled = ChatCompletionResponse(
            model=model,
            choices=[Choice(message=ChatMessageResponse(content=full_content))],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        try:
            await semantic_cache.store(
                model=model,
                messages=messages,
                response=assembled,
                tenant_id=tenant_id,
                cache_isolation=cache_isolation,
            )
        except Exception as e:
            logger.warning("stream_cache_store_failed", error=str(e))


async def _stream_cached_response(response):
    """Convert a cached ChatCompletionResponse into SSE chunks for streaming."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    model = response.model
    content = response.choices[0].message.content

    # Role chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    # Content chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

    # Stop chunk
    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"

    yield "data: [DONE]\n\n"


async def _wrap_stream_with_slot_release(gen, queue_manager, backend_name, model):
    """Release concurrency slot when streaming response completes."""
    try:
        async for chunk in gen:
            yield chunk
    finally:
        from gateway.observability.metrics import ACTIVE_REQUESTS
        ACTIVE_REQUESTS.labels(backend=backend_name).dec()
        await queue_manager.release_slot(backend_name, model)


async def _wrap_stream_with_journal(
    gen,
    journal,
    request_id,
    tenant_id,
    backend_name,
    model,
    messages,
    start_time,
    rate_limiter,
    token_budget_daily,
):
    """Record journal completion and token metrics after streaming finishes."""
    buffered_content = []
    error_status = None
    try:
        async for chunk in gen:
            if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                try:
                    data = json.loads(chunk[6:])
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            buffered_content.append(content)
                except (json.JSONDecodeError, IndexError, KeyError):
                    pass
            yield chunk
    except Exception:
        error_status = 500
        raise
    finally:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        full_content = "".join(buffered_content)
        prompt_text = "\n".join(m.content for m in messages)
        tokens_prompt = count_tokens(prompt_text, model)
        tokens_completion = count_tokens(full_content, model) if full_content else 0

        # Record journal completion
        if journal is not None:
            await journal.record_completion(
                request_id=request_id,
                status=error_status or 200,
                latency_ms=latency_ms,
                backend=backend_name,
                cache_hit=False,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
            )

        # Fix streaming token metrics gap
        TOKENS_CONSUMED.labels(
            tenant=tenant_id, model=model, type="prompt"
        ).inc(tokens_prompt)
        TOKENS_CONSUMED.labels(
            tenant=tenant_id, model=model, type="completion"
        ).inc(tokens_completion)

        # Record token budget for streaming
        if rate_limiter and token_budget_daily:
            try:
                await rate_limiter.record_tokens(
                    tenant_id, tokens_prompt + tokens_completion
                )
            except Exception:
                pass


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
    # Set tenant and model on request state for metrics middleware
    request.state.tenant_id = tenant.id
    request.state.model_name = chat_request.model

    journal = getattr(request.app.state, "journal", None)
    request_start_time = time.perf_counter()

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
                RATE_LIMIT_REJECTIONS.labels(
                    tenant=tenant.id, limit_type=deny_info["limit_type"]
                ).inc()
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
                RATE_LIMIT_REJECTIONS.labels(
                    tenant=tenant.id, limit_type="token_budget_daily"
                ).inc()
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

    # Record journal entry
    if journal is not None:
        prompt_hash = RequestJournal.compute_prompt_hash(chat_request.messages)
        await journal.record_request(
            request_id=request_id,
            tenant_id=tenant.id,
            model=chat_request.model,
            prompt_hash=prompt_hash,
            timestamp=time.time(),
        )

    # Semantic cache check (after auth + rate-limit, before routing)
    semantic_cache = getattr(request.app.state, "semantic_cache", None)
    cache_isolation = getattr(tenant, "cache_isolation", "shared")

    if semantic_cache is not None:
        try:
            cached_response, cache_similarity = await semantic_cache.lookup(
                model=chat_request.model,
                messages=chat_request.messages,
                tenant_id=tenant.id,
                cache_isolation=cache_isolation,
            )
            if cached_response is not None:
                await semantic_cache.record_hit()
                CACHE_OPERATIONS.labels(model=chat_request.model, status="hit").inc()
                request.state.cache_status = "HIT"
                request.state.cache_similarity = cache_similarity
                logger.info(
                    "cache_hit",
                    model=chat_request.model,
                    tenant_id=tenant.id,
                    similarity=cache_similarity,
                    streaming=chat_request.stream,
                )
                if journal is not None:
                    await journal.record_completion(
                        request_id=request_id,
                        status=200,
                        latency_ms=round((time.perf_counter() - request_start_time) * 1000, 2),
                        backend="cache",
                        cache_hit=True,
                        tokens_prompt=cached_response.usage.prompt_tokens,
                        tokens_completion=cached_response.usage.completion_tokens,
                    )
                if chat_request.stream:
                    return StreamingResponse(
                        _stream_cached_response(cached_response),
                        media_type="text/event-stream",
                    )
                return cached_response
            else:
                await semantic_cache.record_miss()
                CACHE_OPERATIONS.labels(model=chat_request.model, status="miss").inc()
                request.state.cache_status = "MISS"
        except Exception as e:
            logger.warning("cache_lookup_failed", error=str(e))
            request.state.cache_status = "MISS"

    # Stampede guard (non-streaming only)
    lock_acquired = False
    lock_key = ""
    if semantic_cache is not None and not chat_request.stream:
        try:
            lock_acquired, lock_key = await semantic_cache.acquire_stampede_lock(
                chat_request.model, chat_request.messages
            )
            if not lock_acquired:
                # Another request is computing this — wait for it
                waited_response, waited_similarity = await semantic_cache.wait_for_cached_result(
                    chat_request.model, chat_request.messages,
                    tenant.id, cache_isolation, timeout=30.0,
                )
                if waited_response is not None:
                    await semantic_cache.record_hit()
                    request.state.cache_status = "HIT"
                    request.state.cache_similarity = waited_similarity
                    logger.info(
                        "stampede_guard_hit",
                        model=chat_request.model,
                        tenant_id=tenant.id,
                        similarity=waited_similarity,
                    )
                    return waited_response
                # Timeout — fall through to backend call
                logger.info("stampede_guard_timeout", model=chat_request.model)
        except Exception as e:
            logger.warning("stampede_guard_failed", error=str(e))

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

        # Queue: acquire concurrency slot
        queue_manager = getattr(request.app.state, "queue_manager", None)
        slot_backend = None
        if queue_manager is not None:
            slot_acquired = await queue_manager.acquire_slot(
                backend.name, backend.max_concurrent
            )
            if not slot_acquired:
                try:
                    await queue_manager.enqueue(
                        chat_request.model, request_id, tenant.priority
                    )
                    QUEUE_DEPTH.labels(model=chat_request.model).inc()
                except QueueFullError:
                    raise HTTPException(
                        status_code=503,
                        detail="Queue full",
                        headers={"Retry-After": "5"},
                    )
                try:
                    queue_wait_ms = await queue_manager.wait_for_slot(request_id)
                    QUEUE_DEPTH.labels(model=chat_request.model).dec()
                    request.state.queue_wait_ms = queue_wait_ms
                except QueueTimeoutError:
                    await queue_manager.remove_from_queue(
                        chat_request.model, request_id
                    )
                    QUEUE_DEPTH.labels(model=chat_request.model).dec()
                    raise HTTPException(
                        status_code=504, detail="Queue timeout"
                    )
                # Re-check circuit breaker after dequeue
                exclude = cb_registry.get_open_backends()
                backend = registry.find_backend_for_model(
                    chat_request.model, routing_key=routing_key, exclude=exclude
                )
                if backend is None:
                    raise HTTPException(
                        status_code=503,
                        detail="All backends unavailable after dequeue",
                    )
                await queue_manager.acquire_slot(
                    backend.name, backend.max_concurrent
                )
            slot_backend = backend.name
            ACTIVE_REQUESTS.labels(backend=slot_backend).inc()

        request.state.backend_name = backend.name

        stream_translator = STREAM_TRANSLATORS.get(backend.provider)
        if stream_translator is None:
            if queue_manager is not None and slot_backend:
                await queue_manager.release_slot(slot_backend, chat_request.model)
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

        wrapped_gen = _wrap_stream_with_circuit_breaker(raw_gen, cb)
        if semantic_cache is not None:
            wrapped_gen = _tee_stream_for_cache(
                wrapped_gen, semantic_cache, chat_request.model,
                chat_request.messages, tenant.id, cache_isolation,
            )
        # Slot release wrapper goes outermost
        if queue_manager is not None and slot_backend:
            wrapped_gen = _wrap_stream_with_slot_release(
                wrapped_gen, queue_manager, slot_backend, chat_request.model
            )
        # Journal wrapper outermost — records completion after stream finishes
        if journal is not None:
            wrapped_gen = _wrap_stream_with_journal(
                wrapped_gen, journal, request_id, tenant.id,
                backend.name, chat_request.model, chat_request.messages,
                request_start_time, rate_limiter, tenant.token_budget_daily,
            )
        return StreamingResponse(wrapped_gen, media_type="text/event-stream")

    # Non-streaming path with failover retry loop
    cb_registry = request.app.state.circuit_breakers
    exclude = cb_registry.get_open_backends()
    last_error: HTTPException | None = None
    queue_manager = getattr(request.app.state, "queue_manager", None)

    for attempt in range(3):
        backend = registry.find_backend_for_model(
            chat_request.model, routing_key=routing_key, exclude=exclude
        )
        if backend is None:
            break

        request.state.backend_name = backend.name

        # Queue: acquire concurrency slot
        slot_backend = None
        if queue_manager is not None:
            slot_acquired = await queue_manager.acquire_slot(
                backend.name, backend.max_concurrent
            )
            if not slot_acquired:
                # At capacity — enqueue for the model
                try:
                    await queue_manager.enqueue(
                        chat_request.model, request_id, tenant.priority
                    )
                    QUEUE_DEPTH.labels(model=chat_request.model).inc()
                except QueueFullError:
                    raise HTTPException(
                        status_code=503,
                        detail="Queue full",
                        headers={"Retry-After": "5"},
                    )
                try:
                    queue_wait_ms = await queue_manager.wait_for_slot(request_id)
                    QUEUE_DEPTH.labels(model=chat_request.model).dec()
                    request.state.queue_wait_ms = queue_wait_ms
                except QueueTimeoutError:
                    await queue_manager.remove_from_queue(
                        chat_request.model, request_id
                    )
                    QUEUE_DEPTH.labels(model=chat_request.model).dec()
                    raise HTTPException(
                        status_code=504, detail="Queue timeout"
                    )
                # Re-check circuit breaker and re-route after dequeue
                exclude = cb_registry.get_open_backends()
                backend = registry.find_backend_for_model(
                    chat_request.model, routing_key=routing_key, exclude=exclude
                )
                if backend is None:
                    raise HTTPException(
                        status_code=503,
                        detail="All backends unavailable after dequeue",
                    )
                request.state.backend_name = backend.name
                await queue_manager.acquire_slot(
                    backend.name, backend.max_concurrent
                )
            slot_backend = backend.name
            ACTIVE_REQUESTS.labels(backend=slot_backend).inc()

        translator = TRANSLATORS.get(backend.provider)
        if translator is None:
            if queue_manager is not None and slot_backend:
                await queue_manager.release_slot(slot_backend, chat_request.model)
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
            TOKENS_CONSUMED.labels(
                tenant=tenant.id, model=chat_request.model, type="prompt"
            ).inc(result.usage.prompt_tokens)
            TOKENS_CONSUMED.labels(
                tenant=tenant.id, model=chat_request.model, type="completion"
            ).inc(result.usage.completion_tokens)

            # Record token usage for daily budget
            if tenant.token_budget_daily and rate_limiter:
                try:
                    await rate_limiter.record_tokens(tenant.id, result.usage.total_tokens)
                except Exception as e:
                    logger.warning("token_recording_failed", error=str(e))

            # Store in cache on miss
            if semantic_cache is not None:
                try:
                    await semantic_cache.store(
                        model=chat_request.model,
                        messages=chat_request.messages,
                        response=result,
                        tenant_id=tenant.id,
                        cache_isolation=cache_isolation,
                    )
                except Exception as e:
                    logger.warning("cache_store_failed", error=str(e))

            # Release stampede lock after cache store
            if lock_acquired and lock_key:
                try:
                    await semantic_cache.release_stampede_lock(lock_key)
                except Exception:
                    pass

            if journal is not None:
                await journal.record_completion(
                    request_id=request_id,
                    status=200,
                    latency_ms=duration_ms,
                    backend=backend.name,
                    cache_hit=False,
                    tokens_prompt=result.usage.prompt_tokens,
                    tokens_completion=result.usage.completion_tokens,
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

        finally:
            # Always release the slot for this attempt
            if queue_manager is not None and slot_backend:
                ACTIVE_REQUESTS.labels(backend=slot_backend).dec()
                await queue_manager.release_slot(slot_backend, chat_request.model)

    # All attempts exhausted or no backend available
    if journal is not None:
        await journal.record_completion(
            request_id=request_id,
            status=last_error.status_code if last_error else 404,
            latency_ms=round((time.perf_counter() - request_start_time) * 1000, 2),
            backend="",
            cache_hit=False,
            tokens_prompt=0,
            tokens_completion=0,
        )
    if last_error:
        raise last_error
    raise HTTPException(
        status_code=404,
        detail=f"No backend available for model: {chat_request.model}",
    )
