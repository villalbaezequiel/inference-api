"""Punto de entrada. `create_app()` permite inyectar dependencias en tests."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.api.middleware.rate_limit import RateLimiter, RateLimitExceeded
from app.api.middleware.request_context import RequestContextMiddleware
from app.api.routes import health, infer, stream
from app.config import Settings, get_settings
from app.core.audit import AuditWriter
from app.core.cache import InferenceCache, create_redis
from app.core.circuit_breaker import CircuitBreaker, CircuitOpenError
from app.core.inference_service import InferenceService
from app.core.llm_client import (
    LLMClient,
    LLMClientError,
    LLMProvider,
    LLMTimeoutError,
    LLMUpstreamError,
    build_provider,
)


def create_app(
    settings: Settings | None = None,
    redis: Redis | None = None,
    provider: LLMProvider | None = None,
    breaker: CircuitBreaker | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Todo lo pesado ocurre UNA vez acá, nunca por request.
        r = redis or create_redis(settings.redis_url, settings.redis_socket_timeout_s, settings.redis_max_connections)
        cb = breaker or CircuitBreaker(
            settings.cb_failure_threshold, settings.cb_failure_window_s, settings.cb_open_duration_s
        )
        llm = LLMClient(provider or build_provider(settings), cb)
        audit = AuditWriter(r, settings.audit_stream_key, settings.audit_stream_maxlen, settings.audit_queue_size)
        await audit.start()

        app.state.settings = settings
        app.state.llm_client = llm
        app.state.audit_writer = audit
        app.state.rate_limiter = RateLimiter(r, settings.rate_limit_per_minute, fail_open=settings.rate_limit_fail_open)
        app.state.inference_service = InferenceService(
            llm,
            InferenceCache(r, settings.cache_ttl_s),
            settings.llm_default_timeout_s,
            settings.llm_max_timeout_s,
            settings.single_flight_lock_ttl_s,
            settings.single_flight_wait_s,
        )
        try:
            yield
        finally:
            await audit.stop()
            await llm.aclose()
            if redis is None:
                await r.aclose()

    app = FastAPI(title="AI Inference API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RequestContextMiddleware)
    app.include_router(infer.router)
    app.include_router(stream.router)
    app.include_router(health.router)
    _register_exception_handlers(app)
    return app


def _error(status: int, error: str, message: str | None = None, retry_after_s: int | None = None, headers=None):
    body: dict = {"error": error}
    if message:
        body["message"] = message
    if retry_after_s is not None:
        body["retry_after_s"] = retry_after_s
    return JSONResponse(body, status_code=status, headers=headers)


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RateLimitExceeded)
    async def _rate_limited(_: Request, exc: RateLimitExceeded):
        return _error(
            429, "rate_limit_exceeded", retry_after_s=exc.retry_after_s, headers={"Retry-After": str(exc.retry_after_s)}
        )

    @app.exception_handler(CircuitOpenError)
    async def _circuit_open(_: Request, exc: CircuitOpenError):
        return _error(
            503, "llm_unavailable", retry_after_s=exc.retry_after_s, headers={"Retry-After": str(exc.retry_after_s)}
        )

    @app.exception_handler(LLMTimeoutError)
    async def _timeout(_: Request, exc: LLMTimeoutError):
        return _error(504, "llm_timeout", str(exc))

    @app.exception_handler(LLMUpstreamError)
    async def _upstream(_: Request, exc: LLMUpstreamError):
        return _error(502, "llm_upstream_error", "the LLM provider failed")

    @app.exception_handler(LLMClientError)
    async def _llm_client(_: Request, exc: LLMClientError):
        return _error(400, "llm_rejected_request", str(exc))

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        if isinstance(exc.detail, dict):
            return JSONResponse(exc.detail, status_code=exc.status_code, headers=exc.headers)
        return _error(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse({"error": "invalid_request", "details": details}, status_code=422)
