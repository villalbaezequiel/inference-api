import asyncio
import time
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_audit_writer, get_llm_client, validate_model
from app.api.middleware.rate_limit import enforce_rate_limit
from app.api.sse import SSEResponse, StreamContext, sse_event
from app.core.audit import AuditWriter
from app.core.circuit_breaker import CircuitOpenError
from app.core.llm_client import LLMClient, LLMClientError, LLMError, LLMTimeoutError, approx_tokens
from app.schemas import ErrorResponse, InferRequest

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post(
    "/stream",
    response_class=SSEResponse,
    # Explícito: FastAPI lo infiere de la firma de __init__ del response y SSEResponse no lo
    # declara -> sin esto /openapi.json falla.
    status_code=200,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Stream SSE de tokens"},
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def stream(
    body: InferRequest,
    request: Request,
    client_id: str = Depends(enforce_rate_limit),
    llm: LLMClient = Depends(get_llm_client),
    audit: AuditWriter = Depends(get_audit_writer),
) -> SSEResponse:
    validate_model(request, body.model)
    request.state.log_extra = {"model": body.model, "cache_hit": False}
    settings = request.app.state.settings
    timeout_s = min(body.config.timeout_s or settings.llm_default_timeout_s, settings.llm_max_timeout_s)
    request_id = request.state.request_id
    ctx = StreamContext()

    async def events():
        tokens: list[str] = []
        status = "success"
        error: str | None = None
        started = time.perf_counter()
        upstream = llm.stream(body.model, body.input, body.config, timeout_s)
        try:
            index = 0
            async for token in upstream:
                tokens.append(token)
                yield sse_event({"token": token, "index": index})
                index += 1
            yield sse_event("[DONE]")
        except CircuitOpenError as exc:
            status, error = "error", "llm_unavailable"
            yield sse_event({"error": "llm_unavailable", "retry_after_s": exc.retry_after_s})
        except LLMTimeoutError:
            status, error = "timeout", "llm_timeout"
            yield sse_event({"error": "llm_timeout", "timeout_s": timeout_s})
        except LLMClientError as exc:
            status, error = "error", "llm_rejected_request"
            yield sse_event({"error": "llm_rejected_request", "message": str(exc)})
        except LLMError:
            status, error = "error", "llm_error"
            yield sse_event({"error": "llm_error"})
        except (asyncio.CancelledError, GeneratorExit):
            status = "cancelled"
            raise
        finally:
            await upstream.aclose()
            if ctx.termination:
                status = "cancelled"
                error = ctx.termination
            request.state.log_extra.update(status=status, tokens_out=len(tokens), error=error)
            # Auditoría: put_nowait en cola acotada -> nunca bloquea el response.
            audit.submit(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "request_id": request_id,
                    "client_id": client_id,
                    "model": body.model,
                    "input": body.input,
                    "output": "".join(tokens),
                    "tokens_in": approx_tokens(body.input),
                    "tokens_out": len(tokens),
                    "status": status,
                    "termination": ctx.termination,
                    "error": error,
                    "latency_ms": round((time.perf_counter() - started) * 1000),
                }
            )

    return SSEResponse(events(), ctx=ctx, stall_timeout_s=settings.stream_stall_timeout_s)
