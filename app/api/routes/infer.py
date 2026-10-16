import time

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_inference_service, validate_model
from app.api.middleware.rate_limit import enforce_rate_limit
from app.core.inference_service import InferenceService
from app.schemas import ErrorResponse, InferRequest, InferResponse, Usage

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post(
    "/infer",
    response_model=InferResponse,
    dependencies=[Depends(enforce_rate_limit)],
    responses={
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse, "description": "Circuit breaker abierto"},
        502: {"model": ErrorResponse, "description": "Error del proveedor LLM"},
        504: {"model": ErrorResponse, "description": "Timeout del proveedor LLM"},
    },
)
async def infer(
    body: InferRequest,
    request: Request,
    service: InferenceService = Depends(get_inference_service),
) -> InferResponse:
    start = time.perf_counter()
    validate_model(request, body.model)
    request.state.log_extra = {"model": body.model}

    result = await service.infer(body)

    request.state.log_extra.update(
        cache_hit=result.cache_hit,
        tokens_in=result.completion.tokens_in,
        tokens_out=result.completion.tokens_out,
    )
    return InferResponse(
        request_id=request.state.request_id,
        output=result.completion.text,
        model=body.model,
        usage=Usage(tokens_in=result.completion.tokens_in, tokens_out=result.completion.tokens_out),
        latency_ms=round((time.perf_counter() - start) * 1000),
        cache_hit=result.cache_hit,
    )
