import time

from fastapi import APIRouter, Depends, Header

from app.api.deps import get_model_registry
from app.api.middleware.auth import require_api_key
from app.core.ml_model import ModelRegistry
from app.schemas import ClassifyRequest, ClassifyResponse, ErrorResponse

router = APIRouter(prefix="/v1", tags=["classification"])


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    dependencies=[Depends(require_api_key)],
    responses={
        400: {"model": ErrorResponse, "description": "Versión de modelo inexistente"},
        401: {"model": ErrorResponse},
    },
)
async def classify(
    body: ClassifyRequest,
    x_model_version: str | None = Header(default=None, alias="X-Model-Version"),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ClassifyResponse:
    start = time.perf_counter()
    version = registry.resolve(x_model_version)  # 400 si no existe (exception handler)
    label, confidence = await registry.classify(body.input, version)
    return ClassifyResponse(
        label=label,
        confidence=round(confidence, 4),
        model_version=version,
        latency_ms=round((time.perf_counter() - start) * 1000),
    )
