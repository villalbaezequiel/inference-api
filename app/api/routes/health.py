import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> JSONResponse:
    state = request.app.state
    try:
        async with asyncio.timeout(0.5):
            await state.redis.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "error"

    circuit = state.llm_client.breaker.state.value
    models = state.model_registry.loaded_versions
    expected = set(state.settings.model_versions_list)

    # Redis caído o circuito abierto => "degraded" pero 200: la API sigue sirviendo
    # (fail-open de cache/rate-limit, /classify no depende del LLM). Sólo si faltan
    # modelos el pod no está listo (503) y el orquestador no debe rutearle tráfico.
    degraded = redis_status != "ok" or circuit != "closed"
    body = HealthResponse(
        status="degraded" if degraded else "ok",
        redis=redis_status,
        llm_circuit=circuit,
        models_loaded=models,
    )
    status_code = 200 if expected.issubset(models) else 503
    return JSONResponse(body.model_dump(), status_code=status_code)
