"""Acceso tipado a los singletons creados en el lifespan (app.state)."""

from fastapi import HTTPException, Request, status

from app.core.inference_service import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


def validate_model(request: Request, model: str) -> None:
    allowed = request.app.state.settings.allowed_models_set
    if model not in allowed:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail={"error": "unsupported_model", "message": f"model must be one of {sorted(allowed)}"},
        )
