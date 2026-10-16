"""Acceso tipado a los singletons creados en el lifespan (app.state)."""

from fastapi import HTTPException, Request, status

from app.core.audit import AuditWriter
from app.core.inference_service import InferenceService
from app.core.llm_client import LLMClient
from app.core.ml_model import ModelRegistry


def get_inference_service(request: Request) -> InferenceService:
    return request.app.state.inference_service


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_audit_writer(request: Request) -> AuditWriter:
    return request.app.state.audit_writer


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def validate_model(request: Request, model: str) -> None:
    allowed = request.app.state.settings.allowed_models_set
    if model not in allowed:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail={"error": "unsupported_model", "message": f"model must be one of {sorted(allowed)}"},
        )
