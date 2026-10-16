"""Logs estructurados en JSON con contexto por request (contextvars)."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
client_id_var: ContextVar[str | None] = ContextVar("client_id", default=None)

# Campos extra que se aceptan vía `logger.info("...", extra={...})`
_EXTRA_FIELDS = (
    "model",
    "model_version",
    "latency_ms",
    "cache_hit",
    "status",
    "endpoint",
    "http_status",
    "error",
    "tokens_in",
    "tokens_out",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "client_id": getattr(record, "client_id", None) or client_id_var.get(),
        }
        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # uvicorn trae sus propios handlers de texto plano: los unificamos.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    # El access log de uvicorn duplica nuestro log de request; lo silenciamos.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
