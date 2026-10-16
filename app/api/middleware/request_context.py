"""Middleware ASGI puro: asigna request_id, lo devuelve en X-Request-ID y loguea cada request.

Es ASGI puro (no BaseHTTPMiddleware) porque BaseHTTPMiddleware bufferiza y rompe
la propagación de desconexión en respuestas streaming (SSE, R2).
"""

import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging_config import client_id_var, request_id_var

logger = logging.getLogger("app.access")

# Sondeados cada pocos segundos por el orquestador: loguearlos es ruido.
_SILENT_PATHS = {"/health"}


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode()[:64]
        request_id = incoming or str(uuid.uuid4())
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        rid_token = request_id_var.set(request_id)
        cid_token = client_id_var.set(None)

        start = time.perf_counter()
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message["headers"] = list(message.get("headers") or []) + [(b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if scope["path"] not in _SILENT_PATHS:
                # Las rutas completan model, cache_hit, etc. en request.state.log_extra.
                extra = dict(state.get("log_extra") or {})
                extra.setdefault(
                    "status",
                    "success" if status_code < 400 else ("timeout" if status_code == 504 else "error"),
                )
                route = scope.get("route")
                logger.info(
                    "request completed",
                    extra={
                        "request_id": request_id,
                        "client_id": state.get("client_id"),
                        "endpoint": getattr(route, "path", scope["path"]),
                        "http_status": status_code,
                        "latency_ms": round((time.perf_counter() - start) * 1000),
                        **extra,
                    },
                )
            request_id_var.reset(rid_token)
            client_id_var.reset(cid_token)
