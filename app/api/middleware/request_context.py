"""Middleware ASGI puro: asigna request_id y lo devuelve en el header X-Request-ID.

Es ASGI puro (no BaseHTTPMiddleware) porque BaseHTTPMiddleware bufferiza y rompe
la propagación de desconexión en respuestas streaming (SSE, R2).
"""

import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send


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
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = list(message.get("headers") or []) + [(b"x-request-id", request_id.encode())]
            await send(message)

        await self.app(scope, receive, send_wrapper)
