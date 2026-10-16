"""Respuesta SSE con detección de desconexión, backpressure y slow-consumer guard.

¿Por qué no usar StreamingResponse tal cual? Con ASGI spec >= 2.4 (uvicorn),
Starlette sólo detecta la desconexión cuando un `send` falla, es decir, recién
al emitir el próximo token. Si el LLM está "pensando" 20s, seguimos pagando esos
20s de generación para nadie. Además, al cortar por excepción no cierra el
generador de forma determinística (queda a merced del GC).

Este response:
  1. Escucha `http.disconnect` en paralelo -> cancela el stream al instante.
  2. Hace `send` con timeout: si el cliente no drena el socket en
     STREAM_STALL_TIMEOUT_S (slow consumer), se aborta.
  3. Siempre hace `aclose()` del generador -> el `finally` del generador cierra
     la conexión upstream con el LLM y encola la auditoría.

Backpressure: el generador es *pull-based* (no hay cola intermedia). Sólo se le
pide al LLM el próximo token cuando el anterior ya se entregó al transporte; si
el cliente lee lento, `send` espera el drain del socket, dejamos de leer del
upstream y el control de flujo TCP frena al proveedor. Memoria por stream: O(1 chunk).
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send


def sse_event(data: Any) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n".encode()


@dataclass
class StreamContext:
    """Estado compartido entre el response (transporte) y el generador (negocio)."""

    termination: str | None = None  # None | "client_disconnected" | "slow_consumer"


class SSEResponse(StreamingResponse):
    media_type = "text/event-stream"

    def __init__(self, content, ctx: StreamContext, stall_timeout_s: float, **kwargs):
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
        headers.update(kwargs.pop("headers", None) or {})
        super().__init__(content, headers=headers, **kwargs)
        self.ctx = ctx
        self.stall_timeout_s = stall_timeout_s

    async def _send_body(self, send: Send) -> None:
        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})
        async for chunk in self.body_iterator:
            try:
                await asyncio.wait_for(
                    send({"type": "http.response.body", "body": chunk, "more_body": True}),
                    timeout=self.stall_timeout_s,
                )
            except TimeoutError:
                self.ctx.termination = "slow_consumer"
                return
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _wait_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        body_task = asyncio.create_task(self._send_body(send))
        disconnect_task = asyncio.create_task(self._wait_disconnect(receive))
        try:
            done, _ = await asyncio.wait({body_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED)
            if disconnect_task in done and not body_task.done():
                self.ctx.termination = self.ctx.termination or "client_disconnected"
        finally:
            for task in (body_task, disconnect_task):
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(body_task, disconnect_task, return_exceptions=True)
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()  # determinístico: cancela upstream
        body_result = results[0]
        if isinstance(body_result, OSError):
            self.ctx.termination = self.ctx.termination or "client_disconnected"
        elif isinstance(body_result, Exception):
            raise body_result
