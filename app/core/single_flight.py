"""Request coalescing en proceso (Escenario A — thundering herd, nivel 1).

N corrutinas pidiendo la misma key al mismo tiempo comparten UNA sola ejecución.
La ejecución corre en su propia Task: si el cliente "líder" se desconecta, los
demás no se caen con él (asyncio.shield) y el resultado igual llega al cache.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class SingleFlight(Generic[T]):
    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Task[T]] = {}

    def _done(self, key: str, task: asyncio.Task[T]) -> None:
        self._inflight.pop(key, None)
        if not task.cancelled():
            task.exception()  # marca la excepción como "retrieved" si nadie esperaba

    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Ejecuta `fn` una sola vez por key concurrente. Devuelve (resultado, shared)."""
        task = self._inflight.get(key)
        shared = task is not None
        if task is None:
            task = asyncio.create_task(fn())
            self._inflight[key] = task
            task.add_done_callback(lambda t, k=key: self._done(k, t))
        return await asyncio.shield(task), shared
