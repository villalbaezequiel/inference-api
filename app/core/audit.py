"""Auditoría de streams sin bloquear el response.

El handler SSE hace `put_nowait` (O(1), nunca espera) en una cola acotada.
Un worker en background (arrancado en el lifespan) drena la cola en lotes y
persiste en un Redis Stream (XADD con MAXLEN aproximado) + log.

Si la cola se llena (Redis caído mucho tiempo), se descarta el evento y se
loguea: preferimos perder auditoría a meter latencia o memoria ilimitada en el
camino del usuario. Para auditoría con garantía de entrega, ver DECISIONS.md
(outbox / Kafka).
"""

import asyncio
import json
import logging
from contextlib import suppress

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger("app.audit")


class AuditWriter:
    def __init__(self, redis: Redis, stream_key: str, maxlen: int, queue_size: int = 10_000):
        self.redis = redis
        self.stream_key = stream_key
        self.maxlen = maxlen
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task | None = None

    def submit(self, event: dict) -> None:
        try:
            self.queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.error("audit queue full, event dropped (request_id=%s)", event.get("request_id"))

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="audit-writer")

    async def stop(self, drain_timeout_s: float = 5.0) -> None:
        if self._task is None:
            return
        with suppress(TimeoutError):
            await asyncio.wait_for(self.queue.join(), timeout=drain_timeout_s)
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task

    async def _run(self) -> None:
        while True:
            batch = [await self.queue.get()]
            while not self.queue.empty() and len(batch) < 100:
                batch.append(self.queue.get_nowait())
            try:
                await self._persist(batch)
            except Exception:  # el worker nunca debe morir
                logger.exception("audit persist failed")
            finally:
                for _ in batch:
                    self.queue.task_done()

    async def _persist(self, batch: list[dict]) -> None:
        for event in batch:
            logger.info(
                "stream audit request_id=%s client_id=%s model=%s status=%s tokens_out=%s",
                event.get("request_id"),
                event.get("client_id"),
                event.get("model"),
                event.get("status"),
                event.get("tokens_out"),
            )
        try:
            async with self.redis.pipeline(transaction=False) as pipe:
                for event in batch:
                    pipe.xadd(
                        self.stream_key,
                        {"data": json.dumps(event, ensure_ascii=False)},
                        maxlen=self.maxlen,
                        approximate=True,
                    )
                await pipe.execute()
        except RedisError as exc:
            logger.warning("audit redis write failed: %r", exc)
