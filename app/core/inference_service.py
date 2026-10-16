"""Orquestación de /v1/infer: cache -> single-flight (local + distribuido) -> LLM.

Flujo ante un cache miss (Escenario A — thundering herd):

  1. Coalescing en proceso: todas las corrutinas del mismo worker con la misma
     key esperan UNA sola Task.
  2. Lock distribuido en Redis (SET NX PX): entre réplicas, sólo el dueño del lock
     llama al LLM. El resto hace polling del cache con backoff.
  3. Si el líder muere sin escribir el cache, el lock expira / se libera y un
     follower toma el liderazgo. Si Redis no está, se degrada a llamar directo.
"""

import asyncio
import logging
from dataclasses import dataclass

from redis.exceptions import RedisError

from app.core.cache import InferenceCache, build_cache_key
from app.core.llm_client import Completion, LLMClient, LLMTimeoutError
from app.core.single_flight import SingleFlight
from app.schemas import InferRequest

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceResult:
    completion: Completion
    cache_hit: bool


class InferenceService:
    def __init__(
        self,
        llm: LLMClient,
        cache: InferenceCache,
        default_timeout_s: float,
        max_timeout_s: float,
        lock_ttl_s: float,
        follower_wait_s: float,
    ):
        self.llm = llm
        self.cache = cache
        self.default_timeout_s = default_timeout_s
        self.max_timeout_s = max_timeout_s
        self.lock_ttl_s = lock_ttl_s
        self.follower_wait_s = follower_wait_s
        self._flight: SingleFlight[Completion] = SingleFlight()

    def timeout_for(self, req: InferRequest) -> float:
        return min(req.config.timeout_s or self.default_timeout_s, self.max_timeout_s)

    async def infer(self, req: InferRequest) -> InferenceResult:
        key = build_cache_key(req.model, req.input, req.config.cache_relevant())

        cached = await self.cache.get(key)
        if cached is not None:
            return InferenceResult(Completion(**cached), cache_hit=True)

        timeout_s = self.timeout_for(req)
        try:
            # El deadline "real" lo aplica LLMClient (y cuenta para el breaker). Este es una red de seguridad para
            # quien espera un resultado compartido, con un pequeño margen para que el
            # timeout interno gane la carrera.
            async with asyncio.timeout(timeout_s + 0.25):
                completion, _ = await self._flight.do(key, lambda: self._fill(key, req, timeout_s))
        except TimeoutError as exc:
            raise LLMTimeoutError(f"no answer within {timeout_s}s") from exc

        # Un resultado compartido o traído por otra réplica no es "cache hit" en
        # sentido estricto para ESTE request (no lo pagó el cache, lo pagó el líder),
        # pero tampoco generó una llamada al LLM. Lo reportamos como hit=False.
        return InferenceResult(completion, cache_hit=False)

    # ------------------------------------------------------------------
    async def _fill(self, key: str, req: InferRequest, timeout_s: float) -> Completion:
        """Corre una sola vez por key y por proceso."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + min(timeout_s, self.follower_wait_s)
        backoff = 0.02

        while True:
            try:
                token = await self.cache.try_lock(key, self.lock_ttl_s)
            except RedisError as exc:
                logger.warning("lock unavailable, calling LLM directly: %r", exc)
                return await self._call_and_store(key, req, timeout_s)

            if token is not None:  # somos líderes entre réplicas
                try:
                    # Double-check: otra réplica pudo haber llenado el cache justo antes.
                    cached = await self.cache.get(key)
                    if cached is not None:
                        return Completion(**cached)
                    return await self._call_and_store(key, req, timeout_s)
                finally:
                    await self.cache.release_lock(key, token)

            # Follower: esperar a que el líder (otra réplica) escriba el cache.
            while loop.time() < deadline:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 0.2)
                cached = await self.cache.get(key)
                if cached is not None:
                    return Completion(**cached)
                try:
                    if not await self.cache.lock_exists(key):
                        break  # el líder terminó/murió sin cachear -> reintentar liderazgo
                except RedisError:
                    return await self._call_and_store(key, req, timeout_s)
            else:
                logger.warning("single-flight wait exceeded, calling LLM directly")
                return await self._call_and_store(key, req, timeout_s)

    async def _call_and_store(self, key: str, req: InferRequest, timeout_s: float) -> Completion:
        completion = await self.llm.complete(req.model, req.input, req.config, timeout_s)
        await self.cache.set(
            key,
            {"text": completion.text, "tokens_in": completion.tokens_in, "tokens_out": completion.tokens_out},
        )
        return completion
