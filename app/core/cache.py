"""Wrapper de Redis: cache de inferencias + locks distribuidos.

Política ante fallas de Redis: **fail-open**. Si Redis no responde dentro de
REDIS_SOCKET_TIMEOUT_S, el cache se comporta como un miss y el request sigue
hacia el LLM. Perder el cache degrada costo/latencia; perder el servicio rompe el SLA.
"""

import hashlib
import json
import logging
import secrets
from typing import Any

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

_RELEASE_LOCK = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""

CACHE_PREFIX = "infer:v1:"
LOCK_PREFIX = "lock:infer:v1:"


def normalize_input(text: str) -> str:
    """Normalización conservadora: colapsa whitespace. NO cambia mayúsculas ni
    puntuación porque eso puede cambiar la semántica del prompt."""
    return " ".join(text.split())


def build_cache_key(model: str, prompt: str, config: dict[str, Any]) -> str:
    material = json.dumps(
        {"model": model, "input": normalize_input(prompt), "config": config},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode()).hexdigest()


def create_redis(url: str, socket_timeout: float, max_connections: int) -> Redis:
    return Redis.from_url(
        url,
        socket_timeout=socket_timeout,
        socket_connect_timeout=socket_timeout,
        max_connections=max_connections,
        decode_responses=True,
        health_check_interval=30,
        # Sin reintentos: redis-py por defecto reintenta con backoff exponencial
        # (segundos). Con Redis caído eso destruye el P95; preferimos fallar rápido
        # y aplicar fail-open en el caller.
        retry=Retry(NoBackoff(), 0),
    )


class InferenceCache:
    def __init__(self, redis: Redis, ttl_s: int):
        self.redis = redis
        self.ttl_s = ttl_s
        self._release_script = redis.register_script(_RELEASE_LOCK)

    async def get(self, key: str) -> dict | None:
        try:
            raw = await self.redis.get(CACHE_PREFIX + key)
        except RedisError as exc:
            logger.warning("cache get failed (fail-open): %r", exc)
            return None
        return json.loads(raw) if raw else None

    async def set(self, key: str, value: dict) -> None:
        try:
            await self.redis.set(CACHE_PREFIX + key, json.dumps(value), ex=self.ttl_s)
        except RedisError as exc:
            logger.warning("cache set failed: %r", exc)

    # ---------------------------------------------------------- locks
    async def try_lock(self, key: str, ttl_s: float) -> str | None:
        """SET NX PX. Devuelve el token si se obtuvo el lock, None si lo tiene otro.

        Lanza RedisError si Redis no está disponible (el caller decide degradar).
        """
        token = secrets.token_hex(8)
        ok = await self.redis.set(LOCK_PREFIX + key, token, nx=True, px=int(ttl_s * 1000))
        return token if ok else None

    async def lock_exists(self, key: str) -> bool:
        return bool(await self.redis.exists(LOCK_PREFIX + key))

    async def release_lock(self, key: str, token: str) -> None:
        try:
            await self._release_script(keys=[LOCK_PREFIX + key], args=[token])
        except RedisError as exc:
            logger.warning("lock release failed (expira por TTL): %r", exc)
