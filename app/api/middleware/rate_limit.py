"""Rate limiting por cliente: ventana deslizante (sliding log) sobre Redis.

Por qué sliding log y no fixed window: con ventana fija un cliente puede hacer
60 req a las 12:00:59 y otras 60 a las 12:01:00 (120 en 1 segundo). El sliding
log garantiza "como máximo N en cualquier ventana de 60s".

Todo ocurre en un único script Lua -> atómico entre réplicas, 1 round-trip.
Costo: O(N) memoria por cliente (N=60), irrelevante a esta escala.
"""

import logging
import math
import time
import uuid

from fastapi import Depends, Request
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.api.middleware.auth import require_api_key

logger = logging.getLogger(__name__)

# KEYS[1] = clave del cliente
# ARGV = now_ms, window_ms, limit, member
# Devuelve {allowed(0|1), retry_after_ms}
_SLIDING_WINDOW = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
local count = redis.call('ZCARD', key)
if count < limit then
    redis.call('ZADD', key, now, ARGV[4])
    redis.call('PEXPIRE', key, window)
    return {1, 0}
end
local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
local retry = window - (now - tonumber(oldest[2]))
return {0, retry}
"""


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_s: int):
        self.retry_after_s = retry_after_s


class RateLimiter:
    def __init__(self, redis: Redis, limit: int, window_s: int = 60, fail_open: bool = True):
        self.redis = redis
        self.limit = limit
        self.window_ms = window_s * 1000
        self.fail_open = fail_open
        self._script = redis.register_script(_SLIDING_WINDOW)

    async def hit(self, client_id: str) -> tuple[bool, int]:
        """Registra un request. Devuelve (allowed, retry_after_s)."""
        now_ms = int(time.time() * 1000)
        try:
            allowed, retry_ms = await self._script(
                keys=[f"ratelimit:{client_id}"],
                args=[now_ms, self.window_ms, self.limit, f"{now_ms}-{uuid.uuid4().hex[:8]}"],
            )
        except RedisError as exc:
            logger.error("rate limiter unavailable: %r", exc)
            if self.fail_open:
                return True, 0
            return False, 1
        if allowed:
            return True, 0
        return False, max(1, math.ceil(int(retry_ms) / 1000))


async def enforce_rate_limit(request: Request, client_id: str = Depends(require_api_key)) -> str:
    limiter: RateLimiter = request.app.state.rate_limiter
    allowed, retry_after_s = await limiter.hit(client_id)
    if not allowed:
        raise RateLimitExceeded(retry_after_s)
    return client_id
