"""R1 — /v1/infer: auth, rate limiting, cache, respuesta, thundering herd, degradación."""

import asyncio
import uuid

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.core.cache import CACHE_PREFIX, InferenceCache, build_cache_key, create_redis
from app.core.circuit_breaker import CircuitBreaker
from app.core.inference_service import InferenceService
from app.core.llm_client import LLMClient, MockProvider
from app.main import create_app
from app.schemas import InferRequest
from tests.conftest import HEADERS, OTHER_KEY, infer_body


@pytest.mark.parametrize(
    ("headers", "error"),
    [({}, "missing_api_key"), ({"X-API-Key": "nope"}, "invalid_api_key")],
)
async def test_auth_required(client, headers, error):
    r = await client.post("/v1/infer", json=infer_body(), headers=headers)
    assert r.status_code == 401
    assert r.json() == {"error": error}


async def test_infer_response_shape(client):
    r = await client.post("/v1/infer", json=infer_body(), headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    uuid.UUID(data["request_id"])
    assert r.headers["x-request-id"] == data["request_id"]
    assert data["model"] == "gpt-4o"
    assert data["output"]
    assert data["usage"]["tokens_in"] > 0 and data["usage"]["tokens_out"] > 0
    assert isinstance(data["latency_ms"], int) and data["latency_ms"] >= 0
    assert data["cache_hit"] is False


async def test_second_identical_request_is_cache_hit(client, provider):
    first = await client.post("/v1/infer", json=infer_body(), headers=HEADERS)
    second = await client.post("/v1/infer", json=infer_body(), headers=HEADERS)
    assert second.json()["cache_hit"] is True
    assert second.json()["output"] == first.json()["output"]
    assert provider.calls == 1


async def test_cache_ttl_is_configurable(make_client, redis):
    client = await make_client(cache_ttl_s=42)
    await client.post("/v1/infer", json=infer_body(), headers=HEADERS)
    key = CACHE_PREFIX + build_cache_key("gpt-4o", "hola mundo", {"temperature": 0.7, "max_tokens": 256})
    assert 0 < await redis.ttl(key) <= 42


async def test_rate_limit_returns_429_with_retry_after(make_client):
    client = await make_client(rate_limit_per_minute=3)
    for i in range(3):
        assert (await client.post("/v1/infer", json=infer_body(f"req {i}"), headers=HEADERS)).status_code == 200
    r = await client.post("/v1/infer", json=infer_body("req 4"), headers=HEADERS)
    assert r.status_code == 429
    body = r.json()
    assert body["error"] == "rate_limit_exceeded"
    assert 1 <= body["retry_after_s"] <= 60
    assert r.headers["retry-after"] == str(body["retry_after_s"])


async def test_rate_limit_is_per_client(make_client):
    client = await make_client(rate_limit_per_minute=1)
    await client.post("/v1/infer", json=infer_body(), headers=HEADERS)
    assert (await client.post("/v1/infer", json=infer_body(), headers=HEADERS)).status_code == 429
    assert (await client.post("/v1/infer", json=infer_body(), headers={"X-API-Key": OTHER_KEY})).status_code == 200


async def test_thundering_herd_single_llm_call(client, provider):
    responses = await asyncio.gather(
        *(client.post("/v1/infer", json=infer_body("herd"), headers=HEADERS) for _ in range(100))
    )
    assert all(r.status_code == 200 for r in responses)
    assert len({r.json()["output"] for r in responses}) == 1
    assert provider.calls == 1


async def test_thundering_herd_across_replicas(redis):
    """Tres réplicas (SingleFlight distinto) comparten Redis: una sola llamada al LLM."""
    provider = MockProvider(latency_ms=100)
    llm = LLMClient(provider, CircuitBreaker())
    replicas = [
        InferenceService(llm, InferenceCache(redis, 60), 5, 10, lock_ttl_s=5, follower_wait_s=5) for _ in range(3)
    ]
    req = InferRequest(model="gpt-4o", input="multi replica")
    results = await asyncio.gather(*(replicas[i % 3].infer(req) for i in range(30)))
    assert provider.calls == 1
    assert len({r.completion.text for r in results}) == 1


async def test_redis_down_fails_open(settings, provider):
    dead = create_redis("redis://127.0.0.1:1/0", socket_timeout=0.05, max_connections=10)
    app = create_app(settings=settings, redis=dead, provider=provider)
    async with (
        LifespanManager(app) as manager,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app), base_url="http://t") as c,
    ):
        r = await c.post("/v1/infer", json=infer_body(), headers=HEADERS)
        assert r.status_code == 200
