import json
import logging

import httpx
from asgi_lifespan import LifespanManager

from app.core.cache import create_redis
from app.logging_config import JsonFormatter
from app.main import create_app
from tests.conftest import HEADERS, infer_body


async def test_health_ok(client):
    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "redis": "ok", "llm_circuit": "closed", "models_loaded": ["v1", "v2"]}


async def test_health_degraded_when_redis_down(settings, provider):
    dead = create_redis("redis://127.0.0.1:1/0", socket_timeout=0.05, max_connections=10)
    app = create_app(settings=settings, redis=dead, provider=provider)
    async with (
        LifespanManager(app) as manager,
        httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app), base_url="http://t") as c,
    ):
        r = await c.get("/health")
        # Degradado pero 200: la API sigue sirviendo con fail-open.
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"
        assert r.json()["redis"] == "error"


async def test_structured_log_has_required_fields(client, caplog):
    caplog.set_level(logging.INFO, logger="app.access")
    r = await client.post("/v1/infer", json=infer_body("log me"), headers=HEADERS)
    record = next(rec for rec in caplog.records if rec.name == "app.access")

    payload = json.loads(JsonFormatter().format(record))
    for field in ("timestamp", "request_id", "client_id", "model", "latency_ms", "cache_hit", "status"):
        assert field in payload, field
    assert payload["request_id"] == r.json()["request_id"]
    assert payload["client_id"] == "client-test"
    assert payload["model"] == "gpt-4o"
    assert payload["status"] == "success"


async def test_structured_log_marks_timeouts(client, caplog):
    caplog.set_level(logging.INFO, logger="app.access")
    r = await client.post("/v1/infer", json=infer_body("__slow__", timeout_s=0.02), headers=HEADERS)
    assert r.status_code == 504
    record = next(rec for rec in caplog.records if rec.name == "app.access")
    assert record.status == "timeout"
