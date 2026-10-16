from collections.abc import AsyncIterator

import fakeredis
import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.core.llm_client import MockProvider
from app.main import create_app

API_KEY = "test-key"
OTHER_KEY = "other-key"
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_keys=f"{API_KEY}:client-test,{OTHER_KEY}:client-other",
        rate_limit_per_minute=1000,
        mock_llm_latency_ms=20,
        llm_timeout_s=5,
    )


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def provider() -> MockProvider:
    return MockProvider(latency_ms=20)


@pytest.fixture
def app_factory(settings, redis, provider):
    def _make(**overrides):
        cfg = settings.model_copy(update=overrides)
        return create_app(settings=cfg, redis=redis, provider=provider)

    return _make


@pytest.fixture
async def make_client(app_factory):
    managers: list[LifespanManager] = []
    clients: list[httpx.AsyncClient] = []

    async def _make(**overrides) -> httpx.AsyncClient:
        app = app_factory(**overrides)
        manager = LifespanManager(app)
        await manager.__aenter__()
        managers.append(manager)
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app), base_url="http://test")
        clients.append(client)
        return client

    yield _make
    for client in clients:
        await client.aclose()
    for manager in managers:
        await manager.__aexit__(None, None, None)


@pytest.fixture
async def client(make_client) -> AsyncIterator[httpx.AsyncClient]:
    yield await make_client()


def infer_body(text: str = "hola mundo", model: str = "gpt-4o", **config) -> dict:
    return {"model": model, "input": text, "config": {"temperature": 0.7, **config}}
