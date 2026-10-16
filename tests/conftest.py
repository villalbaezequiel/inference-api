import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import fakeredis
import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.config import Settings
from app.core.circuit_breaker import CircuitBreaker
from app.core.llm_client import MockProvider
from app.core.train import train
from app.main import create_app

API_KEY = "test-key"
OTHER_KEY = "other-key"
HEADERS = {"X-API-Key": API_KEY}


class FakeClock:
    """Reloj controlable para el circuit breaker: tests deterministas sin sleep."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TrackingProvider(MockProvider):
    """MockProvider que registra cuántos tokens generó y si el stream se cerró."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokens_generated = 0
        self.stream_closed = asyncio.Event()

    async def stream(self, model, prompt, config):
        try:
            async for token in super().stream(model, prompt, config):
                self.tokens_generated += 1
                yield token
        finally:
            self.stream_closed.set()


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models")
    train(path, ["v1", "v2"])
    return path


@pytest.fixture
def settings(models_dir) -> Settings:
    return Settings(
        api_keys=f"{API_KEY}:client-test,{OTHER_KEY}:client-other",
        rate_limit_per_minute=1000,
        models_dir=str(models_dir),
        mock_llm_latency_ms=20,
        mock_llm_token_delay_ms=1,
        llm_default_timeout_s=5,
        stream_stall_timeout_s=2,
        log_level="WARNING",
    )


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def provider() -> TrackingProvider:
    return TrackingProvider(latency_ms=20, token_delay_ms=1)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def breaker(clock) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=5, failure_window_s=30, open_duration_s=60, clock=clock)


@pytest.fixture
def app_factory(settings, redis, provider, breaker):
    def _make(**overrides):
        cfg = settings.model_copy(update=overrides)
        return create_app(settings=cfg, redis=redis, provider=provider, breaker=breaker)

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
