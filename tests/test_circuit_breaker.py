"""R2 — Circuit breaker: 5 fallos en 30s -> OPEN 60s -> HALF_OPEN -> CLOSED/OPEN."""

import asyncio

import pytest

from app.core.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState
from app.core.llm_client import LLMClientError
from tests.conftest import HEADERS, FakeClock, infer_body


@pytest.fixture
def cb(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(failure_threshold=5, failure_window_s=30, open_duration_s=60, clock=clock)


def fail(cb: CircuitBreaker, n: int = 1) -> None:
    for _ in range(n):
        probe = cb.acquire()
        cb.record_failure(probe)


# ------------------------------------------------------------ unit
def test_four_failures_keep_circuit_closed(cb):
    fail(cb, 4)
    assert cb.state is CircuitState.CLOSED


def test_five_failures_within_window_open_circuit(cb, clock):
    for _ in range(5):
        fail(cb)
        clock.advance(5)  # 5 fallos en 20s
    assert cb.state is CircuitState.OPEN


def test_failures_outside_window_do_not_count(cb, clock):
    fail(cb, 4)
    clock.advance(31)  # los 4 anteriores salen de la ventana
    fail(cb)
    assert cb.state is CircuitState.CLOSED


def test_open_circuit_rejects_with_retry_after(cb, clock):
    fail(cb, 5)
    clock.advance(15)
    with pytest.raises(CircuitOpenError) as exc:
        cb.acquire()
    assert exc.value.retry_after_s == 45


def test_half_open_after_open_duration(cb, clock):
    fail(cb, 5)
    clock.advance(59.9)
    assert cb.state is CircuitState.OPEN
    clock.advance(0.1)
    assert cb.state is CircuitState.HALF_OPEN


def test_half_open_allows_single_probe(cb, clock):
    fail(cb, 5)
    clock.advance(60)
    assert cb.acquire() is True  # probe
    with pytest.raises(CircuitOpenError):
        cb.acquire()  # segunda llamada concurrente: rechazada


def test_probe_success_closes_circuit_and_resets_failures(cb, clock):
    fail(cb, 5)
    clock.advance(60)
    cb.record_success(cb.acquire())
    assert cb.state is CircuitState.CLOSED
    fail(cb, 4)
    assert cb.state is CircuitState.CLOSED  # el historial se limpió


def test_probe_failure_reopens_circuit(cb, clock):
    fail(cb, 5)
    clock.advance(60)
    cb.record_failure(cb.acquire())
    assert cb.state is CircuitState.OPEN
    assert cb.retry_after_s() == 60


async def test_guard_ignores_cancellation_and_client_errors(cb):
    for _ in range(10):
        with pytest.raises(asyncio.CancelledError):
            async with cb.guard():
                raise asyncio.CancelledError
        with pytest.raises(LLMClientError):
            async with cb.guard(ignore=(LLMClientError,)):
                raise LLMClientError("bad request")
    assert cb.state is CircuitState.CLOSED


async def test_cancelled_probe_releases_half_open_slot(cb, clock):
    fail(cb, 5)
    clock.advance(60)
    with pytest.raises(asyncio.CancelledError):
        async with cb.guard():
            raise asyncio.CancelledError
    assert cb.state is CircuitState.HALF_OPEN
    assert cb.acquire() is True  # otro request puede ser la nueva prueba


# ----------------------------------------------------- integración HTTP
async def test_circuit_opens_after_five_upstream_failures(client, provider, clock):
    for i in range(5):
        r = await client.post("/v1/infer", json=infer_body(f"__fail__ {i}"), headers=HEADERS)
        assert r.status_code == 502
    calls_before = provider.calls

    r = await client.post("/v1/infer", json=infer_body("healthy prompt"), headers=HEADERS)
    assert r.status_code == 503
    assert r.json() == {"error": "llm_unavailable", "retry_after_s": 60}
    assert r.headers["retry-after"] == "60"
    assert provider.calls == calls_before  # fail-fast: no se tocó el proveedor

    health = (await client.get("/health")).json()
    assert health["llm_circuit"] == "open"
    assert health["status"] == "degraded"


async def test_cache_hits_are_served_while_circuit_is_open(client, clock):
    await client.post("/v1/infer", json=infer_body("cached prompt"), headers=HEADERS)
    for i in range(5):
        await client.post("/v1/infer", json=infer_body(f"__fail__ {i}"), headers=HEADERS)
    r = await client.post("/v1/infer", json=infer_body("cached prompt"), headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["cache_hit"] is True


async def test_circuit_recovers_through_half_open(client, clock):
    for i in range(5):
        await client.post("/v1/infer", json=infer_body(f"__fail__ {i}"), headers=HEADERS)
    clock.advance(60)
    assert (await client.get("/health")).json()["llm_circuit"] == "half-open"
    r = await client.post("/v1/infer", json=infer_body("recovered"), headers=HEADERS)
    assert r.status_code == 200
    assert (await client.get("/health")).json()["llm_circuit"] == "closed"


async def test_timeouts_count_as_failures(client):
    for i in range(5):
        r = await client.post("/v1/infer", json=infer_body(f"__slow__ {i}", timeout_s=0.02), headers=HEADERS)
        assert r.status_code == 504
    assert (await client.get("/health")).json()["llm_circuit"] == "open"


async def test_open_circuit_in_stream_emits_sse_error(client, clock):
    for i in range(5):
        await client.post("/v1/infer", json=infer_body(f"__fail__ {i}"), headers=HEADERS)
    clock.advance(15)
    r = await client.post("/v1/stream", json=infer_body("stream me"), headers=HEADERS)
    assert r.status_code == 200
    assert 'data: {"error": "llm_unavailable", "retry_after_s": 45}' in r.text
