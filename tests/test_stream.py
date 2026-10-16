"""R2 — Streaming SSE: formato, cancelación upstream, slow consumer, timeout, auditoría."""

import asyncio
import json

import pytest
from asgi_lifespan import LifespanManager

from tests.conftest import API_KEY, HEADERS, infer_body


def parse_sse(text: str) -> list:
    events = []
    for block in text.strip().split("\n\n"):
        assert block.startswith("data: ")
        payload = block[len("data: ") :]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


async def read_audit(redis, request_id: str) -> dict:
    deadline = asyncio.get_running_loop().time() + 2.0
    while asyncio.get_running_loop().time() < deadline:
        for _, fields in await redis.xrange("audit:streams"):
            event = json.loads(fields["data"])
            if event["request_id"] == request_id:
                return event
        await asyncio.sleep(0.02)
    raise AssertionError("audit event not found")


async def test_stream_format(client):
    r = await client.post("/v1/stream", json=infer_body("hola"), headers=HEADERS)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"
    events = parse_sse(r.text)
    assert events[-1] == "[DONE]"
    tokens = events[:-1]
    assert [e["index"] for e in tokens] == list(range(len(tokens)))
    assert all(isinstance(e["token"], str) for e in tokens)


async def test_stream_requires_api_key(client):
    r = await client.post("/v1/stream", json=infer_body())
    assert r.status_code == 401


async def test_stream_is_audited_in_background(client, redis):
    r = await client.post("/v1/stream", json=infer_body("audit me"), headers=HEADERS)
    tokens = [e["token"] for e in parse_sse(r.text)[:-1]]
    event = await read_audit(redis, r.headers["x-request-id"])
    assert event["status"] == "success"
    assert event["output"] == "".join(tokens)
    assert event["input"] == "audit me"
    assert event["client_id"] == "client-test"


async def test_stream_timeout_emits_error_event(client):
    r = await client.post("/v1/stream", json=infer_body("__slow__", timeout_s=0.05), headers=HEADERS)
    events = parse_sse(r.text)
    assert {"error": "llm_timeout", "timeout_s": 0.05} in events


# ---------------------------------------------------- ASGI de bajo nivel
def _scope(body: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/stream",
        "raw_path": b"/v1/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"test"),
            (b"content-type", b"application/json"),
            (b"x-api-key", API_KEY.encode()),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }


@pytest.fixture
async def running_app(app_factory):
    def _start(**overrides):
        return LifespanManager(app_factory(**overrides))

    return _start


async def test_client_disconnect_cancels_upstream(running_app, provider, redis):
    provider.token_delay_s = 0.05  # 20+ tokens * 50ms: el stream completo tardaría >1s
    body = json.dumps(infer_body("cancel me")).encode()
    disconnected = asyncio.Event()
    sent: list[dict] = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            disconnected.set()  # el cliente se va después del primer token

    async with running_app() as manager:
        await asyncio.wait_for(manager.app(_scope(body), receive, send), timeout=2)
        await asyncio.wait_for(provider.stream_closed.wait(), timeout=1)
        assert provider.tokens_generated <= 2  # se cortó la generación upstream
        request_id = dict(sent[0]["headers"])[b"x-request-id"].decode()
        event = await read_audit(redis, request_id)
        assert event["status"] == "cancelled"
        assert event["termination"] == "client_disconnected"


async def test_slow_consumer_is_aborted(running_app, provider, redis):
    body = json.dumps(infer_body("slow reader")).encode()
    never = asyncio.Event()
    sent: list[dict] = []
    request_sent = False

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await never.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        body_chunks = [m for m in sent if m["type"] == "http.response.body"]
        if len(body_chunks) >= 2:
            await never.wait()  # socket "lleno": el cliente dejó de leer

    async with running_app(stream_stall_timeout_s=0.1) as manager:
        await asyncio.wait_for(manager.app(_scope(body), receive, send), timeout=2)
        await asyncio.wait_for(provider.stream_closed.wait(), timeout=1)
        request_id = dict(sent[0]["headers"])[b"x-request-id"].decode()
        event = await read_audit(redis, request_id)
        assert event["termination"] == "slow_consumer"
        assert event["status"] == "cancelled"
