"""R3 — Modelo embebido: carga única, versionado por header, inferencia fuera del loop."""

import asyncio
import time

import pytest
from asgi_lifespan import LifespanManager

from app.core.ml_model import LoadedModel, ModelRegistry
from tests.conftest import HEADERS


async def test_classify_v2(client):
    r = await client.post(
        "/v1/classify",
        json={"input": "excelente servicio, lo recomiendo"},
        headers={**HEADERS, "X-Model-Version": "v2"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "positive"
    assert 0.5 < data["confidence"] <= 1
    assert data["model_version"] == "v2"
    assert isinstance(data["latency_ms"], int)


async def test_default_version_when_header_missing(client):
    r = await client.post("/v1/classify", json={"input": "muy bueno"}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["model_version"] == "v1"


async def test_version_header_is_case_insensitive(client):
    r = await client.post("/v1/classify", json={"input": "muy bueno"}, headers={**HEADERS, "X-Model-Version": "V2"})
    assert r.json()["model_version"] == "v2"


async def test_unknown_version_returns_400(client):
    r = await client.post("/v1/classify", json={"input": "x"}, headers={**HEADERS, "X-Model-Version": "v3"})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "unknown_model_version"
    assert "v3" in body["message"] and "v1" in body["message"] and "v2" in body["message"]


async def test_classify_requires_api_key(client):
    r = await client.post("/v1/classify", json={"input": "x"})
    assert r.status_code == 401


async def test_models_are_loaded_once_not_per_request(make_client, monkeypatch):
    calls = []
    original = ModelRegistry._load_sync

    def counting(self, version):
        calls.append(version)
        return original(self, version)

    monkeypatch.setattr(ModelRegistry, "_load_sync", counting)
    client = await make_client()
    assert sorted(calls) == ["v1", "v2"]
    for _ in range(20):
        await client.post("/v1/classify", json={"input": "hola"}, headers={**HEADERS, "X-Model-Version": "v2"})
    assert sorted(calls) == ["v1", "v2"]


async def test_missing_artifact_fails_startup(app_factory, tmp_path):
    app = app_factory(models_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        async with LifespanManager(app):
            pass


async def test_inference_does_not_block_event_loop(models_dir, monkeypatch):
    """Una predicción lenta (CPU-bound simulada) corre fuera del loop: otras corrutinas avanzan."""
    registry = ModelRegistry(str(models_dir), ["v1"], "v1")
    await registry.load()
    original = LoadedModel.predict_batch

    def slow_predict(self, texts):
        time.sleep(0.3)  # bloquea el thread, NO el event loop
        return original(self, texts)

    monkeypatch.setattr(LoadedModel, "predict_batch", slow_predict)
    loop = asyncio.get_running_loop()
    start = loop.time()
    classify = asyncio.create_task(registry.classify("hola", "v1"))
    await asyncio.sleep(0.01)
    assert loop.time() - start < 0.1  # el loop respondió mientras el modelo trabajaba
    assert not classify.done()
    assert (await classify)[0] in {"positive", "negative"}
    await registry.close()
