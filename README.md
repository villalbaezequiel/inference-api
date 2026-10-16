# AI Inference API

API de inferencia para LLMs con streaming SSE, rate limiting, cache, circuit breaker y un
clasificador ML embebido. FastAPI + Redis + scikit-learn.

## Levantar

```bash
docker compose up --build
```

- API: http://localhost:8000 · Swagger: http://localhost:8000/docs
- Variables configurables: ver [`.env.example`](.env.example)

## Endpoints

### `POST /v1/infer` — inferencia sincrónica

```bash
curl -s -X POST localhost:8000/v1/infer \
  -H 'X-API-Key: dev-key-123' -H 'Content-Type: application/json' \
  -d '{"model": "gpt-4o", "input": "Explicá qué es un circuit breaker", "config": {"temperature": 0.7}}'
```

```json
{"request_id": "5c1d...", "output": "...", "model": "gpt-4o",
 "usage": {"tokens_in": 9, "tokens_out": 21}, "latency_ms": 154, "cache_hit": false}
```

Repetir el mismo request devuelve `"cache_hit": true`. Modelos permitidos: `gpt-4o`, `gpt-4o-mini`
(`ALLOWED_MODELS`). `config` acepta `temperature`, `max_tokens`, `top_p` y `timeout_s`.

Rate limit: 60 req/min por API key. Al excederlo:

```json
{"error": "rate_limit_exceeded", "retry_after_s": 12}
```

Errores (formato `{"error": ..., ...}`): 401 `missing_api_key` / `invalid_api_key`,
400 `unsupported_model`, 422 `invalid_request`, 429 `rate_limit_exceeded`, 502 `llm_upstream_error`,
503 `llm_unavailable` (circuit breaker abierto), 504 `llm_timeout`. `config.timeout_s` fija el
timeout por request (default 30 s, máximo 120 s).

### `POST /v1/stream` — streaming SSE

```bash
curl -N -X POST localhost:8000/v1/stream \
  -H 'X-API-Key: dev-key-123' -H 'Content-Type: application/json' \
  -d '{"model": "gpt-4o", "input": "Hola", "config": {"timeout_s": 10}}'
```

```
data: {"token": "La", "index": 0}

data: {"token": " arquitectura", "index": 1}

data: [DONE]
```

Mismo body, auth y rate limit que `/v1/infer`. Errores del LLM durante el stream llegan como evento
(`data: {"error": "llm_unavailable", "retry_after_s": 45}`, `llm_timeout`, `llm_error`). Cortar el
`curl` con Ctrl+C cancela la generación upstream. Cada stream se audita en el Redis Stream
`audit:streams`.

### `POST /v1/classify` — clasificación con modelo ML embebido

```bash
curl -s -X POST localhost:8000/v1/classify \
  -H 'X-API-Key: dev-key-123' -H 'X-Model-Version: v2' -H 'Content-Type: application/json' \
  -d '{"input": "excelente servicio, lo recomiendo"}'
```

```json
{"label": "positive", "confidence": 0.7303, "model_version": "v2", "latency_ms": 3}
```

`X-Model-Version`: `v1` (default si no se envía) o `v2`. Versión inexistente:

```bash
curl -s -X POST localhost:8000/v1/classify -H 'X-API-Key: dev-key-123' \
  -H 'X-Model-Version: v9' -H 'Content-Type: application/json' -d '{"input": "x"}'
# 400 {"error": "unknown_model_version", "message": "model version 'v9' not available; use one of ['v1', 'v2']"}
```

Los modelos se entrenan en el build de la imagen y se cargan una vez al arranque.

### `GET /health` — healthcheck

```bash
curl http://localhost:8000/health
# {"status": "ok", "redis": "ok", "llm_circuit": "closed", "models_loaded": ["v1", "v2"]}
```

Con Redis caído o el circuito abierto responde `"status": "degraded"` (200); si falta un modelo, 503.

### Logs

Una línea JSON por request en stdout (`make logs`), con `timestamp`, `request_id`, `client_id`,
`model`, `latency_ms`, `cache_hit` y `status`. Nivel configurable con `LOG_LEVEL`.

### `GET /metrics` — métricas Prometheus (bonus)

```bash
curl -s localhost:8000/metrics | grep -E '^(http_requests_total|cache_requests_total)'
# http_requests_total{endpoint="/v1/infer",method="POST",status="200"} 2.0
# cache_requests_total{result="hit"} 1.0
# cache_requests_total{result="miss"} 1.0
```

Requests totales, latencia (`http_request_duration_seconds`) y cache hits/misses.

### Carga con locust (bonus)

```bash
make load           # locust headless 60 s a ~105 RPS contra docker compose
```

Levanta la app con rate limit alto (para medir capacidad, no el limitador) y deja CSV y
`report.html` en `loadtest/results/`. Interactivo: `docker compose --profile load up locust` →
http://localhost:8089. Resultado de referencia: 102.6 RPS, 0 errores
([RESULTS.md](loadtest/results/RESULTS.md)).

## Desarrollo

```bash
python -m venv .venv && source .venv/bin/activate
make install        # dependencias + dev
make test           # pytest
make lint           # ruff
```

Correr la app fuera de Docker (con un Redis local en :6379):

```bash
make models         # genera los artefactos del clasificador en ./models
uvicorn --factory app.main:create_app --reload
```

## Estructura

```
app/
  main.py               app
  api/routes/           infer.py, stream.py, classify.py, health.py, metrics.py
  api/middleware/       auth.py, rate_limit.py
  core/llm_client.py    llm_client
  core/ml_model.py      ml_model
  core/train.py         entrenamiento de los artefactos v1/v2
  core/cache.py         cache
tests/                  tests
loadtest/               locustfile + resultados
DECISIONS.md            ← el documento importante
```
