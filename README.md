# AI Inference API

API de inferencia para LLMs con streaming SSE, rate limiting, cache, circuit breaker.
FastAPI + Redis.

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
(`ALLOWED_MODELS`). `config` acepta `temperature`, `max_tokens` y `top_p`.

Rate limit: 60 req/min por API key. Al excederlo:

```json
{"error": "rate_limit_exceeded", "retry_after_s": 12}
```

Errores (formato `{"error": ..., ...}`): 401 `missing_api_key` / `invalid_api_key`,
400 `unsupported_model`, 422 `invalid_request`, 429 `rate_limit_exceeded`, 502 `llm_upstream_error`,
504 `llm_timeout`.

### `POST /v1/stream` — streaming SSE (TBD)

### `POST /v1/classify` — clasificación con modelo ML embebido (TBD)

### `GET /health` — healthcheck

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## Desarrollo

```bash
python -m venv .venv && source .venv/bin/activate
make install        # dependencias + dev
make test           # pytest
make lint           # ruff
```

Correr la app fuera de Docker (con un Redis local en :6379):

```bash
uvicorn --factory app.main:create_app --reload
```

## Estructura

```
app/
  main.py               app
  api/routes/           infer.py, stream.py, classify.py, health.py
  api/middleware/       auth.py, rate_limit.py
  core/llm_client.py    llm_client
  core/ml_model.py      ml_model
  core/cache.py         cache
tests/                  tests
DECISIONS.md            ← el documento importante
```
