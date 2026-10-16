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

### `POST /v1/infer` — inferencia sincrónica (TBD)

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
