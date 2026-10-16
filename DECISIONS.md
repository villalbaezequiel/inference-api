# DECISIONS.md — AI Inference API

Documento de arquitectura y trade-offs. Está escrito para que un TL entienda **qué**
hice, **por qué** y **qué dejé afuera a propósito**. Se actualiza en cada etapa.

> Estado: **etapa 0 — esqueleto**. Estructura, Docker y `/health`. Las decisiones de abajo
> son el diseño de partida; cada etapa las confirma o corrige.

---

## TL

| Tema                | Decisión                                                           | Por qué                                                                                                    |
| ------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Framework / runtime | FastAPI + uvicorn (uvloop, httptools), **1 worker por contenedor** | El estado en proceso (circuit breaker, coalescing) es por proceso; se escala con réplicas, no con workers. |
| HTTP client         | `httpx.AsyncClient` compartido (pool de conexiones)                | Async nativo, timeouts por request, sin `requests` sincrónico.                                             |
| Cache / rate limit  | **Redis** (no in-memory)                                           | Con varias réplicas, un límite de 60 req/min por cliente y una cache en memoria local serían incorrectos.  |
| LLM                 | Interfaz de proveedor con **mock por default**                     | `docker compose up` funciona sin credenciales; la capa HTTP no conoce al proveedor.                        |
| Fallas de Redis     | **Fail-open** (cache = miss, rate limit = permitir)                | Perder cache degrada costo; perder el servicio rompe el SLA de 99.9 %.                                     |
| App factory         | `create_app()` + `uvicorn --factory`                               | Cada test crea una app aislada; el estado se inicializa en el lifespan, no al importar.                    |

---

## Contexto y objetivos

| Métrica             | Objetivo                  |
| ------------------- | ------------------------- |
| Latencia P95 (sync) | < 200 ms                  |
| Concurrencia base   | 300 req/s, picos de miles |
| Streaming           | SSE                       |
| Disponibilidad      | 99.9 %                    |

La latencia P95 < 200 ms solo es alcanzable con cache hit o con el mock: un LLM real tarda
más. Por eso cache y coalescing son parte central del diseño, no una optimización.

---

## Arquitectura

```
cliente ──HTTP──▶ FastAPI (1 proceso por contenedor)
                    │
                    ├─ POST /v1/infer    ─▶ auth ─▶ rate limit ─▶ cache ──miss──▶ LLMClient ─[circuit breaker]─▶ proveedor
                    ├─ POST /v1/stream   ─▶ auth ─▶ rate limit ─▶ SSE ◀── LLMClient.stream
                    ├─ POST /v1/classify ─▶ auth ─▶ ModelRegistry (v1 | v2, cargado una vez al arranque)
                    └─ GET  /health
                                                                       Redis: rate limit, cache
```

**Estructura** (respeta la estructura solicitada):

```
app/
  main.py                    create_app()
  api/
    routes/                  infer.py, stream.py, classify.py, health.py
    middleware/              auth.py, rate_limit.py
  core/
    llm_client.py            abstracción del proveedor + circuit breaker
    ml_model.py              carga + versioning
    cache.py                 wrapper de Redis
tests/
```

---

## Plan por etapas

| Etapa  | Alcance                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 0      | Esqueleto, Docker, `/health`, DECISIONS inicial                                                                                       |
| 1 — R1 | `/v1/infer`: auth `X-API-Key`, rate limit 60 req/min en Redis, cache con TTL configurable                                             |
| 2 — R2 | `/v1/stream`: SSE, cancelación upstream, timeout por request, circuit breaker (5 fallos/30 s → abierto 60 s), auditoría en background |
| 3 — R3 | `/v1/classify`: modelo cargado una vez sin bloquear el loop, routing por `X-Model-Version`                                            |
| 4      | Tests (R1 completo + circuit breaker), healthcheck completo, logs JSON, métricas Prometheus                                           |
| 5      | Completar DECISIONS, verificar `docker compose up`                                                                                    |

---

## Escenarios de carga

- **A — Thundering herd**: a definir en etapa 1 (coalescing de requests idénticos).
- **B — LLM provider down**: a definir en etapa 2 (circuit breaker + timeouts).
- **C — Slow consumer streaming**: a definir en etapa 2 (backpressure + cancelación).

---

## Levantar el proyecto

```bash
docker compose up --build          # app + Redis
curl http://localhost:8000/health
```
