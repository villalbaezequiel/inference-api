# DECISIONS.md — AI Inference API

Documento de arquitectura y trade-offs. Está escrito para que un TL entienda **qué**
hice, **por qué** y **qué dejé afuera a propósito**. Se actualiza en cada etapa.

> Estado: **etapa 1 — R1**. `/v1/infer` con auth, rate limit, cache y escenario A (thundering
> herd). Las decisiones de abajo son el diseño de partida; cada etapa las confirma o corrige.

---

## TL

| Tema                | Decisión                                                            | Por qué                                                                                                    |
| ------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Framework / runtime | FastAPI + uvicorn (uvloop, httptools), **1 worker por contenedor**  | El estado en proceso (circuit breaker, coalescing) es por proceso; se escala con réplicas, no con workers. |
| HTTP client         | `httpx.AsyncClient` compartido (pool de conexiones)                 | Async nativo, timeouts por request, sin `requests` sincrónico.                                             |
| LLM                 | Interfaz de proveedor con **mock por default**                      | `docker compose up` funciona sin credenciales; la capa HTTP no conoce al proveedor.                        |
| Fallas de Redis     | **Fail-open** (cache = miss, rate limit = permitir)                 | Perder cache degrada costo; perder el servicio rompe el SLA de 99.9 %.                                     |
| Rate limiting       | **Sliding window log** en Redis (no in-memory), un único script Lua | Correcto con varias réplicas; sin el "doble burst" de fixed window; atómico, 1 round-trip.                 |
| Cache               | Exact-match sobre `sha256(modelo + input normalizado + config)`     | Determinístico y barato; cumple "mismo input + mismo modelo → misma respuesta".                            |
| Thundering herd     | **Single-flight en 2 niveles**: en proceso + lock `SET NX PX`       | 100 requests idénticos → **1** llamada al LLM, incluso entre réplicas.                                     |
| App factory         | `create_app()` + `uvicorn --factory`                                | Cada test crea una app aislada; el estado se inicializa en el lifespan, no al importar.                    |

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
  main.py                    create_app(), lifespan, handlers de error
  config.py                  Settings (todo por env)
  schemas.py                 modelos Pydantic de request/response
  api/
    deps.py                  acceso a singletons de app.state
    routes/                  infer.py, stream.py, classify.py, health.py
    middleware/              auth.py, rate_limit.py, request_context.py
  core/
    llm_client.py            abstracción del proveedor + circuit breaker
    ml_model.py              carga + versioning
    cache.py                 wrapper de Redis (cache + locks)
    inference_service.py     cache -> single-flight -> LLM
    single_flight.py         coalescing en proceso
tests/
```

### Ciclo de vida

Todo lo costoso ocurre **una vez** en el `lifespan`: pool de Redis, cliente HTTP del proveedor
(pool keep-alive), rate limiter y servicio de inferencia. Las rutas sólo leen singletons de
`app.state`. `create_app()` recibe dependencias opcionales (Redis, proveedor) → los tests inyectan
fakeredis y un proveedor mock con contador de llamadas.

### Auth y rate limit como _dependencias_, no como middleware global

Respeté los archivos `middleware/auth.py` y `rate_limit.py` pero los implementé como dependencias
de FastAPI porque: (1) aplican sólo a las rutas que las declaran (`/health` queda público para el
orquestador), (2) se encadenan: auth resuelve el `client_id` y el rate limit lo recibe tipado,
(3) aparecen en OpenAPI (botón _Authorize_ en `/docs`) y (4) son triviales de overridear en tests.
El único middleware real es `RequestContextMiddleware` (asigna `request_id`), y es **ASGI puro**
porque `BaseHTTPMiddleware` bufferiza y rompería la desconexión de SSE en R2.

---

## Requerimientos

### R1 — `/v1/infer`

**Autenticación.** `X-API-Key` → `client_id`. Comparación con `hmac.compare_digest` (tiempo
constante). Keys por env (`API_KEYS=key:client,...`). _Producción:_ keys hasheadas en una DB o
secret manager, con scopes y rotación; para el challenge alcanza con env.

**Rate limiting: 60 req/min por cliente, sliding window log en Lua.**

- _Fixed window_ (`INCR` + `EXPIRE`) es más barato pero permite 120 requests en 1 segundo en el
  borde de la ventana. Con SLAs exigentes ese es justamente el pico que queremos cortar.
- _Token bucket / GCRA_ serían O(1) en memoria; el sliding log cuesta O(60) entradas por cliente
  (unos KB), irrelevante. Elegí el log porque `retry_after_s` sale **exacto** (cuándo sale de la
  ventana el request más viejo) y es fácil de razonar en review.
- Un único `EVALSHA` hace prune + count + add + expire → **atómico entre réplicas** y 1 RTT.
- Cuenta **todos** los requests autenticados, incluidos cache hits: el límite protege la API, no
  sólo al LLM. Es una decisión de producto.
- Usa `time.time()` del nodo (no `TIME` de Redis) para que el script sea determinístico; asume
  relojes sincronizados con NTP (desvío de ms, irrelevante para ventanas de 60 s).
- 429 con `{"error": "rate_limit_exceeded", "retry_after_s": N}` y header `Retry-After`.

**Cache "semántico".** El enunciado lo define como _mismo input + mismo modelo → misma
respuesta_. Implementación:

```
key = sha256(json({model, input: " ".join(input.split()), config: {temperature, max_tokens, top_p}}))
```

- Normalización **conservadora**: sólo whitespace. No paso a minúsculas ni saco puntuación porque
  eso cambia la semántica de un prompt ("Hola" vs "hola" ya es otro input para el LLM).
- La `config` que altera el output **es parte de la key**: misma pregunta con otra temperatura no
  debe devolver la respuesta cacheada.
- TTL configurable (`CACHE_TTL_S`, default 1 h). Los errores **no** se cachean.
- **Trade-off consciente:** con `temperature > 0` el LLM es no determinístico y cachear "congela"
  una de las respuestas posibles. Lo acepto porque es lo que pide el requerimiento y es lo que baja
  costo/latencia; en producción lo expondría como opt-out por request (`config.cache: false`) o
  cachearía sólo `temperature == 0`.
- Un cache semántico "de verdad" (embeddings + búsqueda por similitud) queda fuera: agrega un
  modelo de embeddings en el camino crítico y riesgo de devolver respuestas a preguntas parecidas
  pero distintas. Es una evolución, no un requisito.
- `cache_hit: true` sólo cuando el resultado salió del cache para **este** request. Los requests
  que esperaron al líder (coalesced, ver escenario A) reportan `false`.

**Respuesta.** `request_id` reutiliza el `X-Request-ID` entrante si existe (trazabilidad end-to-end).

**LLM.** `LLMClient` envuelve un `LLMProvider` (`MockProvider` por default, `OpenAIProvider`
opcional con `LLM_PROVIDER=openai`). En esta etapa aplica un timeout global (`LLM_TIMEOUT_S`, 30 s):
ninguna llamada al proveedor queda sin deadline. El timeout por request y el circuit breaker son R2.

---

## Plan por etapas

| Etapa  | Alcance                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 0      | Esqueleto, Docker, `/health`, DECISIONS inicial                                                                                       |
| 1 — R1 | `/v1/infer`: auth `X-API-Key`, rate limit 60 req/min en Redis, cache con TTL configurable                                             |
| 2 — R2 | `/v1/stream`: SSE, cancelación upstream, timeout por request, circuit breaker (5 fallos/30 s → abierto 60 s), auditoría en background |
| 3 — R3 | `/v1/classify`: modelo cargado una vez sin bloquear el loop, routing por `X-Model-Version`                                            |
| 4      | Tests de circuit breaker, healthcheck completo, logs JSON, métricas Prometheus                                                        |
| 5      | Completar DECISIONS, verificar `docker compose up`                                                                                    |

---

## Escenarios de carga

### A — Thundering herd ✅ implementado

> 100 clientes hacen el mismo request justo después de un cache miss.

Single-flight en dos niveles (`core/single_flight.py`, `core/inference_service.py`):

1. **En proceso:** el primer request con una key crea una `Task`; los siguientes 99 esperan esa
   misma Task con `asyncio.shield`. Si el cliente "líder" se desconecta, la Task sigue, los demás
   reciben su resultado y el cache queda poblado.
2. **Entre réplicas:** antes de llamar al LLM, la Task toma un lock `SET lock:{key} token NX PX`.
   - Si lo obtiene: _double-check_ del cache, llama al LLM, escribe el cache y libera el lock con un
     script Lua compare-and-delete (sólo borra si el token es suyo).
   - Si no: hace polling del cache con backoff exponencial (20 → 200 ms). Si el lock desaparece sin
     valor en cache (el líder falló), intenta tomar el liderazgo. Si se excede el tiempo de espera o
     Redis no responde, **degrada a llamar al LLM directo** (nunca deja al cliente colgado).
   - El TTL del lock (35 s) es mayor que el timeout del LLM (30 s), para que un líder que muere no
     bloquee la key más que eso.
3. Los errores se comparten: si el LLM falla, los 100 reciben el mismo error en vez de reintentar
   en estampida.

Tests: 100 requests concurrentes por HTTP → `provider.calls == 1`; 30 requests repartidos en 3
"réplicas" (3 `InferenceService` con `SingleFlight` distintos y Redis compartido) → 1 llamada.

_Alternativas descartadas:_ stale-while-revalidate (requiere un valor previo, no sirve para el
primer miss); _probabilistic early expiration_ (complementario: evita que todas las keys calientes
expiren a la vez; lo agregaría junto con jitter en el TTL).

### Pendientes

- **B — LLM provider down**: a definir en etapa 2 (circuit breaker + timeouts).
- **C — Slow consumer streaming**: a definir en etapa 2 (backpressure + cancelación).

---

## Levantar el proyecto

Ver [README](README.md).
