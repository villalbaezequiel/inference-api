# DECISIONS.md — AI Inference API

Documento de arquitectura y trade-offs. Está escrito para que un TL entienda **qué**
hice, **por qué** y **qué dejé afuera a propósito**. Se actualiza en cada etapa.

> Estado: **etapa 4 — observabilidad y tests**. R1, R2 y R3 completos; tests del circuit breaker,
> healthcheck completo (`redis`, `llm_circuit`, `models_loaded`) y logs JSON estructurados por
> request. Las decisiones de abajo son el diseño de partida; cada etapa las confirma o corrige.

---

## TL

| Tema                | Decisión                                                                 | Por qué                                                                                                    |
| ------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Framework / runtime | FastAPI + uvicorn (uvloop, httptools), **1 worker por contenedor**       | El estado en proceso (circuit breaker, coalescing) es por proceso; se escala con réplicas, no con workers. |
| HTTP client         | `httpx.AsyncClient` compartido (pool de conexiones)                      | Async nativo, timeouts por request, sin `requests` sincrónico.                                             |
| LLM                 | Interfaz de proveedor con **mock por default**                           | `docker compose up` funciona sin credenciales; la capa HTTP no conoce al proveedor.                        |
| Fallas de Redis     | **Fail-open** (cache = miss, rate limit = permitir)                      | Perder cache degrada costo; perder el servicio rompe el SLA de 99.9 %.                                     |
| Rate limiting       | **Sliding window log** en Redis (no in-memory), un único script Lua      | Correcto con varias réplicas; sin el "doble burst" de fixed window; atómico, 1 round-trip.                 |
| Cache               | Exact-match sobre `sha256(modelo + input normalizado + config)`          | Determinístico y barato; cumple "mismo input + mismo modelo → misma respuesta".                            |
| Thundering herd     | **Single-flight en 2 niveles**: en proceso + lock `SET NX PX`            | 100 requests idénticos → **1** llamada al LLM, incluso entre réplicas.                                     |
| Circuit breaker     | Ventana deslizante 5 fallos/30 s → OPEN 60 s → HALF-OPEN, **1** prueba   | Lo pedido, con prueba única para no golpear a un proveedor que se está recuperando.                        |
| Streaming           | `SSEResponse` propio: listener de desconexión + `send` con timeout       | Cancela el upstream **al instante** si el cliente se va; memoria por stream O(1 chunk).                    |
| Auditoría           | Cola acotada + worker en background → Redis Stream                       | `put_nowait` nunca bloquea el response; si se satura, se descarta y se loguea.                             |
| Modelo ML           | scikit-learn, artefactos generados en el **build**, carga en el lifespan | Una sola carga por proceso vía `to_thread`; si falta un artefacto la app no arranca.                       |
| Inferencia ML       | `ThreadPoolExecutor` **dedicado**                                        | CPU-bound fuera del event loop; no compite con el threadpool default de anyio.                             |
| Logs                | JSON a stdout, **una línea por request** desde el middleware ASGI         | Campos mínimos del enunciado en un solo lugar; las rutas sólo aportan contexto (`model`, `cache_hit`).     |
| Healthcheck         | `degraded` con **200** si Redis o el circuito fallan; **503** sin modelos | La API sigue sirviendo degradada; sólo un pod sin modelos debe salir del balanceo.                         |
| App factory         | `create_app()` + `uvicorn --factory`                                     | Cada test crea una app aislada; el estado se inicializa en el lifespan, no al importar.                    |

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
  logging_config.py          formatter JSON + contextvars (request_id, client_id)
  schemas.py                 modelos Pydantic de request/response
  api/
    deps.py                  acceso a singletons de app.state
    sse.py                   SSEResponse: desconexión, slow consumer, cierre determinístico
    routes/                  infer.py, stream.py, classify.py, health.py
    middleware/              auth.py, rate_limit.py, request_context.py (request_id + log de acceso)
  core/
    llm_client.py            abstracción del proveedor + timeouts
    circuit_breaker.py       closed / open / half-open
    audit.py                 cola + worker de auditoría de streams
    ml_model.py              carga + versioning
    train.py                 genera los artefactos v1/v2 (corre en el build de la imagen)
    cache.py                 wrapper de Redis (cache + locks)
    inference_service.py     cache -> single-flight -> LLM
    single_flight.py         coalescing en proceso
tests/
```

### Ciclo de vida

Todo lo costoso ocurre **una vez** en el `lifespan`: pool de Redis, cliente HTTP del proveedor
(pool keep-alive), circuit breaker, rate limiter, servicio de inferencia, worker de auditoría y
carga de los modelos ML. Las rutas sólo leen singletons de `app.state`. `create_app()` recibe dependencias opcionales (Redis, proveedor, breaker) → los tests
inyectan fakeredis y un proveedor mock con contador de llamadas.

### Auth y rate limit como _dependencias_, no como middleware global

Respeté los archivos `middleware/auth.py` y `rate_limit.py` pero los implementé como dependencias
de FastAPI porque: (1) aplican sólo a las rutas que las declaran (`/health` queda público para el
orquestador), (2) se encadenan: auth resuelve el `client_id` y el rate limit lo recibe tipado,
(3) aparecen en OpenAPI (botón _Authorize_ en `/docs`) y (4) son triviales de overridear en tests.
El único middleware real es `RequestContextMiddleware` (asigna `request_id` y emite el log de acceso), y es **ASGI puro**
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
opcional con `LLM_PROVIDER=openai`). Ninguna llamada al proveedor queda sin deadline; desde R2 el
timeout es por request y las llamadas pasan por el circuit breaker (ver R2).

### R2 — `/v1/stream` (SSE) + resiliencia

**Endpoint:** `POST /v1/stream` con el mismo body que `/v1/infer`, misma auth y mismo rate limit.
Es POST (no GET) porque el input puede ser largo y no debe terminar en logs de proxies vía query
string. Contrapartida: `EventSource` del browser sólo hace GET; los clientes usan `fetch` +
`ReadableStream` (lo normal en APIs de LLM).

**Formato** exactamente el pedido: `data: {"token": "...", "index": n}` … `data: [DONE]`. Los
errores van como evento: `data: {"error": "llm_unavailable", "retry_after_s": 45}`,
`llm_timeout`, `llm_error`. Para errores que ocurren _antes_ del primer token igual se responde
200 + evento de error: el status HTTP ya se envió cuando arranca el stream, por eso el error de
negocio vive en el payload. Auth, rate limit y modelo inválido sí fallan con su status (401, 429,
400) porque se validan antes de abrir el stream.

**Cancelación upstream si el cliente se desconecta — por qué un `SSEResponse` propio.** Con ASGI
spec ≥ 2.4 (uvicorn), `StreamingResponse` de Starlette detecta la desconexión **sólo cuando falla
el próximo `send`**. Si el LLM tarda 20 s en el próximo token, seguimos pagando 20 s de generación
para nadie. Además, al cortar por excepción no cierra el generador de forma determinística (queda
al GC). `SSEResponse` (`api/sse.py`):

1. corre en paralelo un listener de `http.disconnect` → al llegar, cancela la tarea del stream;
2. siempre hace `aclose()` del generador → su `finally` cierra el `async with httpx.stream(...)`
   → **se corta la conexión TCP con el proveedor y deja de generar** (y de facturar tokens);
3. hace cada `send` con timeout (slow consumer, ver escenario C).

Test: `test_client_disconnect_cancels_upstream` desconecta tras el primer token y verifica que el
proveedor generó ≤ 2 tokens y que la auditoría registra `client_disconnected`.

**Timeout configurable por request:** `config.timeout_s` (default `LLM_DEFAULT_TIMEOUT_S=30`,
techo `LLM_MAX_TIMEOUT_S=120` para que un cliente no pueda retener recursos indefinidamente). Aplica
también a `/v1/infer` y no forma parte de la cache key (no cambia el output). En streaming es un
**deadline total** (no por token), aplicado con `wait_for` sobre cada `anext()` con el tiempo
restante — no uso `asyncio.timeout` envolviendo un `yield` porque cancelaría la tarea equivocada
mientras el generador está suspendido.

**Circuit breaker** (`core/circuit_breaker.py`):

- Ventana **deslizante** de timestamps de fallo (deque): 5 fallos dentro de 30 s → OPEN 60 s.
- Pasados los 60 s → HALF-OPEN: deja pasar **una sola** llamada de prueba; las concurrentes
  reciben 503. Éxito → CLOSED (limpia historial); fallo → OPEN otros 60 s.
- Cuenta como fallo: 5xx, 429 del proveedor, errores de transporte, **timeouts**. **No** cuenta:
  4xx del proveedor por request inválido (culpa del cliente, no del proveedor) ni cancelaciones por
  desconexión del cliente o slow consumer. Si una cancelación ocurre durante la prueba de
  HALF-OPEN, se libera el slot para que otro request pruebe.
- Con el circuito abierto: `/v1/infer` → 503 `{"error": "llm_unavailable", "retry_after_s": N}` +
  header `Retry-After`; `/v1/stream` → evento SSE con el mismo payload.
- Sin locks: todas las transiciones ocurren sin `await` intermedio → atómicas en el event loop.
- `retry_after_s` = tiempo restante del OPEN (el ejemplo `45` del enunciado = 15 s después de
  abrir).
- Reloj inyectable → testeable de forma determinística sin `sleep` (`FakeClock` de
  `tests/conftest.py`; `tests/test_circuit_breaker.py` cubre las transiciones y la integración HTTP y SSE).

**Trade-off: breaker por proceso, no compartido.** Con N réplicas, cada una abre su circuito tras
_sus_ 5 fallos: el proveedor caído recibe hasta 5·N requests antes de que todo el cluster falle
rápido, y cada réplica se recupera independientemente. Lo elegí porque no agrega Redis al camino
de cada llamada al LLM, cada réplica reacciona a lo que _ella_ ve y el costo (5·N requests extra)
es bajo. Si el proveedor cobrara caro los fallos o N fuera grande, movería el estado a Redis
(contador en ZSET + flag `open_until` con TTL) con un caché local de 1 s del estado para no pagar
un RTT por request.

**Auditoría sin bloquear el response.** El enunciado sugiere _background task_. Un
`BackgroundTask` de Starlette **no corre si el cliente se desconecta** (justamente el caso que más
interesa auditar) y además se ejecuta dentro del ciclo del request. En su lugar: el `finally` del
generador hace `queue.put_nowait(evento)` (O(1), nunca espera) en una cola acotada; un worker
arrancado en el lifespan la drena en lotes con un pipeline de `XADD` a un **Redis Stream**
(`MAXLEN ~ 100k`) + una línea de log. El evento incluye input, output completo, tokens, estado
(`success|timeout|error|cancelled`) y motivo (`client_disconnected|slow_consumer`). Si la cola se
llena (Redis caído mucho tiempo), se descarta y se loguea: elegí perder auditoría antes que meter
latencia o memoria no acotada en el camino del usuario. En el shutdown se drena la cola (hasta
5 s). Si la auditoría fuera requisito legal (_at-least-once_), usaría outbox transaccional o Kafka
con acks.

### R3 — `/v1/classify` (modelo ML embebido)

**Runtime: scikit-learn.** El enunciado acepta sklearn, ONNX o stub; lo que se evalúa es _cómo se
carga y sirve_. Elegí un modelo real (no stub) para que las dos versiones den resultados
distintos de verdad y para que la carga y la inferencia tengan el costo de CPU que un stub
escondería. `app/core/train.py` entrena dos pipelines sobre un dataset chico y sintético de
sentimiento (`positive` / `negative`):

- **v1:** TF-IDF de palabras (1-2 gramas) + Logistic Regression.
- **v2:** TF-IDF de palabras + n-gramas de caracteres (más robusto a typos y flexiones) + LogReg.

**Artefactos construidos en el build** de la imagen (`RUN python -m app.core.train`): el contenedor
arranca con los `.joblib` listos y la API sólo los _carga_; nunca entrena al arrancar. En
producción el artefacto vendría de un model registry (MLflow, S3 versionado) y se descargaría en
un init container. Fuera de Docker: `make models`.

**Carga única al arranque, sin bloquear el loop.** En el `lifespan`:
`asyncio.gather(to_thread(load v1), to_thread(load v2))` + una predicción de _warm-up_ por versión
(la primera inferencia no paga inicialización perezosa). Deserializar es disco + CPU: hecho en el
loop, frenaría el arranque de los demás componentes. Las rutas sólo leen el `ModelRegistry` de `app.state`. Test:
`test_models_are_loaded_once_not_per_request` cuenta 2 cargas aunque haya 20 requests.

**Fail-fast.** Si falta un artefacto, la app **no arranca** (`FileNotFoundError` en el lifespan).
Prefiero un pod que no pasa el healthcheck a uno que acepta tráfico y responde 500 en
`/v1/classify`.

**Inferencia fuera del event loop.** `predict_proba` es CPU-bound: corre en un
`ThreadPoolExecutor` **dedicado** (`ML_THREADS`, default 2) vía `run_in_executor`. No uso el
threadpool default de anyio porque lo comparten las dependencias sync de FastAPI y un pico de
clasificaciones las dejaría sin threads. sklearn/NumPy liberan el GIL en buena parte del trabajo
numérico, así que los threads alcanzan para un modelo de ~1 ms; para modelos pesados usaría
`ProcessPoolExecutor` u ONNX Runtime (libera el GIL completamente). Test
`test_inference_does_not_block_event_loop`: una predicción que tarda 300 ms no impide que el loop
siga atendiendo otras corrutinas.

**Versionado por header.** `X-Model-Version: v1 | v2`, case-insensitive; sin header se usa
`DEFAULT_MODEL_VERSION=v1` (los clientes existentes no se rompen cuando aparece una versión
nueva). Versión inexistente → 400
`{"error": "unknown_model_version", "message": "model version 'v3' not available; use one of ['v1', 'v2']"}`.
Las versiones disponibles se configuran con `MODEL_VERSIONS` y cada una se carga en memoria al
arranque: cambiar de versión es un lookup en un dict, sin I/O.

**Auth sí, rate limit no.** `/v1/classify` exige `X-API-Key`, pero no consume la cuota de 60
req/min: esa cuota protege al LLM (costo por token y latencia de segundos) y la clasificación es
inferencia local de ~1 ms. Si hiciera falta limitarla, es un `Depends(enforce_rate_limit)`.

**Respuesta.** `latency_ms` mide la inferencia del lado del servidor (resolución de versión +
predicción), igual que en `/v1/infer`. `confidence` es la probabilidad de la clase elegida,
redondeada a 4 decimales.

**Seguridad.** joblib usa pickle: deserializar un artefacto ajeno es ejecución de código. Sólo
cargamos artefactos que genera nuestro propio build; para artefactos de terceros o de un registry
compartido usaría ONNX (formato sin código ejecutable) o verificación de firma.

**Fuera de alcance (bonus):** batching dinámico de requests concurrentes por ventana de 10 ms.

---

## Observabilidad

**Logs estructurados.** `configure_logging()` instala un único handler JSON a stdout (también para
los loggers de uvicorn; su access log se silencia porque duplica el nuestro). Cada request produce
**una** línea `app.access` emitida por `RequestContextMiddleware` al terminar:

```json
{"timestamp": "2026-10-16T20:59:11.673342+00:00", "level": "INFO", "logger": "app.access",
 "message": "request completed", "request_id": "07161888-...", "client_id": "client-demo",
 "model": "gpt-4o", "latency_ms": 167, "cache_hit": false, "status": "success",
 "endpoint": "/v1/infer", "http_status": 200, "tokens_in": 1, "tokens_out": 20}
```

- `request_id` y `client_id` viajan en `contextvars`: cualquier log emitido durante el request
  (cache, breaker, rate limit) sale con ambos sin pasarlos a mano.
- Las rutas aportan el contexto de negocio en `request.state.log_extra` (`model`, `cache_hit`,
  tokens); el middleware agrega `latency_ms` y deriva `status` del código HTTP
  (`2xx/3xx → success`, `504 → timeout`, resto → `error`).
- En `/v1/stream` el status HTTP siempre es 200: el generador escribe el resultado real
  (`success | timeout | error | cancelled`) antes de que el middleware loguee, y la latencia
  incluye todo el stream.
- `/health` no se loguea: lo sondea el orquestador cada pocos segundos y sólo agrega ruido.
- Nivel por `LOG_LEVEL` (default `INFO`).

**Healthcheck.** `GET /health` → `{"status", "redis", "llm_circuit", "models_loaded"}`.

| Situación                     | `status`   | HTTP | Por qué                                                               |
| ----------------------------- | ---------- | ---- | --------------------------------------------------------------------- |
| Todo bien                     | `ok`       | 200  |                                                                       |
| Redis sin respuesta en 0.5 s  | `degraded` | 200  | Cache y rate limit son fail-open: el pod sigue siendo útil.           |
| Circuito `open` o `half-open` | `degraded` | 200  | Sacar el pod no arregla al proveedor; `/classify` y cache hits andan. |
| Falta un modelo configurado   | —          | 503  | El pod no puede cumplir su contrato: que no reciba tráfico.           |

Si un Redis o un proveedor caído hicieran fallar el health, el orquestador reiniciaría **todas** las
réplicas a la vez por una dependencia externa: una falla parcial se convertiría en total.

---

## Plan por etapas

| Etapa  | Alcance                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| 0      | Esqueleto, Docker, `/health`, DECISIONS inicial                                                                                       |
| 1 — R1 | `/v1/infer`: auth `X-API-Key`, rate limit 60 req/min en Redis, cache con TTL configurable                                             |
| 2 — R2 | `/v1/stream`: SSE, cancelación upstream, timeout por request, circuit breaker (5 fallos/30 s → abierto 60 s), auditoría en background |
| 3 — R3 | `/v1/classify`: modelo cargado una vez sin bloquear el loop, routing por `X-Model-Version`                                            |
| 4      | Tests de circuit breaker, healthcheck completo, logs JSON                                                                             |
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
   - El TTL del lock (35 s) es mayor que el timeout default del LLM (30 s), para que un líder que muere no
     bloquee la key más que eso.
3. Los errores se comparten: si el LLM falla, los 100 reciben el mismo error en vez de reintentar
   en estampida.

Tests: 100 requests concurrentes por HTTP → `provider.calls == 1`; 30 requests repartidos en 3
"réplicas" (3 `InferenceService` con `SingleFlight` distintos y Redis compartido) → 1 llamada.

_Alternativas descartadas:_ stale-while-revalidate (requiere un valor previo, no sirve para el
primer miss); _probabilistic early expiration_ (complementario: evita que todas las keys calientes
expiren a la vez; lo agregaría junto con jitter en el TTL).

### B — LLM provider down ✅ implementado

> El proveedor empieza a retornar timeouts. ¿Cómo falla el sistema gracefully?

Cadena de defensa:

1. **Deadline por request** (default 30 s, configurable): ningún request queda colgado; el cliente
   recibe 504 `llm_timeout` (o el evento `llm_timeout` en SSE).
2. Los timeouts **cuentan como fallos** → tras 5 en 30 s el **circuito se abre**: durante 60 s los
   requests que necesitan al LLM responden **503 inmediato** con `retry_after_s` y `Retry-After`,
   en vez de acumular corrutinas y conexiones esperando 30 s cada una (lo que tumba un servicio no
   es el proveedor caído, sino los recursos retenidos esperándolo).
3. **Lo que no depende del LLM sigue funcionando:** los cache hits de `/v1/infer` se siguen
   sirviendo con el circuito abierto: el cache se consulta antes de pasar por el breaker; y
   `/v1/classify` no se entera, porque el modelo corre dentro del proceso.
4. **Recuperación controlada:** HALF-OPEN deja pasar una sola prueba; no se golpea al proveedor con
   todo el tráfico acumulado en el instante en que vuelve.
5. El single-flight (escenario A) evita que N requests idénticos se conviertan en N timeouts.

_Siguiente paso en producción:_ fallback a un proveedor/modelo secundario detrás de la misma
interfaz `LLMProvider`, con un breaker por proveedor; y _retries_ sólo para errores de conexión
(no para timeouts de generación, que ya consumieron el presupuesto de latencia), con jitter.

### C — Slow consumer streaming ✅ implementado

> Un cliente lee el SSE muy lento. ¿Qué pasa con el buffer? ¿Se cancela el request upstream?

- **No hay buffer intermedio que crezca.** El generador es _pull-based_: sólo se le pide al LLM el
  próximo token cuando el anterior ya fue aceptado por el transporte. Si el cliente lee lento, el
  buffer de escritura de uvicorn llega a su _high-water mark_, `send()` espera el drain, dejamos de
  leer del socket del proveedor y el **control de flujo TCP** frena al proveedor. Memoria por
  stream: O(1 chunk) + buffers del kernel. Con una cola intermedia (productor/consumidor) habría
  que acotarla y decidir qué descartar; el pull lo evita por diseño.
- **Pero esperar indefinidamente también es un problema:** un cliente que deja de leer pero no
  cierra retiene una corrutina, una conexión al proveedor y tokens que se siguen facturando. Cada
  `send` tiene un timeout (`STREAM_STALL_TIMEOUT_S=10`): si el cliente no drena en ese tiempo, se
  aborta el stream, **se cierra la conexión upstream** y se audita con
  `termination: slow_consumer`. No cuenta como fallo del proveedor para el breaker.
- Test `test_slow_consumer_is_aborted`: un `send` que nunca retorna después del 2º chunk → el app
  termina en ~100 ms y el stream del proveedor queda cerrado.

_Mejora pendiente:_ comentarios SSE de heartbeat (`: ping`) durante silencios largos del LLM, para
que proxies con idle-timeout no corten la conexión y para detectar antes clientes muertos.

---

## Levantar el proyecto

Ver [README](README.md).
