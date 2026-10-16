# Resultado de carga — locust, 60 s, ~100 RPS

**Comando:** `make load` (locust headless `-u 150 -r 30 -t 60s`, 0.7 req/s por usuario)
**Entorno:** 1 réplica (1 worker uvicorn), Redis 7, `LLM_PROVIDER=mock` (150 ms de latencia
simulada por llamada, 30 ms entre tokens), `RATE_LIMIT_PER_MINUTE` alto para medir capacidad y no
el limitador. Docker Desktop con 6 CPUs; locust corre **en la misma VM** que la API y compite por
CPU: los números son conservadores.

| Endpoint      | Requests | Fallos | P50   | P95    | P99    | req/s     |
| ------------- | -------: | -----: | ----: | -----: | -----: | --------: |
| /v1/infer     |    3 626 |      0 | 49 ms | 230 ms | 300 ms |      60.7 |
| /v1/classify  |    1 602 |      0 | 43 ms |  76 ms | 160 ms |      26.8 |
| /v1/stream¹   |      903 |      0 | 50 ms |  95 ms | 180 ms |      15.1 |
| **Total**     | **6 131**|  **0** | 47 ms | 210 ms | 300 ms | **102.6** |

¹ Con `stream=True` locust mide hasta los headers de respuesta (≈ TTFT), no hasta el último token.

## Lectura

- **0 errores a 102.6 RPS sostenidos** en una sola réplica.
- `/v1/infer` es bimodal: ~69 % de los requests son _cache hits_ (P50 49 ms con locust compitiendo
  por CPU); los _misses_ pagan los 150 ms del LLM simulado, y eso es lo que define el P95 de
  ~230 ms. El P95 < 200 ms del SLA depende de la latencia del proveedor en los misses, no de la API.
- Métricas de `GET /metrics` al final de la corrida:
  - `cache_requests_total`: hit=2501, miss=1130 → **hit ratio 69 %**
  - `http_requests_total{status="200"}`: infer 3631, classify 1602, stream 917 (incluye requests
    que locust no llegó a contabilizar al cortar en 60 s)
- `/v1/classify` P50 43 ms está inflado por la contención de CPU con locust; la inferencia del
  modelo en sí es de un dígito de ms.

CSV crudos de la corrida: `run_stats.csv`, `run_stats_history.csv`. El `report.html` que genera
`make load` no se versiona.
