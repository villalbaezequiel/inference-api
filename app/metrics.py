"""Métricas Prometheus (bonus), expuestas en GET /metrics.

Cache hit ratio en PromQL:
    sum(rate(cache_requests_total{result="hit"}[5m]))
      / sum(rate(cache_requests_total[5m]))
"""

from prometheus_client import Counter, Histogram

REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total de requests HTTP",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Latencia de requests HTTP",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2.5, 5, 10, 30),
)

CACHE_REQUESTS = Counter(
    "cache_requests_total",
    "Lookups al cache de inferencia",
    ["result"],  # hit | miss
)
