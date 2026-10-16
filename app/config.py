"""Configuración centralizada. Todo se puede sobreescribir por variables de entorno."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = "INFO"

    # --- Redis -------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    # Timeouts cortos: Redis está en el camino crítico de cada request.
    # Si Redis tarda más que esto, preferimos degradar (fail-open) a romper el P95.
    redis_socket_timeout_s: float = 0.25
    redis_max_connections: int = 200

    # --- Auth --------------------------------------------------------------
    # Formato: "api_key:client_id,api_key2:client_id2"
    api_keys: str = "dev-key-123:client-demo,dev-key-456:client-b"

    # --- Rate limiting -----------------------------------------------------
    rate_limit_per_minute: int = 60
    # Si Redis no responde: True = dejar pasar (disponibilidad) / False = rechazar.
    rate_limit_fail_open: bool = True

    # --- Cache -------------------------------------------------------------
    cache_ttl_s: int = 3600
    # Single-flight distribuido (Escenario A): TTL del lock y cuánto espera un "follower".
    single_flight_lock_ttl_s: float = 35.0
    single_flight_wait_s: float = 30.0

    # --- LLM ---------------------------------------------------------------
    # "mock" (default, sin credenciales) | "openai"
    llm_provider: str = "mock"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    # Timeout por request: default si el cliente no manda `config.timeout_s`, y techo.
    llm_default_timeout_s: float = 30.0
    llm_max_timeout_s: float = 120.0
    allowed_models: str = "gpt-4o,gpt-4o-mini"

    # Mock provider: latencia simulada (total en sync, por token en streaming).
    mock_llm_latency_ms: int = 150
    mock_llm_token_delay_ms: int = 30

    # --- Circuit breaker ---------------------------------------------------
    # N fallos dentro de una ventana de W segundos -> abierto D segundos.
    cb_failure_threshold: int = 5
    cb_failure_window_s: float = 30.0
    cb_open_duration_s: float = 60.0

    # --- Streaming ---------------------------------------------------------
    # Si el cliente no consume un chunk en este tiempo, se lo considera
    # "slow consumer" y se aborta el stream (y el request upstream).
    stream_stall_timeout_s: float = 10.0

    # --- Audit -------------------------------------------------------------
    audit_queue_size: int = 10_000
    audit_stream_key: str = "audit:streams"
    audit_stream_maxlen: int = 100_000

    # --- ML ----------------------------------------------------------------
    # Artefactos `classifier-{version}.joblib` (se generan con `python -m app.core.train`).
    models_dir: str = "models"
    model_versions: str = "v1,v2"
    default_model_version: str = "v1"
    ml_threads: int = Field(default=2, ge=1)

    # ----------------------------------------------------------------------
    @property
    def api_key_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for pair in self.api_keys.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, _, client = pair.partition(":")
            result[key.strip()] = (client or key).strip()
        return result

    @property
    def allowed_models_set(self) -> set[str]:
        return {m.strip() for m in self.allowed_models.split(",") if m.strip()}

    @property
    def model_versions_list(self) -> list[str]:
        return [v.strip() for v in self.model_versions.split(",") if v.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
