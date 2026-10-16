"""Configuración centralizada. Todo se puede sobreescribir por variables de entorno."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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
    llm_timeout_s: float = 30.0
    allowed_models: str = "gpt-4o,gpt-4o-mini"

    # Mock provider: latencia simulada.
    mock_llm_latency_ms: int = 150

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
