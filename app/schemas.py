from pydantic import BaseModel, ConfigDict, Field


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)

    def cache_relevant(self) -> dict:
        """Parámetros que cambian el output (forman parte de la cache key)."""
        return self.model_dump(exclude_none=True)


class InferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1, max_length=64)
    input: str = Field(min_length=1, max_length=32_000)
    config: InferenceConfig = Field(default_factory=InferenceConfig)


class Usage(BaseModel):
    tokens_in: int
    tokens_out: int


class InferResponse(BaseModel):
    request_id: str
    output: str
    model: str
    usage: Usage
    latency_ms: int
    cache_hit: bool


class ErrorResponse(BaseModel):
    error: str
    message: str | None = None
    retry_after_s: int | None = None
