"""Abstracción del proveedor LLM + timeout.

    LLMClient  ──(deadline)──▶  LLMProvider
                                 ├─ MockProvider   (default, sin credenciales)
                                 └─ OpenAIProvider (httpx async)

Las rutas sólo conocen `LLMClient`; cambiar de proveedor no toca la capa HTTP.
"""

import asyncio
import hashlib
import random
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.schemas import InferenceConfig


# ============================================================== errores
class LLMError(Exception):
    """Error atribuible al proveedor."""


class LLMTimeoutError(LLMError):
    pass


class LLMUpstreamError(LLMError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMClientError(Exception):
    """Error 4xx del proveedor causado por el request."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(slots=True)
class Completion:
    text: str
    tokens_in: int
    tokens_out: int


def approx_tokens(text: str) -> int:
    # Aproximación barata (~4 chars/token) para proveedores que no informan usage.
    return max(1, round(len(text) / 4))


# ============================================================ proveedores
class LLMProvider(Protocol):
    async def complete(self, model: str, prompt: str, config: InferenceConfig) -> Completion: ...

    async def aclose(self) -> None: ...


class MockProvider:
    """Proveedor determinístico para desarrollo y tests (sin credenciales)."""

    _VOCAB = (
        "La arquitectura propuesta prioriza la latencia y la resiliencia del sistema "
        "manteniendo un consumo de recursos acotado frente a picos de tráfico"
    ).split()

    def __init__(self, latency_ms: int = 150):
        self.latency_s = latency_ms / 1000
        self.calls = 0  # útil para tests (thundering herd)

    def _tokens(self, model: str, prompt: str, config: InferenceConfig) -> list[str]:
        seed = int(hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        n = min(config.max_tokens, 12 + seed % 20)
        words = [rng.choice(self._VOCAB) for _ in range(n)]
        return [words[0]] + [f" {w}" for w in words[1:]]

    async def complete(self, model: str, prompt: str, config: InferenceConfig) -> Completion:
        self.calls += 1
        await asyncio.sleep(self.latency_s)
        tokens = self._tokens(model, prompt, config)
        return Completion("".join(tokens), approx_tokens(prompt), len(tokens))

    async def aclose(self) -> None:
        return None


class OpenAIProvider:
    """Cliente OpenAI-compatible (Chat Completions) sobre httpx async."""

    def __init__(self, api_key: str, base_url: str, max_connections: int = 200):
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            # El deadline real lo maneja LLMClient; esto es un techo de seguridad.
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=50),
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        code = response.status_code
        if code < 400:
            return
        if code == 429 or code >= 500:
            raise LLMUpstreamError(f"upstream error {code}", status_code=code)
        raise LLMClientError(f"provider rejected request ({code})", status_code=code)

    async def complete(self, model: str, prompt: str, config: InferenceConfig) -> Completion:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], **config.cache_relevant()}
        try:
            response = await self._client.post("/chat/completions", json=body)
        except httpx.TransportError as exc:
            raise LLMUpstreamError(f"transport error: {exc!r}") from exc
        self._raise_for_status(response)
        data = response.json()
        usage = data.get("usage") or {}
        text = data["choices"][0]["message"]["content"] or ""
        return Completion(
            text,
            usage.get("prompt_tokens", approx_tokens(prompt)),
            usage.get("completion_tokens", approx_tokens(text)),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# ============================================================== cliente
class LLMClient:
    def __init__(self, provider: LLMProvider, timeout_s: float):
        self.provider = provider
        self.timeout_s = timeout_s

    async def complete(self, model: str, prompt: str, config: InferenceConfig) -> Completion:
        try:
            async with asyncio.timeout(self.timeout_s):
                return await self.provider.complete(model, prompt, config)
        except TimeoutError as exc:
            raise LLMTimeoutError(f"LLM did not answer within {self.timeout_s}s") from exc

    async def aclose(self) -> None:
        await self.provider.aclose()


def build_provider(settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requiere OPENAI_API_KEY")
        return OpenAIProvider(settings.openai_api_key, settings.openai_base_url)
    return MockProvider(latency_ms=settings.mock_llm_latency_ms)
