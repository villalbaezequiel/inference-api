"""Abstracción del proveedor LLM + circuit breaker + timeouts.

    LLMClient  ──(guard: circuit breaker, deadline)──▶  LLMProvider
                                                         ├─ MockProvider   (default, sin credenciales)
                                                         └─ OpenAIProvider (httpx async, SSE upstream)

Las rutas sólo conocen `LLMClient`; cambiar de proveedor no toca la capa HTTP.
"""

import asyncio
import hashlib
import json
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.core.circuit_breaker import CircuitBreaker
from app.schemas import InferenceConfig


# ============================================================== errores
class LLMError(Exception):
    """Error del proveedor que cuenta como fallo para el circuit breaker."""


class LLMTimeoutError(LLMError):
    pass


class LLMUpstreamError(LLMError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMClientError(Exception):
    """Error 4xx del proveedor causado por el request (NO abre el circuito)."""

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

    def stream(self, model: str, prompt: str, config: InferenceConfig) -> AsyncIterator[str]: ...

    async def aclose(self) -> None: ...


class MockProvider:
    """Proveedor determinístico para desarrollo y tests (sin credenciales).

    Disparadores (sólo en el mock) para demostrar resiliencia sin un proveedor real:
      - input contiene "__fail__"  -> error 503 upstream
      - input contiene "__slow__"  -> tarda 10x más
    """

    _VOCAB = (
        "La arquitectura propuesta prioriza la latencia y la resiliencia del sistema "
        "manteniendo un consumo de recursos acotado frente a picos de tráfico"
    ).split()

    def __init__(self, latency_ms: int = 150, token_delay_ms: int = 30):
        self.latency_s = latency_ms / 1000
        self.token_delay_s = token_delay_ms / 1000
        self.calls = 0  # útil para tests (thundering herd)

    @staticmethod
    def _maybe_fail(prompt: str) -> None:
        if "__fail__" in prompt:
            raise LLMUpstreamError("mock provider failure", status_code=503)

    def _tokens(self, model: str, prompt: str, config: InferenceConfig) -> list[str]:
        seed = int(hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        n = min(config.max_tokens, 12 + seed % 20)
        words = [rng.choice(self._VOCAB) for _ in range(n)]
        return [words[0]] + [f" {w}" for w in words[1:]]

    async def complete(self, model: str, prompt: str, config: InferenceConfig) -> Completion:
        self.calls += 1
        await asyncio.sleep(self.latency_s * (10 if "__slow__" in prompt else 1))
        self._maybe_fail(prompt)
        tokens = self._tokens(model, prompt, config)
        return Completion("".join(tokens), approx_tokens(prompt), len(tokens))

    async def stream(self, model: str, prompt: str, config: InferenceConfig) -> AsyncIterator[str]:
        self.calls += 1
        await asyncio.sleep(self.latency_s / 3)  # time-to-first-token
        self._maybe_fail(prompt)
        delay = self.token_delay_s * (10 if "__slow__" in prompt else 1)
        for token in self._tokens(model, prompt, config):
            await asyncio.sleep(delay)
            yield token

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
    def _body(model: str, prompt: str, config: InferenceConfig, stream: bool) -> dict:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **config.cache_relevant(),
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        code = response.status_code
        if code < 400:
            return
        if code == 429 or code >= 500:
            raise LLMUpstreamError(f"upstream error {code}", status_code=code)
        raise LLMClientError(f"provider rejected request ({code})", status_code=code)

    async def complete(self, model: str, prompt: str, config: InferenceConfig) -> Completion:
        try:
            response = await self._client.post("/chat/completions", json=self._body(model, prompt, config, False))
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

    async def stream(self, model: str, prompt: str, config: InferenceConfig) -> AsyncIterator[str]:
        try:
            # Al salir del `async with` (fin, error o cancelación) httpx cierra la
            # conexión -> el proveedor deja de generar (cancelación upstream).
            async with self._client.stream(
                "POST", "/chat/completions", json=self._body(model, prompt, config, True)
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    chunk = json.loads(data)
                    for choice in chunk.get("choices") or []:
                        token = (choice.get("delta") or {}).get("content")
                        if token:
                            yield token
        except httpx.TransportError as exc:
            raise LLMUpstreamError(f"transport error: {exc!r}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


# ============================================================== cliente
class LLMClient:
    def __init__(self, provider: LLMProvider, breaker: CircuitBreaker):
        self.provider = provider
        self.breaker = breaker

    async def complete(self, model: str, prompt: str, config: InferenceConfig, timeout_s: float) -> Completion:
        async with self.breaker.guard(ignore=(LLMClientError,)):
            try:
                async with asyncio.timeout(timeout_s):
                    return await self.provider.complete(model, prompt, config)
            except TimeoutError as exc:
                raise LLMTimeoutError(f"LLM did not answer within {timeout_s}s") from exc

    async def stream(self, model: str, prompt: str, config: InferenceConfig, timeout_s: float) -> AsyncIterator[str]:
        """Itera tokens con deadline total. Lanza CircuitOpenError antes del primer token.

        El consumidor debe cerrar el generador (`aclose`) si abandona: eso cancela el
        request upstream y libera el permiso del circuit breaker sin contarlo como fallo.
        """
        probe = self.breaker.acquire()
        deadline = asyncio.get_running_loop().time() + timeout_s
        upstream = self.provider.stream(model, prompt, config)
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise LLMTimeoutError(f"stream exceeded {timeout_s}s")
                try:
                    token = await asyncio.wait_for(anext(upstream), timeout=remaining)
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    raise LLMTimeoutError(f"stream exceeded {timeout_s}s") from exc
                yield token
            self.breaker.record_success(probe)
        except LLMClientError:
            self.breaker.release(probe)
            raise
        except (asyncio.CancelledError, GeneratorExit):
            # Cliente desconectado / slow consumer: no es culpa del proveedor.
            self.breaker.release(probe)
            raise
        except Exception:
            self.breaker.record_failure(probe)
            raise
        finally:
            await upstream.aclose()  # garantiza cancelación upstream

    async def aclose(self) -> None:
        await self.provider.aclose()


def build_provider(settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requiere OPENAI_API_KEY")
        return OpenAIProvider(settings.openai_api_key, settings.openai_base_url)
    return MockProvider(latency_ms=settings.mock_llm_latency_ms, token_delay_ms=settings.mock_llm_token_delay_ms)
