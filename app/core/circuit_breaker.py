"""Circuit breaker hacia el proveedor LLM.

Regla: N fallos dentro de una ventana deslizante de W segundos -> OPEN por D segundos.
Pasado D, el circuito queda HALF_OPEN y deja pasar **un único** request de prueba:
  - si tiene éxito -> CLOSED (se limpia el historial de fallos)
  - si falla       -> OPEN otra vez por D segundos

No usa locks: todo el estado se modifica sin `await` intermedios, así que dentro de
un event loop las transiciones son atómicas. El estado es por proceso (ver DECISIONS.md).
"""

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from enum import StrEnum

logger = logging.getLogger(__name__)


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitOpenError(Exception):
    def __init__(self, retry_after_s: int):
        super().__init__(f"circuit open, retry after {retry_after_s}s")
        self.retry_after_s = retry_after_s


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        failure_window_s: float = 30.0,
        open_duration_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        name: str = "llm",
    ):
        self.failure_threshold = failure_threshold
        self.failure_window_s = failure_window_s
        self.open_duration_s = open_duration_s
        self._clock = clock
        self.name = name

        self._failures: deque[float] = deque()
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._probe_in_flight = False

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> CircuitState:
        """Estado actual; resuelve la transición OPEN -> HALF_OPEN por tiempo."""
        if (
            self._state is CircuitState.OPEN
            and self._opened_at is not None
            and self._clock() - self._opened_at >= self.open_duration_s
        ):
            self._transition(CircuitState.HALF_OPEN)
        return self._state

    def retry_after_s(self) -> int:
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            remaining = self.open_duration_s - (self._clock() - self._opened_at)
            return max(1, math.ceil(remaining))
        return 1

    def _transition(self, new: CircuitState) -> None:
        if new is self._state:
            return
        old, self._state = self._state, new
        if new is CircuitState.OPEN:
            self._opened_at = self._clock()
        elif new is CircuitState.CLOSED:
            self._opened_at = None
            self._failures.clear()
        logger.warning("circuit breaker %s: %s -> %s", self.name, old.value, new.value)

    def _prune(self, now: float) -> None:
        cutoff = now - self.failure_window_s
        while self._failures and self._failures[0] < cutoff:
            self._failures.popleft()

    # --------------------------------------------------------------- protocol
    def acquire(self) -> bool:
        """Pide permiso para llamar. Devuelve True si es el request de prueba (half-open).

        Lanza CircuitOpenError si el circuito no deja pasar.
        """
        state = self.state
        if state is CircuitState.CLOSED:
            return False
        if state is CircuitState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        # OPEN, o HALF_OPEN con la prueba ya en curso
        raise CircuitOpenError(self.retry_after_s() if state is CircuitState.OPEN else 1)

    def record_success(self, probe: bool = False) -> None:
        if probe:
            self._probe_in_flight = False
        if self._state is CircuitState.HALF_OPEN:
            self._transition(CircuitState.CLOSED)

    def record_failure(self, probe: bool = False) -> None:
        now = self._clock()
        if probe:
            self._probe_in_flight = False
        if self._state is CircuitState.HALF_OPEN:
            self._transition(CircuitState.OPEN)
            return
        if self._state is CircuitState.OPEN:
            return
        self._failures.append(now)
        self._prune(now)
        if len(self._failures) >= self.failure_threshold:
            self._transition(CircuitState.OPEN)

    def release(self, probe: bool) -> None:
        """El request terminó sin veredicto (p.ej. el cliente canceló)."""
        if probe:
            self._probe_in_flight = False

    @asynccontextmanager
    async def guard(self, ignore: tuple[type[BaseException], ...] = ()) -> AsyncIterator[None]:
        """`async with breaker.guard(): ...` — registra éxito/fallo automáticamente.

        - asyncio.CancelledError (cliente desconectado) no cuenta como fallo.
        - Las excepciones en `ignore` (errores del cliente, 4xx) tampoco.
        """
        probe = self.acquire()
        try:
            yield
        except asyncio.CancelledError:
            self.release(probe)
            raise
        except ignore:
            self.release(probe)
            raise
        except BaseException:
            self.record_failure(probe)
            raise
        else:
            self.record_success(probe)
