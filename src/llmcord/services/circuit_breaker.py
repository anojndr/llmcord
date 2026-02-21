"""Simple async circuit breaker for transient provider outages.

This is intentionally lightweight: it tracks consecutive-ish failures within a
rolling time window and opens the circuit for a cooldown period.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass(slots=True)
class CircuitState:
    """State for a single circuit key."""

    failure_count: int = 0
    window_started_at: float = 0.0
    opened_until: float = 0.0


@dataclass(slots=True)
class CircuitBreaker:
    """Async circuit breaker with a rolling failure window and cooldown."""

    failure_threshold: int = 3
    window_seconds: float = 60.0
    open_seconds: float = 90.0
    _states: dict[str, CircuitState] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def is_open(self, key: str) -> bool:
        """Return True when the circuit for `key` is currently open."""
        now = time.monotonic()
        async with self._lock:
            state = self._states.get(key)
            return bool(state and state.opened_until > now)

    async def record_success(self, key: str) -> None:
        """Record a success and reset circuit state for `key`."""
        async with self._lock:
            self._states.pop(key, None)

    async def record_failure(self, key: str) -> bool:
        """Record a failure and return True if the circuit is open after recording."""
        now = time.monotonic()
        async with self._lock:
            state = self._states.get(key)
            if state is None:
                state = CircuitState(failure_count=0, window_started_at=now)
                self._states[key] = state

            if state.opened_until > now:
                return True

            if (now - state.window_started_at) > self.window_seconds:
                state.failure_count = 0
                state.window_started_at = now

            state.failure_count += 1
            if state.failure_count >= self.failure_threshold:
                state.opened_until = now + self.open_seconds
                return True

            return False


GLOBAL_PROVIDER_CIRCUIT_BREAKER = CircuitBreaker()
