"""A circuit breaker for the VLM dependency.

`identify_product` sits on the critical path of every job whose product
class was not declared (DECISIONS.md 17), so an OpenAI outage used to fail
100% of such jobs — each one first paying the full retry budget from
entry 20, and each one recorded as FAILED, which is an operations alarm
for something operations cannot fix.

The breaker turns a repeated, already-diagnosed failure into a fast,
honest one: after `failure_threshold` consecutive failures it opens, and
while open calls are refused immediately instead of re-proving the outage
per job. After `cooldown_s` it half-opens and lets exactly one probe
through; success closes it, failure re-opens it for another cooldown.

Deliberately not thread-safe beyond a simple lock, and deliberately
in-process: the worker is single-threaded and one replica's view of the
API's health is the view that matters for its own routing decisions. A
shared breaker in Redis would couple replicas together and add a network
round trip to the fast path it exists to protect.
"""
from __future__ import annotations

import threading
import time
from enum import Enum

import structlog

logger = structlog.get_logger()


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised instead of attempting a call the breaker believes will fail."""


class CircuitBreaker:
    """Consecutive-failure breaker with a half-open probe."""

    def __init__(
        self,
        name: str,
        failure_threshold: int,
        cooldown_s: float,
        clock=time.monotonic,
    ):
        self.name = name
        self.failure_threshold = max(1, failure_threshold)
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False

    @property
    def state(self) -> BreakerState:
        with self._lock:
            return self._state_unlocked()

    def _state_unlocked(self) -> BreakerState:
        if self._opened_at is None:
            return BreakerState.CLOSED
        if self._clock() - self._opened_at >= self.cooldown_s:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def allows(self) -> bool:
        """True if a call should be attempted now.

        In half-open only one probe is allowed at a time; a second caller
        is refused rather than mounting a thundering herd against an API
        that has not yet proven it recovered.
        """
        with self._lock:
            state = self._state_unlocked()
            if state is BreakerState.CLOSED:
                return True
            if state is BreakerState.HALF_OPEN and not self._half_open_in_flight:
                self._half_open_in_flight = True
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            was = self._state_unlocked()
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_in_flight = False
        if was is not BreakerState.CLOSED:
            logger.info("circuit_closed", breaker=self.name)

    def record_failure(self) -> None:
        with self._lock:
            self._half_open_in_flight = False
            self._consecutive_failures += 1
            should_open = self._consecutive_failures >= self.failure_threshold
            if should_open:
                already_open = self._opened_at is not None
                self._opened_at = self._clock()
                failures = self._consecutive_failures
        if should_open and not already_open:
            logger.warning(
                "circuit_opened",
                breaker=self.name,
                consecutive_failures=failures,
                cooldown_s=self.cooldown_s,
            )

    def guard(self) -> None:
        """Raise CircuitOpenError if a call must not be attempted."""
        if not self.allows():
            raise CircuitOpenError(
                f"{self.name} circuit is open; refusing call without attempting it"
            )
