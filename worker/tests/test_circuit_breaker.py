"""F4: an OpenAI outage must not fail every job.

identify_product is on the critical path of every job without a declared
product_class, so an outage failed 100% of them — each paying the full
retry budget first, and each recorded as FAILED, which pages someone
about a dependency they cannot fix.

The breaker makes the second and subsequent failures fast, and D2's
NEEDS_REVIEW gives them somewhere honest to land. Entirely offline: the
clock is injected, so no test sleeps.
"""
from __future__ import annotations

import pytest

from retina_worker.circuit_breaker import (
    BreakerState,
    CircuitBreaker,
    CircuitOpenError,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def breaker(clock) -> CircuitBreaker:
    return CircuitBreaker(
        name="test", failure_threshold=3, cooldown_s=60.0, clock=clock
    )


class TestClosedState:
    def test_starts_closed(self, breaker):
        assert breaker.state is BreakerState.CLOSED
        assert breaker.allows() is True

    def test_failures_below_the_threshold_keep_it_closed(self, breaker):
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.state is BreakerState.CLOSED
        assert breaker.allows() is True

    def test_a_success_resets_the_streak(self, breaker):
        """Consecutive, not cumulative: intermittent failures against a
        healthy API must not eventually trip it."""
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.state is BreakerState.CLOSED


class TestOpening:
    def test_opens_at_the_threshold(self, breaker):
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state is BreakerState.OPEN

    def test_open_refuses_without_attempting(self, breaker):
        for _ in range(3):
            breaker.record_failure()

        assert breaker.allows() is False
        with pytest.raises(CircuitOpenError):
            breaker.guard()

    def test_threshold_is_configurable(self, clock):
        eager = CircuitBreaker(
            name="t", failure_threshold=1, cooldown_s=10.0, clock=clock
        )

        eager.record_failure()

        assert eager.state is BreakerState.OPEN


class TestHalfOpen:
    def test_stays_open_until_the_cooldown_elapses(self, breaker, clock):
        for _ in range(3):
            breaker.record_failure()

        clock.advance(59.0)

        assert breaker.state is BreakerState.OPEN
        assert breaker.allows() is False

    def test_half_opens_after_the_cooldown(self, breaker, clock):
        for _ in range(3):
            breaker.record_failure()

        clock.advance(60.0)

        assert breaker.state is BreakerState.HALF_OPEN

    def test_exactly_one_probe_is_admitted(self, breaker, clock):
        """A queue of waiting jobs must not all rush an API that has not
        yet proven it recovered."""
        for _ in range(3):
            breaker.record_failure()
        clock.advance(60.0)

        assert breaker.allows() is True
        assert breaker.allows() is False

    def test_a_successful_probe_closes_it(self, breaker, clock):
        for _ in range(3):
            breaker.record_failure()
        clock.advance(60.0)
        breaker.allows()

        breaker.record_success()

        assert breaker.state is BreakerState.CLOSED
        assert breaker.allows() is True

    def test_a_failed_probe_reopens_for_another_cooldown(self, breaker, clock):
        for _ in range(3):
            breaker.record_failure()
        clock.advance(60.0)
        breaker.allows()

        breaker.record_failure()

        assert breaker.state is BreakerState.OPEN
        clock.advance(59.0)
        assert breaker.state is BreakerState.OPEN
        clock.advance(1.0)
        assert breaker.state is BreakerState.HALF_OPEN


class TestWorkerDegradesToNeedsReview:
    def test_settings_expose_the_thresholds(self):
        from retina_worker.config import Settings

        s = Settings(
            vlm_breaker_failure_threshold=7, vlm_breaker_cooldown_s=12.0
        )

        assert s.vlm_breaker_failure_threshold == 7
        assert s.vlm_breaker_cooldown_s == 12.0

    def test_worker_builds_a_breaker_from_config(self, patched_redis, monkeypatch):
        from retina_worker.config import Settings
        from retina_worker.worker import Worker

        monkeypatch.setattr(
            "retina_worker.worker.get_default_registry", lambda *a, **k: object()
        )
        monkeypatch.setattr("retina_worker.worker.VLMRouter", lambda *a, **k: object())
        worker = Worker(
            settings=Settings(
                vlm_breaker_failure_threshold=2, vlm_breaker_cooldown_s=5.0
            )
        )

        assert worker.vlm_breaker.failure_threshold == 2
        assert worker.vlm_breaker.cooldown_s == 5.0
        assert worker.vlm_breaker.state is BreakerState.CLOSED
