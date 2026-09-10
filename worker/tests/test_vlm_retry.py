"""F20: every OpenAI call is timed and bounded.

The client was constructed with no timeout and no call site passed one, so
a hung request stalled the single-threaded poll loop for as long as the
SDK's default allowed. No job of any kind progressed meanwhile. There was
also no retry, so a single 429 failed the job outright.

All offline: the client is replaced with a stub, so nothing here talks to
OpenAI or spends money.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from openai import APITimeoutError, InternalServerError, RateLimitError

from retina_worker.config import Settings
from retina_worker.models.vlm_router import VLMRouter, VLMUnavailableError


def _router(**overrides) -> VLMRouter:
    settings = Settings(openai_api_key="sk-test", **overrides)
    return VLMRouter(settings=settings)


class _Recorder:
    """Stands in for client.chat.completions.create."""

    def __init__(self, *, fail_times: int = 0, exc=None):
        self.calls: list[dict] = []
        self.fail_times = fail_times
        self.exc = exc or RateLimitError

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.fail_times:
            raise _make(self.exc)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]
        )


def _make(exc_type):
    """Construct an openai exception without a real HTTP response."""
    return exc_type.__new__(exc_type)


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Backoff is asserted by inspection, not by actually waiting."""
    monkeypatch.setattr(time, "sleep", lambda _s: None)


class TestTimeoutIsAlwaysPassed:
    def test_client_is_constructed_with_a_timeout(self):
        router = _router(openai_timeout_s=12.5)

        assert router.client.timeout == 12.5

    def test_sdk_retries_are_disabled_so_ours_are_authoritative(self):
        """Two retry layers would multiply: 3 of ours x 2 of the SDK's is
        six calls and six times the backoff budget."""
        assert _router().client.max_retries == 0

    def test_every_attempt_carries_a_timeout(self, monkeypatch):
        router = _router()
        rec = _Recorder()
        monkeypatch.setattr(router.client.chat.completions, "create", rec)

        router._call_with_retry("probe", model="gpt-4o", messages=[])

        assert rec.calls[0]["timeout"] > 0


class TestBoundedRetry:
    @pytest.mark.parametrize(
        "exc", [RateLimitError, APITimeoutError, InternalServerError]
    )
    def test_transient_failures_are_retried(self, monkeypatch, exc):
        router = _router()
        rec = _Recorder(fail_times=1, exc=exc)
        monkeypatch.setattr(router.client.chat.completions, "create", rec)

        router._call_with_retry("probe", model="gpt-4o", messages=[])

        assert len(rec.calls) == 2

    def test_retries_are_capped(self, monkeypatch):
        router = _router(gpt4v_max_retries=3)
        rec = _Recorder(fail_times=99)
        monkeypatch.setattr(router.client.chat.completions, "create", rec)

        with pytest.raises(VLMUnavailableError):
            router._call_with_retry("probe", model="gpt-4o", messages=[])

        assert len(rec.calls) == 3

    def test_exhaustion_raises_a_distinct_error(self, monkeypatch):
        """Not a parse error and not a generic exception: the breaker needs
        to count 'the API did not answer' specifically."""
        router = _router()
        monkeypatch.setattr(
            router.client.chat.completions, "create", _Recorder(fail_times=99)
        )

        with pytest.raises(VLMUnavailableError):
            router._call_with_retry("probe", model="gpt-4o", messages=[])

    def test_a_non_transient_error_is_not_retried(self, monkeypatch):
        """A malformed request fails identically next time; retrying it just
        burns the deadline."""
        router = _router()
        calls = []

        def _raise_value_error(**kwargs):
            calls.append(kwargs)
            raise ValueError("bad request")

        monkeypatch.setattr(
            router.client.chat.completions, "create", _raise_value_error
        )

        with pytest.raises(ValueError):
            router._call_with_retry("probe", model="gpt-4o", messages=[])

        assert len(calls) == 1


class TestWallClockDeadline:
    def test_deadline_stops_further_attempts(self, monkeypatch):
        """Per-attempt timeouts alone do not bound the total: the deadline
        is what stops one job monopolising the loop."""
        router = _router(openai_total_deadline_s=0.0, gpt4v_max_retries=5)
        rec = _Recorder(fail_times=99)
        monkeypatch.setattr(router.client.chat.completions, "create", rec)

        with pytest.raises(VLMUnavailableError):
            router._call_with_retry("probe", model="gpt-4o", messages=[])

        assert rec.calls == []

    def test_attempt_timeout_never_exceeds_remaining_budget(self, monkeypatch):
        router = _router(openai_timeout_s=30.0, openai_total_deadline_s=5.0)
        rec = _Recorder()
        monkeypatch.setattr(router.client.chat.completions, "create", rec)

        router._call_with_retry("probe", model="gpt-4o", messages=[])

        assert rec.calls[0]["timeout"] <= 5.0
