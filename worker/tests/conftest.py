"""Shared pytest fixtures for the worker test suite.

Path setup: retina_worker is not pip-installed in the dev environment, so
worker/src goes on sys.path here rather than requiring `pip install -e .`
before the suite will collect.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKER_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = WORKER_ROOT.parent

sys.path.insert(0, str(WORKER_ROOT / "src"))

import fakeredis  # noqa: E402
import redis as real_redis  # noqa: E402

from retina_worker import redis_client as redis_client_module  # noqa: E402
from retina_worker.config import Settings  # noqa: E402

# Why fakeredis rather than the alternatives:
#
# RedisClient leans on real Redis semantics that a naive double cannot
# emulate — consumer groups (XREADGROUP/XACK against a pending entries
# list), sorted-set range trimming for the active-learning pool, and TTL
# expiry on result hashes. fakeredis implements all of these in-process
# (verified: XADD/XGROUP_CREATE/XREADGROUP/XACK/XPENDING/ZADD/EXPIRE).
#
# Rejected: unittest.mock.MagicMock. It records calls but has no storage
# semantics, so `xreadgroup` returns whatever the test author asserts it
# returns. That makes the test a restatement of the implementation rather
# than a check on it — a test suite that would pass against a client
# physically incapable of consuming a stream.
#
# Also rejected: spawning a real redis-server subprocess per session.
# Highest fidelity, and the binary happens to exist on this machine, but
# it makes the suite silently unrunnable anywhere redis-server is not
# installed. The point of this slot is a gate that runs everywhere.


@pytest.fixture
def fake_redis_server() -> fakeredis.FakeServer:
    """A single in-process Redis backing store, shared by every client
    created within one test."""
    return fakeredis.FakeServer()


@pytest.fixture
def patched_redis(monkeypatch, fake_redis_server):
    """Redirect redis.from_url to fakeredis for the duration of a test.

    RedisClient.__init__ calls redis.from_url directly (redis_client.py:80)
    with no injection seam, so the seam is made here instead of changing
    application code.
    """
    def _from_url(url: str, **kwargs):
        kwargs.pop("decode_responses", None)
        return fakeredis.FakeRedis(
            server=fake_redis_server, decode_responses=True, **kwargs
        )

    monkeypatch.setattr(redis_client_module.redis, "from_url", _from_url)
    yield fake_redis_server
    monkeypatch.setattr(redis_client_module.redis, "from_url", real_redis.from_url)


@pytest.fixture
def settings() -> Settings:
    """Worker settings pointed at the (patched) local Redis."""
    return Settings(redis_url="redis://localhost:6379", consumer_name="test-worker")


@pytest.fixture
def redis_client(patched_redis, settings):
    """A RedisClient wired to fakeredis, consumer group already created."""
    return redis_client_module.RedisClient(settings)
