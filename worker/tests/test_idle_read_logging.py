"""B3: an idle queue must not look like a failure in the logs.

read_job blocks for block_ms waiting for work. redis-py 7.x reports an
expired BLOCK by returning an empty list; 8.x raises TimeoutError, which
subclasses RedisError, so the catch-all in read_job logged every idle
interval at ERROR — one every 5 seconds forever on a quiet queue. The
container resolved 8.x while the dev machine ran 7.x, so the noise only
appeared once the worker was containerized.

Both client behaviours must mean the same thing to the caller, and
neither may emit an ERROR. Genuine Redis failures still must.
"""
from __future__ import annotations

import redis
from structlog.testing import capture_logs


def _errors(logs: list[dict]) -> list[dict]:
    return [entry for entry in logs if entry.get("log_level") == "error"]


class TestIdleReadIsNotAnError:
    def test_empty_read_returns_none_without_an_error_record(self, redis_client):
        """The redis-py 7.x shape: an empty return from a blocking read."""
        with capture_logs() as logs:
            result = redis_client.read_job(block_ms=1)

        assert result is None
        assert _errors(logs) == []

    def test_block_timeout_returns_none_without_an_error_record(
        self, redis_client, monkeypatch
    ):
        """The redis-py 8.x shape: the same condition raised as TimeoutError."""
        def _raise_timeout(*args, **kwargs):
            raise redis.exceptions.TimeoutError("Timeout reading from socket")

        monkeypatch.setattr(redis_client.client, "xreadgroup", _raise_timeout)

        with capture_logs() as logs:
            result = redis_client.read_job(block_ms=1)

        assert result is None
        assert _errors(logs) == []

    def test_idle_read_is_still_observable_at_debug(
        self, redis_client, monkeypatch
    ):
        """Demoted, not silenced — the signal is still there when wanted."""
        def _raise_timeout(*args, **kwargs):
            raise redis.exceptions.TimeoutError("Timeout reading from socket")

        monkeypatch.setattr(redis_client.client, "xreadgroup", _raise_timeout)

        with capture_logs() as logs:
            redis_client.read_job(block_ms=1)

        assert [e for e in logs if e.get("event") == "read_job_idle"]


class TestRealFailuresStillLogAtError:
    def test_connection_error_still_logs_an_error(self, redis_client, monkeypatch):
        """A real outage must not be demoted along with the idle case."""
        def _raise_conn(*args, **kwargs):
            raise redis.exceptions.ConnectionError("connection refused")

        monkeypatch.setattr(redis_client.client, "xreadgroup", _raise_conn)

        with capture_logs() as logs:
            result = redis_client.read_job(block_ms=1)

        assert result is None
        assert len(_errors(logs)) == 1

    def test_response_error_still_logs_an_error(self, redis_client, monkeypatch):
        def _raise_response(*args, **kwargs):
            raise redis.exceptions.ResponseError("NOGROUP no such consumer group")

        monkeypatch.setattr(redis_client.client, "xreadgroup", _raise_response)

        with capture_logs() as logs:
            redis_client.read_job(block_ms=1)

        assert len(_errors(logs)) == 1

    def test_a_real_job_still_comes_through(self, redis_client):
        """Guards against fixing the noise by breaking the read."""
        from datetime import datetime

        from retina_worker.schemas import (
            InferenceJob,
            JobStatus,
            ModelType,
            PipelineStage,
        )

        job = InferenceJob(
            job_id="b3job",
            image_id="b3" + "a" * 60,
            model_type=ModelType.PATCHCORE,
            stage=PipelineStage.UNSUPERVISED,
            status=JobStatus.PENDING,
            submitted_at=datetime.utcnow(),
        )
        redis_client.client.xadd(
            "retina:jobs:queue", {"job_data": job.model_dump_json()}
        )

        got = redis_client.read_job(block_ms=1)

        assert got is not None
        assert got[1].job_id == "b3job"
