"""Regression tests for the worker poll loop and its failure paths.

All offline: the VLM router is never called (inference is stubbed) and Redis
is fakeredis via the `patched_redis` fixture.
"""
from __future__ import annotations

from datetime import datetime

import pytest
import redis

from retina_worker.redis_client import JOB_QUEUE_STREAM, KEY_PREFIX, WORKER_GROUP
from retina_worker.schemas import (
    ActiveLearningMeta,
    InferenceJob,
    InferenceResult,
    JobStatus,
    ModelType,
    PipelineStage,
)


def _job(job_id: str = "job1") -> InferenceJob:
    return InferenceJob(
        job_id=job_id,
        image_id=job_id,
        model_type=ModelType.PATCHCORE,
        stage=PipelineStage.UNSUPERVISED,
        status=JobStatus.PENDING,
        submitted_at=datetime.utcnow(),
        image_path=f"/tmp/{job_id}.png",
    )


def _completed(job: InferenceJob, uncertainty: float = 0.9) -> InferenceResult:
    return InferenceResult(
        job_id=job.job_id,
        image_id=job.image_id,
        status=JobStatus.COMPLETED,
        created_at=datetime.utcnow(),
        stage=job.stage,
        anomaly_score=0.55,
        is_anomaly=True,
        confidence=1.0 - uncertainty,
        active_learning=ActiveLearningMeta(uncertainty_score=uncertainty),
    )


@pytest.fixture
def worker(patched_redis, settings, monkeypatch):
    """A Worker whose Redis is fakeredis and whose VLM router is never used."""
    from retina_worker.worker import Worker

    monkeypatch.setattr(
        "retina_worker.worker.VLMRouter", lambda **kwargs: object(),
    )
    return Worker(settings.model_copy(update={"openai_api_key": "sk-test"}))


class TestResultDurability:
    def test_pool_failure_does_not_downgrade_a_completed_result(
        self, worker, monkeypatch
    ):
        """F3: a Redis error from add_to_labeling_pool must not overwrite the
        already-durable COMPLETED record with a FAILED one."""
        job = _job()
        worker.redis.client.xadd(
            JOB_QUEUE_STREAM, {"job_data": job.model_dump_json()}
        )
        monkeypatch.setattr(worker, "_run_inference", lambda j: _completed(j))

        def _boom(**kwargs):
            raise redis.RedisError("pool write failed")

        monkeypatch.setattr(worker.redis, "add_to_labeling_pool", _boom)

        worker._process_next_job()

        stored = worker.redis.get_result(job.job_id)
        assert stored is not None
        assert stored["status"] == JobStatus.COMPLETED.value
        assert stored["error"] is None

    def test_stats_failure_does_not_downgrade_a_completed_result(
        self, worker, monkeypatch
    ):
        job = _job("job2")
        worker.redis.client.xadd(
            JOB_QUEUE_STREAM, {"job_data": job.model_dump_json()}
        )
        monkeypatch.setattr(worker, "_run_inference", lambda j: _completed(j, 0.1))
        monkeypatch.setattr(
            worker.redis,
            "increment_completed_jobs",
            lambda: (_ for _ in ()).throw(redis.RedisError("stats down")),
        )

        worker._process_next_job()

        assert worker.redis.get_result("job2")["status"] == JobStatus.COMPLETED.value

    def test_inference_failure_still_writes_a_failed_result(self, worker, monkeypatch):
        job = _job("job3")
        worker.redis.client.xadd(
            JOB_QUEUE_STREAM, {"job_data": job.model_dump_json()}
        )

        def _explode(j):
            raise RuntimeError("model blew up")

        monkeypatch.setattr(worker, "_run_inference", _explode)

        worker._process_next_job()

        stored = worker.redis.get_result("job3")
        assert stored["status"] == JobStatus.FAILED.value
        assert stored["error"]["code"] == "INFERENCE_ERROR"
        assert (
            worker.redis.client.xpending(JOB_QUEUE_STREAM, WORKER_GROUP)["pending"] == 0
        )
