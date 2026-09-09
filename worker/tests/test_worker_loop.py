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


class TestLabelingPoolGating:
    """F21: uncertainty is computed pre-Stage-2, so a confidently resolved
    case used to land in the pool anyway."""

    def _run(self, worker, job, result, monkeypatch):
        worker.redis.client.xadd(
            JOB_QUEUE_STREAM, {"job_data": job.model_dump_json()}
        )
        monkeypatch.setattr(worker, "_run_inference", lambda j: result)
        worker._process_next_job()
        return worker.redis.client.zrange(f"{KEY_PREFIX}:al:pool", 0, -1)

    def test_confident_stage2_verdict_keeps_the_sample_out_of_the_pool(
        self, worker, monkeypatch
    ):
        job = _job("resolved")
        result = _completed(job, uncertainty=0.9)
        result.stage2_verdict = "rejected_false_positive"
        result.stage2_confidence = 0.95

        assert self._run(worker, job, result, monkeypatch) == []

    def test_unsure_stage2_verdict_still_pools_the_sample(self, worker, monkeypatch):
        job = _job("unsure")
        result = _completed(job, uncertainty=0.9)
        result.stage2_verdict = "uncertain"
        result.stage2_confidence = 0.4

        assert self._run(worker, job, result, monkeypatch) == ["unsure"]

    def test_stage2_confidence_damps_the_uncertainty_score(self, worker):
        job = _job("blend")
        without = worker._build_result(
            job=job, anomaly_score=0.5, is_anomaly=True, product_class="wood",
            product_confidence=0.9, natural_description=None, defect_type=None,
            defect_location=None, defect_severity=None, routing_reason="x",
            vlm_model_used=None, vlm_api_cost_estimate_usd=0.0, heatmap=None,
            model_used=ModelType.PATCHCORE, t_start=0.0,
        )
        with_stage2 = worker._build_result(
            job=job, anomaly_score=0.5, is_anomaly=True, product_class="wood",
            product_confidence=0.9, natural_description=None, defect_type=None,
            defect_location=None, defect_severity=None, routing_reason="x",
            vlm_model_used=None, vlm_api_cost_estimate_usd=0.0, heatmap=None,
            model_used=ModelType.PATCHCORE, t_start=0.0,
            stage2_verdict="confirmed_anomaly", stage2_confidence=0.8,
        )

        assert without.active_learning.uncertainty_score == pytest.approx(1.0)
        assert with_stage2.active_learning.uncertainty_score == pytest.approx(0.2)


class TestScoreClamping:
    """F26: GPT-4o writes anomaly_score as a free float, but
    InferenceResult declares ge=0.0 le=1.0 — an out-of-range value used to
    fail the whole job on a ValidationError."""

    def _build(self, worker, score, model_used=ModelType.GPT4V):
        return worker._build_result(
            job=_job("clamp"), anomaly_score=score, is_anomaly=True,
            product_class="unknown", product_confidence=None,
            natural_description=None, defect_type=None, defect_location=None,
            defect_severity=None, routing_reason="unknown_product_zero_shot",
            vlm_model_used="gpt-4o", vlm_api_cost_estimate_usd=0.005,
            heatmap=None, model_used=model_used, t_start=0.0,
        )

    @pytest.mark.parametrize(
        ("raw", "expected"), [(1.7, 1.0), (-0.4, 0.0), (0.62, 0.62)]
    )
    def test_zero_shot_score_is_clamped(self, worker, raw, expected):
        assert self._build(worker, raw).anomaly_score == pytest.approx(expected)


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

    def test_poll_loop_reclaims_an_abandoned_entry(
        self, worker, settings, monkeypatch
    ):
        """F2: an entry left pending by a dead replica is picked up by the
        sweep at the top of the poll loop, not stranded forever."""
        from retina_worker.redis_client import RedisClient

        dead = RedisClient(settings.model_copy(update={"consumer_name": "dead-1"}))
        dead.client.xadd(
            JOB_QUEUE_STREAM, {"job_data": _job("orphan").model_dump_json()}
        )
        dead.read_job(block_ms=1)
        worker.settings = settings.model_copy(update={"job_reclaim_idle_ms": 0})
        monkeypatch.setattr(worker, "_run_inference", lambda j: _completed(j, 0.1))

        worker._process_next_job()

        assert worker.redis.get_result("orphan")["status"] == JobStatus.COMPLETED.value
        assert (
            worker.redis.client.xpending(JOB_QUEUE_STREAM, WORKER_GROUP)["pending"] == 0
        )

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
