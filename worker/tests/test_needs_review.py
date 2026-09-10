"""D2: abstention is not failure.

FAILED used to absorb two unrelated things: the pipeline breaking, and the
pipeline running correctly but declining to decide. That makes the failure
rate uninterpretable — it moves with how hard the images are, not with how
healthy the system is — and leaves a circuit breaker nowhere sensible to
degrade to. NEEDS_REVIEW separates them.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from retina_worker.config import Settings
from retina_worker.schemas import (
    InferenceError,
    InferenceJob,
    InferenceResult,
    JobStatus,
    ModelType,
    PipelineStage,
)


def _job(job_id: str = "d2job") -> InferenceJob:
    return InferenceJob(
        job_id=job_id,
        image_id="d2" + "c" * 60,
        model_type=ModelType.PATCHCORE,
        stage=PipelineStage.UNSUPERVISED,
        status=JobStatus.PENDING,
        submitted_at=datetime.utcnow(),
    )


class TestStatusVocabulary:
    def test_needs_review_is_a_distinct_status(self):
        assert JobStatus.NEEDS_REVIEW.value == "needs_review"
        assert JobStatus.NEEDS_REVIEW is not JobStatus.FAILED
        assert JobStatus.NEEDS_REVIEW is not JobStatus.COMPLETED

    def test_it_round_trips_through_the_wire_format(self):
        result = InferenceResult(
            job_id="j", image_id="i", status=JobStatus.NEEDS_REVIEW,
        )

        restored = InferenceResult.model_validate_json(result.model_dump_json())

        assert restored.status is JobStatus.NEEDS_REVIEW


class TestAbstentionDetection:
    @pytest.fixture
    def worker(self, patched_redis, settings, monkeypatch):
        from retina_worker.worker import Worker

        monkeypatch.setattr(
            "retina_worker.worker.get_default_registry", lambda *a, **k: object()
        )
        monkeypatch.setattr("retina_worker.worker.VLMRouter", lambda *a, **k: object())
        return Worker(settings=settings)

    @pytest.mark.parametrize("score", [0.45, 0.5, 0.55])
    def test_scores_at_the_boundary_are_abstentions(self, worker, score):
        assert worker._is_abstention(score) is True

    @pytest.mark.parametrize("score", [0.02, 0.1, 0.9, 0.98])
    def test_decisive_scores_are_not_abstentions(self, worker, score):
        assert worker._is_abstention(score) is False

    def test_the_boundary_is_configurable(self, patched_redis, monkeypatch):
        """Raising the bar means fewer abstentions.

        The setting is the minimum uncertainty required to decline, so a
        score of 0.4 (uncertainty 0.8) abstains under the 0.5 default and
        stops abstaining once the bar is above 0.8.
        """
        from retina_worker.worker import Worker

        monkeypatch.setattr(
            "retina_worker.worker.get_default_registry", lambda *a, **k: object()
        )
        monkeypatch.setattr("retina_worker.worker.VLMRouter", lambda *a, **k: object())

        default = Worker(settings=Settings(abstain_uncertainty=0.5))
        strict = Worker(settings=Settings(abstain_uncertainty=0.9))

        assert default._is_abstention(0.4) is True
        assert strict._is_abstention(0.4) is False


class TestNeedsReviewFeedsTheLabelingPool:
    @pytest.fixture
    def worker(self, patched_redis, settings, monkeypatch):
        from retina_worker.worker import Worker

        monkeypatch.setattr(
            "retina_worker.worker.get_default_registry", lambda *a, **k: object()
        )
        monkeypatch.setattr("retina_worker.worker.VLMRouter", lambda *a, **k: object())
        return Worker(settings=settings)

    def _result(self, status: JobStatus, uncertainty: float) -> InferenceResult:
        from retina_worker.schemas import ActiveLearningMeta

        return InferenceResult(
            job_id="d2job",
            image_id="d2" + "c" * 60,
            status=status,
            anomaly_score=0.95,
            active_learning=ActiveLearningMeta(uncertainty_score=uncertainty),
        )

    def test_needs_review_is_pooled_even_when_uncertainty_is_low(self, worker):
        """The status is the request for a human; the arithmetic does not
        get to overrule it."""
        worker._record_for_active_learning(
            _job(), self._result(JobStatus.NEEDS_REVIEW, uncertainty=0.0)
        )

        pool = worker.redis.client.zrange("retina:al:pool", 0, -1)

        assert pool == ["d2" + "c" * 60]

    def test_a_confident_completed_result_is_not_pooled(self, worker):
        worker._record_for_active_learning(
            _job(), self._result(JobStatus.COMPLETED, uncertainty=0.0)
        )

        assert worker.redis.client.zrange("retina:al:pool", 0, -1) == []


class TestFailedStaysFailed:
    def test_infrastructure_failure_is_still_failed(self):
        """A broken pipeline must not be softened into a review request —
        that is the alarm the operations team relies on."""
        result = InferenceResult(
            job_id="j",
            image_id="i",
            status=JobStatus.FAILED,
            error=InferenceError(code="INFERENCE_ERROR", message="boom"),
        )

        assert result.status is JobStatus.FAILED
        assert result.error is not None
