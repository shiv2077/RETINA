"""D1: a declared product_class must bypass VLM identification entirely.

Inferring the product per image put a paid external API on the hot path of
every request, to determine something the submitting station already knows.
The router is meant to be the cold-start fallback CLAUDE.md 1.1 describes;
declaring the class is what makes that true.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from retina_worker.config import Settings, available_categories
from retina_worker.schemas import (
    InferenceJob,
    JobStatus,
    ModelType,
    PipelineStage,
)


def _job(product_class: str | None = None) -> InferenceJob:
    return InferenceJob(
        job_id="d1job",
        image_id="d1" + "b" * 60,
        model_type=ModelType.PATCHCORE,
        stage=PipelineStage.UNSUPERVISED,
        status=JobStatus.PENDING,
        submitted_at=datetime.utcnow(),
        product_class=product_class,
    )


class TestSchema:
    def test_product_class_defaults_to_none(self):
        assert _job().product_class is None

    def test_product_class_survives_serialization(self):
        restored = InferenceJob.model_validate_json(_job("bottle").model_dump_json())

        assert restored.product_class == "bottle"


class TestAvailableCategories:
    def test_lists_categories_with_checkpoints(self, tmp_path):
        for name in ("bottle", "hazelnut"):
            (tmp_path / f"patchcore_{name}.ckpt").touch()
        (tmp_path / "notes.txt").touch()

        assert available_categories(tmp_path) == ["bottle", "hazelnut"]

    def test_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert available_categories(tmp_path / "nope") == []

    def test_registry_and_api_agree(self, tmp_path):
        """Both sides must derive the set the same way, or the API will
        accept a class the worker cannot serve."""
        (tmp_path / "patchcore_tile.ckpt").touch()
        settings = Settings(patchcore_checkpoint_path=str(tmp_path))

        assert available_categories(settings.resolved_checkpoint_dir()) == ["tile"]


class TestWorkerRouting:
    """The worker must not call identify_product when the class is declared."""

    @pytest.fixture
    def worker(self, patched_redis, settings, monkeypatch):
        from retina_worker.worker import Worker

        monkeypatch.setattr(
            "retina_worker.worker.get_default_registry", lambda *a, **k: object()
        )
        monkeypatch.setattr(
            "retina_worker.worker.VLMRouter", lambda *a, **k: object()
        )
        return Worker(settings=settings)

    def test_declared_class_does_not_touch_the_session_cache(self, worker):
        """A declared class belongs to its own job. Caching it would steer
        later jobs that declared nothing."""
        worker._set_cached_product_class("carpet", 0.9)

        assert worker._get_cached_product_class() == "carpet"
        # Declaring a different class on one job must not overwrite it.
        job = _job("bottle")
        assert job.product_class == "bottle"
        assert worker._get_cached_product_class() == "carpet"
