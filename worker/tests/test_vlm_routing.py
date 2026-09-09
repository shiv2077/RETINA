"""Offline unit tests for VLMRouter's Stage 2 routing band.

No API calls: constructing an OpenAI client with a dummy key touches no
network, and should_run_stage2 is pure arithmetic over settings.
"""
from __future__ import annotations

import pytest

from retina_worker.config import Settings
from retina_worker.models.vlm_router import VLMRouter


def _router(**overrides) -> VLMRouter:
    return VLMRouter(
        settings=Settings(openai_api_key="sk-test-not-used", **overrides)
    )


class TestStage2Band:
    def test_lower_edge_follows_the_anomaly_threshold(self):
        """F17: at anomaly_threshold=0.3 a 0.4 score is flagged, so it must
        be eligible for Stage 2. The hardcoded 0.5 lower bound skipped it."""
        router = _router(anomaly_threshold=0.3)

        assert router.should_run_stage2(0.4) is True

    def test_scores_below_the_flag_threshold_do_not_trigger(self):
        router = _router(anomaly_threshold=0.3)

        assert router.should_run_stage2(0.29) is False

    def test_upper_bound_defaults_to_zero_point_nine(self):
        router = _router()

        assert router.stage2_trigger_max == pytest.approx(0.9)
        assert router.should_run_stage2(0.89) is True
        assert router.should_run_stage2(0.9) is False

    def test_upper_bound_is_configurable(self):
        router = _router(stage2_trigger_max=0.75)

        assert router.should_run_stage2(0.8) is False
        assert router.should_run_stage2(0.7) is True

    def test_missing_api_key_still_raises(self):
        with pytest.raises(ValueError, match="OpenAI API key"):
            VLMRouter(settings=Settings(openai_api_key=""))
