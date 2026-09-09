"""Integration tests for VLMRouter.

Every test here makes a real, billed OpenAI API call, so the whole module
is marked `integration` and is deselected by the default addopts in
pyproject.toml. Run deliberately:

    python -m pytest -m integration tests/test_vlm_router.py

Tests the router under its correct usage contract: the VLM explains images
PatchCore has already flagged, it is not a standalone verdict.

Converted from a print-based script (formerly scripts/test_vlm_router.py)
that reported PASS/FAIL to stdout without ever asserting, so a regression
could not fail a build.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = pytest.mark.integration


def _load_api_key() -> str | None:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    env = REPO_ROOT / ".env"
    if not env.is_file():
        return None
    for line in env.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            key = line.split("=", 1)[1].strip()
            os.environ["OPENAI_API_KEY"] = key
            return key
    return None


def _require(path: str) -> bytes:
    full = REPO_ROOT / path
    if not full.is_file():
        pytest.skip(f"fixture image not present: {path}")
    return full.read_bytes()


@pytest.fixture(scope="module")
def router():
    if _load_api_key() is None:
        pytest.skip("no OPENAI_API_KEY in environment or .env")
    from retina_worker.models.vlm_router import VLMRouter

    return VLMRouter()


@pytest.mark.parametrize(
    ("image", "expected_class"),
    [
        ("mvtec/bottle/test/good/000.png", "bottle"),
        ("mvtec/leather/test/good/000.png", "leather"),
        ("mvtec/wood/test/good/000.png", "wood"),
    ],
)
def test_identify_product(router, image, expected_class):
    result = router.identify_product(_require(image))

    assert result.product_class == expected_class
    assert result.is_known_category


def test_identify_bottle_is_confident(router):
    result = router.identify_product(_require("mvtec/bottle/test/good/000.png"))

    assert result.confidence > 0.7


def test_describe_defect_on_high_scoring_anomaly(router):
    result = router.describe_defect(
        _require("mvtec/bottle/test/broken_large/000.png"),
        product_class="bottle",
        anomaly_score=0.92,
    )

    assert result.has_defect


def test_low_score_short_circuits_the_api_call(router):
    """The runtime guard must fire before the VLM is consulted, so a clean
    image with a low score cannot be talked into having a defect."""
    result = router.describe_defect(
        _require("mvtec/bottle/test/good/000.png"),
        product_class="bottle",
        anomaly_score=0.08,
    )

    assert not result.has_defect
    assert result.confidence > 0.9


def test_zero_shot_catches_an_obvious_defect(router):
    result = router.zero_shot_detect(
        _require("mvtec/pill/test/crack/000.png"),
        product_description="pharmaceutical pill",
    )

    assert result.is_anomaly


@pytest.mark.xfail(
    reason="Documented limitation: cable_swap is physically intact. Zero-shot "
    "cannot know the correct wiring convention without a reference image, so "
    "this class of defect is missed. Locked in as xfail rather than asserted "
    "as a pass, so it flips to XPASS if reference support is ever added.",
    strict=False,
)
def test_zero_shot_misses_reference_dependent_defect(router):
    result = router.zero_shot_detect(
        _require("mvtec/cable/test/cable_swap/000.png"),
        product_description="electrical cable",
    )

    assert result.is_anomaly


def test_zero_shot_on_a_good_image_is_advisory_only(router):
    """VLM priming bias sometimes false-positives on clean images. That is
    expected and is why production routes through PatchCore first, so this
    asserts only that the call returns a well-formed result."""
    result = router.zero_shot_detect(
        _require("mvtec/pill/test/good/000.png"),
        product_description="pharmaceutical pill",
    )

    assert isinstance(result.is_anomaly, bool)
    assert 0.0 <= result.anomaly_score <= 1.0
