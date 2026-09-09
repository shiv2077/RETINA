"""Integration tests for PatchCoreRegistry.

Needs torch, anomalib and the ~230 MB per-category checkpoints in
checkpoints/, so the module is marked `integration` and deselected by the
default addopts in pyproject.toml. Run deliberately:

    python -m pytest -m integration tests/test_patchcore_registry.py

Converted from a print-based script (formerly
scripts/test_patchcore_registry.py) that never asserted.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

pytestmark = pytest.mark.integration

EXPECTED_CATEGORIES = sorted([
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
])


@pytest.fixture(scope="module")
def registry():
    pytest.importorskip("torch", reason="torch not installed")
    pytest.importorskip("anomalib", reason="anomalib not installed")
    if not CHECKPOINT_DIR.is_dir() or not any(CHECKPOINT_DIR.glob("patchcore_*.ckpt")):
        pytest.skip(f"no patchcore checkpoints in {CHECKPOINT_DIR}")

    from retina_worker.models.patchcore_registry import PatchCoreRegistry

    return PatchCoreRegistry(checkpoint_dir=CHECKPOINT_DIR, max_cached=2)


def test_available_categories_lists_all_trained_checkpoints(registry):
    assert registry.available_categories() == EXPECTED_CATEGORIES


def test_has_checkpoint_discriminates_known_from_unknown(registry):
    assert registry.has_checkpoint("bottle")
    assert not registry.has_checkpoint("pastry")


def test_get_returns_a_patchcore_instance(registry):
    from anomalib.models import Patchcore

    assert isinstance(registry.get("bottle"), Patchcore)


def test_second_get_is_a_cache_hit(registry):
    first = registry.get("bottle")

    start = time.time()
    second = registry.get("bottle")
    elapsed = time.time() - start

    assert second is first
    assert elapsed < 0.05


def test_lru_evicts_the_oldest_at_max_cached(registry):
    registry.get("bottle")
    registry.get("leather")
    registry.get("wood")  # should evict bottle

    stats = registry.stats()

    assert stats["cache_size"] == 2
    assert set(stats["cached_categories"]) == {"leather", "wood"}


def test_inference_returns_a_finite_score(registry):
    import torch
    from PIL import Image
    from torchvision.transforms.v2 import functional as TVF  # noqa: N812

    img_path = REPO_ROOT / "mvtec/bottle/test/good/000.png"
    if not img_path.is_file():
        pytest.skip("mvtec bottle test image not present")

    tensor = TVF.to_image(Image.open(img_path).convert("RGB"))
    batch = TVF.to_dtype(tensor, torch.float32, scale=True).unsqueeze(0)
    if torch.cuda.is_available():
        batch = batch.cuda()

    model = registry.get("bottle")
    with torch.no_grad():
        out = model(batch)

    assert out.pred_score is not None
    score = float(out.pred_score.item())
    assert score == score  # not NaN
