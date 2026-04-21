"""End-to-end tests for PatchCoreRegistry.

Validates the registry works before we wire it into worker.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker" / "src"))

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F  # noqa: N812

from anomalib.models import Patchcore  # noqa: E402
from retina_worker.models.patchcore_registry import PatchCoreRegistry  # noqa: E402


EXPECTED_CATEGORIES = sorted([
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
])

results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}  {detail}")
    results.append((name, condition, detail))


def preprocess(img_path: Path) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    tensor = F.to_image(img)
    tensor = F.to_dtype(tensor, torch.float32, scale=True)
    batch = tensor.unsqueeze(0)
    if torch.cuda.is_available():
        batch = batch.cuda()
    return batch


def main() -> None:
    reg = PatchCoreRegistry(checkpoint_dir=ROOT / "checkpoints", max_cached=2)

    # T1
    avail = reg.available_categories()
    check(
        "T1 available_categories returns 15 trained categories",
        avail == EXPECTED_CATEGORIES,
        f"got {len(avail)}: {avail}",
    )

    # T2
    check(
        "T2 has_checkpoint bottle=True, pastry=False",
        reg.has_checkpoint("bottle") and not reg.has_checkpoint("pastry"),
    )

    # T3
    t0 = time.time()
    m = reg.get("bottle")
    load_s = time.time() - t0
    check(
        "T3 get('bottle') returns a Patchcore instance",
        isinstance(m, Patchcore),
        f"load={load_s:.2f}s",
    )

    # T4 — second call is a cache hit, must be fast
    t0 = time.time()
    m2 = reg.get("bottle")
    hit_s = time.time() - t0
    check(
        "T4 cache hit <0.05s and same object",
        (m2 is m) and hit_s < 0.05,
        f"hit={hit_s*1000:.2f}ms",
    )

    # T5 — LRU eviction with max_cached=2
    reg.get("leather")
    reg.get("wood")  # should evict bottle
    stats = reg.stats()
    check(
        "T5 LRU evicts oldest at max_cached=2",
        stats["cache_size"] == 2
        and "bottle" not in stats["cached_categories"]
        and set(stats["cached_categories"]) == {"leather", "wood"},
        f"cached={stats['cached_categories']}",
    )

    # T6 — real inference
    # Reload bottle for inference (was evicted); will also evict leather.
    bottle = reg.get("bottle")
    img_path = ROOT / "mvtec/bottle/test/good/000.png"
    batch = preprocess(img_path)
    with torch.no_grad():
        out = bottle(batch)
    score = float(out.pred_score.item()) if out.pred_score is not None else None
    amap_shape = tuple(out.anomaly_map.shape) if out.anomaly_map is not None else None
    check(
        "T6 inference on bottle/good/000.png returns a finite score",
        score is not None and (0.0 <= score <= 1.0 or score == score),  # not NaN
        f"score={score:.4f} anomaly_map_shape={amap_shape}",
    )

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n==== {passed}/{len(results)} passed ====")
    for name, ok, detail in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}  {detail}")


if __name__ == "__main__":
    main()
