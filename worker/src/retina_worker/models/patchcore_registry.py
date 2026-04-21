"""Per-category PatchCore checkpoint registry with LRU GPU caching.

One instance serves the whole worker process. Checkpoints live at
`./checkpoints/patchcore_{category}.ckpt` (anomalib format, produced by
`scripts/train_anomalib.py`). Models are loaded lazily on first `get()` and
kept in a thread-safe LRU cache sized for the 6 GB RTX 3060 budget.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

import structlog
import torch
from anomalib.models import Patchcore

logger = structlog.get_logger()


CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_NAMING = "patchcore_{category}.ckpt"

# Tuned for RTX 3060 Laptop (6 GB). Each checkpoint is ~230 MB on disk; the
# in-GPU footprint at inference is roughly 1.5 GB including the WideResNet50
# backbone and coreset memory bank. Two hot models fit; more risks OOM.
MAX_CACHED_MODELS = 2


class PatchCoreRegistry:
    """Thread-safe LRU cache of per-category Patchcore models."""

    def __init__(
        self,
        checkpoint_dir: Path = CHECKPOINT_DIR,
        max_cached: int = MAX_CACHED_MODELS,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_cached = max_cached
        self._cache: OrderedDict[str, Patchcore] = OrderedDict()
        self._lock = threading.Lock()
        self._load_times: dict[str, float] = {}

    def available_categories(self) -> list[str]:
        if not self.checkpoint_dir.exists():
            return []
        return sorted(
            f.stem.removeprefix("patchcore_")
            for f in self.checkpoint_dir.glob("patchcore_*.ckpt")
        )

    def has_checkpoint(self, category: str) -> bool:
        return (self.checkpoint_dir /
                CHECKPOINT_NAMING.format(category=category)).is_file()

    def get(self, category: str) -> Patchcore:
        """Return a loaded, eval-mode, GPU-resident model for `category`.

        Raises FileNotFoundError if no checkpoint exists on disk.
        """
        with self._lock:
            if category in self._cache:
                self._cache.move_to_end(category)
                logger.debug("patchcore_cache_hit", category=category)
                return self._cache[category]

            ckpt_path = self.checkpoint_dir / CHECKPOINT_NAMING.format(
                category=category,
            )
            if not ckpt_path.is_file():
                raise FileNotFoundError(
                    f"No Patchcore checkpoint for category '{category}' at "
                    f"{ckpt_path}. Available: {self.available_categories()}"
                )

            t0 = time.time()
            logger.info("patchcore_loading", category=category, path=str(ckpt_path))

            # PyTorch 2.7 defaults torch.load to weights_only=True, which rejects
            # Lightning checkpoints containing anomalib custom types (PrecisionType,
            # ModelSize, etc.). These checkpoints were created by this same
            # codebase and are trusted. Temporarily override the default for this
            # load call only.
            _original_torch_load = torch.load

            def _load_with_weights_only_false(*args, **kwargs):
                # Force-override: Lightning passes weights_only=True explicitly,
                # so setdefault does not take effect. We know the file is ours.
                kwargs["weights_only"] = False
                return _original_torch_load(*args, **kwargs)

            torch.load = _load_with_weights_only_false
            try:
                model = Patchcore.load_from_checkpoint(str(ckpt_path))
            finally:
                torch.load = _original_torch_load

            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            load_s = time.time() - t0
            self._load_times[category] = load_s
            logger.info(
                "patchcore_loaded",
                category=category,
                seconds=round(load_s, 2),
            )

            while len(self._cache) >= self.max_cached:
                evicted_cat, evicted_model = self._cache.popitem(last=False)
                del evicted_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("patchcore_evicted", category=evicted_cat)

            self._cache[category] = model
            return model

    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_categories": list(self._cache.keys()),
                "cache_size": len(self._cache),
                "max_cached": self.max_cached,
                "available_on_disk": self.available_categories(),
                "load_times_s": dict(self._load_times),
            }


_default_registry: PatchCoreRegistry | None = None


def get_default_registry() -> PatchCoreRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = PatchCoreRegistry()
    return _default_registry
