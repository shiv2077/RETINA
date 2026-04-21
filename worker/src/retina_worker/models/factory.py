"""
Model Factory — DEPRECATED
==========================

.. deprecated::
    The worker no longer uses this factory. Stage 1 routing now goes through
    ``models.patchcore_registry.PatchCoreRegistry`` (per-category checkpoints)
    and ``models.vlm_router.VLMRouter`` (product identification, defect
    description, zero-shot fallback). See ``docs/vlm_router_design.md`` and
    ``docs/integration_plan.md``.

    This module is retained for backward compatibility with tests and any
    external callers still importing ``get_model`` / ``ModelType`` dispatch.
    Do not add new call sites.

Registry
--------
- patchcore → PatchCoreReal  (real ResNet50 / Anomalib implementation)
- padim     → PatchCoreStub  (stub — PaDiM integration pending)
- winclip   → WinCLIPStub    (legacy stub — superseded by GPT-4V)
- gpt4v     → GPT4VDetector  (real GPT-4o vision API)
- pushpull  → PushPullStub   (stub — supervised model pending)

To test without API keys or GPU, swap back to stubs:
    _MODEL_REGISTRY[ModelType.PATCHCORE] = PatchCoreStub
    _MODEL_REGISTRY[ModelType.GPT4V] = WinCLIPStub
"""

from __future__ import annotations

import structlog

from ..config import get_settings
from ..schemas import ModelType
from .base import AnomalyDetector
from .gpt4v_detector import GPT4VDetector
from .patchcore_real import PatchCoreReal
from .patchcore_stub import PatchCoreStub
from .pushpull_stub import PushPullStub
from .winclip_stub import WinCLIPStub

logger = structlog.get_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Registry & cache
# ─────────────────────────────────────────────────────────────────────────────

# Maps ModelType → class.  Swap entries here to switch implementations.
_MODEL_REGISTRY: dict[ModelType, type[AnomalyDetector]] = {
    ModelType.PATCHCORE: PatchCoreReal,    # Real k-NN memory-bank model
    ModelType.PADIM: PatchCoreStub,        # TODO: real PaDiM via Anomalib
    ModelType.WINCLIP: WinCLIPStub,        # Legacy stub (kept for compat)
    ModelType.GPT4V: GPT4VDetector,        # Real GPT-4o VLM detector
    ModelType.PUSHPULL: PushPullStub,      # TODO: real supervised model
}

# Loaded model instances (one per ModelType, created lazily)
_MODEL_CACHE: dict[ModelType, AnomalyDetector] = {}


def get_model(model_type: ModelType, cached: bool = True) -> AnomalyDetector:
    """
    Return an anomaly detection model instance, initialising it if needed.

    Parameters
    ----------
    model_type : ModelType
        Which model to load.
    cached : bool
        If True (default), reuse the cached instance to avoid reloading
        weights on every job.

    Returns
    -------
    AnomalyDetector
        Model instance ready for inference.

    Raises
    ------
    ValueError
        If model_type is not in the registry.
    RuntimeError
        If the model fails to load (e.g. missing API key or checkpoint).
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type!r}. "
            f"Available: {[mt.value for mt in _MODEL_REGISTRY]}"
        )

    if cached and model_type in _MODEL_CACHE:
        logger.debug("Returning cached model", model_type=model_type.value)
        return _MODEL_CACHE[model_type]

    model_class = _MODEL_REGISTRY[model_type]
    settings = get_settings()

    # Pass constructor kwargs for models that need config values.
    if model_type == ModelType.GPT4V:
        model = GPT4VDetector(
            api_key=settings.openai_api_key,
            product_type=settings.gpt4v_product_type,
            max_retries=settings.gpt4v_max_retries,
            anomaly_threshold=settings.anomaly_threshold,
        )
    elif model_type == ModelType.PATCHCORE:
        model = PatchCoreReal(
            checkpoint_path=settings.patchcore_checkpoint_path,
            anomaly_threshold=settings.anomaly_threshold,
        )
    else:
        model = model_class()

    logger.info("Loading model", model_type=model_type.value, class_name=model_class.__name__)
    model.load_model()

    if cached:
        _MODEL_CACHE[model_type] = model

    return model


def clear_model_cache() -> None:
    """Clear cached model instances (useful in tests or memory-constrained envs)."""
    global _MODEL_CACHE
    _MODEL_CACHE = {}
    logger.info("Model cache cleared")


def get_cached_models() -> list[str]:
    """Return list of currently cached model names."""
    return [mt.value for mt in _MODEL_CACHE]
