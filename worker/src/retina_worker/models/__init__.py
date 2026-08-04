"""
Anomaly Detection Models Package
================================

Two live modules, both used directly by ``retina_worker.worker.Worker``:

- ``patchcore_registry``: per-category PatchCore checkpoint loading with a
  2-model LRU GPU cache. Stage 1 verdict.
- ``vlm_router``: GPT-4o orchestration (identify_product, describe_defect,
  zero_shot_detect, stage2_refine).

Nothing else in this package is imported by the worker. There is no model
factory or registry pattern here — ``worker.py`` imports each module
directly.
"""
