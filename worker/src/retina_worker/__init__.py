"""
RETINA ML Worker Package
========================

This package implements the ML inference worker for the RETINA multi-stage
visual anomaly detection system.

Architecture Overview
---------------------

The worker:
1. Consumes inference jobs from a Redis Stream (XREADGROUP)
2. Identifies the product via GPT-4o-mini, then runs the matching
   per-category PatchCore checkpoint (Stage 1)
3. For anomalous, mid-confidence scores, refines the verdict via GPT-4o
   in-context few-shot classification (Stage 2)
4. Stores results back to Redis and adds uncertain samples to the active
   learning pool

See Also
--------
- ``retina_worker.models.patchcore_registry``: per-category PatchCore
  checkpoint loading with a 2-model LRU GPU cache
- ``retina_worker.models.vlm_router``: GPT-4o orchestration layer
- ``retina_worker.worker``: Main worker loop implementation
"""

__version__ = "0.1.0"
__author__ = ""
