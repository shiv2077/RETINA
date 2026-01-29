"""
RETINA ML Worker Package
========================

This package implements the ML inference worker for the RETINA multi-stage
visual anomaly detection system.

Architecture Overview
---------------------

The worker:
1. Consumes inference jobs from a Redis Stream
2. Runs anomaly detection models (currently stubbed)
3. Stores results back to Redis
4. Adds uncertain samples to the active learning pool

Research Context
----------------

This worker supports a two-stage anomaly detection pipeline:

**Stage 1 - Unsupervised/Zero-shot Detection**
- PatchCore: Memory-bank based, uses pretrained CNN features + k-NN
- PaDiM: Gaussian modeling of patch-level feature distributions  
- WinCLIP: Zero-shot detection using CLIP text-image similarity

**Stage 2 - Supervised Classification**
- Push-Pull: Contrastive learning with labeled anomaly samples
- Activated after collecting ~100-200 labels via active learning

Implementation Status
---------------------

Currently, model inference is **stubbed** with deterministic placeholders
that simulate the expected output format. This allows:

- End-to-end pipeline testing
- Frontend development
- Architecture validation

Real model integration is planned via the Anomalib framework, which provides
production-ready implementations of PatchCore, PaDiM, and other methods.

See Also
--------
- ``retina_worker.models.base``: Base class for anomaly detectors
- ``retina_worker.models.patchcore_stub``: PatchCore placeholder
- ``retina_worker.worker``: Main worker loop implementation
"""

__version__ = "0.1.0"
__author__ = ""
