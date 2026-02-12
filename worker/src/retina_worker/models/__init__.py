"""
Anomaly Detection Models Package
================================

This package contains the anomaly detection model implementations.

Implementation Status
---------------------

**Currently Implemented (Stubs):**
- PatchCoreStub: Deterministic placeholder for PatchCore
- WinCLIPStub: Deterministic placeholder for WinCLIP  
- PushPullStub: Deterministic placeholder for supervised model

**Planned (Real Models):**
- Integration with Anomalib for production-ready implementations
- Custom push-pull contrastive learning model

Research Notes
--------------

The stub implementations generate deterministic anomaly scores based on
a hash of the image ID. This allows consistent testing while maintaining
the expected output format. The scores are designed to produce:

- ~60% normal samples (score < 0.5)
- ~30% anomalies (score > 0.5)
- ~10% high-uncertainty cases (score ≈ 0.5)

This distribution simulates realistic industrial inspection scenarios.

Integration Path
----------------

When integrating real models:

1. Inherit from ``AnomalyDetector`` base class
2. Implement ``load_model()`` and ``predict()`` methods
3. Use the same return types (AnomalyPrediction)
4. Register in the model factory (``get_model()``)

See ``base.py`` for the abstract interface definition.
"""

from .base import AnomalyDetector, AnomalyPrediction
from .factory import get_model
from .patchcore_stub import PatchCoreStub
from .winclip_stub import WinCLIPStub
from .pushpull_stub import PushPullStub
from .supervised_classifier import SupervisedClassifier, get_supervised_classifier

__all__ = [
    "AnomalyDetector",
    "AnomalyPrediction",
    "get_model",
    "PatchCoreStub",
    "WinCLIPStub", 
    "PushPullStub",
]
