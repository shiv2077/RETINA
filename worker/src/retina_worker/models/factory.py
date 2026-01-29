"""
Model Factory
=============

Factory function for creating anomaly detection model instances.

This module provides a single entry point for obtaining model instances,
making it easy to swap between stub and real implementations.
"""

import structlog

from ..schemas import ModelType
from .base import AnomalyDetector
from .patchcore_stub import PatchCoreStub
from .pushpull_stub import PushPullStub
from .winclip_stub import WinCLIPStub

logger = structlog.get_logger()


# Model registry: maps model type to class
_MODEL_REGISTRY: dict[ModelType, type[AnomalyDetector]] = {
    ModelType.PATCHCORE: PatchCoreStub,
    ModelType.PADIM: PatchCoreStub,  # Reuse PatchCore stub for PaDiM (similar approach)
    ModelType.WINCLIP: WinCLIPStub,
    ModelType.PUSHPULL: PushPullStub,
}

# Cache of loaded model instances
_MODEL_CACHE: dict[ModelType, AnomalyDetector] = {}


def get_model(model_type: ModelType, cached: bool = True) -> AnomalyDetector:
    """
    Get an anomaly detection model instance.
    
    This factory function returns the appropriate model implementation
    for the given model type. By default, models are cached to avoid
    repeated loading of weights.
    
    Parameters
    ----------
    model_type : ModelType
        The type of model to retrieve
    cached : bool
        If True, return cached instance if available.
        If False, create a new instance.
        
    Returns
    -------
    AnomalyDetector
        Model instance ready for inference
        
    Raises
    ------
    ValueError
        If model_type is not recognized
        
    Example
    -------
    >>> from retina_worker.schemas import ModelType
    >>> model = get_model(ModelType.PATCHCORE)
    >>> result = model.predict("test_image")
    
    Notes
    -----
    When integrating real models, update the registry:
    
    ```python
    _MODEL_REGISTRY[ModelType.PATCHCORE] = RealPatchCore
    ```
    
    The factory pattern allows seamless swapping between stub and
    real implementations without changing client code.
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Return cached instance if available and caching enabled
    if cached and model_type in _MODEL_CACHE:
        logger.debug("Returning cached model", model_type=model_type.value)
        return _MODEL_CACHE[model_type]
    
    # Create new instance
    model_class = _MODEL_REGISTRY[model_type]
    model = model_class()
    
    # Load the model
    logger.info("Loading model", model_type=model_type.value, model_class=model_class.__name__)
    model.load_model()
    
    # Cache if enabled
    if cached:
        _MODEL_CACHE[model_type] = model
    
    return model


def clear_model_cache() -> None:
    """
    Clear the model cache.
    
    Useful for testing or when memory needs to be freed.
    """
    global _MODEL_CACHE
    _MODEL_CACHE = {}
    logger.info("Model cache cleared")


def get_cached_models() -> list[str]:
    """
    Get list of currently cached model names.
    
    Returns
    -------
    list[str]
        Names of cached models
    """
    return [mt.value for mt in _MODEL_CACHE.keys()]
