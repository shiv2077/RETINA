"""
Base Anomaly Detector Interface
===============================

Defines the abstract interface for all anomaly detection models.
Both stub implementations and real models must implement this interface.

Design Rationale
----------------

The interface is designed to:

1. **Decouple model implementation from worker logic**
   - Workers don't need to know model internals
   - Models can be swapped without changing worker code

2. **Support both Stage 1 and Stage 2 models**
   - Unified prediction interface
   - Stage-specific outputs handled via optional fields

3. **Enable easy integration with Anomalib**
   - Anomalib models can be wrapped in this interface
   - Minimal adaptation layer required

4. **Support uncertainty estimation for active learning**
   - All models must provide uncertainty scores
   - Used for sample selection in the labeling pool
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnomalyPrediction:
    """
    Result of anomaly detection inference.
    
    This dataclass captures all outputs from anomaly detection models,
    supporting both Stage 1 (unsupervised) and Stage 2 (supervised) methods.
    
    Attributes
    ----------
    anomaly_score : float
        Overall anomaly score in range [0, 1].
        0.0 = definitely normal, 1.0 = definitely anomalous.
        
    is_anomaly : bool
        Binary classification result (anomaly_score > threshold).
        
    confidence : float
        Model's confidence in the prediction, range [0, 1].
        Low confidence indicates uncertainty.
        
    uncertainty : float
        Uncertainty measure for active learning.
        Higher uncertainty = more valuable for labeling.
        Typically calculated as: 1 - abs(2 * anomaly_score - 1)
        
    heatmap : Optional[bytes]
        Spatial anomaly heatmap as PNG bytes (if generated).
        Shows localization of detected anomalies.
        
    feature_distance : Optional[float]
        Distance in feature space from normal distribution.
        Used by PatchCore (k-NN distance) and PaDiM (Mahalanobis).
        
    clip_similarity : Optional[float]
        CLIP text-image similarity score (WinCLIP only).
        
    defect_category : Optional[str]
        Predicted defect category for multi-class classification.
        Only available in Stage 2 (supervised) mode.
        
    category_probabilities : Optional[dict]
        Probability distribution over defect categories.
        Only available in Stage 2 (supervised) mode.
        
    embedding_distance : Optional[float]
        Distance from decision boundary in contrastive embedding space.
        Only available for push-pull model (Stage 2).
    """
    
    # Core prediction outputs (required)
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    uncertainty: float
    
    # Stage 1 specific outputs (optional)
    heatmap: Optional[bytes] = None
    feature_distance: Optional[float] = None
    clip_similarity: Optional[float] = None
    
    # Stage 2 specific outputs (optional)
    defect_category: Optional[str] = None
    category_probabilities: Optional[dict[str, float]] = field(default_factory=dict)
    embedding_distance: Optional[float] = None


class AnomalyDetector(ABC):
    """
    Abstract base class for anomaly detection models.
    
    All anomaly detection models (both stubs and real implementations)
    must inherit from this class and implement the required methods.
    
    Example
    -------
    >>> class MyModel(AnomalyDetector):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_model"
    ...     
    ...     def load_model(self) -> None:
    ...         # Load pretrained weights
    ...         pass
    ...     
    ...     def predict(self, image_id: str, image_data: bytes | None) -> AnomalyPrediction:
    ...         # Run inference
    ...         return AnomalyPrediction(...)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name/identifier of this model.
        
        Returns
        -------
        str
            Model identifier (e.g., "patchcore", "padim", "winclip")
        """
        ...
    
    @property
    def is_loaded(self) -> bool:
        """
        Check if the model weights are loaded and ready for inference.
        
        Returns
        -------
        bool
            True if model is ready, False otherwise
        """
        return True  # Stubs are always "loaded"
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load model weights and prepare for inference.
        
        This method should:
        - Load pretrained weights from disk or download
        - Initialize any required preprocessing pipelines
        - Move model to appropriate device (CPU/GPU)
        
        Raises
        ------
        RuntimeError
            If model loading fails
        """
        ...
    
    @abstractmethod
    def predict(
        self,
        image_id: str,
        image_data: Optional[bytes] = None,
    ) -> AnomalyPrediction:
        """
        Run anomaly detection inference on an image.
        
        Parameters
        ----------
        image_id : str
            Unique identifier for the image.
            Used for deterministic stub outputs and logging.
            
        image_data : Optional[bytes]
            Raw image data (JPEG/PNG bytes).
            Currently None as we use ID-only references.
            Will be required when real image storage is implemented.
        
        Returns
        -------
        AnomalyPrediction
            Prediction result with anomaly score and metadata.
            
        Raises
        ------
        ValueError
            If image_id is empty or invalid
        RuntimeError
            If inference fails
        """
        ...
    
    def calculate_uncertainty(self, anomaly_score: float) -> float:
        """
        Calculate uncertainty score from anomaly score.
        
        The uncertainty is highest when the anomaly score is near 0.5
        (the decision boundary), and lowest at the extremes.
        
        This is used for active learning sample selection - samples
        with high uncertainty are most valuable for labeling.
        
        Parameters
        ----------
        anomaly_score : float
            Anomaly score in range [0, 1]
            
        Returns
        -------
        float
            Uncertainty score in range [0, 1]
            
        Notes
        -----
        The formula used is: 1 - |2 * score - 1|
        
        This produces:
        - uncertainty = 1.0 when score = 0.5 (maximum uncertainty)
        - uncertainty = 0.0 when score = 0.0 or 1.0 (minimum uncertainty)
        """
        return 1.0 - abs(2.0 * anomaly_score - 1.0)
