from functools import lru_cache
from PIL import Image
import io
from anomalib.deploy import TorchInferencer

from app.api.routers.anomalyService import DummyInferencer


"""
It should be used in the backend pipeline after an anomaly is detected.
"""

@lru_cache(maxsize=1)
def get_classification_model():
    try:
        """Load the anomaly classification model once (singleton)."""
        # Adjust the path to your model weights and config
        return TorchInferencer(
            path="models/classifier_model.pt"         # path to your classification model checkpoint
        )
    except FileNotFoundError:
        print("Model file not found, returning dummy object")
        return DummyInferencer()

def run_classification_inference(image_bytes: bytes):
    """
    Run supervised classification on the given image bytes and return the predicted class.

    Returns a dictionary with:
        - 'label': predicted class label
        - 'confidence': (optional) prediction score or confidence
    """
    inferencer = get_classification_model()

    # Convert byte stream to image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Run classification
    prediction = inferencer.predict(image=image)

    # Extract results
    # note: Adjust depending on your classification model's output structure.
    label = prediction.pred_label
    score = float(prediction.pred_score) if hasattr(prediction, "pred_score") else None

    result = {
        "label": label,
        "confidence": score
    }
    return result
