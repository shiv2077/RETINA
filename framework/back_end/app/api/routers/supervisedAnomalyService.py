from functools import lru_cache
from PIL import Image
import io
from anomalib.deploy import TorchInferencer

from app.api.routers.anomalyService import DummyInferencer


@lru_cache(maxsize=1)
def get_supervised_model():
    # replace
    try:
        return TorchInferencer(
            path="models/supervised_model.pt",
        )
    except FileNotFoundError:
        print("Model file not found, returning dummy object")
        return DummyInferencer()

def run_supervised_inference(image_bytes: bytes):
    """run supervised anomaly classification on the given image bytes and return the result."""
    inferencer = get_supervised_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prediction = inferencer.predict(image=image)

    # note : The structure of the prediction depends on the model type 
    # (classification or scoring)

    anomaly_score = float(prediction.pred_score)  # tensor to float
    # classify as anomaly if score > threshold (you might determine a threshold via validation)
    is_anomalous = anomaly_score > 0.5  # anything
    result = {
        "anomaly_score": anomaly_score,
        "is_anomalous": is_anomalous
    }
    return result