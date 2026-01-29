# ensure to have the models trained and ready for inference
# anomalib will produce a model checkpoint or TorchScript (.pt) file and a config (.yaml)

from anomalib.deploy import TorchInferencer
from functools import lru_cache
import os
from app.api.routers.anomalyService import DummyInferencer

from app.models.unsupervised.WinCLIP.setup_winclip import WinClipInferencer


# lru_cache to ensure the model is loaded only once (singleton)
@lru_cache(maxsize=1)
def get_model():
    return TorchInferencer(path="E:/ML - Self Study/MV4QC_back_end/app/models/unsupervised/output/patchcore/export/patchcore.pt")
    #return TorchInferencer(path="app.models.unsupervised.output\padim\export\padim.pt")
    #return WinClipInferencer("app.models.unsupervised.configs/config_winclip.yaml")
    #return TorchInferencer(
    #    path="models/padim_config.pt"         # path to your classification model checkpoint
    #)

     

"""
asynchronous queue - run_inference will not happen in the /predict request handle : faster

it will be called by a background worker by worker.py
"""

def run_inference(image_id: str, image_bytes: bytes):
    """Run anomaly detection on the given image bytes and return the result."""
    # load model - reuse cached instance after first call
    inferencer = get_model()
    # bytes -> PIL Image
    from PIL import Image
    import io
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    prediction = inferencer.predict(image=image)

    anomaly_score = float(prediction.pred_score)  # tensor to float
    # classify as anomaly if score > threshold (you might determine a threshold via validation)
    is_anomalous = anomaly_score > 0.5  # anything
    result = {
        "image_id": image_id,
        "anomaly_score": anomaly_score,
        "is_anomalous": is_anomalous
        # You could also include heatmap or mask if needed: e.g. prediction.anomaly_map
    }
    return result
