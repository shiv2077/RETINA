# ensure to have the models trained and ready for inference
# anomalib will produce a model checkpoint or TorchScript (.pt) file and a config (.yaml)

from anomalib.deploy import TorchInferencer
from functools import lru_cache
import os
os.environ["TRUST_REMOTE_CODE"] = "1"

from WinCLIP.setup_winclip import WinClipInferencer
#from app.api.routers.anomalyService import DummyInferencer

# lru_cache to ensure the model is loaded only once (singleton)
@lru_cache(maxsize=1)
def get_model():
    try:
        #return TorchInferencer(path="output\patchcore\export\patchcore.pt")
        #return TorchInferencer(path="output\padim\export\padim.pt")
        return WinClipInferencer("configs/config_winclip.yaml")
    except FileNotFoundError:
        print("Model file not found, returning dummy object")
        #return DummyInferencer()
     

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

if __name__ == "__main__":
    # Lees de afbeelding in als bytes
    img_path = r"C:\MSVDP\Unsupervised_Models\dataset\decospan_small\test\anomaly\PO22-34546_5_1_img001_patch013.jpg"
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    
    # Run inferentie
    result = run_inference("test_image_001", img_bytes)
    print(result)