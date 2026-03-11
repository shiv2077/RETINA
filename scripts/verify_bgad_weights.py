import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.backend.models.bgad import BGADModel

def verify_model():
    model_path = PROJECT_ROOT / "models" / "weights" / "bgad_production.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "output" / "bgad_production.pt"
        
    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
    
    # Inspect State Dict
    print("\n--- Model State Dict Keys ---")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Format: Dictionary with 'model_state_dict'")
    elif isinstance(checkpoint, dict) and "encoder_state" in checkpoint:
        state_dict = checkpoint["encoder_state"]
        print("Format: Dictionary with 'encoder_state', 'center', etc.")
        if "center" in checkpoint:
            print(f"Trained Center Vector sum: {checkpoint['center'].sum().item():.4f}")
        if "threshold" in checkpoint:
            print(f"Threshold: {checkpoint['threshold']}")
    else:
        state_dict = checkpoint
        print("Format: Raw state dict")
    
    print(f"Total parameter tensors: {len(state_dict)}")
        
    # Load Model
    model = BGADModel(device="cpu", backbone="resnet18")
    if isinstance(checkpoint, dict) and "encoder_state" in checkpoint:
        model.load(model_path)
    else:
        # Provide compatible load
        # Filter unmatched keys if needed, but try strict defaults
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning on state dictionary load: {e}")
            
    model.eval()
    
    # Validation Data Setup
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test on MVTec
    mvtec_dir = PROJECT_ROOT / "mvtec_anomaly_detection" / "bottle" / "train" / "good"
    print("\n--- Testing on MVTec Dataset ---")
    if mvtec_dir.exists():
        mvtec_imgs = list(mvtec_dir.glob("*.png"))[:5]
        for img_path in mvtec_imgs:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            score = model.predict(tensor)
            print(f"MVTec image {img_path.name}: Anomaly Score = {score:.4f}")
    else:
        print("MVTec directory not found.")
        
    # Test on Decospan
    decospan_dir = PROJECT_ROOT / "decospan_small" / "train" / "good"
    print("\n--- Testing on Decospan Dataset ---")
    if decospan_dir.exists():
        deco_imgs = list(decospan_dir.glob("*.png"))[:5]
        if not deco_imgs:
            deco_imgs = list(decospan_dir.rglob("*.png"))[:5]
            if not deco_imgs:
                deco_imgs = list(decospan_dir.rglob("*.jpg"))[:5]
        for img_path in deco_imgs:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            score = model.predict(tensor)
            print(f"Decospan image {img_path.name}: Anomaly Score = {score:.4f}")
    else:
        print("Decospan directory not found.")
        
    unified_dir = PROJECT_ROOT / "MSVPD_Unified_Dataset" / "train" / "good"
    print("\n--- Testing on Unified Dataset ---")
    if unified_dir.exists():
        unified_imgs = list(unified_dir.rglob("*.png"))[:5]
        for img_path in unified_imgs:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0)
            score = model.predict(tensor)
            print(f"Unified image {img_path.name}: Anomaly Score = {score:.4f}")
    else:
        print("Unified directory not found.")

if __name__ == '__main__':
    verify_model()
