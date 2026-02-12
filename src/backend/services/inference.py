"""
Inference Service
Unified interface for running anomaly detection models.
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from datetime import datetime

try:
    from ..models import PatchCoreModel, BGADModel
    from ..config import config, MODELS_DIR
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models import PatchCoreModel, BGADModel
    from config import config, MODELS_DIR


class InferenceService:
    """
    Unified inference service for anomaly detection.
    Supports both PatchCore (unsupervised) and BGAD (supervised) models.
    """
    
    def __init__(self):
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Models cache
        self.patchcore_models: Dict[str, PatchCoreModel] = {}
        self.bgad_model: Optional[BGADModel] = None
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.patchcore.img_size, config.patchcore.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Inference history
        self.history: List[Dict] = []
    
    def load_patchcore(self, category: str, model_path: Optional[Path] = None) -> PatchCoreModel:
        """Load or create PatchCore model for a category."""
        if category in self.patchcore_models:
            return self.patchcore_models[category]
        
        model = PatchCoreModel(
            backbone=config.patchcore.backbone,
            layers=config.patchcore.layers,
            img_size=config.patchcore.img_size,
            coreset_ratio=config.patchcore.coreset_ratio,
            num_neighbors=config.patchcore.num_neighbors,
            device=str(self.device)
        )
        
        if model_path and model_path.exists():
            model.load(model_path)
        elif (MODELS_DIR / f"patchcore_{category}.pth").exists():
            model.load(MODELS_DIR / f"patchcore_{category}.pth")
        
        self.patchcore_models[category] = model
        return model
    
    def load_bgad(self, model_path: Optional[Path] = None) -> BGADModel:
        """Load or create BGAD model."""
        if self.bgad_model is not None:
            return self.bgad_model
        
        model = BGADModel(
            backbone=config.bgad.backbone,
            feature_dim=config.bgad.feature_dim,
            margin=config.bgad.margin,
            device=str(self.device)
        )
        
        if model_path and model_path.exists():
            model.load(model_path)
        elif (MODELS_DIR / "bgad_model.pth").exists():
            model.load(MODELS_DIR / "bgad_model.pth")
        
        self.bgad_model = model
        return model
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for inference."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        return self.transform(image)
    
    def run_patchcore(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        category: str,
        return_heatmap: bool = True
    ) -> Dict:
        """
        Run PatchCore inference on a single image.
        
        Args:
            image: Input image
            category: MVTec category name
            return_heatmap: Whether to return anomaly heatmap
        
        Returns:
            Dict with anomaly score, prediction, and optional heatmap
        """
        model = self.load_patchcore(category)
        
        if not model.is_fitted:
            raise RuntimeError(f"PatchCore model for {category} is not trained")
        
        # Preprocess
        if not isinstance(image, torch.Tensor):
            tensor = self.preprocess_image(image)
        else:
            tensor = image
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Inference
        scores, maps = model.predict(tensor.to(self.device))
        
        result = {
            "model": "patchcore",
            "category": category,
            "anomaly_score": float(scores[0].item()),
            "is_anomaly": scores[0].item() > config.patchcore.anomaly_threshold,
            "threshold": config.patchcore.anomaly_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        if return_heatmap:
            result["anomaly_map"] = maps[0].numpy()
        
        # Log
        self.history.append(result)
        
        return result
    
    def run_bgad(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor]
    ) -> Dict:
        """
        Run BGAD inference on a single image.
        
        Args:
            image: Input image
        
        Returns:
            Dict with anomaly score and prediction
        """
        model = self.load_bgad()
        
        if not model.center_initialized:
            raise RuntimeError("BGAD model is not trained")
        
        # Preprocess
        if not isinstance(image, torch.Tensor):
            tensor = self.preprocess_image(image)
        else:
            tensor = image
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # Inference
        result = model.predict_single(tensor.to(self.device))
        
        output = {
            "model": "bgad",
            "anomaly_score": result["anomaly_score"],
            "is_anomaly": result["is_anomaly"],
            "threshold": model.threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        # Log
        self.history.append(output)
        
        return output
    
    def run_ensemble(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        category: str,
        weights: Dict[str, float] = {"patchcore": 0.6, "bgad": 0.4}
    ) -> Dict:
        """
        Run ensemble inference combining PatchCore and BGAD.
        
        Args:
            image: Input image
            category: MVTec category
            weights: Model weights for score combination
        
        Returns:
            Combined result
        """
        patchcore_result = self.run_patchcore(image, category, return_heatmap=True)
        
        try:
            bgad_result = self.run_bgad(image)
            bgad_available = True
        except RuntimeError:
            bgad_result = None
            bgad_available = False
        
        # Normalize scores (approximate)
        pc_score = min(patchcore_result["anomaly_score"] / 2.0, 1.0)  # PatchCore scores vary
        
        if bgad_available:
            bgad_score = min(bgad_result["anomaly_score"] / 2.0, 1.0)
            combined_score = (
                weights["patchcore"] * pc_score +
                weights["bgad"] * bgad_score
            )
        else:
            combined_score = pc_score
        
        return {
            "model": "ensemble",
            "category": category,
            "combined_score": combined_score,
            "patchcore_score": patchcore_result["anomaly_score"],
            "bgad_score": bgad_result["anomaly_score"] if bgad_available else None,
            "is_anomaly": combined_score > 0.5,
            "anomaly_map": patchcore_result.get("anomaly_map"),
            "timestamp": datetime.now().isoformat()
        }
    
    def batch_inference(
        self,
        images: List[Union[str, Path]],
        category: str,
        model: str = "patchcore",
        batch_size: int = 8
    ) -> List[Dict]:
        """Run batch inference."""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_paths = images[i:i + batch_size]
            batch_tensors = torch.stack([self.preprocess_image(p) for p in batch_paths])
            
            if model == "patchcore":
                pc_model = self.load_patchcore(category)
                scores, maps = pc_model.predict(batch_tensors.to(self.device))
                
                for j, path in enumerate(batch_paths):
                    results.append({
                        "image_path": str(path),
                        "model": "patchcore",
                        "category": category,
                        "anomaly_score": float(scores[j].item()),
                        "is_anomaly": scores[j].item() > config.patchcore.anomaly_threshold,
                        "anomaly_map": maps[j].numpy()
                    })
            
            elif model == "bgad":
                bgad_model = self.load_bgad()
                result = bgad_model.predict(batch_tensors.to(self.device))
                
                for j, path in enumerate(batch_paths):
                    results.append({
                        "image_path": str(path),
                        "model": "bgad",
                        "anomaly_score": float(result["scores"][j]),
                        "is_anomaly": bool(result["predictions"][j])
                    })
        
        return results
    
    def get_history(self, limit: int = 100) -> List[Dict]:
        """Get recent inference history."""
        return self.history[-limit:]
    
    def clear_history(self):
        """Clear inference history."""
        self.history = []
