# pyre-ignore-all-errors
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
import logging
import base64
import json
import os
from time import time

# Setup logger
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI client not installed. GPT-4V features disabled.")

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
        self.vlm_model: Optional[object] = None  # Zero-shot VLM (AdaCLIP/WinCLIP)
        
        # OpenAI Client
        self.openai_client = None
        self.gpt4v_model = "gpt-4-vision-preview"  # or "gpt-4o" for production
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.openai_client = OpenAI(api_key=api_key)
                logger.info("✅ OpenAI client initialized for GPT-4V")
            else:
                logger.warning("⚠️  OPENAI_API_KEY environment variable not set. GPT-4V disabled.")
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.patchcore.img_size, config.patchcore.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Inference history
        self.history: List[Dict] = []
        
        # Cascade routing statistics
        self.cascade_stats = {
            "confident_normal": 0,
            "confident_anomaly": 0,
            "uncertain_routed_to_vlm": 0,
            "vlm_flagged_anomaly": 0,
        }
    
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
    
    def load_vlm(self, model_path: Optional[Path] = None) -> object:
        """
        Load zero-shot Vision-Language Model (AdaCLIP/WinCLIP) for fallback.
        
        Args:
            model_path: Optional path to pretrained VLM weights
        
        Returns:
            Loaded VLM model instance
        """
        if self.vlm_model is not None:
            return self.vlm_model
        
        try:
            # Try to load AdaCLIP or WinCLIP
            from ..models.vlm import AdaCLIP  # Adjust import path as needed
            
            model = AdaCLIP()
            if model_path and model_path.exists():
                model.load(model_path)
            
            self.vlm_model = model
            logger.info("✓ VLM model loaded successfully")
            return model
            
        except ImportError:
            logger.warning("⚠ VLM model not available. Cascade will skip VLM fallback.")
            return None
    
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for inference."""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        return self.transform(image)
    
    def predict_with_cascade(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        normal_threshold: float = 0.2,
        anomaly_threshold: float = 0.8,
        use_vlm_fallback: bool = True
    ) -> Dict:
        """
        Asynchronous Cascade Router: Fast BGAD with VLM fallback for uncertain predictions.
        
        This method implements an intelligent inference pipeline that:
        1. Runs fast BGAD model first (edge computing optimized)
        2. Routes to heavy VLM only for uncertain (novel/drift) cases
        3. Automatically flags ambiguous cases for expert labeling
        
        Args:
            image: Input image (path, PIL Image, numpy array, or tensor)
            normal_threshold: Score below which BGAD result is considered confident normal
                           Recommended: 0.2 (Case A)
            anomaly_threshold: Score above which BGAD result is considered confident anomaly
                            Recommended: 0.8 (Case B)
            use_vlm_fallback: Whether to use VLM for uncertain predictions (Case C)
        
        Returns:
            Standardized prediction dictionary with keys:
            {
                "model_used": "bgad" | "vlm" | "ensemble",
                "anomaly_score": float,
                "is_anomaly": bool,
                "confidence": float (0.0-1.0),
                "threshold": float,
                "requires_expert_labeling": bool,
                "routing_case": "A_confident_normal" | "B_confident_anomaly" | "C_uncertain",
                "vlm_result": dict (optional, if VLM was used),
                "timestamp": str,
            }
        
        Cascade Logic:
            Case A (Confident Normal): score < normal_threshold
                → Returns BGAD prediction immediately (~milliseconds on RTX 3060)
            
            Case B (Confident Anomaly): score > anomaly_threshold
                → Returns BGAD prediction immediately
            
            Case C (Uncertain/Novel Defect): normal_threshold <= score <= anomaly_threshold
                → Routes to heavy VLM for zero-shot analysis
                → Flags for expert annotation if VLM also detects anomaly
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Load BGAD model
        bgad_model = self.load_bgad()
        
        # Preprocess image if needed
        if not isinstance(image, torch.Tensor):
            tensor = self.preprocess_image(image)
        else:
            tensor = image
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        # ========== PRIMARY INFERENCE: Fast BGAD ==========
        logger.info("🚀 Running BGAD primary inference...")
        bgad_result = bgad_model.predict_single(tensor.to(self.device))
        bgad_score = bgad_result["anomaly_score"]
        
        logger.debug(f"   BGAD Score: {bgad_score:.4f}")
        logger.debug(f"   Thresholds: Normal={normal_threshold}, Anomaly={anomaly_threshold}")
        
        # ========== UNCERTAINTY ROUTER: Threshold Logic ==========
        
        if bgad_score < normal_threshold:
            # ===== CASE A: Confident Normal =====
            logger.info(f"✓ CASE A (Confident Normal): Score {bgad_score:.4f} < {normal_threshold}")
            self.cascade_stats["confident_normal"] += 1
            
            # Math fix: Strict bounds to prevent confidence > 100% or < 0%
            confidence_val = float(1.0 - (bgad_score / normal_threshold))
            confidence_val = max(0.0, min(1.0, confidence_val))
            
            result = {
                "model_used": "bgad",
                "anomaly_score": bgad_score,
                "is_anomaly": False,
                "confidence": confidence_val,
                "threshold": normal_threshold,
                "requires_expert_labeling": False,
                "routing_case": "A_confident_normal",
                "vlm_result": None,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"   → Returning BGAD prediction immediately (latency: <10ms)")
            
        elif bgad_score > anomaly_threshold:
            # ===== CASE B: Confident Anomaly =====
            logger.info(f"✓ CASE B (Confident Anomaly): Score {bgad_score:.4f} > {anomaly_threshold}")
            self.cascade_stats["confident_anomaly"] += 1
            
            # Math fix: Strict bounds matching exactly expected limits and preventing out of bounds percentage
            if anomaly_threshold < 1.0:
                confidence_val = float((bgad_score - anomaly_threshold) / (1.0 - anomaly_threshold))
            else:
                # Fallback if anomaly threshold is extremely high
                confidence_val = 1.0
            confidence_val = max(0.0, min(1.0, confidence_val))
            
            result = {
                "model_used": "bgad",
                "anomaly_score": bgad_score,
                "is_anomaly": True,
                "confidence": confidence_val,
                "threshold": anomaly_threshold,
                "requires_expert_labeling": False,
                "routing_case": "B_confident_anomaly",
                "vlm_result": None,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"   → Returning BGAD prediction immediately (latency: <10ms)")
            
        else:
            # ===== CASE C: Uncertain / Novel Defect =====
            logger.warning(f"⚠ CASE C (Uncertain): {normal_threshold} <= Score {bgad_score:.4f} <= {anomaly_threshold}")
            logger.warning(f"   → BGAD prediction uncertain. Data drift or novel defect detected.")
            self.cascade_stats["uncertain_routed_to_vlm"] += 1
            
            # --- VLM Fallback: GPT-4V Zero-shot Analysis ---
            if use_vlm_fallback:
                logger.info(f"🔄 Routing to GPT-4V for zero-shot analysis...")
                
                import tempfile
                import os
                # Save image to temp file for GPT-4V
                temp_image_path = None
                try:
                    if isinstance(image, torch.Tensor):
                        img_tensor = image.cpu()
                        if img_tensor.dim() == 4:
                            img_tensor = img_tensor[0]
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_tensor = img_tensor * std + mean
                        from torchvision import transforms
                        img_pil = transforms.ToPILImage()(img_tensor)
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                            img_pil.save(tmp.name)
                            temp_image_path = tmp.name
                    else:
                        temp_image_path = str(image)
                    
                    vlm_result = self.call_gpt4v_zero_shot(temp_image_path)
                    
                    vlm_is_anomaly = vlm_result.get("is_anomaly", False)
                    vlm_score = vlm_result.get("confidence", 0.5)
                    vlm_reasoning = vlm_result.get("reasoning", "")
                    
                    if vlm_is_anomaly:
                        logger.warning(f"   🚨 VLM ALSO DETECTED ANOMALY → Flagging for expert labeling")
                        self.cascade_stats["vlm_flagged_anomaly"] += 1
                        requires_labeling = True
                    else:
                        logger.info(f"   ✓ VLM classified as normal → No labeling needed (unless vlm failed)")
                        requires_labeling = False if vlm_result.get("status") == "success" else True
                    
                    ensemble_score = (bgad_score + vlm_score) / 2.0
                    
                    result = {
                        "model_used": "ensemble",
                        "anomaly_score": ensemble_score,
                        "is_anomaly": vlm_is_anomaly or (bgad_score > anomaly_threshold),
                        "confidence": vlm_score,
                        "threshold": (normal_threshold + anomaly_threshold) / 2.0,
                        "requires_expert_labeling": requires_labeling,
                        "routing_case": "C_uncertain_vlm_routed",
                        "vlm_score": float(vlm_score),
                        "vlm_reasoning": str(vlm_reasoning),
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error(f"   ❌ VLM inference failed: {str(e)}")
                    result = {
                        "model_used": "bgad",
                        "anomaly_score": bgad_score,
                        "is_anomaly": bgad_score > 0.5,
                        "confidence": 0.3,
                        "threshold": normal_threshold,
                        "requires_expert_labeling": True,
                        "routing_case": "C_uncertain_vlm_failed",
                        "vlm_score": 0.0,
                        "vlm_reasoning": f"API Error: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    }
                finally:
                    if temp_image_path and isinstance(image, torch.Tensor) and os.path.exists(temp_image_path):
                        os.unlink(temp_image_path)
            
            else:
                # VLM not available or disabled → flag for manual labeling
                logger.warning(f"   VLM fallback unavailable → Flagging for expert annotation")
                
                result = {
                    "model_used": "bgad",
                    "anomaly_score": bgad_score,
                    "is_anomaly": bgad_score > 0.5,
                    "confidence": 0.5,
                    "threshold": normal_threshold,
                    "requires_expert_labeling": True,  # Always flag uncertain when VLM unavailable
                    "routing_case": "C_uncertain_no_vlm",
                    "vlm_score": 0.0,
                    "vlm_reasoning": "VLM Disabled",
                    "timestamp": datetime.now().isoformat()
                }
        
        # ========== LOGGING & HISTORY ==========
        logger.info(f"✓ Cascade complete: {result['routing_case']}")
        logger.info(f"   → Model used: {result['model_used']}")
        logger.info(f"   → Requires labeling: {result['requires_expert_labeling']}")
        
        # Add to history
        self.history.append(result)
        
        return result
    
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
    
    def get_cascade_statistics(self) -> Dict:
        """
        Get cascade routing statistics for monitoring and debugging.
        
        Returns:
            Dictionary with cascade performance metrics:
            {
                "confident_normal": int,          # Case A: Score < normal_threshold
                "confident_anomaly": int,         # Case B: Score > anomaly_threshold
                "uncertain_routed_to_vlm": int,   # Case C: Routed to VLM
                "vlm_flagged_anomaly": int,       # Among Case C, VLM also detected anomaly
                "vlm_accuracy": float,            # Percentage of uncertain cases caught by VLM
                "total_inferences": int,          # Total predictions processed
                "edge_model_utilization": float,  # Percentage staying on edge (Cases A+B)
            }
        """
        total = (
            self.cascade_stats["confident_normal"] +
            self.cascade_stats["confident_anomaly"] +
            self.cascade_stats["uncertain_routed_to_vlm"]
        )
        
        edge_utilization = (
            (self.cascade_stats["confident_normal"] + 
             self.cascade_stats["confident_anomaly"]) / total * 100
        ) if total > 0 else 0.0
        
        vlm_accuracy = (
            (self.cascade_stats["vlm_flagged_anomaly"] / 
             self.cascade_stats["uncertain_routed_to_vlm"] * 100)
        ) if self.cascade_stats["uncertain_routed_to_vlm"] > 0 else 0.0
        
        return {
            "confident_normal": self.cascade_stats["confident_normal"],
            "confident_anomaly": self.cascade_stats["confident_anomaly"],
            "uncertain_routed_to_vlm": self.cascade_stats["uncertain_routed_to_vlm"],
            "vlm_flagged_anomaly": self.cascade_stats["vlm_flagged_anomaly"],
            "vlm_catch_rate": vlm_accuracy,
            "total_inferences": total,
            "edge_model_utilization": edge_utilization,
        }
    
    def reset_cascade_statistics(self):
        """Reset cascade routing statistics."""
        self.cascade_stats = {
            "confident_normal": 0,
            "confident_anomaly": 0,
            "uncertain_routed_to_vlm": 0,
            "vlm_flagged_anomaly": 0,
        }
        logger.info("✓ Cascade statistics reset")
    
    # ========================================================================
    # GPT-4V ZERO-SHOT ANOMALY DETECTION
    # ========================================================================
    
    def call_gpt4v_zero_shot(self, image_path: str) -> Dict:
        """
        Call GPT-4V for zero-shot industrial anomaly detection.
        
        Production-ready method for uncertain case fallback in the cascade.
        Converts local image to Base64, calls GPT-4V with specialized system prompt,
        and safely parses JSON response. Includes comprehensive error handling and
        graceful fallback to "uncertain" status.
        
        Args:
            image_path: Path to local image file
        
        Returns:
            Standardized anomaly detection result:
            {
                "model_used": "gpt-4-vision",
                "is_anomaly": bool,
                "defect_type": str or None (e.g., "knot", "scratch", "discoloration"),
                "confidence": float (0.0-1.0),
                "reasoning": str (explanation of analysis),
                "status": "success" | "uncertain" (fallback on error),
                "timestamp": str (ISO format),
                "api_error": str (optional, if API call failed),
                "latency_ms": float (wall clock time)
            }
        
        System Prompt (Specialized for Decospan Wood QC):
            Instructs model to analyze for industrial defects specific to wood:
            - Knots (growth defects)
            - Deep scratches (surface damage)
            - Discolorations (staining, bleaching)
            - Warping, cracks, foreign materials
            Enforces JSON response format for safe parsing.
        
        Error Handling:
            - API timeout: Logs and returns uncertain status
            - Malformed JSON: Attempts recovery, falls back to uncertain
            - Missing API key: Returns error in result dict
            - Rate limiting: Handled by OpenAI client retry logic
            - Network errors: Returns uncertain with error logged
        
        Example Usage:
            ```python
            inference_service = InferenceService()
            result = inference_service.call_gpt4v_zero_shot("path/to/image.jpg")
            
            if result["status"] == "success":
                if result["is_anomaly"]:
                    print(f"Anomaly detected: {result['defect_type']}")
                    print(f"Confidence: {result['confidence']:.0%}")
                else:
                    print("Image appears normal")
            else:
                print(f"VLM analysis failed: {result.get('api_error', 'unknown')}")
            ```
        """
        start_time = time()
        
        logger.info(f"📸 Calling GPT-4V for zero-shot analysis: {image_path}")
        
        # ========== VALIDATION & SETUP ==========
        
        if not self.openai_client:
            logger.error("❌ OpenAI client not initialized. Check OPENAI_API_KEY environment variable.")
            return {
                "model_used": "gpt-4-vision",
                "is_anomaly": None,
                "defect_type": None,
                "confidence": 0.0,
                "reasoning": "GPT-4V API not configured",
                "status": "uncertain",
                "api_error": "OpenAI client not initialized",
                "timestamp": datetime.now().isoformat(),
                "latency_ms": (time() - start_time) * 1000
            }
        
        try:
            # ========== IMAGE LOADING & ENCODING ==========
            
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # Read and encode image to Base64
            with open(image_path, "rb") as img_file:
                image_data = base64.standard_b64encode(img_file.read()).decode("utf-8")
            
            # Determine media type
            suffix = image_path.suffix.lower()
            media_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            media_type = media_type_map.get(suffix, "image/jpeg")
            
            logger.debug(f"   ✓ Image encoded (size: {len(image_data)} bytes, type: {media_type})")
            
            # ========== SYSTEM PROMPT (SPECIALIZED FOR DECOSPAN WOOD QC) ==========
            
            system_prompt = (
                "You are an elite industrial quality control architect inspecting Decospan wood products. "
                "Analyze the provided image for anomalies, novel defects, and out-of-distribution textures. Look for: "
                "1. Knots (tree growth defects) - large or unexpected circular marks\n"
                "2. Deep scratches or surface damage - linear gouges\n"
                "3. Discolorations - severe staining, bleaching, or uneven finishing\n"
                "Provide your exact diagnostic analysis as a strictly valid JSON object.\n"
                "Return EXACTLY these keys:\n"
                '{"is_anomaly": boolean, "confidence": float, "reasoning": "string"}\n'
                "DO NOT include markdown block markers like ```json. DO NOT include trailing commas. "
                "Return exclusively the raw JSON literal to ensure programmatic parsing succeeds."
            )
            
            # ========== API CALL WITH RETRY LOGIC ==========
            
            logger.debug(f"   🔄 Sending request to {self.gpt4v_model}...")
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.openai_client.chat.completions.create(
                        model=self.gpt4v_model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{media_type};base64,{image_data}"
                                        }
                                    },
                                    {
                                        "type": "text",
                                        "text": "Analyze this wood product image for quality defects. Reply with JSON ONLY."
                                    }
                                ]
                            }
                        ],
                        max_tokens=500,
                        temperature=0.2  # Low temperature for consistent, deterministic responses
                    )
                    
                    logger.debug("   ✓ API response received")
                    break
                    
                except Exception as api_error:
                    if attempt < max_retries - 1:
                        logger.warning(f"   ⚠️  API call attempt {attempt + 1} failed: {str(api_error)}")
                        if "timeout" in str(api_error).lower():
                            logger.warning("   → Retrying (timeout)...")
                            import time as time_module
                            time_module.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    raise
            
            # ========== RESPONSE PARSING ==========
            
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"   📝 Raw response: {response_text[:200]}...")
            
            # Attempt to extract JSON from response
            # Handle cases where model wraps JSON in markdown or extra text
            json_str = response_text
            
            # Try to find JSON object in response
            if "{" in json_str:
                json_start = json_str.find("{")
                json_end = json_str.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = json_str[json_start:json_end]
            
            # Parse JSON
            parsed_response = json.loads(json_str)
            
            logger.debug(f"   ✓ JSON parsed successfully")
            
            # ========== RESPONSE VALIDATION & SANITIZATION ==========
            
            # Extract fields with defaults and type conversion
            is_anomaly = bool(parsed_response.get("is_anomaly", False))
            defect_type = parsed_response.get("defect_type")
            
            # Sanitize defect_type - convert None to null string
            if defect_type is None:
                defect_type = None
            else:
                defect_type = str(defect_type).strip()
                if defect_type.lower() in ["null", "none", ""]:
                    defect_type = None
            
            # Confidence score - clamp to [0, 1]
            try:
                confidence = float(parsed_response.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.5
            
            reasoning = str(parsed_response.get("reasoning", ""))[:500]  # Limit length
            
            # ========== BUILD RESULT ==========
            
            latency_ms = (time() - start_time) * 1000
            
            result = {
                "model_used": "gpt-4-vision",
                "is_anomaly": is_anomaly,
                "defect_type": defect_type,
                "confidence": confidence,
                "reasoning": reasoning,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "latency_ms": latency_ms
            }
            
            # Log success
            if is_anomaly:
                logger.warning(
                    f"   🚨 ANOMALY DETECTED by GPT-4V: {defect_type} "
                    f"(confidence: {confidence:.0%}, latency: {latency_ms:.0f}ms)"
                )
            else:
                logger.info(
                    f"   ✓ Image appears normal (confidence: {confidence:.0%}, "
                    f"latency: {latency_ms:.0f}ms)"
                )
            
            return result
        
        # ========== ERROR HANDLING ==========
        
        except json.JSONDecodeError as json_error:
            """JSON parsing failed - model returned malformed JSON."""
            logger.error(f"   ❌ JSON parsing failed: {str(json_error)}")
            logger.error(f"   → Raw response was: {response_text[:300]}")
            
            # Attempt recovery: check if response contains clear signals
            lower_text = response_text.lower()
            attempted_parse = {
                "is_anomaly": ("anomal" in lower_text or "defect" in lower_text or 
                             "damage" in lower_text or "issue" in lower_text),
                "defect_type": None,
                "confidence": 0.3,  # Lower confidence due to parse failure
                "reasoning": f"Parse error: {str(json_error)}. Fallback analysis suggests {'anomaly' if 'anomal' in lower_text else 'uncertain'}."
            }
            
            return {
                "model_used": "gpt-4-vision",
                "is_anomaly": attempted_parse["is_anomaly"],
                "defect_type": attempted_parse["defect_type"],
                "confidence": attempted_parse["confidence"],
                "reasoning": attempted_parse["reasoning"],
                "status": "uncertain",
                "api_error": f"JSON parse error: {str(json_error)}",
                "timestamp": datetime.now().isoformat(),
                "latency_ms": (time() - start_time) * 1000
            }
        
        except FileNotFoundError as file_error:
            """Image file not found."""
            logger.error(f"   ❌ Image file not found: {image_path}")
            
            return {
                "model_used": "gpt-4-vision",
                "is_anomaly": None,
                "defect_type": None,
                "confidence": 0.0,
                "reasoning": f"Image loading failed: {str(file_error)}",
                "status": "uncertain",
                "api_error": str(file_error),
                "timestamp": datetime.now().isoformat(),
                "latency_ms": (time() - start_time) * 1000
            }
        
        except Exception as e:
            """Catch-all for unexpected errors (network, API, etc.)."""
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Classify error for logging
            if "timeout" in error_msg.lower():
                log_level = "warning"
                error_category = "Timeout"
            elif "rate_limit" in error_msg.lower() or "429" in error_msg:
                log_level = "warning"
                error_category = "Rate Limited"
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                log_level = "error"
                error_category = "Network Error"
            elif "unauthorized" in error_msg.lower() or "401" in error_msg:
                log_level = "error"
                error_category = "Authentication Error"
            else:
                log_level = "error"
                error_category = "API Error"
            
            log_func = getattr(logger, log_level)
            log_func(f"   ❌ {error_category}: {error_type}: {error_msg}")
            
            return {
                "model_used": "gpt-4-vision",
                "is_anomaly": None,
                "defect_type": None,
                "confidence": 0.0,
                "reasoning": f"GPT-4V analysis failed due to {error_category.lower()}",
                "status": "uncertain",
                "api_error": f"{error_type}: {error_msg}",
                "timestamp": datetime.now().isoformat(),
                "latency_ms": (time() - start_time) * 1000
            }
    
    def cascade_predict_with_gpt4v(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor],
        normal_threshold: float = 0.2,
        anomaly_threshold: float = 0.8,
        save_image_for_vlm: bool = True
    ) -> Dict:
        """
        Enhanced cascade prediction using GPT-4V instead of local VLM.
        
        Same 3-tier routing as cascade_predict(), but routes uncertain cases
        to GPT-4V (cloud) instead of local AdaCLIP/WinCLIP.
        
        Args:
            image: Input image (path, PIL, numpy, or tensor)
            normal_threshold: BGAD score below this = confident normal
            anomaly_threshold: BGAD score above this = confident anomaly
            save_image_for_vlm: If True, saves tensor to temp file for GPT-4V
        
        Returns:
            Same format as cascade_predict() but with gpt-4-vision in vlm_result
        
        Usage:
            ```python
            inference = InferenceService()
            result = inference.cascade_predict_with_gpt4v(
                "path/to/image.jpg",
                normal_threshold=0.2,
                anomaly_threshold=0.8
            )
            
            if result["requires_expert_labeling"]:
                print("Expert annotation needed")
                print(f"GPT-4V confidence: {result['vlm_result']['confidence']:.0%}")
            ```
        """
        import tempfile
        
        logger.info("🚀 Starting cascade prediction with GPT-4V fallback...")
        
        # ========== PRIMARY INFERENCE: Fast BGAD ==========
        
        bgad_model = self.load_bgad()
        
        # Preprocess image
        if not isinstance(image, torch.Tensor):
            tensor = self.preprocess_image(image)
        else:
            tensor = image
        
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        bgad_result = bgad_model.predict_single(tensor.to(self.device))
        bgad_score = bgad_result["anomaly_score"]
        
        logger.debug(f"   BGAD Score: {bgad_score:.4f}")
        
        # ========== ROUTING DECISION ==========
        
        if bgad_score < normal_threshold:
            # Case A: Confident Normal
            logger.info(f"✓ CASE A: Score {bgad_score:.4f} < {normal_threshold} (Confident Normal)")
            
            return {
                "model_used": "bgad",
                "anomaly_score": bgad_score,
                "is_anomaly": False,
                "confidence": 1.0 - (bgad_score / normal_threshold),
                "threshold": normal_threshold,
                "requires_expert_labeling": False,
                "routing_case": "A_confident_normal",
                "vlm_result": None,
                "timestamp": datetime.now().isoformat()
            }
        
        elif bgad_score > anomaly_threshold:
            # Case B: Confident Anomaly
            logger.info(f"✓ CASE B: Score {bgad_score:.4f} > {anomaly_threshold} (Confident Anomaly)")
            
            return {
                "model_used": "bgad",
                "anomaly_score": bgad_score,
                "is_anomaly": True,
                "confidence": (bgad_score - anomaly_threshold) / 2.0,
                "threshold": anomaly_threshold,
                "requires_expert_labeling": False,
                "routing_case": "B_confident_anomaly",
                "vlm_result": None,
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            # Case C: Uncertain - Route to GPT-4V
            logger.warning(
                f"⚠ CASE C: {normal_threshold} <= Score {bgad_score:.4f} <= {anomaly_threshold} "
                f"(Uncertain/Novel Defect)"
            )
            logger.info("🔄 Routing to GPT-4V for zero-shot analysis...")
            
            # Save image to temp file if needed
            if save_image_for_vlm:
                # If input is tensor, need to convert to image and save
                if isinstance(image, torch.Tensor):
                    # Denormalize and convert to PIL
                    img_tensor = image.cpu()
                    if img_tensor.dim() == 4:
                        img_tensor = img_tensor[0]
                    
                    # Denormalize: reverse (x - mean) / std
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    
                    # Convert to PIL
                    img_pil = transforms.ToPILImage()(img_tensor)
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        img_pil.save(tmp.name)
                        temp_image_path = tmp.name
                else:
                    temp_image_path = str(image)
            else:
                temp_image_path = str(image)
            
            # Call GPT-4V
            try:
                gpt4v_result = self.call_gpt4v_zero_shot(temp_image_path)
                logger.info(f"   ✓ GPT-4V analysis complete")
                
                # Check if GPT-4V also detected anomaly
                if gpt4v_result["status"] == "success" and gpt4v_result["is_anomaly"]:
                    logger.warning(f"   🚨 GPT-4V ALSO DETECTED ANOMALY: {gpt4v_result['defect_type']}")
                    requires_labeling = True
                else:
                    logger.info(f"   ✓ GPT-4V classified as normal or uncertain")
                    requires_labeling = False if gpt4v_result["status"] == "success" else True
                
                # Ensemble score
                gpt4v_score = gpt4v_result["confidence"] if gpt4v_result["status"] == "success" else 0.5
                ensemble_score = (bgad_score + gpt4v_score) / 2.0
                
                result = {
                    "model_used": "ensemble",
                    "anomaly_score": ensemble_score,
                    "is_anomaly": gpt4v_result.get("is_anomaly", bgad_score > 0.5),
                    "confidence": gpt4v_result["confidence"],
                    "threshold": (normal_threshold + anomaly_threshold) / 2.0,
                    "requires_expert_labeling": requires_labeling,
                    "routing_case": "C_uncertain_gpt4v_routed",
                    "vlm_result": gpt4v_result,
                    "bgad_score": bgad_score,
                    "gpt4v_score": gpt4v_score,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"   ❌ GPT-4V call failed: {str(e)}")
                # Fallback to BGAD with labeling flag
                result = {
                    "model_used": "bgad",
                    "anomaly_score": bgad_score,
                    "is_anomaly": bgad_score > 0.5,
                    "confidence": 0.3,
                    "threshold": normal_threshold,
                    "requires_expert_labeling": True,
                    "routing_case": "C_uncertain_gpt4v_failed",
                    "vlm_result": {"error": str(e)},
                    "timestamp": datetime.now().isoformat()
                }
            
            # Clean up temp file
            if save_image_for_vlm and isinstance(image, torch.Tensor):
                try:
                    os.remove(temp_image_path)
                except:
                    pass  # Best effort cleanup
        
        logger.info(f"✓ Cascade decision: {result['routing_case']}")
        
        # Log to history
        self.history.append(result)
        
        return result
