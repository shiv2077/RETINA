"""
RETINA API Server
FastAPI backend for multi-stage anomaly detection.
"""
import os
import sys
import io
import base64
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import torch

# Handle both direct execution and module import
try:
    from .config import config, UPLOADS_DIR, ANNOTATIONS_DIR, MODELS_DIR
    from .services import PipelineService, LabelingService, InferenceService
    from .services.pipeline import PipelineConfig
except ImportError:
    from config import config, UPLOADS_DIR, ANNOTATIONS_DIR, MODELS_DIR
    from services import PipelineService, LabelingService, InferenceService
    from services.pipeline import PipelineConfig


# ============================================================================
# API Models
# ============================================================================

class LabelSubmission(BaseModel):
    image_id: str
    label: str  # "normal", "anomaly", "uncertain"
    defect_type: Optional[str] = None
    defect_types: Optional[List[str]] = None
    bounding_boxes: Optional[List[Dict]] = None
    confidence: str = "high"
    notes: str = ""


class TrainRequest(BaseModel):
    category: str
    fast_sampling: bool = True


class InferenceRequest(BaseModel):
    category: str
    model: str = "patchcore"  # "patchcore", "bgad", "ensemble"


class EvaluationRequest(BaseModel):
    category: str


class CascadeResponse(BaseModel):
    model_used: str
    anomaly_score: float
    is_anomaly: bool
    confidence: float
    threshold: float
    requires_expert_labeling: bool
    routing_case: str
    vlm_score: Optional[float] = None
    vlm_reasoning: Optional[str] = None
    timestamp: str
    queue_info: Optional[Dict] = None

# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title="RETINA API",
    description="Multi-Stage Industrial Anomaly Detection System",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services
pipeline_service = PipelineService(PipelineConfig())
inference_service = InferenceService()
labeling_service = LabelingService(ANNOTATIONS_DIR)


# ============================================================================
# Health & Status Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "RETINA API",
        "version": "2.0.0",
        "status": "running",
        "device": str(inference_service.device)
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None
    }


@app.get("/status")
async def get_status():
    """Get full system status."""
    return {
        "pipeline": pipeline_service.get_status(),
        "labeling": labeling_service.get_progress(),
        "models": {
            "patchcore": list(inference_service.patchcore_models.keys()),
            "bgad": inference_service.bgad_model is not None
        }
    }


# ============================================================================
# Dataset Endpoints
# ============================================================================

@app.get("/categories")
async def list_categories():
    """List available MVTec AD categories."""
    mvtec_path = Path(config.mvtec_path)
    if not mvtec_path.exists():
        raise HTTPException(status_code=404, detail="MVTec dataset not found")
    
    categories = sorted([
        d.name for d in mvtec_path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    
    return {"categories": categories, "total": len(categories)}


@app.get("/categories/{category}/stats")
async def get_category_stats(category: str):
    """Get statistics for a category."""
    mvtec_path = Path(config.mvtec_path) / category
    if not mvtec_path.exists():
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
    
    train_path = mvtec_path / "train" / "good"
    test_path = mvtec_path / "test"
    
    train_count = len(list(train_path.glob("*.png"))) if train_path.exists() else 0
    
    test_stats = {}
    if test_path.exists():
        for defect_dir in test_path.iterdir():
            if defect_dir.is_dir():
                count = len(list(defect_dir.glob("*.png")))
                test_stats[defect_dir.name] = count
    
    return {
        "category": category,
        "train_samples": train_count,
        "test_samples": test_stats,
        "total_test": sum(test_stats.values())
    }


# ============================================================================
# Pipeline Endpoints
# ============================================================================

@app.post("/pipeline/stage1/train")
async def train_stage1(request: TrainRequest):
    """Stage 1: Train PatchCore on normal samples."""
    try:
        result = pipeline_service.run_stage1_unsupervised(
            category=request.category,
            fast_sampling=request.fast_sampling
        )
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/stage2/samples")
async def get_labeling_samples(batch_size: int = Query(default=20)):
    """Stage 2: Get samples for labeling."""
    result = pipeline_service.run_stage2_labeling(batch_size=batch_size)
    
    # Convert samples to JSON-serializable format
    batch = []
    for sample in result["batch"]:
        item = {
            "image_id": sample["image_id"],
            "anomaly_score": sample["anomaly_score"],
            "status": sample["status"]
        }
        
        # Encode anomaly map as base64 if available
        if sample.get("anomaly_map") is not None:
            # Normalize and convert to image
            amap = sample["anomaly_map"]
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            amap = (amap * 255).astype(np.uint8)
            
            img = Image.fromarray(amap)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            item["anomaly_map_base64"] = base64.b64encode(buffer.getvalue()).decode()
        
        batch.append(item)
    
    return {
        "batch": batch,
        "batch_size": result["batch_size"],
        "progress": result["progress"],
        "ready_for_stage3": result["ready_for_stage3"]
    }


@app.post("/pipeline/stage3/train")
async def train_stage3(epochs: int = Query(default=30)):
    """Stage 3: Train BGAD with labeled data."""
    try:
        result = pipeline_service.run_stage3_supervised(epochs=epochs)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/evaluate")
async def evaluate_pipeline():
    """Evaluate the full pipeline."""
    try:
        result = pipeline_service.evaluate_full_pipeline()
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Labeling Endpoints
# ============================================================================

@app.post("/labels/session/start")
async def start_labeling_session(annotator: str = Query(default="expert")):
    """Start a new labeling session."""
    session_id = labeling_service.start_session(annotator)
    return {"session_id": session_id, "annotator": annotator}


@app.post("/labels/submit")
async def submit_label(submission: LabelSubmission):
    """Submit a label for an image."""
    result = labeling_service.submit_label(
        image_id=submission.image_id,
        label=submission.label,
        defect_type=submission.defect_type,
        defect_types=submission.defect_types,
        bounding_boxes=submission.bounding_boxes,
        confidence=submission.confidence,
        notes=submission.notes
    )
    return result


@app.post("/labels/skip/{image_id}")
async def skip_sample(image_id: str):
    """Skip a sample without labeling."""
    return labeling_service.skip_sample(image_id)


# ============================================================================
# Cascade Inference & Active Learning Queue Endpoints
# ============================================================================

@app.post("/api/predict/cascade", response_model=CascadeResponse)
async def predict_cascade(
    file: UploadFile = File(...),
    normal_threshold: float = 0.2,
    anomaly_threshold: float = 0.8,
    use_vlm_fallback: bool = True
):
    """
    Cascade prediction endpoint.
    
    When uncertain (score between normal_threshold and anomaly_threshold):
    - Routes to VLM for additional validation
    - Automatically queues for expert annotation if flagged
    
    This is the primary entry point for active learning.
    """
    try:
        # Read and parse image
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Prepare tensor
        from torchvision.transforms import Compose, Resize, Normalize, ToTensor
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(pil_image).unsqueeze(0)
        
        # Run cascade prediction
        cascade_result = inference_service.predict_with_cascade(
            image=image_tensor,
            normal_threshold=normal_threshold,
            anomaly_threshold=anomaly_threshold,
            use_vlm_fallback=use_vlm_fallback
        )
        
        # If flagged for labeling, add to queue
        if cascade_result.get("requires_expert_labeling"):
            queue_result = labeling_service.add_to_cascade_queue(
                image_path=str(file.filename),
                bgad_score=cascade_result.get("anomaly_score", 0.0),
                vlm_score=cascade_result.get("vlm_score"),
                routing_case=cascade_result.get("routing_case"),
                metadata={
                    "bgad_confidence": cascade_result.get("confidence"),
                    "vlm_reasoning": cascade_result.get("vlm_reasoning"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            cascade_result["queue_info"] = queue_result
        
        return cascade_result
        
    except Exception as e:
        import logging
        logging.error(f"Cascade prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/labeling/cascade/queue")
async def get_cascade_queue(limit: int = Query(default=None)):
    """
    Fetch pending images from cascade annotation queue.
    
    Returns:
    - Queue items with image paths and metadata
    - Queue statistics (total pending, labeled, skipped)
    - Each item ready for annotation studio
    """
    result = labeling_service.get_cascade_queue(limit=limit)
    return result


@app.post("/api/labeling/cascade/submit")
async def submit_cascade_annotation(
    image_id: str = Form(...),
    label: str = Form(...),  # "normal" or "anomaly"
    bounding_boxes: str = Form(default="[]"),  # JSON string
    defect_types: str = Form(default="[]"),  # JSON list
    notes: str = Form(default="")
):
    """
    Submit annotation for cascade queue item.
    
    When submitted:
    - Image is marked as labeled
    - Annotation stored with cascade metadata
    - Image removed from pending queue
    """
    try:
        import json
        bboxes = json.loads(bounding_boxes)
        defects = json.loads(defect_types)
        
        result = labeling_service.mark_cascade_labeled(
            image_id=image_id,
            label=label,
            bounding_boxes=bboxes,
            defect_types=defects,
            notes=notes
        )
        return result
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/labeling/cascade/skip/{image_id}")
async def skip_cascade_item(image_id: str):
    """Skip a cascade queue item without labeling."""
    result = labeling_service.skip_cascade_item(image_id)
    return result


@app.get("/api/labeling/cascade/stats")
async def get_cascade_stats():
    """Get cascade queue statistics and annotation store stats."""
    result = labeling_service.get_cascade_stats()
    return result


@app.get("/labels/progress")
async def get_labeling_progress():
    """Get current labeling progress."""
    return labeling_service.get_progress()


@app.get("/labels/stats")
async def get_labeling_stats():
    """Get labeling statistics."""
    return labeling_service.store.get_stats()


@app.get("/labeling/queue")
async def get_labeling_queue(
    category: str = Query(default="bottle"),
    limit: int = Query(default=50)
):
    """Get images for labeling from the uncertain sample pool."""
    try:
        # Get samples from pipeline
        if pipeline_service.category == category and pipeline_service.patchcore is not None:
            # Get uncertain samples
            mvtec_path = Path(config.mvtec_path) / category / "test"
            samples = []
            
            for defect_dir in mvtec_path.iterdir():
                if not defect_dir.is_dir():
                    continue
                for img_path in defect_dir.glob("*.png")[:limit // 2]:
                    image = Image.open(img_path).convert("RGB")
                    result = inference_service.run_patchcore(image, category)
                    
                    # Calculate uncertainty (higher when score is near 0.5)
                    uncertainty = 1.0 - abs(result["anomaly_score"] - 0.5) * 2
                    
                    samples.append({
                        "image_id": f"{category}_{defect_dir.name}_{img_path.stem}",
                        "image_path": str(img_path.relative_to(Path(config.mvtec_path))),
                        "anomaly_score": result["anomaly_score"],
                        "uncertainty_score": uncertainty,
                    })
            
            # Sort by uncertainty (highest first)
            samples.sort(key=lambda x: x["uncertainty_score"], reverse=True)
            return {"samples": samples[:limit]}
        else:
            return {"samples": [], "message": "Train PatchCore first to generate samples"}
    except Exception as e:
        return {"samples": [], "error": str(e)}


@app.post("/labeling/submit")
async def submit_labeling(
    category: str = Form(...),
    annotation: str = Form(...)  # JSON string
):
    """Submit a labeling annotation."""
    import json
    try:
        annotation_data = json.loads(annotation)
        result = labeling_service.submit_label(
            image_id=annotation_data["image_id"],
            label=annotation_data["label"],
            defect_types=annotation_data.get("defect_types"),
            bounding_boxes=annotation_data.get("bounding_boxes"),
            notes=annotation_data.get("notes", "")
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/labeling/stats")
async def get_detailed_labeling_stats(category: str = Query(default=None)):
    """Get detailed labeling statistics."""
    stats = labeling_service.store.get_stats()
    return {
        "total": stats.get("total", 0),
        "normal": stats.get("by_label", {}).get("normal", 0),
        "anomaly": stats.get("by_label", {}).get("anomaly", 0),
        "uncertain": stats.get("by_label", {}).get("uncertain", 0),
    }


@app.get("/evaluations")
async def get_evaluations():
    """Get all stored evaluation results."""
    eval_file = ANNOTATIONS_DIR / "evaluations.json"
    if eval_file.exists():
        import json
        with open(eval_file) as f:
            return {"evaluations": json.load(f)}
    return {"evaluations": {}}


@app.get("/labels/export/{format}")
async def export_labels(format: str = "json"):
    """Export labels in specified format."""
    if format not in ["json", "coco", "yolo"]:
        raise HTTPException(status_code=400, detail=f"Unknown format: {format}")
    
    output_path = labeling_service.export(format=format)
    return {"success": True, "path": str(output_path)}


# ============================================================================
# Inference Endpoints
# ============================================================================

@app.post("/inference/image")
async def run_inference(
    file: UploadFile = File(...),
    category: str = Form(...),
    model: str = Form(default="patchcore")
):
    """Run inference on an uploaded image."""
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Save upload
    upload_path = UPLOADS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    image.save(upload_path)
    
    try:
        if model == "patchcore":
            result = inference_service.run_patchcore(image, category)
        elif model == "bgad":
            result = inference_service.run_bgad(image)
        elif model == "ensemble":
            result = inference_service.run_ensemble(image, category)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
        
        # Encode anomaly map if present
        if "anomaly_map" in result and result["anomaly_map"] is not None:
            amap = result["anomaly_map"]
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            amap = (amap * 255).astype(np.uint8)
            
            img = Image.fromarray(amap)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            result["anomaly_map_base64"] = base64.b64encode(buffer.getvalue()).decode()
            del result["anomaly_map"]
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Alias for frontend compatibility
@app.post("/inference/predict")
async def predict_inference(
    image: UploadFile = File(...),
    category: str = Form(default="bottle")
):
    """Run inference on an uploaded image (frontend-friendly endpoint)."""
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    start_time = datetime.now()
    
    try:
        # Try PatchCore first
        result = inference_service.run_patchcore(pil_image, category)
        
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        response = {
            "anomaly_score": result["anomaly_score"],
            "is_anomaly": result["is_anomaly"],
            "confidence": 1.0 - abs(result["anomaly_score"] - 0.5) * 2,  # Higher confidence at extremes
            "processing_time_ms": processing_time,
            "model_used": "PatchCore",
        }
        
        # Encode heatmap
        if "anomaly_map" in result and result["anomaly_map"] is not None:
            amap = result["anomaly_map"]
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
            amap = (amap * 255).astype(np.uint8)
            
            # Create colormap
            import cv2
            heatmap = cv2.applyColorMap(amap, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(heatmap)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            response["heatmap_base64"] = base64.b64encode(buffer.getvalue()).decode()
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inference/history")
async def get_inference_history(limit: int = Query(default=100)):
    """Get recent inference history."""
    return {"history": inference_service.get_history(limit)}


# ============================================================================
# Model Management Endpoints
# ============================================================================

@app.get("/models")
async def list_models():
    """List available models."""
    models = []
    
    # Check for PatchCore models
    for model_file in MODELS_DIR.glob("patchcore_*.pth"):
        category = model_file.stem.replace("patchcore_", "")
        models.append({
            "type": "patchcore",
            "category": category,
            "path": str(model_file),
            "size_mb": model_file.stat().st_size / 1e6
        })
    
    # Check for BGAD model
    bgad_path = MODELS_DIR / "bgad_model.pth"
    if bgad_path.exists():
        models.append({
            "type": "bgad",
            "category": "all",
            "path": str(bgad_path),
            "size_mb": bgad_path.stat().st_size / 1e6
        })
    
    return {"models": models, "total": len(models)}


@app.delete("/models/{model_type}/{category}")
async def delete_model(model_type: str, category: str):
    """Delete a model."""
    if model_type == "patchcore":
        model_path = MODELS_DIR / f"patchcore_{category}.pth"
    elif model_type == "bgad":
        model_path = MODELS_DIR / "bgad_model.pth"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
    
    if model_path.exists():
        model_path.unlink()
        return {"success": True, "deleted": str(model_path)}
    else:
        raise HTTPException(status_code=404, detail="Model not found")


# ============================================================================
# Run Server
# ============================================================================

def main():
    import uvicorn
    uvicorn.run(
        "src.backend.app:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug
    )


if __name__ == "__main__":
    main()
