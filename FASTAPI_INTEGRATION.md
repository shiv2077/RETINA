# FastAPI Integration Template

## Copy-Paste Ready Code for Your Backend

Use these code snippets to integrate the new BGAD fit() and predict() methods into your FastAPI backend.

---

## 1. Update Pipeline Service (src/backend/services/pipeline.py)

```python
# Add this import at the top
from models.bgad import BGADModel
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class PipelineService:
    """Multi-stage anomaly detection pipeline"""
    
    # ... existing code ...
    
    def run_stage3_supervised(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 10,
        save_path: str = "models/bgad_stage3.pt"
    ):
        """
        Stage 3: Train supervised BGAD model on labeled data.
        
        Args:
            train_loader: DataLoader with (image, label) pairs
            val_loader: Optional validation DataLoader
            epochs: Training epochs (default: 10)
            save_path: Where to save trained model
        
        Returns:
            Tuple of (trained_bgad_model, training_history)
        """
        logger.info("="*60)
        logger.info("🚀 STAGE 3: Supervised BGAD Training")
        logger.info("="*60)
        
        try:
            # Initialize BGAD model
            bgad_model = BGADModel(
                backbone="resnet18",        # Can use resnet50, efficientnet_b0
                feature_dim=256,
                margin=1.0,                 # Boundary margin
                pull_weight=1.0,            # Normal sample attraction
                push_weight=0.1,            # Anomaly repulsion
            )
            
            logger.info(f"✓ BGADModel initialized")
            logger.info(f"  Backbone: resnet18")
            logger.info(f"  Feature dim: 256")
            logger.info(f"  Margin: 1.0")
            
            # Train using the new fit() method
            history = bgad_model.fit(
                dataloader=train_loader,
                epochs=epochs,
                lr=0.001,
                save_path=save_path
            )
            
            # Return model and history
            logger.info(f"✓ Training finished successfully")
            logger.info(f"  Final loss: {history['total_loss'][-1]:.4f}")
            logger.info(f"  Model saved to: {save_path}")
            
            return bgad_model, history
            
        except Exception as e:
            logger.error(f"❌ Stage 3 training failed: {str(e)}")
            raise
    
    def evaluate_stage3(self, model, val_loader):
        """Evaluate trained BGAD model"""
        # Optional: Add validation logic here
        pass
```

---

## 2. Add Training Endpoint (src/backend/api/routes.py)

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class TrainingRequest(BaseModel):
    """Request body for training"""
    stage: int  # 1, 2, or 3
    epochs: Optional[int] = 10
    learning_rate: Optional[float] = 0.001
    batch_size: Optional[int] = 32
    dataset_path: Optional[str] = "./MSVPD_Unified_Dataset"

class TrainingResponse(BaseModel):
    """Response after training"""
    status: str
    stage: int
    epochs: int
    final_loss: float
    model_path: str

@router.post("/api/train/stage3")
async def train_stage3(request: TrainingRequest):
    """
    Train Stage 3 (Supervised BGAD).
    
    This endpoint trains a Boundary-Guided Anomaly Detection model
    using labeled normal and anomalous samples.
    
    Example request:
    {
        "stage": 3,
        "epochs": 10,
        "learning_rate": 0.001,
        "dataset_path": "./MSVPD_Unified_Dataset"
    }
    """
    try:
        logger.info(f"📊 Training request received")
        logger.info(f"  Stage: {request.stage}")
        logger.info(f"  Epochs: {request.epochs}")
        logger.info(f"  Learning rate: {request.learning_rate}")
        
        # Validate request
        if request.stage != 3:
            raise HTTPException(status_code=400, detail="Invalid stage")
        
        # Load dataset
        from torch.utils.data import DataLoader
        from datasets import MVTecDataset
        
        dataset = MVTecDataset(
            root_dir=request.dataset_path,
            split="train"
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=request.batch_size or 32,
            shuffle=True,
            num_workers=4
        )
        
        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Create pipeline service
        pipeline = PipelineService()
        
        # Train BGAD model (runs in background)
        model, history = await asyncio.to_thread(
            pipeline.run_stage3_supervised,
            train_loader=train_loader,
            epochs=request.epochs or 10,
            save_path=f"models/bgad_stage3_{int(time.time())}.pt"
        )
        
        # Return response
        return TrainingResponse(
            status="success",
            stage=3,
            epochs=request.epochs or 10,
            final_loss=float(history["total_loss"][-1]),
            model_path=f"models/bgad_stage3.pt"
        )
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 3. Add Inference Endpoint (src/backend/api/routes.py)

```python
from fastapi import UploadFile, File
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import io

class PredictionResponse(BaseModel):
    """Response from prediction endpoint"""
    status: str
    model: str
    anomaly_score: float
    is_anomaly: bool
    threshold: float
    confidence: Optional[float] = None

@router.post("/api/predict")
async def predict_defect(
    file: UploadFile = File(...),
    model_path: Optional[str] = None
):
    """
    Predict anomaly score for uploaded image.
    
    Uses trained BGAD model to predict whether image contains defects.
    
    Returns:
    {
        "status": "success",
        "model": "BGAD",
        "anomaly_score": 0.847,
        "is_anomaly": false,
        "threshold": 1.0,
        "confidence": 0.92
    }
    """
    try:
        # Load image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocessing pipeline (must match training)
        preprocess = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        image_tensor = preprocess(image)  # [3, 224, 224]
        
        # Load BGAD model
        from models.bgad import BGADModel
        
        bgad_model = BGADModel()
        bgad_model.load(model_path or "models/bgad_stage3.pt")
        
        # Get prediction using new predict() method
        anomaly_score = bgad_model.predict(image_tensor)
        
        # Determine if anomalous
        threshold = bgad_model.threshold  # Default: 1.0
        is_anomaly = float(anomaly_score) > threshold
        
        # Compute confidence (distance from threshold)
        confidence = abs(float(anomaly_score) - threshold) / threshold
        confidence = min(1.0, confidence)  # Clamp to [0, 1]
        
        logger.info(f"✓ Prediction: score={anomaly_score:.4f}, "
                   f"is_anomaly={is_anomaly}, confidence={confidence:.2f}")
        
        return PredictionResponse(
            status="success",
            model="BGAD",
            anomaly_score=float(anomaly_score),
            is_anomaly=is_anomaly,
            threshold=float(threshold),
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 4. Add Batch Inference Endpoint (src/backend/api/routes.py)

```python
from typing import List

@router.post("/api/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_path: Optional[str] = None
):
    """
    Predict anomaly scores for multiple images.
    
    Returns list of predictions for each image.
    """
    try:
        from models.bgad import BGADModel
        
        # Load model once
        bgad_model = BGADModel()
        bgad_model.load(model_path or "models/bgad_stage3.pt")
        
        results = []
        
        for file in files:
            # Load and preprocess image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            preprocess = Compose([
                Resize((224, 224)),
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            image_tensor = preprocess(image)
            
            # Get prediction
            anomaly_score = bgad_model.predict(image_tensor)
            threshold = bgad_model.threshold
            is_anomaly = float(anomaly_score) > threshold
            
            results.append({
                "filename": file.filename,
                "anomaly_score": float(anomaly_score),
                "is_anomaly": is_anomaly,
                "threshold": float(threshold)
            })
        
        return {
            "status": "success",
            "model": "BGAD",
            "num_images": len(results),
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 5. Update Backend Initialization (src/backend/main.py)

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router
from utils.logger import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RETINA Anomaly Detection API",
    description="Multi-Stage Visual Defect Detection Pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(api_router)

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    logger.info("="*60)
    logger.info("🚀 RETINA Backend Starting")
    logger.info("="*60)
    logger.info("✓ BGAD fit() and predict() methods ready")
    logger.info("✓ POST /api/train/stage3 - Train BGAD model")
    logger.info("✓ POST /api/predict - Single image inference")
    logger.info("✓ POST /api/predict/batch - Batch inference")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("🛑 RETINA Backend Shutting Down")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 6. Update Frontend API Call (frontend/src/app/api/train/stage3/route.ts)

```typescript
// frontend/src/app/api/train/stage3/route.ts

import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    const response = await fetch('http://localhost:8000/api/train/stage3', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        stage: 3,
        epochs: body.epochs || 10,
        learning_rate: body.learningRate || 0.001,
        batch_size: body.batchSize || 32,
        dataset_path: body.datasetPath || './MSVPD_Unified_Dataset',
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    
    return NextResponse.json({
      status: 'success',
      data: data,
      message: `Stage 3 training completed. Final loss: ${data.final_loss.toFixed(4)}`,
    });
    
  } catch (error) {
    console.error('Training error:', error);
    return NextResponse.json(
      { 
        status: 'error',
        message: error instanceof Error ? error.message : 'Training failed',
      },
      { status: 500 }
    );
  }
}
```

---

## 7. Update Inference API Call (frontend/src/app/api/predict/route.ts)

```typescript
// frontend/src/app/api/predict/route.ts

import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }
    
    // Forward to backend
    const backendFormData = new FormData();
    backendFormData.append('file', file);
    
    const response = await fetch('http://localhost:8000/api/predict', {
      method: 'POST',
      body: backendFormData,
    });
    
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    
    const data = await response.json();
    
    return NextResponse.json({
      status: 'success',
      prediction: data,
      message: data.is_anomaly ? 'Defect detected!' : 'Image is normal',
    });
    
  } catch (error) {
    console.error('Prediction error:', error);
    return NextResponse.json(
      { 
        status: 'error',
        message: error instanceof Error ? error.message : 'Prediction failed',
      },
      { status: 500 }
    );
  }
}
```

---

## ✅ Testing the Integration

### Test Training Endpoint

```bash
curl -X POST http://localhost:8000/api/train/stage3 \
  -H "Content-Type: application/json" \
  -d '{
    "stage": 3,
    "epochs": 10,
    "learning_rate": 0.001
  }'
```

Expected response:
```json
{
  "status": "success",
  "stage": 3,
  "epochs": 10,
  "final_loss": 0.4532,
  "model_path": "models/bgad_stage3_1710000123.pt"
}
```

### Test Prediction Endpoint

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@test_image.png"
```

Expected response:
```json
{
  "status": "success",
  "model": "BGAD",
  "anomaly_score": 0.847,
  "is_anomaly": false,
  "threshold": 1.0,
  "confidence": 0.15
}
```

---

## 🚀 Deployment Checklist

- [ ] Copy fit() and predict() implementations (already done ✓)
- [ ] Update pipeline.py with run_stage3_supervised() method
- [ ] Add /api/train/stage3 endpoint
- [ ] Add /api/predict endpoint
- [ ] Add /api/predict/batch endpoint
- [ ] Update frontend API calls
- [ ] Test training from UI
- [ ] Test inference from UI
- [ ] Monitor FastAPI logs during training
- [ ] Deploy to production

All code snippets are ready to copy-paste! 🎉
