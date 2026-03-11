# 📋 Cascade Inference → Active Learning Queue Integration Guide

## Overview

Your RETINA system now has **end-to-end connectivity** from cascade inference through expert annotation:

```
┌─────────────────────┐
│  User Image Input   │
│   (from camera)     │
└──────────┬──────────┘
           │
           ▼
     ┌─────────────┐
     │ BGAD Model  │  ← Fast edge inference (<10ms)
     │ (RTX 3060)  │
     └──────┬──────┘
            │
     ┌──────▼─────────┐
     │ Cascade Router │
     │ 3-Tier Logic   │
     └──────┬─────────┘
            │
     ┌──────▼────────────────┐
     │ Score Comparison       │
     └┬─────────┬──────────┬──┘
      │         │          │
 Score<0.2  0.2-0.8   Score>0.8
   Conf.    Uncertain  Conf.
  Normal    (→VLM)    Anomaly
      │         │          │
      └─────────┼──────────┘
              │
     ┌────────▼──────────┐
     │ VLM Fallback      │  ← For uncertain cases
     │ (AdaCLIP/WinCLIP) │    (~650ms)
     └────────┬──────────┘
              │
        ┌─────▼──────────┐
        │ Requires Lab?  │
        └─────┬──────────┘
              │
     ┌────────▼──────────────┐
     │ Add to Queue           │
     │ (Cascade Queue)        │
     │ ✨ Auto-flagged        │
     └────────┬──────────────┘
              │
     ┌────────▼──────────────┐
     │ Annotation Studio      │
     │ (Next.js Dashboard)    │
     │ • Canvas drawing       │
     │ • Defect labeling      │
     │ • Bounding boxes       │
     └────────┬──────────────┘
              │
     ┌────────▼──────────────┐
     │ Production Dataset     │
     │ (with cascade meta)    │
     └───────────────────────┘
```

---

## System Architecture

### Backend Services (Python/FastAPI)

#### 1. **Inference Service** (`src/backend/services/inference.py`)

**Method**: `predict_with_cascade()`

- Takes image tensor as input
- Runs BGAD on edge device
- Routes based on thresholds:
  - **Case A**: Score < 0.2 → Return normal immediately
  - **Case B**: Score > 0.8 → Return anomaly immediately
  - **Case C**: 0.2 ≤ Score ≤ 0.8 → Route to VLM
- Returns structured response with `requires_expert_labeling` flag

**Output Format**:
```python
{
    "model_used": "bgad" | "ensemble" | "vlm",
    "anomaly_score": 0.0-2.0,  # BGAD score
    "is_anomaly": bool,
    "confidence": 0.0-1.0,
    "routing_case": "A_confident_normal" | "B_confident_anomaly" | "C_uncertain_vlm_routed",
    "requires_expert_labeling": bool,  # ← KEY FLAG!
    "vlm_result": {...},
    "bgad_score": 0.0-2.0,
    "vlm_score": 0.0-1.0,
    "timestamp": "ISO-8601",
    "processing_time_ms": int
}
```

#### 2. **Labeling Service** (`src/backend/services/labeling.py`)

**New Methods** (Thread-safe with `threading.RLock()`):

| Method | Purpose |
|--------|---------|
| `add_to_cascade_queue()` | Add flagged images to queue |
| `get_cascade_queue()` | Fetch pending annotations |
| `mark_cascade_labeled()` | Mark as labeled + create annotation |
| `skip_cascade_item()` | Skip without labeling |
| `get_cascade_stats()` | Monitor queue |

**Queue Item Format**:
```python
{
    "image_id": "unique_id",
    "image_path": "/path/to/image.png",
    "bgad_score": 0.51,
    "vlm_score": 0.55,
    "routing_case": "C_uncertain_vlm_routed",
    "status": "pending" | "labeled" | "skipped",
    "created_at": "ISO-8601",
    "metadata": {...}
}
```

**Queue File**: `annotations/cascade_queue.json` (persists across restarts)

#### 3. **FastAPI Endpoints** (`src/backend/app.py`)

| Endpoint | Method | Purpose | Response |
|----------|--------|---------|----------|
| `/api/predict/cascade` | POST | Run cascade prediction | CascadeResponse + queue_info |
| `/api/labeling/cascade/queue` | GET | Fetch pending items | CascadeQueueResponse |
| `/api/labeling/cascade/submit` | POST | Submit annotation | Success + remaining count |
| `/api/labeling/cascade/skip/{image_id}` | POST | Skip item | Success status |
| `/api/labeling/cascade/stats` | GET | Queue statistics | CascadeQueueStats |

### Frontend Services (TypeScript/Next.js)

#### 1. **API Client** (`frontend/src/lib/api.ts`)

**New Functions**:

```typescript
// Run cascade prediction
export async function predictCascade(
  imageFile: File,
  options?: {
    normal_threshold?: number;
    anomaly_threshold?: number;
    use_vlm_fallback?: boolean;
  }
): Promise<CascadeResponse>

// Fetch queue
export async function fetchAnnotationQueue(
  limit?: number
): Promise<CascadeQueueResponse>

// Submit annotation
export async function submitCascadeAnnotation(
  submission: CascadeAnnotationSubmission
): Promise<{ success: boolean; image_id: string; remaining_in_queue: number }>

// Skip item
export async function skipCascadeItem(
  image_id: string
): Promise<{ success: boolean; image_id: string; remaining_in_queue: number }>

// Get stats
export async function getCascadeStats(
): Promise<CascadeQueueStats>
```

#### 2. **Annotation Studio** (`frontend/src/app/label/page.tsx`)

**New State**:
- `isCascadeMode`: Switch between cascade queue and standard pipeline
- `cascadeStats`: Real-time queue statistics
- Auto-dropdown selector with "cascade" option

**Cascade-Specific Features**:
- Sample info panel showing BGAD/VLM scores and routing case
- Skip button for uncertain items
- Auto-reload when queue depletes
- Real-time stats (pending count, avg score)

---

## End-to-End Workflow

### 1️⃣ Inference Stage (Backend)

**User uploads image via inference endpoint**:
```python
POST /api/predict/cascade
Content-Type: multipart/form-data

file: <image.png>
normal_threshold: 0.2
anomaly_threshold: 0.8
use_vlm_fallback: True
```

**Backend processes**:
1. Load image, convert to tensor
2. Run BGAD on RTX 3060
3. Get anomaly_score (0-2.0)
4. Compare against thresholds
5. If Case C: Route to VLM
6. Return result with `requires_expert_labeling` flag

**Response Example** (Case C - Uncertain):
```json
{
  "model_used": "ensemble",
  "anomaly_score": 0.51,
  "is_anomaly": true,
  "confidence": 0.50,
  "routing_case": "C_uncertain_vlm_routed",
  "requires_expert_labeling": true,
  "vlm_result": {
    "classification": "anomaly",
    "confidence": 0.72
  },
  "bgad_score": 0.51,
  "vlm_score": 0.55,
  "timestamp": "2026-03-11T14:30:00",
  "processing_time_ms": 650,
  "queue_info": {
    "success": true,
    "image_id": "image_001",
    "queue_position": 0,
    "queue_size": 3
  }
}
```

### 2️⃣ Queue Stage (Backend)

**When `requires_expert_labeling=True`**:
- Image automatically added to cascade queue
- Queue persisted to `annotations/cascade_queue.json`
- Thread-safe: Uses RLock to prevent race conditions

**Queue state** (one item):
```json
{
  "image_id": "image_001",
  "image_path": "path/to/image.png",
  "bgad_score": 0.51,
  "vlm_score": 0.55,
  "routing_case": "C_uncertain_vlm_routed",
  "status": "pending",
  "created_at": "2026-03-11T14:30:00",
  "source": "cascade_inference",
  "metadata": {
    "bgad_confidence": 0.50,
    "vlm_result": {...},
    "timestamp": "2026-03-11T14:30:00"
  }
}
```

### 3️⃣ Annotation Studio (Frontend)

**Expert opens annotation studio**:

1. Navigate to label page: `http://localhost:3000/label`
2. Change dropdown from "bottle" to **"cascade"**
3. UI loads pending items from queue
4. Shows sample info panel with:
   - Image ID
   - BGAD score (with color: green if <0.8, red if >0.8)
   - VLM score (if available)
   - Routing case (for debugging)

**Expert annotates**:
1. Canvas shows image
2. Expert draws bounding box (keyboard shortcut or mouse)
3. Selects defect type (scratch, crack, dent, etc.)
4. Optionally adds notes
5. Clicks "Anomaly" or "Normal"
6. Submission triggered

### 4️⃣ Submission Stage (Backend)

**POST /api/labeling/cascade/submit**:
```json
{
  "image_id": "image_001",
  "label": "anomaly",
  "bounding_boxes": [
    {
      "x": 0.1,
      "y": 0.15,
      "width": 0.3,
      "height": 0.25,
      "defect_type": "scratch",
      "confidence": 1.0
    }
  ],
  "defect_types": ["scratch"],
  "notes": "Clear surface scratch visible"
}
```

**Backend processes**:
1. Find image in cascade queue (thread-safe)
2. Create Annotation object with cascade metadata
3. Save to annotation store
4. Mark queue item as "labeled"
5. Return remaining count

**Response**:
```json
{
  "success": true,
  "image_id": "image_001",
  "label": "anomaly",
  "remaining_in_queue": 2
}
```

### 5️⃣ Production Dataset (Persistent)

**Annotation stored with cascade context**:
```python
Annotation(
    image_id="image_001",
    image_path="path/to/image.png",
    label="anomaly",
    bounding_boxes=[...],
    annotation_score=0.51,  # BGAD score
    metadata={
        "cascade_source": True,
        "bgad_score": 0.51,
        "vlm_score": 0.55,
        "routing_case": "C_uncertain_vlm_routed",
        "surface_model": "RetNet18"
    }
)
```

---

## Usage Examples

### Example 1: Python Backend (Direct API)

```python
from src.backend.services.inference import InferenceService
from src.backend.services.labeling import LabelingService
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

# Initialize services
inference_service = InferenceService()
labeling_service = LabelingService("annotations")

# Load image
img = Image.open("test_image.png").convert("RGB")
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = transform(img).unsqueeze(0)

# Run cascade prediction
result = inference_service.predict_with_cascade(
    image=image_tensor,
    normal_threshold=0.2,
    anomaly_threshold=0.8,
    use_vlm_fallback=True
)

print(f"Prediction: {result['routing_case']}")
print(f"BGAD Score: {result['anomaly_score']:.3f}")

# If flagged, automatically queued in labeling_service.cascade_queue
if result["requires_expert_labeling"]:
    print(f"✨ Added to annotation queue!")
    
    # Later, expert submits annotation
    labeling_service.mark_cascade_labeled(
        image_id=result.get("image_id"),
        label="anomaly",
        bounding_boxes=[{
            "x": 0.1, "y": 0.15, "width": 0.3, "height": 0.25,
            "defect_type": "scratch", "confidence": 1.0
        }],
        defect_types=["scratch"],
        notes="Clear scratch"
    )
    print(f"✅ Annotation saved!")
```

### Example 2: Frontend (TypeScript)

```typescript
import * as api from '@/lib/api';

// User uploads image for cascade prediction
const file = new File([imageData], 'test.png', { type: 'image/png' });

const response = await api.predictCascade(file, {
  normal_threshold: 0.2,
  anomaly_threshold: 0.8,
  use_vlm_fallback: true
});

console.log(`Model used: ${response.model_used}`);
console.log(`Routing case: ${response.routing_case}`);

if (response.requires_expert_labeling) {
  console.log(`✨ Image queued for annotation!`);
  console.log(`Queue position: ${response.queue_info?.queue_position}`);
}

// Later: Expert fetches queue
const queueResponse = await api.fetchAnnotationQueue(limit: 10);
console.log(`Pending items: ${queueResponse.queue_size}`);

// Expert submits annotation
const submission: api.CascadeAnnotationSubmission = {
  image_id: 'image_001',
  label: 'anomaly',
  bounding_boxes: [{
    x: 0.1,
    y: 0.15,
    width: 0.3,
    height: 0.25,
    defect_type: 'scratch',
    confidence: 1.0
  }],
  defect_types: ['scratch'],
  notes: 'Clear surface scratch'
};

const result = await api.submitCascadeAnnotation(submission);
console.log(`✅ Annotation submitted! Remaining: ${result.remaining_in_queue}`);
```

### Example 3: cURL (Testing)

```bash
# 1. Run cascade prediction
curl -X POST http://localhost:8000/api/predict/cascade \
  -F "file=@test_image.png" \
  -F "normal_threshold=0.2" \
  -F "anomaly_threshold=0.8" \
  -F "use_vlm_fallback=true"

# Response:
# {
#   "model_used": "ensemble",
#   "anomaly_score": 0.51,
#   "routing_case": "C_uncertain_vlm_routed",
#   "requires_expert_labeling": true,
#   "queue_info": {...}
# }

# 2. Fetch cascade queue
curl http://localhost:8000/api/labeling/cascade/queue?limit=10

# Response:
# {
#   "success": true,
#   "queue": [
#     {
#       "image_id": "image_001",
#       "bgad_score": 0.51,
#       "vlm_score": 0.55,
#       "routing_case": "C_uncertain_vlm_routed",
#       "status": "pending"
#     }
#   ],
#   "stats": {"total_in_queue": 1, "pending": 1, ...}
# }

# 3. Submit annotation
curl -X POST http://localhost:8000/api/labeling/cascade/submit \
  -F "image_id=image_001" \
  -F "label=anomaly" \
  -F "bounding_boxes=[{\"x\":0.1,\"y\":0.15,\"width\":0.3,\"height\":0.25,\"defect_type\":\"scratch\"}]" \
  -F "defect_types=[\"scratch\"]" \
  -F "notes=Clear scratch"

# Response:
# {
#   "success": true,
#   "image_id": "image_001",
#   "remaining_in_queue": 0
# }

# 4. Get cascade stats
curl http://localhost:8000/api/labeling/cascade/stats

# Response:
# {
#   "total_queued": 1,
#   "pending": 0,
#   "labeled": 1,
#   "skipped": 0,
#   "avg_bgad_score": 0.51,
#   "annotation_store_stats": {...}
# }
```

---

## Configuration & Tuning

### Cascade Thresholds

Adjust based on your use case:

```python
# Conservative (more VLM usage)
normal_threshold = 0.1      # Lower = more routed
anomaly_threshold = 0.9     # Higher = more routed
# Expected: 5-10% routed to VLM

# Moderate (balanced)
normal_threshold = 0.2      # Default
anomaly_threshold = 0.8     # Default
# Expected: 1-3% routed to VLM

# Aggressive (faster, less VLM)
normal_threshold = 0.3      # Higher = fewer routed
anomaly_threshold = 0.7     # Lower = fewer routed
# Expected: <1% routed to VLM
```

### Queue Behavior

**Auto-Queue Properties**:
- Items inserted at **front** (most recent first)
- Persisted to disk (survives restarts)
- Thread-safe (RLock prevents race conditions)
- No duplicates (checks if already pending)

**Clear Queue**:
```bash
rm annotations/cascade_queue.json
# Reload to fetch fresh queue
```

---

## Performance Metrics

### Expected Throughput

With single RTX 3060:

| Component | Latency | Throughput |
|-----------|---------|-----------|
| BGAD forward pass | 5-10ms | 100-200 img/s |
| VLM forward pass | 600-800ms | 1-2 img/s |
| Queue management | <1ms | N/A (async) |
| Annotation submission | 10-50ms | N/A (user-driven) |

### Queue Statistics

```json
{
  "total_queued": 150,
  "pending": 12,
  "labeled": 138,
  "skipped": 0,
  "avg_bgad_score": 0.5234,
  "annotation_store_stats": {
    "total": 138,
    "by_label": {
      "normal": 45,
      "anomaly": 93,
      "uncertain": 0
    },
    "with_bboxes": 93
  }
}
```

---

## Troubleshooting

### Queue not updating?

1. Check file exists: `annotations/cascade_queue.json`
2. Verify labeling service initialized: `LabelingService(ANNOTATIONS_DIR)`
3. Check for async issues: Use RLock when accessing cascade_queue

### Images not appearing in studio?

1. Verify `requires_expert_labeling=True` returned
2. Check queue endpoint: `GET /api/labeling/cascade/queue`
3. Ensure category dropdown set to "cascade"
4. Reload page (F5)

### VLM not being called?

1. Check: `use_vlm_fallback=True` in cascade call
2. Verify VLM loaded: `inference_service.load_vlm()`
3. Score must be between thresholds (0.2-0.8)
4. Check logs for VLM errors

### Annotation not saving?

1. Check POST body format (JSON arrays)
2. Verify `image_id` matches queue item
3. Check annotation directory permissions
4. Review error response

---

## Next Steps

1. **Test in isolation**:
   ```bash
   python -c "
   from src.backend.services.inference import InferenceService
   from PIL import Image
   service = InferenceService()
   img = Image.open('test.png')
   result = service.predict_with_cascade(...)
   print(result)"
   ```

2. **Test queue operations**:
   ```bash
   curl http://localhost:8000/api/labeling/cascade/stats
   ```

3. **Test full flow**:
   - Upload image → Check queue → Annotate → Verify saved

4. **Monitor production**:
   - Call `/api/labeling/cascade/stats` periodically
   - Track routing_case distribution
   - Monitor avg BGAD score

---

## Summary

✅ **You now have**:
- Cascade inference that auto-queues uncertain images
- Thread-safe queue that persists across restarts
- API to fetch and annotate queued items
- Next.js UI integrated with cascade queue
- End-to-end pipeline from prediction to labeled dataset

🚀 **Ready for production!**
