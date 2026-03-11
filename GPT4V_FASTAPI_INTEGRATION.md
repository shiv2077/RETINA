# GPT-4V FastAPI Integration Guide

Shows how to expose GPT-4V methods via HTTP endpoints for easy use in production.

---

## Implementation

Add these endpoints to `src/backend/app.py`:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import tempfile
from src.backend.services.inference import InferenceService

app = FastAPI(title="RETINA Anomaly Detection")
inference_service = InferenceService()

# ============================================================================
# GPT-4V ENDPOINTS
# ============================================================================

@app.post("/inference/gpt4v/zero-shot")
async def gpt4v_zero_shot(file: UploadFile = File(...)):
    """
    Zero-shot anomaly detection using GPT-4V.
    
    Upload an image for real-time GPT-4V analysis.
    Returns: is_anomaly, defect_type, confidence, reasoning
    
    Usage:
        curl -X POST -F "file=@image.jpg" http://localhost:8000/inference/gpt4v/zero-shot
    
    Response:
        {
            "status": "success",
            "is_anomaly": true,
            "defect_type": "knot",
            "confidence": 0.85,
            "reasoning": "Visible circular knot...",
            "latency_ms": 1234
        }
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        # Call GPT-4V
        result = inference_service.call_gpt4v_zero_shot(tmp_path)
        
        # Return result
        return JSONResponse(
            status_code=200 if result["status"] == "success" else 400,
            content=result
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        import os
        try:
            os.remove(tmp_path)
        except:
            pass


@app.post("/inference/cascade-gpt4v")
async def cascade_with_gpt4v(
    file: UploadFile = File(...),
    normal_threshold: float = Query(0.2, ge=0, le=1),
    anomaly_threshold: float = Query(0.8, ge=0, le=1)
):
    """
    Complete cascade prediction with GPT-4V fallback.
    
    Implements 3-tier routing:
    - BGAD < normal_threshold: Confident normal (fast)
    - BGAD > anomaly_threshold: Confident anomaly (fast)
    - In between: Routed to GPT-4V (accurate)
    
    Usage:
        curl -X POST \\
          -F "file=@image.jpg" \\
          -F "normal_threshold=0.2" \\
          -F "anomaly_threshold=0.8" \\
          http://localhost:8000/inference/cascade-gpt4v
    
    Response:
        {
            "routing_case": "C_uncertain_gpt4v_routed",
            "requires_expert_labeling": true,
            "vlm_result": {
                "is_anomaly": true,
                "defect_type": "scratch",
                "confidence": 0.75,
                "reasoning": "Deep surface scratch detected"
            },
            "bgad_score": 0.65,
            "gpt4v_score": 0.75
        }
    """
    # Validate thresholds
    if normal_threshold >= anomaly_threshold:
        raise HTTPException(
            status_code=400,
            detail="normal_threshold must be < anomaly_threshold"
        )
    
    # Save uploaded file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    
    try:
        # Call cascade with GPT-4V
        result = inference_service.cascade_predict_with_gpt4v(
            tmp_path,
            normal_threshold=normal_threshold,
            anomaly_threshold=anomaly_threshold
        )
        
        return JSONResponse(
            status_code=200,
            content=result
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up
        import os
        try:
            os.remove(tmp_path)
        except:
            pass


@app.get("/inference/gpt4v/status")
async def gpt4v_status():
    """
    Check GPT-4V API status and configuration.
    
    Response:
        {
            "gpt4v_available": true,
            "model": "gpt-4-vision-preview",
            "api_key_configured": true,
            "cascade_stats": {...}
        }
    """
    return {
        "gpt4v_available": inference_service.openai_client is not None,
        "model": inference_service.gpt4v_model,
        "api_key_configured": bool(inference_service.openai_client),
        "cascade_stats": inference_service.get_cascade_statistics()
    }


@app.get("/inference/gpt4v/stats")
async def gpt4v_stats():
    """
    Get cascade routing statistics for monitoring.
    
    Response:
        {
            "confident_normal": 1234,
            "confident_anomaly": 567,
            "uncertain_routed_to_vlm": 89,
            "vlm_flagged_anomaly": 45,
            "edge_model_utilization": 95.2,
            "vlm_catch_rate": 50.6
        }
    """
    return inference_service.get_cascade_statistics()


@app.post("/inference/gpt4v/reset-stats")
async def reset_gpt4v_stats():
    """Reset cascade routing statistics."""
    inference_service.reset_cascade_statistics()
    return {"status": "Statistics reset"}
```

---

## Usage Examples

### Example 1: Direct Zero-Shot Analysis

**cURL:**
```bash
curl -X POST \
  -F "file=@wood_sample.jpg" \
  http://localhost:8000/inference/gpt4v/zero-shot
```

**Python:**
```python
import requests

with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/inference/gpt4v/zero-shot",
        files=files
    )

result = response.json()
print(f"Anomaly: {result['is_anomaly']}")
print(f"Defect: {result['defect_type']}")
print(f"Confidence: {result['confidence']:.0%}")
```

**JavaScript/Fetch:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch(
    'http://localhost:8000/inference/gpt4v/zero-shot',
    {
        method: 'POST',
        body: formData
    }
);

const result = await response.json();
console.log(`Anomaly: ${result.is_anomaly}`);
console.log(`Defect: ${result.defect_type}`);
```

### Example 2: Cascade Routing with Custom Thresholds

**cURL:**
```bash
curl -X POST \
  -F "file=@image.jpg" \
  -F "normal_threshold=0.25" \
  -F "anomaly_threshold=0.75" \
  http://localhost:8000/inference/cascade-gpt4v
```

**Python:**
```python
import requests

with open("image.jpg", "rb") as f:
    files = {"file": f}
    params = {
        "normal_threshold": 0.25,
        "anomaly_threshold": 0.75
    }
    response = requests.post(
        "http://localhost:8000/inference/cascade-gpt4v",
        files=files,
        params=params
    )

result = response.json()
if result["requires_expert_labeling"]:
    print(f"📋 Expert review needed: {result['routing_case']}")
```

### Example 3: Check Status

**cURL:**
```bash
curl http://localhost:8000/inference/gpt4v/status
```

**Response:**
```json
{
    "gpt4v_available": true,
    "model": "gpt-4-vision-preview",
    "api_key_configured": true,
    "cascade_stats": {
        "confident_normal": 1234,
        "confident_anomaly": 567,
        "uncertain_routed_to_vlm": 89,
        "vlm_flagged_anomaly": 45,
        "edge_model_utilization": 95.2,
        "vlm_catch_rate": 50.6
    }
}
```

### Example 4: Monitor Cascade Statistics

**cURL:**
```bash
curl http://localhost:8000/inference/gpt4v/stats
```

**Interpretation:**
```
Confident Normal:     1234 images (70%)  - Accepted on edge
Confident Anomaly:    567 images (20%)   - Rejected on edge
Uncertain → GPT-4V:   89 images (5%)     - Analyzed by cloud
VLM Flagged Anomaly:  45 images (51%)    - Of the uncertain cases, 51% confirmed

Edge Utilization: 95.2% (Cases A+B stay on device)
VLM Catch Rate: 50.6% (GPT-4V agrees with 51% of uncertain cases as anomalies)
```

---

## Integration with Frontend

### React Component Example

```javascript
import React, { useState } from 'react';

export function GPT4VAnalysis() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async (e) => {
    e.preventDefault();
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('normal_threshold', 0.2);
    formData.append('anomaly_threshold', 0.8);

    try {
      const response = await fetch(
        'http://localhost:8000/inference/cascade-gpt4v',
        {
          method: 'POST',
          body: formData
        }
      );
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="gpt4v-analysis">
      <form onSubmit={handleAnalyze}>
        <input
          type="file"
          accept="image/*"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? 'Analyzing...' : 'Analyze with GPT-4V'}
        </button>
      </form>

      {result && (
        <div className="results">
          <h3>Cascade Routing: {result.routing_case}</h3>
          
          {result.routing_case === 'A_confident_normal' && (
            <p>✅ Normal image (BGAD < 0.2)</p>
          )}
          
          {result.routing_case === 'B_confident_anomaly' && (
            <p>❌ Anomaly detected (BGAD > 0.8)</p>
          )}
          
          {result.routing_case.includes('gpt4v_routed') && (
            <div>
              <p>🤔 GPT-4V Analysis:</p>
              <p>Anomaly: {result.vlm_result.is_anomaly ? '✓' : '✗'}</p>
              <p>Defect: {result.vlm_result.defect_type}</p>
              <p>Confidence: {(result.vlm_result.confidence * 100).toFixed(0)}%</p>
              <p>Reason: {result.vlm_result.reasoning}</p>
            </div>
          )}

          {result.requires_expert_labeling && (
            <button className="flag-for-review">
              Flag for Expert Review
            </button>
          )}
        </div>
      )}
    </div>
  );
}
```

---

## OpenAPI Documentation

The endpoints are automatically documented at:
```
http://localhost:8000/docs
```

This provides an interactive swagger UI where you can:
- Test endpoints directly
- See request/response schemas
- Check parameter validation
- View error messages

---

## Performance Considerations

### Rate Limiting

Add rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/inference/gpt4v/zero-shot")
@limiter.limit("10/minute")
async def gpt4v_zero_shot(request: Request, file: UploadFile = File(...)):
    # Implementation
    pass
```

### Async Processing

For high-volume scenarios, consider async task queue:

```python
from celery import Celery

celery_app = Celery('retina')

@celery_app.task
def analyze_gpt4v(image_path: str):
    """Async task for GPT-4V analysis"""
    result = inference_service.call_gpt4v_zero_shot(image_path)
    return result

@app.post("/inference/gpt4v/analyze-async")
async def gpt4v_analyze_async(file: UploadFile = File(...)):
    """Fire and forget"""
    # Save file
    # Queue task
    # Return job_id
    pass
```

### Caching

Cache results for identical images:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_result(image_hash: str):
    """Cache GPT-4V results by image hash"""
    pass

# Before calling GPT-4V:
image_hash = hashlib.sha256(image_data).hexdigest()
cached = get_cached_result(image_hash)
```

---

## Deployment

### Docker Setup

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Set OpenAI API key
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV PYTHONUNBUFFERED=1

# Run server
CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build & Run:**
```bash
docker build -t retina-gpt4v .

docker run \
  -e OPENAI_API_KEY="sk-proj-..." \
  -p 8000:8000 \
  retina-gpt4v
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: retina-gpt4v
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: retina
        image: retina-gpt4v:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

---

## Monitoring

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

gpt4v_calls = Counter('gpt4v_calls_total', 'Total GPT-4V calls')
gpt4v_latency = Histogram('gpt4v_latency_seconds', 'GPT-4V latency')

@app.post("/inference/gpt4v/zero-shot")
async def gpt4v_zero_shot(file: UploadFile = File(...)):
    gpt4v_calls.inc()
    with gpt4v_latency.time():
        result = inference_service.call_gpt4v_zero_shot(...)
    return result
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

@app.post("/inference/gpt4v/zero-shot")
async def gpt4v_zero_shot(file: UploadFile = File(...)):
    logger.info(f"GPT-4V request: {file.filename}")
    result = inference_service.call_gpt4v_zero_shot(...)
    logger.info(f"GPT-4V result: {result['status']}, latency: {result['latency_ms']}ms")
    return result
```

---

## Testing Endpoints

### Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from src.backend.app import app

client = TestClient(app)

def test_gpt4v_status():
    response = client.get("/inference/gpt4v/status")
    assert response.status_code == 200
    assert "gpt4v_available" in response.json()

@pytest.mark.asyncio
async def test_gpt4v_zero_shot(test_image_path):
    with open(test_image_path, "rb") as f:
        response = client.post(
            "/inference/gpt4v/zero-shot",
            files={"file": f}
        )
    assert response.status_code in [200, 400]  # 400 if API fails
```

---

## Summary

**What was created:**
- 5 FastAPI endpoints for GPT-4V
- OpenAPI/Swagger documentation (auto)
- Frontend integration example
- Deployment guides (Docker, Kubernetes)
- Monitoring examples

**Endpoints:**
- `POST /inference/gpt4v/zero-shot` - Direct analysis
- `POST /inference/cascade-gpt4v` - Cascade routing
- `GET /inference/gpt4v/status` - Health check
- `GET /inference/gpt4v/stats` - Statistics
- `POST /inference/gpt4v/reset-stats` - Reset stats

**Next steps:**
1. Copy endpoints to `src/backend/app.py`
2. Test with `curl` examples
3. Integrate with frontend
4. Deploy to production

---

**Status:** ✅ Ready for HTTP integration
