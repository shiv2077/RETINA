# GPT-4V Integration Guide

Production-ready integration of OpenAI's GPT-4V for zero-shot industrial anomaly detection in RETINA's cascade router.

## Overview

Two new methods have been added to `InferenceService`:

1. **`call_gpt4v_zero_shot(image_path: str)`** - Direct GPT-4V API call
2. **`cascade_predict_with_gpt4v(...)`** - Enhanced cascade with GPT-4V fallback

### Why GPT-4V for Cascade Fallback?

RETINA's cascade router needs a fallback for uncertain cases:

```
BGAD Inference (Edge, Fast)
    ↓
Score < 0.2?  → Normal (70%)
Score > 0.8?  → Anomaly (20%)
0.2-0.8?      → UNCERTAIN (10%) ← Need fallback
                        ↓
                   GPT-4V Zero-Shot
                   (Cloud, Slow but accurate)
```

GPT-4V provides:
- ✅ **Zero-shot capability** - No training on Decospan data needed
- ✅ **Domain understanding** - Industrial QC expertise from training
- ✅ **Novel defect detection** - Catches defects the model hasn't seen
- ✅ **Interpretability** - Returns reasoning, not just scores
- ✅ **Production-ready** - Includes error handling, retry logic, timeouts

---

## Setup

### 1. Install OpenAI Client

```bash
pip install openai
```

### 2. Set OpenAI API Key

```bash
# Option A: Environment variable (recommended)
export OPENAI_API_KEY="sk-proj-..."

# Option B: In .env file
echo 'OPENAI_API_KEY=sk-proj-...' >> .env
source .env

# Option C: In code (NOT recommended - security risk)
# See examples below
```

Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### 3. Verify Setup

```python
from src.backend.services.inference import InferenceService

inference = InferenceService()

if inference.openai_client:
    print("✅ OpenAI client initialized (GPT-4V available)")
else:
    print("❌ OpenAI API key not set")
```

---

## Usage

### Method 1: Direct Zero-Shot Analysis

Use this when you have an uncertain BGAD prediction and want GPT-4V analysis.

```python
from src.backend.services.inference import InferenceService

inference = InferenceService()

# Analyze a single image
result = inference.call_gpt4v_zero_shot("path/to/image.jpg")

# Check result
if result["status"] == "success":
    if result["is_anomaly"]:
        print(f"❌ Anomaly detected: {result['defect_type']}")
        print(f"   Confidence: {result['confidence']:.0%}")
        print(f"   Analysis: {result['reasoning']}")
    else:
        print(f"✅ Image appears normal")
        print(f"   Confidence: {result['confidence']:.0%}")
else:
    print(f"⚠️  Analysis uncertain: {result['api_error']}")

print(f"Latency: {result['latency_ms']:.0f}ms")
```

**Response Format:**
```python
{
    "model_used": "gpt-4-vision",
    "is_anomaly": bool,
    "defect_type": "knot" | "scratch" | "discoloration" | None,
    "confidence": float (0.0-1.0),
    "reasoning": "Detailed explanation of analysis",
    "status": "success" | "uncertain",
    "api_error": str (optional, if status == "uncertain"),
    "timestamp": "2024-03-11T...",
    "latency_ms": float
}
```

### Method 2: Cascade Prediction with GPT-4V Fallback

Use this for complete inference with automatic routing to GPT-4V.

```python
from src.backend.services.inference import InferenceService

inference = InferenceService()

# Complete cascade with GPT-4V fallback
result = inference.cascade_predict_with_gpt4v(
    image="path/to/image.jpg",
    normal_threshold=0.2,    # BGAD confident normal below this
    anomaly_threshold=0.8    # BGAD confident anomaly above this
)

# Routing decision
if result["routing_case"] == "A_confident_normal":
    print(f"✓ CASE A (Confident Normal)")
    print(f"  BGAD score: {result['anomaly_score']:.4f}")
    
elif result["routing_case"] == "B_confident_anomaly":
    print(f"✗ CASE B (Confident Anomaly)")
    print(f"  BGAD score: {result['anomaly_score']:.4f}")
    
elif result["routing_case"] == "C_uncertain_gpt4v_routed":
    print(f"? CASE C (Uncertain - Routed to GPT-4V)")
    print(f"  BGAD score: {result['bgad_score']:.4f}")
    print(f"  GPT-4V defect: {result['vlm_result']['defect_type']}")
    print(f"  GPT-4V confidence: {result['vlm_result']['confidence']:.0%}")

# Check if expert labeling needed
if result["requires_expert_labeling"]:
    print(f"🚨 FLAG FOR EXPERT LABELING")
    print(f"   Routing case: {result['routing_case']}")
    print(f"   Reason: {result['vlm_result'].get('reasoning', 'uncertain')}")
```

**Response Format (Cascade):**
```python
{
    "model_used": "bgad" | "ensemble",
    "anomaly_score": float,
    "is_anomaly": bool,
    "confidence": float (0.0-1.0),
    "threshold": float,
    "requires_expert_labeling": bool,
    "routing_case": "A_confident_normal" | "B_confident_anomaly" | "C_uncertain_gpt4v_routed" | "C_uncertain_gpt4v_failed",
    "vlm_result": {  # Only for Case C
        "model_used": "gpt-4-vision",
        "is_anomaly": bool,
        "defect_type": str | None,
        "confidence": float,
        "reasoning": str,
        "status": "success" | "uncertain",
        "latency_ms": float
    },
    "bgad_score": float (Case C only),
    "gpt4v_score": float (Case C only),
    "timestamp": "2024-03-11T..."
}
```

### Method 3: API Integration

Use in FastAPI endpoints for HTTP access:

```python
# src/backend/app.py

from fastapi import FastAPI, File, UploadFile
from src.backend.services.inference import InferenceService

app = FastAPI()
inference = InferenceService()

@app.post("/inference/gpt4v-zero-shot")
async def gpt4v_analysis(file: UploadFile = File(...)):
    """
    Zero-shot anomaly detection using GPT-4V.
    
    Use for images where BGAD is uncertain.
    """
    # Save uploaded file temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        result = inference.call_gpt4v_zero_shot(tmp_path)
        return result
    finally:
        import os
        os.remove(tmp_path)

@app.post("/inference/cascade-with-gpt4v")
async def cascade_with_gpt4v(file: UploadFile = File(...)):
    """
    Complete cascade prediction with GPT-4V fallback for uncertain cases.
    
    Returns routing decision (A/B/C) and expert labeling flag.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        result = inference.cascade_predict_with_gpt4v(tmp_path)
        return result
    finally:
        import os
        os.remove(tmp_path)
```

---

## System Prompt

The specialized system prompt instructs GPT-4V to act as a Decospan wood QC expert:

```
You are an industrial quality control expert specializing in Decospan wood products. 
Analyze the provided image for manufacturing defects and anomalies. Look for: 
1. Knots (tree growth defects) - small to large circular/oval marks 
2. Deep scratches or surface damage - linear marks or gouges 
3. Discolorations - staining, bleaching, or uneven coloring 
4. Warping or curvature - deviation from flat surface 
5. Cracks or splits - breaks in the wood structure 
6. Foreign materials - dust, debris, or contamination on surface 
7. Finish defects - bubbles, drips, or uneven coating 

Provide your analysis as a VALID JSON object with EXACTLY these keys 
(use double quotes, no trailing commas): 
{"is_anomaly": boolean, "defect_type": "string_or_null", "confidence": float, "reasoning": "string"}

Return ONLY the JSON object, nothing else. No markdown, no explanation, just valid JSON.
```

### Customization

To modify for different products, change the defect list in the system prompt:

```python
# In call_gpt4v_zero_shot(), replace the system_prompt string:

system_prompt = (
    "You are an industrial QC expert for [PRODUCT]. "
    "Analyze for these defects: "
    "1. [Defect Type 1] - [description] "
    "2. [Defect Type 2] - [description] "
    # ... etc
)
```

---

## Error Handling

The implementation includes comprehensive error handling:

| Error | Behavior | Fallback |
|-------|----------|----------|
| **API Key Missing** | Returns `uncertain` status | User prompted to set OPENAI_API_KEY |
| **Network Timeout** | Retries 2x with exponential backoff | Returns `uncertain` after timeout |
| **Rate Limiting (429)** | Exponential backoff retry | Returns `uncertain` if persists |
| **Malformed JSON** | Attempts recovery from raw response | Heuristic parse (looks for "anomal", "defect") |
| **File Not Found** | Returns `uncertain` with error message | Logged for debugging |
| **API Error (401, 500, etc.)** | Categorized and logged | Returns `uncertain` status |

**Example Error Response:**
```python
{
    "status": "uncertain",
    "api_error": "ConnectTimeout: Connection timed out after 30s",
    "is_anomaly": None,
    "confidence": 0.0,
    "reasoning": "GPT-4V analysis failed due to network timeout",
    "timestamp": "2024-03-11T14:32:45.123456"
}
```

---

## Performance Characteristics

### Latency (from cascade_predict_with_gpt4v)

```
BGAD (Edge):    0-50ms
├─ Case A/B:    Exit immediately
└─ Case C:      Route to GPT-4V
    └─ GPT-4V:  800-2000ms
       ├─ Image encoding:  50-200ms
       ├─ API call:        500-1500ms
       └─ JSON parsing:    10-50ms
```

### Token Usage

| Image | Tokens | Approx Cost |
|-------|--------|-----|
| 224x224 | ~100 tokens | $0.0005 |
| 512x512 | ~200 tokens | $0.001 |
| 1024x1024 | ~400 tokens | $0.002 |

**Optimization:** If cost is high, adjust:
- Route fewer cases to GPT-4V (increase normal_threshold, decrease anomaly_threshold)
- Use lower resolution images for VLM (224x224 vs 1024x1024)
- Compress images before encoding

---

## Production Checklist

- [ ] OpenAI API key configured in environment
- [ ] API key has `gpt-4-vision` model access (check platform.openai.com)
- [ ] Tested manually with `call_gpt4v_zero_shot("path/to/test.jpg")`
- [ ] Integrated into cascade router via `cascade_predict_with_gpt4v`
- [ ] Logging configured to track GPT-4V latency and errors
- [ ] API error handling tested (simulate timeout, rate limit, auth error)
- [ ] Cost monitoring set up (OpenAI dashboard)
- [ ] Fallback behavior verified (VLM failure → flag for labeling)

---

## Monitoring & Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

inference = InferenceService()
result = inference.call_gpt4v_zero_shot("test.jpg")
# Now you'll see detailed logs with:
# - Image encoding details
# - API request/response
# - JSON parsing
# - Error categorization
```

### Check GPT-4V Usage

```bash
# View API usage in OpenAI dashboard
https://platform.openai.com/account/billing/overview

# The inference logs include latency metrics
grep "latency_ms" logs/*.log
```

### Test without Real Calls

```python
# Mock the API for testing:
from unittest.mock import patch, MagicMock

with patch("src.backend.services.inference.OpenAI") as mock_openai:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = '{"is_anomaly": true, "defect_type": "knot", "confidence": 0.85, "reasoning": "Visible knot"}'
    mock_client.chat.completions.create.return_value = mock_response
    
    inference = InferenceService()
    inference.openai_client = mock_client
    
    result = inference.call_gpt4v_zero_shot("test.jpg")
    assert result["is_anomaly"] == True
    assert result["defect_type"] == "knot"
```

---

## vs. Local VLM (AdaCLIP/WinCLIP)

| Feature | GPT-4V (Cloud) | Local VLM |
|---------|----------------|-----------|
| **Setup** | Requires API key | Pre-downloaded |
| **Latency** | 800-2000ms | 100-500ms |
| **Cost** | $0.001 per image | $0 (after model) |
| **Accuracy** | Very High (SOTA) | Good |
| **Internet** | Required | Not required |
| **Privacy** | Image sent to OpenAI | Local processing |
| **Can customize prompt** | Yes | Limited |

**Decision Tree:**
- Use **GPT-4V** if: Cost < latency, need SOTA accuracy, can handle external API
- Use **Local VLM** if: Latency critical, cost-conscious, privacy required, no internet

**Hybrid approach:** Use local VLM by default, fall back to GPT-4V only for cases local VLM is uncertain.

---

## Troubleshooting

### Issue: "OpenAI client not initialized"
```
Solution: Check OPENAI_API_KEY environment variable
$ echo $OPENAI_API_KEY
If empty:
  export OPENAI_API_KEY="sk-proj-..."
Then test:
  python -c "from openai import OpenAI; OpenAI(api_key=...)"
```

### Issue: "JSON parsing failed"
```
Solution: Check raw response in logs
  Result shows: api_error: "JSON parse error: ..."
  Look at: logs/response_text[:300]
GPT-4V sometimes wraps JSON in markdown code blocks or extra text
The method attempts recovery automatically
```

### Issue: "Image not found"
```
Solution: Verify image path
  Path must be absolute or relative to current working directory
  Check: ls -la path/to/image.jpg
```

### Issue: "Rate limit (429)"
```
Solution: Backoff strategy
  The method retries 2x with exponential backoff
  For production: implement request queuing or cache results
  Check: https://platform.openai.com/account/rate-limits
```

### Issue: "High API costs"
```
Solution: Optimize routing
  1. Increase normal_threshold (fewer images to GPT-4V)
  2. Decrease anomaly_threshold (fewer images to GPT-4V)
  3. Use lower resolution images
  4. Cache results for identical images
  Example:
    result = cascade_predict_with_gpt4v(
        image,
        normal_threshold=0.3,   # More aggressive
        anomaly_threshold=0.7   # More aggressive
    )
```

---

## Next Steps

1. **Test with real images**: Run cascade_predict_with_gpt4v on Decospan dataset
2. **Monitor metrics**: Track latency, accuracy, cost over time
3. **Fine-tune thresholds**: Adjust normal/anomaly thresholds based on production data
4. **Integrate with annotation queue**: Flag anomalies detected by GPT-4V for human verification
5. **Cost optimization**: If expenses are high, implement result caching or local VLM hybrid

---

## Examples

### Full End-to-End Example

```python
import logging
from pathlib import Path
from src.backend.services.inference import InferenceService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize service
inference = InferenceService()

# Test images
test_images = [
    "data/decospan_small/train/good/image_001.jpg",  # Should be normal
    "data/decospan_small/test/defective/image_042.jpg",  # Should be anomaly
]

for image_path in test_images:
    if not Path(image_path).exists():
        continue
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path}")
    print(f"{'='*60}")
    
    # Method 1: Direct GPT-4V
    print("\n[METHOD 1] Direct GPT-4V Zero-Shot")
    result_direct = inference.call_gpt4v_zero_shot(image_path)
    
    if result_direct["status"] == "success":
        print(f"  Anomaly: {result_direct['is_anomaly']}")
        print(f"  Defect: {result_direct['defect_type']}")
        print(f"  Confidence: {result_direct['confidence']:.0%}")
        print(f"  Latency: {result_direct['latency_ms']:.0f}ms")
    else:
        print(f"  Status: {result_direct['status']}")
        print(f"  Error: {result_direct['api_error']}")
    
    # Method 2: Cascade with GPT-4V fallback
    print("\n[METHOD 2] Cascade with GPT-4V Fallback")
    result_cascade = inference.cascade_predict_with_gpt4v(image_path)
    
    print(f"  Routing: {result_cascade['routing_case']}")
    print(f"  BGAD Score: {result_cascade.get('bgad_score', result_cascade['anomaly_score']):.4f}")
    
    if "gpt4v_score" in result_cascade:
        print(f"  GPT-4V Score: {result_cascade['gpt4v_score']:.4f}")
        print(f"  Ensemble Score: {result_cascade['anomaly_score']:.4f}")
    
    if result_cascade["requires_expert_labeling"]:
        print(f"  🚨 FLAGGED FOR EXPERT LABELING")
        if result_cascade["vlm_result"]:
            print(f"     Reason: {result_cascade['vlm_result'].get('reasoning', 'uncertain')}")

# Statistics
print(f"\n{'='*60}")
print("Cascade Statistics")
print(f"{'='*60}")
stats = inference.get_cascade_statistics()
for key, value in stats.items():
    print(f"  {key}: {value}")
```

Run with:
```bash
export OPENAI_API_KEY="sk-proj-..."
python examples/gpt4v_example.py
```

---

## API Reference

### call_gpt4v_zero_shot()

```python
def call_gpt4v_zero_shot(self, image_path: str) -> Dict:
    """
    Call GPT-4V for zero-shot industrial anomaly detection.
    
    Args:
        image_path (str): Path to local image file
    
    Returns:
        Dict with keys:
        - model_used: "gpt-4-vision"
        - is_anomaly: bool | None
        - defect_type: str | None
        - confidence: float (0.0-1.0)
        - reasoning: str
        - status: "success" | "uncertain"
        - api_error: str (if status="uncertain")
        - timestamp: str (ISO format)
        - latency_ms: float
    """
```

### cascade_predict_with_gpt4v()

```python
def cascade_predict_with_gpt4v(
    self,
    image: Union[str, Path, Image.Image, torch.Tensor],
    normal_threshold: float = 0.2,
    anomaly_threshold: float = 0.8,
    save_image_for_vlm: bool = True
) -> Dict:
    """
    Enhanced cascade prediction using GPT-4V instead of local VLM.
    
    Args:
        image: Input image (path, PIL, numpy, or tensor)
        normal_threshold: BGAD score below = confident normal
        anomaly_threshold: BGAD score above = confident anomaly
        save_image_for_vlm: Whether to save tensor to temp file for GPT-4V
    
    Returns:
        Cascade routing decision with routing_case and requires_expert_labeling
    """
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-03-11 | Initial release - GPT-4V integration with cascade routing |
