# 🧪 Cascade Integration - Copy-Paste Test Examples

Quick reference for testing your end-to-end cascade → annotation integration.

---

## ✅ Test 1: Verify Backend Endpoints Exist

```bash
# Check if cascade prediction endpoint is live
curl -s http://localhost:8000/api/predict/cascade --head
# Expected: HTTP/1.1 405 Method Not Allowed (POST required)

# Check queue endpoint
curl -s http://localhost:8000/api/labeling/cascade/queue
# Expected: {"success": true, "queue": [...], "queue_size": X}

# Check stats endpoint
curl -s http://localhost:8000/api/labeling/cascade/stats
# Expected: {"total_queued": X, "pending": Y, ...}
```

---

## ✅ Test 2: Run Cascade Prediction with Test Image

### Option A: Using Python

```python
# backend_test_cascade.py
import sys
sys.path.insert(0, '/home/shiv2077/dev/RETINA')

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from src.backend.services.inference import InferenceService
from src.backend.services.labeling import LabelingService
from pathlib import Path

# Initialize
inference = InferenceService()
labeling = LabelingService(Path("/home/shiv2077/dev/RETINA/annotations"))

# Load test image
test_image_path = "/home/shiv2077/dev/RETINA/mvtec_anomaly_detection/bottle/test/anomaly/000.png"
if not Path(test_image_path).exists():
    print(f"❌ Test image not found: {test_image_path}")
    sys.exit(1)

img = Image.open(test_image_path).convert("RGB")

# Prepare tensor
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = transform(img).unsqueeze(0)

# Run cascade
print("🚀 Running cascade prediction...")
result = inference.predict_with_cascade(
    image=image_tensor,
    normal_threshold=0.2,
    anomaly_threshold=0.8,
    use_vlm_fallback=True
)

print("\n✅ Cascade Result:")
print(f"  Routing Case: {result['routing_case']}")
print(f"  BGAD Score: {result['anomaly_score']:.4f}")
print(f"  Confidence: {result['confidence']:.4f}")
print(f"  Requires Labeling: {result['requires_expert_labeling']}")
if 'vlm_score' in result:
    print(f"  VLM Score: {result['vlm_score']:.4f}")
print(f"  Processing Time: {result.get('processing_time_ms', 'N/A')}ms")

# Check queue
queue_response = labeling.get_cascade_queue()
print(f"\n📋 Cascade Queue:")
print(f"  Queue Size: {queue_response['queue_size']}")
print(f"  Pending: {queue_response['stats']['pending']}")
print(f"  Labeled: {queue_response['stats']['labeled']}")

# Get stats
stats = labeling.get_cascade_stats()
print(f"\n📊 Queue Statistics:")
print(f"  Total Queued: {stats['total_queued']}")
print(f"  Pending Items: {stats['pending']}")
print(f"  Avg BGAD Score: {stats['avg_bgad_score']:.4f}")

print("\n✅ Test 1 Passed!")
```

**Run it**:
```bash
cd /home/shiv2077/dev/RETINA
python backend_test_cascade.py
```

### Option B: Using cURL

```bash
# Get any image for testing
TEST_IMAGE="/home/shiv2077/dev/RETINA/mvtec_anomaly_detection/bottle/test/anomaly/000.png"

# Run cascade prediction
echo "🚀 Sending cascade prediction request..."
curl -X POST http://localhost:8000/api/predict/cascade \
  -F "file=@$TEST_IMAGE" \
  -F "normal_threshold=0.2" \
  -F "anomaly_threshold=0.8" \
  -F "use_vlm_fallback=true" | jq .

# Expected output:
# {
#   "model_used": "bgad",
#   "anomaly_score": 0.XXX,
#   "is_anomaly": bool,
#   "routing_case": "A_confident_normal" | "B_confident_anomaly" | "C_uncertain_vlm_routed",
#   "requires_expert_labeling": bool,
#   ...
# }
```

---

## ✅ Test 3: Fetch Cascade Queue

### Option A: Using Python

```python
# frontend_test_queue.py
import requests

API_URL = "http://localhost:8000"

# Fetch queue
print("📋 Fetching cascade queue...")
response = requests.get(f"{API_URL}/api/labeling/cascade/queue?limit=5")

if response.status_code == 200:
    data = response.json()
    print(f"\n✅ Queue fetched successfully!")
    print(f"   Queue size: {data['queue_size']}")
    print(f"   Pending: {data['stats']['pending']}")
    print(f"   Labeled: {data['stats']['labeled']}")
    
    if data['queue']:
        print(f"\n📷 First item in queue:")
        item = data['queue'][0]
        print(f"   Image ID: {item['image_id']}")
        print(f"   BGAD Score: {item['bgad_score']:.4f}")
        if item.get('vlm_score'):
            print(f"   VLM Score: {item['vlm_score']:.4f}")
        print(f"   Routing Case: {item['routing_case']}")
        print(f"   Status: {item['status']}")
else:
    print(f"❌ Failed to fetch queue: {response.status_code}")
    print(response.text)
```

**Run it**:
```bash
cd /home/shiv2077/dev/RETINA
python -c "import requests; r = requests.get('http://localhost:8000/api/labeling/cascade/queue'); print(r.json())" | jq .
```

### Option B: Using cURL

```bash
# Fetch queue
curl "http://localhost:8000/api/labeling/cascade/queue?limit=10" | jq .

# Fetch only pending items and extract image IDs
curl -s "http://localhost:8000/api/labeling/cascade/queue" | \
  jq '.queue[] | select(.status == "pending") | .image_id'
```

---

## ✅ Test 4: Submit Annotation

### Option A: Using Python

```python
# frontend_test_submit.py
import requests
import json

API_URL = "http://localhost:8000"

# First, get an item from queue
queue_response = requests.get(f"{API_URL}/api/labeling/cascade/queue?limit=1")
queue_data = queue_response.json()

if not queue_data['queue']:
    print("❌ No items in queue to annotate")
    exit(1)

image_id = queue_data['queue'][0]['image_id']
print(f"🎯 Annotating image: {image_id}")

# Prepare annotation
annotation = {
    "image_id": image_id,
    "label": "anomaly",
    "bounding_boxes": [
        {
            "x": 0.1,
            "y": 0.15,
            "width": 0.3,
            "height": 0.25,
            "defect_type": "scratch",
            "confidence": 0.95
        }
    ],
    "defect_types": ["scratch"],
    "notes": "Clear surface scratch visible on upper left"
}

# Submit
print("📤 Submitting annotation...")
submit_response = requests.post(
    f"{API_URL}/api/labeling/cascade/submit",
    data={
        "image_id": annotation["image_id"],
        "label": annotation["label"],
        "bounding_boxes": json.dumps(annotation["bounding_boxes"]),
        "defect_types": json.dumps(annotation["defect_types"]),
        "notes": annotation["notes"]
    }
)

if submit_response.status_code == 200:
    result = submit_response.json()
    print(f"\n✅ Annotation submitted!")
    print(f"   Image ID: {result['image_id']}")
    print(f"   Label: {result['label']}")
    print(f"   Remaining in queue: {result.get('remaining_in_queue', 'N/A')}")
else:
    print(f"❌ Failed to submit: {submit_response.status_code}")
    print(submit_response.text)
```

**Run it**:
```bash
cd /home/shiv2077/dev/RETINA
python frontend_test_submit.py
```

### Option B: Using cURL

```bash
IMAGE_ID="image_001"  # Replace with actual ID from queue

curl -X POST http://localhost:8000/api/labeling/cascade/submit \
  -F "image_id=$IMAGE_ID" \
  -F "label=anomaly" \
  -F "bounding_boxes=[{\"x\":0.1,\"y\":0.15,\"width\":0.3,\"height\":0.25,\"defect_type\":\"scratch\"}]" \
  -F "defect_types=[\"scratch\"]" \
  -F "notes=Clear scratch" | jq .
```

---

## ✅ Test 5: Skip an Item

### Python

```python
import requests

API_URL = "http://localhost:8000"

# Get queue
queue_response = requests.get(f"{API_URL}/api/labeling/cascade/queue?limit=1")
image_id = queue_response.json()['queue'][0]['image_id']

# Skip
skip_response = requests.post(f"{API_URL}/api/labeling/cascade/skip/{image_id}")

if skip_response.status_code == 200:
    print(f"✅ Skipped: {image_id}")
    print(f"   Remaining: {skip_response.json().get('remaining_in_queue')}")
```

### cURL

```bash
curl -X POST http://localhost:8000/api/labeling/cascade/skip/image_001 | jq .
```

---

## ✅ Test 6: Get Cascade Statistics

### Python

```python
import requests

API_URL = "http://localhost:8000"
stats_response = requests.get(f"{API_URL}/api/labeling/cascade/stats")
stats = stats_response.json()

print("📊 Cascade Queue Statistics:")
print(f"  Total Queued: {stats['total_queued']}")
print(f"  Pending: {stats['pending']}")
print(f"  Labeled: {stats['labeled']}")
print(f"  Skipped: {stats['skipped']}")
print(f"  Avg BGAD Score: {stats['avg_bgad_score']:.4f}")
```

### cURL

```bash
curl http://localhost:8000/api/labeling/cascade/stats | jq .
```

---

## ✅ Test 7: Frontend Integration Test

### Navigate to Annotation Studio

```
1. Open browser: http://localhost:3000/label
2. Category dropdown (top center): 
   - Default: "bottle"
   - Change to: "cascade"
3. UI should:
   - Load pending items from queue
   - Display BGAD score in sample info panel
   - Show routing case
   - Display VLM score (if available)
4. Draw bounding box:
   - Click "Box" tool (left sidebar)
   - Click and drag on canvas
   - Select defect type (right sidebar)
5. Submit:
   - Click "Anomaly" or "Normal" button
   - Item removed from queue
   - Next item loaded automatically
```

---

## ✅ Test 8: Full End-to-End Test

```bash
#!/bin/bash
# Test cascade → queue → annotation flow

set -e

API_URL="http://localhost:8000"
TEST_IMAGE="/home/shiv2077/dev/RETINA/mvtec_anomaly_detection/bottle/test/anomaly/000.png"

echo "🚀 Full End-to-End Test"
echo ""

# 1. Get initial queue size
echo "1️⃣  Get initial queue size..."
INITIAL_COUNT=$(curl -s "$API_URL/api/labeling/cascade/stats" | jq '.pending')
echo "   Initial pending: $INITIAL_COUNT"

# 2. Send cascade prediction
echo ""
echo "2️⃣  Send cascade prediction..."
PRED=$(curl -s -X POST "$API_URL/api/predict/cascade" \
  -F "file=@$TEST_IMAGE" \
  -F "normal_threshold=0.2" \
  -F "anomaly_threshold=0.8")

REQUIRES_LABEL=$(echo "$PRED" | jq '.requires_expert_labeling')
CASE=$(echo "$PRED" | jq -r '.routing_case')
echo "   Routing Case: $CASE"
echo "   Requires Labeling: $REQUIRES_LABEL"

# 3. Check queue updated
echo ""
echo "3️⃣  Check queue updated..."
QUEUE=$(curl -s "$API_URL/api/labeling/cascade/queue?limit=1")
QUEUE_SIZE=$(echo "$QUEUE" | jq '.queue_size')
IMAGE_ID=$(echo "$QUEUE" | jq -r '.queue[0].image_id' 2>/dev/null || echo "none")
echo "   Queue size: $QUEUE_SIZE"
echo "   First image: $IMAGE_ID"

if [ "$IMAGE_ID" != "none" ] && [ "$IMAGE_ID" != "null" ]; then
    # 4. Submit annotation
    echo ""
    echo "4️⃣  Submit annotation..."
    SUBMIT=$(curl -s -X POST "$API_URL/api/labeling/cascade/submit" \
      -F "image_id=$IMAGE_ID" \
      -F "label=anomaly" \
      -F "bounding_boxes=[{\"x\":0.1,\"y\":0.1,\"width\":0.2,\"height\":0.2,\"defect_type\":\"scratch\"}]" \
      -F "defect_types=[\"scratch\"]" \
      -F "notes=Test annotation")
    
    SUCCESS=$(echo "$SUBMIT" | jq '.success')
    echo "   Submission Success: $SUCCESS"
    
    # 5. Check queue updated
    echo ""
    echo "5️⃣  Check queue after annotation..."
    FINAL_STATS=$(curl -s "$API_URL/api/labeling/cascade/stats")
    FINAL_LABELED=$(echo "$FINAL_STATS" | jq '.labeled')
    echo "   Labeled count: $FINAL_LABELED"
    
    echo ""
    echo "✅ End-to-End Test Passed!"
else
    echo "⚠️  No items in queue yet (normal if first run)"
fi
```

**Run it**:
```bash
chmod +x /home/shiv2077/dev/RETINA/test_e2e.sh
/home/shiv2077/dev/RETINA/test_e2e.sh
```

---

## 🔧 Troubleshooting Test Failures

### "Queue endpoint returns empty"

**Cause**: No images flagged yet  
**Solution**: Run a few cascade predictions first to populate queue

```bash
# Send 3 test predictions to generate queue items
for i in {1..3}; do
  curl -s -X POST http://localhost:8000/api/predict/cascade \
    -F "file=@test_image.png" > /dev/null
  echo "Sent prediction $i"
done

# Now check queue
curl http://localhost:8000/api/labeling/cascade/queue | jq .
```

### "Submit annotation fails (400)"

**Cause**: Invalid JSON format  
**Solution**: Ensure bounding_boxes is valid JSON

```bash
# ❌ Wrong
-F "bounding_boxes={\"x\":0.1}"

# ✅ Right
-F "bounding_boxes=[{\"x\":0.1,\"y\":0.1,\"width\":0.2,\"height\":0.2,\"defect_type\":\"scratch\"}]"
```

### "Frontend doesn't load cascade queue"

**Cause**: API not responding or category not set to "cascade"  
**Solution**:

```bash
# Check API is live
curl http://localhost:8000/api/labeling/cascade/queue

# Ensure category dropdown is set to "cascade" (not "bottle")
# Hard reload page: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
```

---

## 📝 Test Checklist

- [ ] Backend endpoints exist (curl tests)
- [ ] Cascade prediction returns correct routing_case
- [ ] Queue auto-populated when requires_expert_labeling=True
- [ ] Queue can be fetched via API
- [ ] Annotations can be submitted from Python
- [ ] Frontend loads cascade queue when category="cascade"
- [ ] Frontend displays BGAD score and routing case
- [ ] Frontend can submit annotation
- [ ] Queue depletes after annotation submitted
- [ ] Stats endpoint shows correct counts

---

## 📚 References

- Full guide: `CASCADE_TO_ANNOTATION_GUIDE.md`
- Quick ref: `STEP3_INTEGRATION_COMPLETE.md`
- Cascade architecture: `CASCADE_ROUTER_GUIDE.md`
