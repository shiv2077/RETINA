# GPT-4V Quick Reference Card

## Setup (2 minutes)

```bash
# 1. Install
pip install openai>=1.3.0

# 2. Set API Key
export OPENAI_API_KEY="sk-proj-..."

# 3. Verify
python -c "from src.backend.services.inference import InferenceService; \
  print('✅ Ready!' if InferenceService().openai_client else '❌ API key needed')"
```

## Method 1: Direct Zero-Shot

```python
from src.backend.services.inference import InferenceService

inference = InferenceService()
result = inference.call_gpt4v_zero_shot("image.jpg")

# Check result
if result["status"] == "success":
    print(f"Anomaly: {result['is_anomaly']}")
    print(f"Defect: {result['defect_type']}")
    print(f"Confidence: {result['confidence']:.0%}")
```

**Response:**
```json
{
  "status": "success",
  "is_anomaly": true,
  "defect_type": "knot",
  "confidence": 0.85,
  "reasoning": "Visible circular knot in wood surface",
  "latency_ms": 1234
}
```

## Method 2: Cascade Routing

```python
result = inference.cascade_predict_with_gpt4v(
    image="image.jpg",
    normal_threshold=0.2,
    anomaly_threshold=0.8
)

# Three possible outcomes:
if result["routing_case"] == "A_confident_normal":
    print("✅ Normal (BGAD < 0.2)")
elif result["routing_case"] == "B_confident_anomaly":
    print("❌ Anomaly (BGAD > 0.8)")
else:  # C_uncertain_gpt4v_routed
    print("🤔 Uncertain - GPT-4V analyzed")
    print(f"Expert labeling needed: {result['requires_expert_labeling']}")
```

## Test Script

```bash
# Run example
python examples/gpt4v_example.py image.jpg

# Try cascade
python examples/gpt4v_example.py image.jpg --cascade

# Run both
python examples/gpt4v_example.py image.jpg --both
```

## Routing Logic (Cascade)

```
BGAD Score
    ↓
< 0.2?  → CASE A: Normal       (BGAD only, <10ms)
> 0.8?  → CASE B: Anomaly      (BGAD only, <10ms)
Other?  → CASE C: Uncertain    (Route to GPT-4V, ~1500ms)
             ↓
          GPT-4V Analysis
             ↓
          is_anomaly=T → Flag for expert labeling
          is_anomaly=F → Accept as normal
          Failed?      → Flag for labeling (safe fallback)
```

## Error Handling

| Error | Behavior |
|-------|----------|
| API Key Missing | Logs warning, GPT-4V disabled |
| Network Timeout | Retries 2×, then fallback |
| Rate Limit (429) | Exponential backoff |
| Malformed JSON | Recovery attempt, lower confidence |
| API Error (401, 500, etc.) | Fallback to uncertain |

**All errors return:** `status: "uncertain"` for safety

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| BGAD (Case A/B) | <20ms | Edge device |
| GPT-4V (Case C) | 800-2000ms | Includes network |
| Blended (avg) | ~170ms | 70% fast, 10% slow |

## Cost

| Volume | Cost/Day |
|--------|----------|
| 100 images (10% to GPT-4V) | ~$0.10 |
| 1000 images (10% to GPT-4V) | ~$1.00 |
| 1000 images (100% to GPT-4V) | ~$10.00 |

**Optimize:** Increase thresholds to reduce GPT-4V calls

## Defect Types Recognized

Default system prompt checks for:
1. Knots (tree growth)
2. Scratches (surface damage)
3. Discolorations (staining, bleaching)
4. Warping (curvature)
5. Cracks (splits)
6. Foreign materials
7. Finish defects

**Customize:** Edit system prompt in `call_gpt4v_zero_shot()`

## Integration Example

```python
# In cascade router:
result = inference.cascade_predict_with_gpt4v(image)

if result["requires_expert_labeling"]:
    labeling_service.add_to_cascade_queue(
        image_id=image_id,
        image_path=image_path,
        bgad_score=result.get("bgad_score", result["anomaly_score"]),
        vlm_score=result.get("gpt4v_score", result["anomaly_score"]),
        routing_case=result["routing_case"]
    )
```

## Monitoring

```python
# Get statistics
stats = inference.get_cascade_statistics()

print(f"Case A: {stats['confident_normal']}")
print(f"Case B: {stats['confident_anomaly']}")
print(f"Case C: {stats['uncertain_routed_to_vlm']}")
print(f"Edge Util: {stats['edge_model_utilization']:.1f}%")
print(f"VLM Catch: {stats['vlm_catch_rate']:.1f}%")
```

## Troubleshooting

```bash
# Check API key
echo $OPENAI_API_KEY

# Check installation
python -c "import openai; print(openai.__version__)"

# Debug logs
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"
# Then run inference

# View API usage
# https://platform.openai.com/account/billing/overview
```

## Files & Documentation

| File | Purpose |
|------|---------|
| `src/backend/services/inference.py` | Implementation (500+ new lines) |
| `GPT4V_INTEGRATION.md` | Full documentation (800 lines) |
| `QUICKSTART_GPT4V.md` | Quick setup guide |
| `examples/gpt4v_example.py` | Runnable test script |
| `GPT4V_IMPLEMENTATION_SUMMARY.md` | This implementation summary |

## API Reference (Minimal)

### call_gpt4v_zero_shot(image_path: str) → Dict

```python
result = inference.call_gpt4v_zero_shot("image.jpg")
# Returns: {status, is_anomaly, defect_type, confidence, reasoning, latency_ms}
```

### cascade_predict_with_gpt4v(image, normal_threshold=0.2, anomaly_threshold=0.8) → Dict

```python
result = inference.cascade_predict_with_gpt4v(image)
# Returns: {routing_case, requires_expert_labeling, vlm_result, ...}
```

### get_cascade_statistics() → Dict

```python
stats = inference.get_cascade_statistics()
# Returns: {confident_normal, confident_anomaly, uncertain_routed_to_vlm, edge_model_utilization, vlm_catch_rate}
```

## Next Steps

1. ✅ Install openai: `pip install openai>=1.3.0`
2. ✅ Set API key: `export OPENAI_API_KEY="sk-..."`
3. ✅ Test: `python examples/gpt4v_example.py test.jpg`
4. ✅ Integrate into cascade router
5. ✅ Monitor logs and costs
6. ✅ Tune thresholds

## Support

- Full API docs: See `GPT4V_INTEGRATION.md`
- Example code: `examples/gpt4v_example.py`
- Troubleshooting: `QUICKSTART_GPT4V.md`
- Implementation details: `GPT4V_IMPLEMENTATION_SUMMARY.md`

---

**Status:** ✅ Production Ready
