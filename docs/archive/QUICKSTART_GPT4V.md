# GPT-4V Installation & Quick Start

## Installation

### 1. Install OpenAI Client

```bash
pip install openai>=1.3.0
```

Verify installation:
```bash
python -c "import openai; print(f'OpenAI {openai.__version__} installed')"
```

### 2. Set OpenAI API Key

Get your API key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

**Option A: Environment Variable (Recommended)**

```bash
export OPENAI_API_KEY="sk-proj-..."
```

Add to `.bashrc` or `.zshrc` for persistence:
```bash
echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.bashrc
source ~/.bashrc
```

**Option B: .env File**

Create `.env` in project root:
```
OPENAI_API_KEY=sk-proj-...
```

Load with:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Verify Setup

```python
from src.backend.services.inference import InferenceService

inference = InferenceService()
if inference.openai_client:
    print("✅ GPT-4V is ready!")
else:
    print("❌ Check OPENAI_API_KEY")
```

## Quick Start

### Test 1: Direct Zero-Shot Analysis

```python
from src.backend.services.inference import InferenceService

inference = InferenceService()

result = inference.call_gpt4v_zero_shot("path/to/image.jpg")

if result["status"] == "success":
    print(f"Anomaly: {result['is_anomaly']}")
    print(f"Defect: {result['defect_type']}")
    print(f"Confidence: {result['confidence']:.0%}")
else:
    print(f"Error: {result['api_error']}")
```

### Test 2: Cascade with Fallback

```python
result = inference.cascade_predict_with_gpt4v(
    image="path/to/image.jpg",
    normal_threshold=0.2,
    anomaly_threshold=0.8
)

print(f"Routing: {result['routing_case']}")
print(f"Requires expert labeling: {result['requires_expert_labeling']}")
```

### Test 3: Use Example Script

```bash
# Make executable
chmod +x examples/gpt4v_example.py

# Run with direct GPT-4V
python examples/gpt4v_example.py path/to/image.jpg

# Run with cascade
python examples/gpt4v_example.py path/to/image.jpg --cascade

# Run both
python examples/gpt4v_example.py path/to/image.jpg --both
```

## Requirements

```
openai>=1.3.0
```

Add to `requirements.txt`:
```bash
echo "openai>=1.3.0" >> requirements.txt
pip install -r requirements.txt
```

## API Costs

| Task | Cost per Image | Volume Discount |
|------|---|---|
| Zero-shot analysis | ~$0.001 | Free tier available |
| 100 images/day | ~$0.10 | Included in free tier |
| 1000 images/month | ~$30-40 | Bulk pricing available |

**Optimize costs:**
- Increase normal_threshold to skip more images
- Only route truly uncertain cases (0.2-0.8 range)
- Use lower resolution images

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "OpenAI client not initialized" | Set OPENAI_API_KEY environment variable |
| File not found error | Use absolute paths or check working directory |
| API timeout | Network issue or API overloaded. Retries automatically. |
| Rate limiting (429) | Wait a few minutes before retrying |
| Malformed JSON response | Method attempts recovery. Check logs. |

## Next: Integration

See [GPT4V_INTEGRATION.md](../GPT4V_INTEGRATION.md) for:
- Complete API reference
- Production deployment checklist
- Monitoring & debugging
- Performance tuning
- Cost optimization
