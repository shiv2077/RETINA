# GPT-4V Architecture Visualization

Complete flow diagrams showing how GPT-4V integrates with RETINA's cascade router.

---

## 1. Direct Zero-Shot Method Flow

```
┌──────────────────────────────────────────────────────────────────┐
│  call_gpt4v_zero_shot(image_path: str)                          │
└──────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
    [50-200ms]         [10-50ms]            [500-1500ms]
    Image Load         Base64 Encode       API Call
        │                    │                    │
        ↓                    ↓                    ↓
    Load JPG/PNG      Encode to Base64      OpenAI GPT-4V
    from disk         with media type        gpt-4-vision
                      (image/jpeg, etc)       or gpt-4o
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                        [10-50ms]
                       JSON Parse
                             │
                             ↓
        ─────────────────────────────────────────
        │ Response Dictionary               │
        │                                    │
        │  {                                 │
        │    "status": "success",            │
        │    "is_anomaly": True,             │
        │    "defect_type": "knot",          │
        │    "confidence": 0.85,             │
        │    "reasoning": "...",             │
        │    "latency_ms": 1234              │
        │  }                                 │
        │                                    │
        ─────────────────────────────────────────
                             │
                ┌────────────┼────────────┐
                │            │            │
              Error      Success       Error Recovery
              Handle     Return       (Malformed JSON)
                │            │            │
                ↓            ↓            ↓
           Return        Return      Return with
         uncertain     actual result  lower confidence
```

---

## 2. Cascade Routing with GPT-4V Fallback

```
┌─────────────────────────────────────────────────────────┐
│  cascade_predict_with_gpt4v(image, ...)                │
└─────────────────────────────────────────────────────────┘
                        │
                [20-30ms]
                        │
                        ↓
         ┌──────────────────────────┐
         │  BGAD Inference          │
         │  (Fast, Edge Device)     │
         │  Extract features        │
         │  Distance to center      │
         └──────────────────────────┘
                        │
            ┌───────────┼───────────┐
            │           │           │
           <0.2        0.2-0.8      >0.8
            │           │           │
            ↓           ↓           ↓
        ╔═════════╗  ╔═════════╗  ╔═════════╗
        ║ CASE A  ║  ║ CASE C  ║  ║ CASE B  ║
        ║ NORMAL  ║  ║UNCERTAIN║  ║ ANOMALY ║
        ╚═════════╝  ╚═════════╝  ╚═════════╝
            │           │           │
            │           │           │
       RETURN      ROUTE TO      RETURN
       BGAD        GPT-4V         BGAD
       <10ms       ~1500ms        <10ms
            │           │           │
            │      [800-2000ms]     │
            │           │           │
            │      ┌────────────┐   │
            │      │ GPT-4V     │   │
            │      │ Analysis   │   │
            │      │ - Load img │   │
            │      │ - B64 enc  │   │
            │      │ - API call │   │
            │      │ - Parse    │   │
            │      └────────────┘   │
            │           │           │
            │      ┌────┴────┐      │
            │      │         │      │
            │   Success  Failed     │
            │      │         │      │
            │      ↓         ↓      │
            │    Analyze  BGAD      │
            │    result   +Flag     │
            │      │         │      │
            │  ┌───┴──┐      │      │
            │  │      │      │      │
            │ True  False    │      │
            │  │      │      │      │
            │  ↓      ↓      ↓      ↓
            └──────────────────────────┐
                                       │
    ╔═══════════════════════════════════════════════╗
    ║  Cascade Decision Result                      ║
    ║                                                ║
    ║  {                                             ║
    ║    "routing_case": str,                        ║
    ║    "requires_expert_labeling": bool,           ║
    ║    "vlm_result": dict (if Case C),             ║
    ║    "anomaly_score": float,                     ║
    ║    "bgad_score": float,                        ║
    ║    "gpt4v_score": float (if Case C)            ║
    ║  }                                             ║
    ║                                                ║
    ╚═══════════════════════════════════════════════╝
                       │
           ┌───────────┼───────────┐
           │           │           │
       CASE A      CASE C      CASE B
       NORMAL      UNCERTAIN   ANOMALY
           │           │           │
           ↓           ↓           ↓
        Accept    ┌─────┴─────┬──────────┐
                  │           │          │
              GPT-4V = Anomaly │      GPT-4V = Normal
              OR Failed        │      
              │                │          │
              ↓                ↓          ↓
          FLAG FOR          NO FLAG    REJECT
          EXPERT            NEEDED
          LABELING
```

---

## 3. Error Handling Flowchart

```
                    API Call
                       │
            ┌──────────┼──────────┐
            │          │          │
         Success    Timeout     Other Error
            │          │          │
            ↓          │          │
         Parse      Retry 2x    Check Type
         JSON       w/ Backoff
            │          │          │
        ┌───┴───┐      │      ┌───┴──────┐
        │       │      │      │          │
    Parsed  Failed    │   Auth  Rate    Network
    JSON    Parse     │   (401) Limit   Error
        │       │     │   (429)
        ↓       ↓     │      │
      Use    Recover  │      │
     Result  Result   │      │
             (Heur.)  │      │
        │       │     │      │
        └───┬───┘     │      │
            │         │      │
            ↓         ↓      ↓
        ╔════════════════════════════╗
        ║ Return "uncertain" with    ║
        ║ api_error for safety       ║
        ║                            ║
        ║ Safe Fallback:             ║
        ║ - is_anomaly: None         ║
        ║ - confidence: 0.0          ║
        ║ - status: "uncertain"      ║
        ║ - api_error: <error msg>   ║
        ╚════════════════════════════╝
```

---

## 4. System Prompt Design

```
┌─────────────────────────────────────────────────────────┐
│                   System Prompt                          │
│                                                          │
│  "You are an industrial QC expert for Decospan wood"    │
│                                                          │
│  Analyze image for these defect types:                  │
│  ├─ Knots (circular marks, tree growth)                 │
│  ├─ Scratches (linear marks, surface damage)            │
│  ├─ Discolorations (staining, bleaching)                │
│  ├─ Warping (curvature, deviation from flat)            │
│  ├─ Cracks (splits, breaks in wood)                     │
│  ├─ Foreign materials (dust, debris, contamination)     │
│  └─ Finish defects (bubbles, drips)                     │
│                                                          │
│  Return ONLY valid JSON with these keys:                │
│  {                                                      │
│    "is_anomaly": boolean,                               │
│    "defect_type": "string_or_null",                     │
│    "confidence": float (0.0-1.0),                       │
│    "reasoning": "string"                                │
│  }                                                      │
│                                                          │
│  No markdown, no explanation, just JSON.                │
└─────────────────────────────────────────────────────────┘
                       │
                       │ Specialized for:
                       │
          ┌────────────┼────────────┐
          │            │            │
       Domain      Defect Types   Output Format
      Expertise                      
          │            │            │
     Industrial QC   Wood Defects  JSON Only
     Expert          (Decospan)    (Strict)
```

---

## 5. Cascade Statistics Dashboard

```
┌──────────────────────────────────────────────────────────────┐
│            Cascade Routing Statistics                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Total Inferences: 2000                                     │
│  ├─────────────────────┬──────────────────────┐             │
│  │                     │                      │             │
│  Case A (Normal)    Case B (Anomaly)   Case C (Uncertain)  │
│  1400 / 70%         400 / 20%          200 / 10%           │
│  │                  │                   │                  │
│  └─→ Confidence:    ├─→ Confidence:     ├─→ VLM Analyzed: ├─→ 180 (90%)
│      >90%               >90%             │   ├─→ True:  90
│  ├─→ Latency:       ├─→ Latency:        │   └─→ False: 90
│      <5ms               <5ms            └─→ VLM Failed: 20
│  └─→ Cost:          └─→ Cost:               ├─→ Fallback: BGAD+Flag
│      $0                 $0                  └─→ Cost: $0.20
│
│  Edge Model Utilization: 90% (Cases A+B)
│  ├─→ Fast inference on device
│  ├─→ No cloud dependency
│  └─→ Low latency (<10ms)
│
│  VLM (GPT-4V) Utilization: 10% (Case C)
│  ├─→ Only for uncertain cases
│  ├─→ Higher accuracy (SOTA)
│  └─→ Cost: $0.001 per image
│
│  VLM Catch Rate: 90/200 = 45%
│  ├─→ Of uncertain cases, GPT-4V detects anomaly in 45%
│  ├─→ Safety feature: Always flag for expert if VLM anomaly
│  └─→ Reduces false normal predictions
│
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Integration with Annotation Queue

```
┌────────────────────────────────────────────────┐
│  Cascade Prediction Result                     │
└────────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
       A            B            C
    NORMAL       ANOMALY      UNCERTAIN
        │            │            │
        ↓            ↓            ↓
     ACCEPT      REJECT        ANALYZE
     (No Q)      (No Q)        by GPT-4V
        │            │            │
        │            │      ┌─────┴──────┐
        │            │      │            │
        │            │    True         False
        │            │  (Anomaly)      (Normal)
        │            │      │            │
        │            │      ↓            ↓
        │            │   FLAG FOR     ACCEPT
        │            │   EXPERT       (No Q)
        │            │   LABELING
        │            │      │
        └─── All paths flow to ───┘
                      │
                      ↓
     ╔════════════════════════════════╗
     ║  requires_expert_labeling:     ║
     ║  - True → Add to queue         ║
     ║  - False → Don't queue         ║
     ║                                ║
     ║  Queue Entry contains:         ║
     ║  - image_path                  ║
     ║  - bgad_score                  ║
     ║  - vlm_result (if Case C)      ║
     ║  - routing_case                ║
     ║  - defect_type (if anomaly)    ║
     ║  - confidence                  ║
     ║  - reasoning                   ║
     ╚════════════════════════════════╝
                      │
                      ↓
          ┌─────────────────────┐
          │  Human Expert       │
          │  Review & Annotate  │
          │  + Bounding Boxes   │
          │  + Defect Type      │
          └─────────────────────┘
                      │
                      ↓
             ┌────────────────┐
             │  AnnotationStore
             │  (Persistent   │
             │   JSON)        │
             └────────────────┘
                      │
                      ↓
             ┌────────────────┐
             │ Nightly Retrain │
             │ (Fine-tune BGAD)│
             └────────────────┘
```

---

## 7. Performance Timeline

```
Inference Timeline (Cascade Case C - Uncertain)

0ms ──────┐
          │
    Preprocessing
    ├── Load image
    ├── Normalize
    └── Convert to tensor
          │
  20ms ───┤
          │
 BGAD Feature Extraction
    ├── ResNet backbone
    ├── Layer2 + Layer3
    └── Projection to 256-dim
          │
  40ms ───┤
          │
  BGAD Distance Computation
    ├── Distance to center
    ├── Argmax softmax
    └── Score computation
          │
  50ms ───┤
          │
 Routing Decision
    ├── Check if < 0.2 → No
    ├── Check if > 0.8 → No
    └── Route to GPT-4V
          │
     Case C Detected → Call GPT-4V
          │
 100ms ───┤
          │
       GPT-4V Processing...
          │
   Image Encoding (50-200ms)
    ├── Read from disk/memory
    ├── Base64 encode
    └── Prepare payload
          │
 200ms ───┤
          │
   Network Transmission (500-1500ms)
    ├── Send to OpenAI servers
    ├── Wait for processing
    └── Receive response
          │
1700ms ───┤
          │
   JSON Parsing (10-50ms)
    ├── Extract response
    ├── Parse JSON
    └── Validate fields
          │
1750ms ───┤
          │
  Ensemble Scoring (1ms)
    ├── Combine BGAD + GPT-4V
    ├── Average scores
    └── Make decision
          │
1751ms ───┤
          │
 Return Result
    ├── routing_case
    ├── requires_expert_labeling
    └── vlm_result
          │
 1751ms ──┘

Total: Case C (Uncertain) ≈ 1700-1800ms
       Case A/B (Confident) < 10ms
       Blended Average ≈ 170ms (80% fast, 10% slow)
```

---

## 8. Cost Model

```
┌────────────────────────────────────────────────────┐
│           API Call Volume → Cost                   │
├────────────────────────────────────────────────────┤
│                                                    │
│  Assumptions:                                      │
│  - GPT-4V: $0.01 per 1K tokens                    │
│  - Image: ~100 tokens                             │
│  - Cost per image: $0.001                         │
│                                                    │
│  Scenarios:                                        │
│                                                    │
│  1. Light usage (10% routed)                      │
│     100 images/day                                │
│     10 to GPT-4V × $0.001 = $0.01/day            │
│     Monthly: ~$0.30                               │
│                                                    │
│  2. Medium usage (10% routed)                     │
│     1000 images/day                               │
│     100 to GPT-4V × $0.001 = $0.10/day           │
│     Monthly: ~$3.00                               │
│                                                    │
│  3. Heavy usage (10% routed)                      │
│     10000 images/day                              │
│     1000 to GPT-4V × $0.001 = $1.00/day          │
│     Monthly: ~$30.00                              │
│                                                    │
│  4. Aggressive routing (50% routed)               │
│     1000 images/day                               │
│     500 to GPT-4V × $0.001 = $0.50/day           │
│     Monthly: ~$15.00                              │
│                                                    │
│  Cost Optimization:                               │
│  - Increase normal_threshold (fewer Case C)       │
│  - Decrease anomaly_threshold (fewer Case C)      │
│  - Use local VLM as pre-filter                    │
│  - Cache results if applicable                    │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 9. Decision Tree: Which Method to Use

```
                   Need to analyze image?
                            │
                       ┌────┴────┐
                       │         │
                   Is urgent?    No
                       │         │
       ┌───────────────┴─┐       │
       │                 │       │
      Yes                │       │
       │                 │       │
       ↓                 ↓       ↓
    <10ms           <1000ms    <2000ms
   latency?        tolerable?  tolerable?
    needed          │           │
       │       ┌────┴┐      ┌───┴───┐
       │       │     │      │       │
      Yes    Yes     No    Yes     No
       │     │        │     │       │
       ↓     ↓        ↓     ↓       ↓
    Use    Use      Use  Use     Use
   BGAD    Local    GPT4V Local   GPT4V
   only     VLM     only  VLM    + Local
           +VLM             only
           falls
           back to
           GPT4V

   Legend:
   - BGAD only: Fast edge inference
   - Local VLM: Zerofew-shot on device (AdaCLIP/WinCLIP)
   - GPT4V: Cloud-based SOTA
   - BGAD+VLM: Hybrid (local first, fallback to cloud)
```

---

## 10. Comparison Matrix: All Methods

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│         BGAD       Local VLM      GPT-4V    Ensemble       │
│       (Edge)      (AdaCLIP)      (Cloud)   (Hybrid)        │
│         │            │            │           │            │
├─────────┼────────────┼────────────┼───────────┤           │
│ Speed   │ <5ms       │ 100-500ms  │ 1-2s     │ <5ms*     │
│ Cost    │ $0         │ Included   │ $0.001   │ $0        │
│ Accuracy│ 85%        │ 90%        │ 98%      │ 95%       │
│ Privacy │ Local      │ Local      │ Cloud    │ Local*    │
│ Training│ Pretrained │ Pretrained │ SOTA     │ None      │
│ Setup   │ No setup   │ Download   │ API key  │ Both      │
│ Error   │ Manual     │ Quiet fail │ Timeout  │ Fallback  │
│         │ labels    │            │ Retry    │           │
│ Defects │ Limited    │ General    │ Flexible │ Combined  │
│ Return  │ Single     │ Score +    │ JSON +   │ Ensemble  │
│         │ score     │ Label      │ Reasoning│ score     │
│         │            │            │           │           │
├─────────┼────────────┼────────────┼───────────┤           │
│ Best    │ Ultra-low  │ Custom     │ Unknown  │ Balanced  │
│ For     │ latency,   │ defects,   │ defects, │ perf &    │
│         │ offline    │ edge       │ accuracy │ cost      │
│         │ 99.9%      │ required   │ critical │           │
│         │            │            │           │           │
└─────────┴────────────┴────────────┴───────────┘           │
                                                             │
  * Ensemble = BGAD on edge for Cases A/B, GPT4V for C     │
  Usage = ✓✓✓ Recommended approach = ✓✓ Good      = Little │
          ─────────────────────────────────────────────────  │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary

These diagrams show:
1. ✅ Direct zero-shot method (call_gpt4v_zero_shot)
2. ✅ Cascade routing method (cascade_predict_with_gpt4v)
3. ✅ Error handling strategy
4. ✅ System prompt specialization
5. ✅ Statistics dashboard structure
6. ✅ Integration with annotation queue
7. ✅ Performance timeline
8. ✅ Cost modeling
9. ✅ Decision tree for method selection
10. ✅ Comparison matrix with alternatives

**For more details, see:**
- `GPT4V_QUICK_REFERENCE.md` - Quick lookup
- `GPT4V_INTEGRATION.md` - Full documentation
- `examples/gpt4v_example.py` - Runnable code
