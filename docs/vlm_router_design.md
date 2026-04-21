# VLM Router Design Pattern

## Intent

Use the VLM (GPT-4o / GPT-4o-mini) for **orchestration and explanation**, never
as the primary anomaly verdict. PatchCore — trained on normal samples with a
known coreset memory bank — owns the anomaly decision. The VLM runs only on
images PatchCore has already flagged, and its job is to turn a numeric score
into an operator-actionable sentence.

This asymmetry is not a limitation we are coping with; it is the design. A
specialist classifier plus a generalist explainer is strictly stronger than
either alone, as long as the seam between them is respected.

## Task decomposition

| Task | Owner | Reason |
|---|---|---|
| Anomaly score on a known product | PatchCore checkpoint | Trained on that product's normal distribution; calibrated threshold; reproducible numeric score. |
| Product identification (route-to-checkpoint) | GPT-4o-mini | General visual recognition; cheap (~$0.00015/call); cacheable by image hash. |
| Natural-language defect description | GPT-4o | Strong vision reasoning; turns `score=0.92` into `"severe crack on the rim"` for the operator UI. |
| Zero-shot anomaly on an *unknown* product | GPT-4o | Day-one capability when no PatchCore checkpoint exists. Best-effort; not production-grade. |
| Multi-class defect classification on a known product | Stage 2 supervised model (Push-Pull / BGAD) | Requires labeled training data; VLM cannot reliably discriminate between subtle defect classes. |

## Failure modes accepted by design

### 1. Priming bias on good images
When GPT-4o is asked *"this product was flagged — describe the defect"*, it
tends to find one even if the image is clean (T5 in the original test suite:
model invented "discoloration" on a pristine bottle rim).

**Mitigation — runtime guard in `describe_defect`:** if `anomaly_score < 0.3`,
skip the API call entirely and return `has_defect=False` with a note. This is
both a correctness guard (no hallucinated defects on clean images) and a cost
guard (~$0.005 saved per bypassed call).

The caller is still responsible for only *invoking* the function when
PatchCore has flagged the image. The guard is a belt-and-braces, not a
substitute for correct routing.

### 2. Reference-dependent defects
Some defect classes are defined by deviation from a known convention rather
than by a visible flaw. MVTec's `cable/cable_swap` is the canonical example:
the cable is physically intact, but the colored wires are in the wrong order.
A zero-shot VLM cannot catch this — it has no prior about the correct wiring
convention for *this specific part number*.

This is an intrinsic limit of zero-shot methods, not a prompt-engineering
problem. In production on such defect classes, either:
- collect enough labels to train a specialist, or
- provide a reference image in the prompt (few-shot), or
- accept that these defects are out of scope for the Stage 1 VLM fallback.

Test `T6_reference_dependent_defect_limitation` asserts the miss to lock the
behavior in and prevent someone "fixing" it with prompt hacks that regress
other cases.

### 3. False positives in zero-shot on good images
Same priming bias as #1 but in the `zero_shot_detect` path. We do *not* add a
score guard here because there is no PatchCore score to gate on — zero-shot
is the fallback *because* PatchCore is absent.

Accepted behavior: occasional false positives on clean images. Mitigation in
production is that flagged images go to the expert-review queue, not
straight to reject — so a false positive costs one operator glance, not a
rejected unit. Test `T8` is marked ADVISORY, not FAIL.

### 4. Cost per inference
- Product identification: ~$0.00015 with `gpt-4o-mini` (cached per image hash)
- Defect description: ~$0.005 with `gpt-4o`
- Zero-shot detection: ~$0.005 with `gpt-4o`

At <5% flag rate on a 1000-image/day line and full cache hits on
identification, total daily spend is ≈ $0.25 — acceptable.

## Worker integration contract

```
on job received:
    image_bytes = load(job.image_path)

    # Step 1 — route to the right specialist.
    product = router.identify_product(image_bytes)

    if product.is_known_category:
        # Step 2a — run the specialist. Numeric verdict comes from here.
        score = patchcore[product.product_class].predict(image_bytes)

        if score < 0.3:
            emit(is_anomaly=False, score=score, model="patchcore")
            return

        if score > PATCHCORE_THRESHOLD (≈ 0.7):
            # Step 3 — VLM explains WHY it is anomalous.
            desc = router.describe_defect(
                image_bytes,
                product_class=product.product_class,
                anomaly_score=score,
            )
            emit(is_anomaly=True, score=score, description=desc,
                 model="patchcore+gpt4o")
        else:
            # Ambiguous band — send to expert review.
            enqueue_for_review(image_bytes, score, product.product_class)
    else:
        # Step 2b — zero-shot fallback on unknown products.
        zs = router.zero_shot_detect(image_bytes,
                                     product_description=product.reasoning)
        emit(is_anomaly=zs.is_anomaly, score=zs.anomaly_score,
             reasoning=zs.reasoning, model="gpt4o-zeroshot")
        if zs.is_anomaly:
            enqueue_for_review(image_bytes, zs.anomaly_score, "unknown")
```

### Thresholds
- `< 0.3` — clean; no VLM call.
- `0.3 – 0.7` — ambiguous; no VLM call, route to expert review.
- `> 0.7` — flagged; call `describe_defect` for the operator UI.

These are initial defaults. The lower bound matches the runtime guard in
`describe_defect`. The upper bound should be re-tuned per deployment once a
labeled validation set exists.

### Invariants enforced by the router
- `identify_product` caches by SHA256 of image bytes — identical images incur
  no duplicate API spend within a process lifetime.
- `describe_defect` short-circuits on low scores — callers can be sloppy
  without blowing up the API budget or hallucinating defects.
- All three methods return Pydantic models — downstream code gets typed
  access, no dict-fishing.

## What this pattern is not

It is not a generic "LLM in the loop" pattern. It is specifically a
*specialist-first, explainer-second* layout. The VLM never makes the
classification call on a known product. Swapping this around (e.g. "let the
VLM decide, use PatchCore as a sanity check") would reintroduce all three
failure modes above as production bugs.
