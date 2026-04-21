# VLM Router Integration Plan

## 1. Current worker orchestration — findings

### 1.1 worker.py — main loop
- **Concurrency:** synchronous, single-threaded. `while self.running:` at
  [worker.py:121](../worker/src/retina_worker/worker.py#L121). One job in-flight at a time.
- **Shutdown:** SIGTERM/SIGINT handled at [worker.py:92-93](../worker/src/retina_worker/worker.py#L92-L93), sets `self.running=False` so the current job finishes cleanly.
- **Job pull:** blocking `XREADGROUP` with 5 s timeout at [worker.py:138](../worker/src/retina_worker/worker.py#L138).
- **Per-job pipeline:** status→processing ([:155](../worker/src/retina_worker/worker.py#L155)), `_run_inference` ([:161](../worker/src/retina_worker/worker.py#L161)), store result ([:168](../worker/src/retina_worker/worker.py#L168)), AL pool add ([:171-176](../worker/src/retina_worker/worker.py#L171-L176)), increment counter ([:179](../worker/src/retina_worker/worker.py#L179)), XACK in `finally` ([:207](../worker/src/retina_worker/worker.py#L207)).

### 1.2 Where `model.predict()` is called
- Single call site: [worker.py:260-263](../worker/src/retina_worker/worker.py#L260-L263) with `image_id` and `image_data`.
- Result consumed at [worker.py:266-308](../worker/src/retina_worker/worker.py#L266-L308) to build `Stage1Output` / `Stage2Output` and `InferenceResult`.

### 1.3 How the model is chosen
- Via factory at [worker.py:235](../worker/src/retina_worker/worker.py#L235): `model = get_model(job.model_type)`.
- `job.model_type` comes straight from the backend's enqueued JSON — no per-worker logic picks it today.

### 1.4 Cold-start GPT-4V fallback
- [worker.py:230-242](../worker/src/retina_worker/worker.py#L230-L242). If requested model is `PATCHCORE` *and* the instance is `PatchCoreReal` *and* `has_memory_bank` is false → swap to `ModelType.GPT4V`. Hardcoded, PatchCore-specific.

### 1.5 How results land in Redis
- `self.redis.store_result(result)` at [worker.py:168](../worker/src/retina_worker/worker.py#L168) (InferenceResult JSON in `retina:results:{job_id}`).
- `self.redis.add_to_labeling_pool(...)` at [worker.py:172-176](../worker/src/retina_worker/worker.py#L172-L176) (ZADD to `retina:al:pool`, HSET sample meta).
- `self.redis.acknowledge_job(entry_id)` in `finally` at [worker.py:207](../worker/src/retina_worker/worker.py#L207) (XACK).

### 1.6 factory.py — registry + usage
- Registry at [factory.py:40-46](../worker/src/retina_worker/models/factory.py#L40-L46):

  | ModelType    | Class            | Status |
  |---|---|---|
  | PATCHCORE    | `PatchCoreReal`  | Real; uses ckpt at `settings.patchcore_checkpoint_path` |
  | PADIM        | `PatchCoreStub`  | Stub; not wired |
  | WINCLIP      | `WinCLIPStub`    | Stub; legacy |
  | GPT4V        | `GPT4VDetector`  | Real but monolithic; pre-dates VLMRouter |
  | PUSHPULL     | `PushPullStub`   | Stub; Stage-2 placeholder |
- Call sites of `get_model` in the worker: exactly two, both in [worker.py:235, 242](../worker/src/retina_worker/worker.py#L235). Nothing else uses the factory.
- **Open/closed check:** adding a new `ModelType` and a registry row is additive — existing callers unaffected. Constructor-kwarg branches at [factory.py:90-103](../worker/src/retina_worker/models/factory.py#L90-L103) need one more `elif` per new model that takes non-default args; otherwise fine.

### 1.7 patchcore_real.py — predict signature
- Class declared at [patchcore_real.py:69](../worker/src/retina_worker/models/patchcore_real.py#L69), inherits `AnomalyDetector`.
- Constructor takes `checkpoint_path`, `anomaly_threshold` ([:74-80](../worker/src/retina_worker/models/patchcore_real.py#L74-L80)).
- **predict contract** (from base class + worker call site): `predict(image_id: str, image_data: Optional[bytes]) -> AnomalyPrediction`. Returns the pydantic `AnomalyPrediction` used everywhere (fields: `anomaly_score`, `is_anomaly`, `confidence`, `uncertainty`, `heatmap`, `feature_distance`, `clip_similarity`, `defect_category`, `category_probabilities`, `embedding_distance`, `defect_description`, `defect_location`, `gpt4v_reasoning`).

### 1.8 schemas.py — key pydantic models
- `ModelType`: `{patchcore, padim, winclip, gpt4v, pushpull}` ([schemas.py:29-33](../worker/src/retina_worker/schemas.py#L29-L33)).
- `InferenceJob` ([schemas.py:58-75](../worker/src/retina_worker/schemas.py#L58-L75)): `job_id, image_id, model_type, stage, priority, status, submitted_at, metadata, image_path`.
- `InferenceResult` ([schemas.py:106-142](../worker/src/retina_worker/schemas.py#L106-L142)): core outputs + `stage1_output` + `stage2_output` + `active_learning` + VLM-specific `defect_description, defect_location, gpt4v_reasoning`.

### 1.9 shared/schemas/result.json — VLM field availability
- `defect_description`, `defect_location`, `gpt4v_reasoning` are **in the pydantic** model but **not in the JSON schema** ([result.json:141](../shared/schemas/result.json#L141) closes `properties` without them).
- No `product_class`, `product_confidence`, `natural_description`, or `severity` fields in either place.

## 2. Flow diagrams

### 2.1 Current flow

```
XREADGROUP ──► job ──► get_model(job.model_type)
                         │
                         ├── PATCHCORE + no memory_bank ──► get_model(GPT4V)
                         │
                         └── else ──► (whichever class)
                                       │
                                       ▼
                         model.predict(image_id, image_data)
                                       │
                                       ▼
                         AnomalyPrediction
                                       │
                                       ▼
                  InferenceResult ──► store_result ──► maybe add_to_labeling_pool ──► XACK
```
Shortcomings for the sprint:
- Model choice is decided by the *backend* via `job.model_type`. The worker has no way to pick a per-product checkpoint.
- GPT-4V is a full standalone detector; no separation between "explain" and "decide."
- Cold-start fallback is hardcoded and only fires when PatchCore has *no* memory bank at all — not per-category.

### 2.2 Target flow with VLMRouter

```
XREADGROUP ──► job ──► load image_bytes from job.image_path
                         │
                         ▼
             VLMRouter.identify_product(image_bytes)      [gpt-4o-mini, cached]
                         │
          ┌──────────────┴──────────────┐
          │ is_known_category=True      │ is_known_category=False
          ▼                             ▼
  load PatchCore[product_class]   VLMRouter.zero_shot_detect(...)
          │                             │
   predict(image_bytes)                 ▼
          │                      AnomalyPrediction (score + reasoning, zero-shot)
          ▼                             │
  score = AnomalyPrediction.anomaly_score
          │
  ┌───────┼─────────────┐
  │ <0.3  │ 0.3–0.7     │ >0.7
  ▼       ▼             ▼
 clean   ambiguous   VLMRouter.describe_defect(image_bytes, product_class, score)
                         │                                  [guard short-circuits <0.3]
                         ▼
                    DefectDescription
                         │
                         ▼
  InferenceResult  (adds product_class, product_confidence,
                    natural_description, severity, routing_reason)
                         │
                         ▼
             store_result ──► enqueue_for_review if ambiguous ──► XACK
```

## 3. Files that need to change

| File | Change |
|---|---|
| [shared/schemas/result.json](../shared/schemas/result.json) | **Extend schema.** Add `product_class`, `product_confidence`, `natural_description`, `defect_severity`, `routing_reason`, `vlm_model_used` under `properties`. Non-breaking (all optional). |
| [worker/src/retina_worker/schemas.py](../worker/src/retina_worker/schemas.py) | Mirror the new optional fields on `InferenceResult`. Keep `defect_description`/`defect_location`/`gpt4v_reasoning` for legacy callers. |
| [backend/src/db/models.rs](../backend/src/db/models.rs) + [backend/src/services/redis.rs](../backend/src/services/redis.rs) | Add deserialisation of the new optional fields. (DB columns exist for description/reasoning; add migrations for the new columns or store them in a JSON blob column — decision pending.) |
| [worker/src/retina_worker/config.py](../worker/src/retina_worker/config.py) | Add `patchcore_checkpoints_dir` (default `./checkpoints`) and three thresholds `low_score_cutoff=0.3`, `review_band_upper=0.7`, `describe_threshold=0.7`. |
| [worker/src/retina_worker/models/patchcore_real.py](../worker/src/retina_worker/models/patchcore_real.py) | Add a classmethod or constructor variant that loads **per-category** anomalib `.ckpt` files from `{checkpoints_dir}/patchcore_{category}.ckpt`. Keep the single-checkpoint path for backward compat. |
| [worker/src/retina_worker/models/factory.py](../worker/src/retina_worker/models/factory.py) | Add a `get_patchcore_for(category: str)` helper that caches one PatchCoreReal per category. Leave existing `get_model(ModelType)` untouched. |
| [worker/src/retina_worker/worker.py](../worker/src/retina_worker/worker.py) | Replace `_run_inference` body with the routing from §2.2. Cold-start fallback rewritten: if identify_product returns unknown-or-no-ckpt → zero_shot_detect; if known + ckpt absent → zero_shot_detect with a warning log. |
| [worker/src/retina_worker/models/vlm_router.py](../worker/src/retina_worker/models/vlm_router.py) | **No change** — contract already matches. |
| [scripts/test_integration.py](../scripts/test_integration.py) **(new)** | End-to-end: push a job to Redis, wait for `retina:results:{id}`, assert the new fields are populated. |
| [scripts/test_vlm_router.py](../scripts/test_vlm_router.py) | No change (isolation tests still valid). |

## 4. Risks

1. **Blocking API calls on the hot path.** `identify_product` runs per job; even at 0.2 s latency it halves worker throughput vs pure PatchCore. Mitigation: hash cache already in VLMRouter; consider a local fast classifier later if throughput becomes an issue.
2. **OpenAI outage / rate limit.** If `identify_product` fails, the worker currently has no "known-category default" to fall back on. Decision needed: fail the job, retry with exponential backoff, or default to a configured category.
3. **Cost regression.** Every job triggers a gpt-4o-mini call even if PatchCore would have scored it as clean. At 1000 img/day, ~$0.15/day — small, but worth surfacing.
4. **Schema drift (the audit's existing concern).** Adding fields to `schemas.py` without updating [shared/schemas/result.json](../shared/schemas/result.json) and the Rust struct will corrupt Redis payloads silently. Mitigation: follow the §1.2 update protocol in CLAUDE.md (JSON → Rust → Python, build both services).
5. **PatchCore per-category memory.** 15 checkpoints × ~230 MB = 3.4 GB if all loaded simultaneously. Mitigation: LRU-cache the `PatchCoreReal` instances in the factory helper, keep ≤3 hot.
6. **GPT-4V legacy path.** Cold-start fallback previously routed to `ModelType.GPT4V` (the old `GPT4VDetector` class). After integration, that path is orphaned but not removed. Decision: leave `GPT4VDetector` + `ModelType.GPT4V` intact for compatibility, stop using them in `_run_inference`. Clean-up is a later step.
7. **Heatmap contract.** `Stage1Output.heatmap_available=True` is currently set only when PatchCore returns a heatmap. Anomalib PatchCore checkpoints *do* produce heatmaps; wiring them through is a separate item and out of scope here.

## 5. Schema gaps

Fields the new flow emits that have no home today:

| Field | Type | Purpose |
|---|---|---|
| `product_class` | string | Routed category (e.g., `bottle`) or `unknown` |
| `product_confidence` | float 0-1 | Identification confidence from VLMRouter |
| `natural_description` | string | Operator-facing sentence from `describe_defect` |
| `defect_severity` | string (minor / moderate / severe) | From `describe_defect` |
| `routing_reason` | string | Which branch fired: `patchcore_known`, `zero_shot_unknown`, `review_ambiguous`, `guard_low_score` |
| `vlm_model_used` | string | Which VLM call, if any, produced the description |

Proposal: put them at the top level of `InferenceResult` (flat, queryable) rather than under `stage1_output`, because they can also fire on the zero-shot branch where there is no Stage 1 numeric score.

## 6. Change order

1. **[DONE 2026-04-21]** **Schema first** (result.json → Rust struct → Python pydantic, each verified with `cargo check` / `pytest` import). Do not touch worker logic yet.
   - 11 top-level fields added to `InferenceResult` in all three sources.
   - Drift fixed: `defect_description` / `defect_location` / `gpt4v_reasoning` are now top-level in Rust (previously nested in `Stage1Output`), matching Python and the JSON schema.
   - Python round-trip verified (26 total fields, no MISSING).
   - Rust `cargo check` not run — cargo toolchain is not installed on this host; manual audit found zero remaining `InferenceResult { ... }` construction sites besides `pending()` and zero reads of the removed `Stage1Output.defect_*` fields.
2. **Per-category PatchCore loader** in `patchcore_real.py` + factory helper. Add a tiny unit test that loads `checkpoints/patchcore_bottle.ckpt` and runs `predict` on one image. Proves the loader works in isolation.
3. **Worker rewrite** — replace `_run_inference` with the §2.2 routing. Keep the old function around as `_run_inference_legacy` until the new path is exercised.
4. **Integration test** — push a job, wait for result, assert new fields appear. Covers the schema + worker contract together.
5. **Clean up** — delete `_run_inference_legacy`, drop the `WINCLIP`/old-`GPT4V` code paths from the worker's import list (factory keeps them for external callers).
6. **Frontend + Rust API** — surface the new fields in `api.ts` and label/page.tsx. Out of scope for this plan; flagged for a later prompt.

## 7. Not in scope here

- Multi-class Stage 2 model (`PUSHPULL`) — still a stub; integration blocked on that model's own prompt.
- Heatmap key storage in Redis — orthogonal to router work.
- Removal of the legacy `gpt4v_detector.py` — parked until after the worker is migrated and verified.
- Rust backend route changes (`/api/predict/cascade` etc. from the phantom-routes list).
