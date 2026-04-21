# FastAPI Backend Revival — Plan

Purpose: get [legacy/fastapi_backend/](../legacy/fastapi_backend/) serving on port 3001 with Redis-only storage, compatible with the existing worker + frontend, in the 48-hour sprint window.

## Audit summary

### Routes defined ([app.py](../legacy/fastapi_backend/app.py))
| Method | Path | Purpose | Multipart? |
|---|---|---|---|
| GET | `/` | Banner with device | no |
| GET | `/health` | `{status, timestamp, gpu_available, gpu_name}` | no |
| GET | `/status` | Pipeline+labeling+models snapshot | no |
| GET | `/categories` | Enumerate mvtec categories on disk | no |
| GET | `/categories/{category}/stats` | Train/test image counts | no |
| POST | `/pipeline/stage1/train` | Trigger PatchCore training in-process | no |
| GET | `/pipeline/stage2/samples` | Return labeling batch with base64 anomaly maps | no |
| POST | `/pipeline/stage3/train` | Train BGAD in-process | no |
| GET | `/pipeline/evaluate` | Run evaluation on held-out set | no |
| POST | `/labels/session/start` | Start labeling session | no |
| POST | `/labels/submit` | Submit label (JSON body) | no |
| POST | `/labels/skip/{image_id}` | Skip sample | no |
| POST | `/api/predict/cascade` | Cascade routing prediction | **yes** (`file: UploadFile`) |
| GET | `/api/labeling/cascade/queue` | Items pending cascade review | no |
| POST | `/api/labeling/cascade/submit` | Submit cascade annotation | yes (Form fields) |
| POST | `/api/labeling/cascade/skip/{image_id}` | Skip cascade item | no |
| GET | `/api/labeling/cascade/stats` | Cascade queue stats | no |
| GET | `/labels/progress` | Progress counter | no |
| GET | `/labels/stats` | Label distribution | no |
| GET | `/labeling/queue` | Uncertainty-sorted samples | no |
| POST | `/labeling/submit` | Alt label submit (form + JSON string) | yes |
| GET | `/labeling/stats` | Detailed stats | no |
| GET | `/evaluations` | Stored evaluation results | no |
| GET | `/labels/export/{format}` | Export JSON / COCO / YOLO | no |
| POST | `/inference/image` | Inference on upload | **yes** |
| POST | `/inference/predict` | Inference on upload (frontend alias) | **yes** |
| GET | `/inference/history` | Recent inferences | no |
| GET | `/models` | List models on disk | no |
| DELETE | `/models/{model_type}/{category}` | Remove a weight file | no |

**Multipart upload: yes**, three routes (`/api/predict/cascade`, `/inference/image`, `/inference/predict`).

### Storage dependencies
- **Redis: none.** `grep -i redis` returns zero matches across the whole legacy tree.
- **Postgres/SQLAlchemy/asyncpg: none.**
- **Filesystem only.** `data/annotations/*.json`, `data/uploads/*.png`, `models/weights/*.pth`, `data/exports/*`.
- `LabelingService` writes JSON files ([services/labeling.py](../legacy/fastapi_backend/services/labeling.py)).
- `InferenceService` loads `{MODELS_DIR}/patchcore_{category}.pth` and `{MODELS_DIR}/bgad_model.pth` ([services/inference.py:99-120](../legacy/fastapi_backend/services/inference.py#L99-L120)).

### Hardcoded paths / configs that break in this repo
- [config.py:10](../legacy/fastapi_backend/config.py#L10) `BASE_DIR = Path(__file__).parent.parent.parent` → resolves to `/home/shiv2077/dev/RETINA/legacy`, **not the project root**. Consequence: `MODELS_DIR = legacy/models/weights/`, `DATA_DIR = legacy/data/`, `mvtec_path = legacy/mvtec_anomaly_detection` (dataset lives at `mvtec/` in the project root).
- [config.py:49](../legacy/fastapi_backend/config.py#L49) `ServerConfig.port = 8000`. Frontend + CLAUDE.md require **3001**.
- [app.py:677](../legacy/fastapi_backend/app.py#L677) `uvicorn.run("src.backend.app:app", ...)` — references the pre-Phase-3 path that no longer exists.
- [app.py:23-29](../legacy/fastapi_backend/app.py#L23-L29) and [services/__init__.py:1-11](../legacy/fastapi_backend/services/__init__.py#L1-L11) use a try/except dual-import idiom that only works when launched from specific CWDs; fragile.
- PatchCore load path format is `patchcore_{category}.pth`. Our anomalib checkpoints are `patchcore_{category}.ckpt` at `./checkpoints/` — **different loader, different file extension**.
- `InferenceService.run_patchcore` instantiates the legacy `PatchCoreModel` class (custom — from `src.models.patchcore`, which no longer exists after Phase 3). Grep confirms the import in [services/inference.py:32,37](../legacy/fastapi_backend/services/inference.py#L32) points at `..config` only; the `PatchCoreModel` class reference earlier in the file will fail to resolve.
- No JWT / auth middleware. CORS is wide-open (`*` + localhost). Acceptable for local dev; not for external.
- Result payloads (e.g. `CascadeResponse` at [app.py:60-71](../legacy/fastapi_backend/app.py#L60-L71)) do **not** match the 26-field `InferenceResult` schema we just finalised in step 1.

### Dependency check
Imports used that are **not** in the worker's [pyproject.toml](../worker/pyproject.toml):
- `fastapi`
- `uvicorn`
- `scikit-learn` (used by `services/pipeline.py` for `roc_auc_score`, `precision_recall_fscore_support`, `confusion_matrix`)
- `tqdm`
- `opencv-python` (for `cv2.applyColorMap` at [app.py:600-601](../legacy/fastapi_backend/app.py#L600-L601))

Already available in the `ml` conda env: `torch`, `torchvision`, `numpy`, `Pillow`, `pydantic`, `openai`, `structlog`.

Expected env vars: none from config (all defaults in dataclasses). `OPENAI_API_KEY` is used only if the cascade code path calls the VLM — pulled from `os.environ` at runtime via `openai` SDK default.

### Step-C live import test
```
$ cd legacy/fastapi_backend && python -c "from app import app"
ModuleNotFoundError: No module named 'fastapi'
```
Fails at line 15. Uvicorn test skipped — can't boot without deps.

## Big architectural note before any fixes

The legacy FastAPI backend was built as a **single-process monolith**: the HTTP handlers load PatchCore/BGAD weights into the request process and run inference synchronously. It has no concept of the Redis job queue or the `retina_worker` process.

The current sprint target ("FastAPI backend alongside the worker + frontend, labels in Redis") forces a choice:

**Option A — Run FastAPI monolithically, retire the worker for the demo.**
- FastAPI does inference directly, no Redis queue, no worker process.
- Simplest path; matches how the legacy code was designed.
- Labels still go to Redis if we rewrite `LabelingService` (small change; its storage backend is already an abstraction).
- Downside: the audit's two-stage flow through Redis goes unused for the demo.

**Option B — Keep the worker, make FastAPI a thin layer over Redis.**
- FastAPI enqueues jobs to `retina:jobs:queue` and polls `retina:results:{id}`.
- Preserves the architecture.
- Requires gutting the in-process inference from FastAPI and rewriting `InferenceService` to be a Redis client.
- Estimated 1-2 days of work — kills the sprint budget.

Recommended: **Option A**. Rationale: the sprint goal is a working demo, not an architecturally pure one. `InferenceService` + the trained anomalib checkpoints already sit in one Python process — adding Redis queuing between them adds latency and complexity for no demo benefit. Keep the worker for the AL pool + label writes only. All actual inference runs inside the FastAPI process.

The plan below assumes Option A. If you pick B, scratch fixes §6.1-6.4 and open a new plan.

## Fixes needed — numbered

### Installs (project env)
1. **Install backend deps.** `pip install fastapi uvicorn scikit-learn tqdm opencv-python` in the `ml` conda env. Reason: four of these are absent and the import crashes on line 15 of app.py without them.

### Config / paths
2. **[legacy/fastapi_backend/config.py:10](../legacy/fastapi_backend/config.py#L10)** — change `BASE_DIR = Path(__file__).parent.parent.parent` to `Path(__file__).resolve().parent.parent.parent` (one more level up) so `BASE_DIR` lands on the project root. Reason: current value resolves to `legacy/`, so MVTec path, models dir, annotations dir all miss.
3. **[legacy/fastapi_backend/config.py:61](../legacy/fastapi_backend/config.py#L61)** — change `mvtec_path` default from `mvtec_anomaly_detection` to `mvtec`. Reason: dataset is at `mvtec/` in this repo.
4. **[legacy/fastapi_backend/config.py:12](../legacy/fastapi_backend/config.py#L12)** — change `MODELS_DIR = BASE_DIR / "models" / "weights"` to `BASE_DIR / "checkpoints"`. Reason: the anomalib `.ckpt` files trained in the last session live at `checkpoints/patchcore_{category}.ckpt`.
5. **[legacy/fastapi_backend/config.py:49](../legacy/fastapi_backend/config.py#L49)** — change `ServerConfig.port` default from `8000` to `3001`. Reason: frontend `NEXT_PUBLIC_API_URL` and CLAUDE.md port contract require 3001.
6. **[legacy/fastapi_backend/app.py:677](../legacy/fastapi_backend/app.py#L677)** — change `uvicorn.run("src.backend.app:app", ...)` to `uvicorn.run("app:app", ...)`. Reason: old Phase-3 module path no longer exists; current `app` is the module itself when launched via `uvicorn app:app`.
7. **[legacy/fastapi_backend/app.py:23-29](../legacy/fastapi_backend/app.py#L23-L29)** — collapse the try/except dual-import block to a single absolute-import form that works whether launched as `uvicorn app:app` from `legacy/fastapi_backend/` or as `uvicorn legacy.fastapi_backend.app:app` from the repo root. Reason: reduces startup fragility; current form is opaque when it fails.

### PatchCore loader bridge (biggest single fix)
8. **[legacy/fastapi_backend/services/inference.py:83-102](../legacy/fastapi_backend/services/inference.py#L83-L102)** `load_patchcore()` — replace the legacy `PatchCoreModel` loader with the anomalib path: `anomalib.models.Patchcore.load_from_checkpoint(f"checkpoints/patchcore_{category}.ckpt")`. Reason: current code expects a `.pth` file produced by a class that Phase 3 deleted; our 15 trained checkpoints are anomalib `.ckpt` files.
9. **[services/inference.py:389-428](../legacy/fastapi_backend/services/inference.py#L389-L428)** `run_patchcore()` — rewrite to call the anomalib model's `forward()` (or wrap in `engine.predict()`), unpack `anomaly_map` and `pred_score` from the anomalib batch format, return the legacy dict keys (`anomaly_score`, `is_anomaly`, `threshold`, `anomaly_map`) unchanged. Reason: the caller contract must stay the same so `/inference/image`, `/inference/predict`, `/api/predict/cascade` routes don't need edits.
10. **[services/inference.py:119-120](../legacy/fastapi_backend/services/inference.py#L119-L120)** BGAD loader — gate it behind `if bgad_path.exists()` and just skip gracefully when `bgad_model.pth` is absent. Reason: we never trained BGAD in this sprint; absence should be a no-op, not a crash.

### Label storage → Redis
11. **New file `legacy/fastapi_backend/services/redis_label_store.py`** — thin wrapper that writes `retina:labels:{image_id}` hashes and reads `retina:al:pool` using the same key contract as the existing worker's `redis_client.py`. Reason: user wants labels in Redis, not JSON files; the worker already owns this namespace.
12. **[services/labeling.py](../legacy/fastapi_backend/services/labeling.py)** — replace the internal `AnnotationStore` filesystem writes with the new `RedisLabelStore`. Keep the `LabelingService` public methods identical (`submit_label`, `skip_sample`, `get_progress`, etc.) so [app.py](../legacy/fastapi_backend/app.py) route handlers are untouched. Reason: preserve API surface while swapping the storage backend.
13. **Frontend expectation check.** Verify `frontend/src/lib/api.ts` calls match the legacy FastAPI paths exactly (e.g. `POST /labels/submit` vs the Rust `POST /api/labels/submit`). If the frontend expects `/api/labels/...` prefixes, add a handful of alias routes in app.py rather than edit the frontend. Reason: the frontend was written against the Rust routes; swapping backends shouldn't require a frontend-wide rename.

### Contract alignment (optional for demo, required for data integrity)
14. **[legacy/fastapi_backend/app.py](../legacy/fastapi_backend/app.py)** `CascadeResponse` and `/inference/predict` response — extend to include the 11 top-level fields added to `shared/schemas/result.json` in step 1 (`product_class`, `natural_description`, `routing_reason`, `vlm_model_used`, etc.). Populate `routing_reason="patchcore_high_confidence"` etc. based on which branch fires. Reason: shared/schemas/result.json is the wire contract; FastAPI emitting a divergent shape re-creates the exact drift we just fixed.
15. **VLM router integration.** Wire `worker/src/retina_worker/models/vlm_router.py` into `/api/predict/cascade` and `/inference/predict` — replace whatever ad-hoc VLM call the legacy `InferenceService.predict_with_cascade` does today. Reason: single source of truth for the VLM contract, and we just verified it with 7 passing tests.

### Cleanup / nice-to-haves
16. **Remove `main()` at [app.py:674-681](../legacy/fastapi_backend/app.py#L674-L681)** or rewrite. Reason: broken reference; easier to launch with `uvicorn` directly.
17. **Drop `/pipeline/stage1/train`, `/pipeline/stage3/train`, `/pipeline/evaluate`** — or mark them as returning 501 for the demo. Reason: they call `PipelineService` which is coupled to the deleted legacy training pipeline; training happens via `scripts/train_anomalib.py` now.
18. **Remove `/models` and `DELETE /models/{type}/{category}`** or rewrite to point at `checkpoints/*.ckpt`. Reason: current implementation globs `MODELS_DIR/patchcore_*.pth` which will be empty.
19. **Add a `/api/system/status` alias** that wraps the existing `/status` — frontend calls `getSystemStatus()` on `/api/system/status` per the audit. Reason: minimise frontend diff.

## Order of operations
1. Fixes #1, #5, #6, #7 — get `uvicorn app:app --port 3001` to boot even if half the routes 500. Validate with `curl /health` returning JSON.
2. Fixes #2, #3, #4 — categories and models endpoints start seeing real data.
3. Fixes #8, #9, #10 — PatchCore inference actually runs on the sprint's anomalib checkpoints.
4. Fixes #11, #12, #13 — labels land in Redis; frontend label page stops 404'ing.
5. Fixes #14, #15 — response shape + VLM router integration, for schema hygiene and operator UX.
6. Fixes #16-19 — cleanup pass once everything else is green.

## Risks flagged for the decision point
- **PatchCoreModel deletion (Phase 3).** Fix #8 assumes anomalib swap-in works cleanly. If the legacy `run_patchcore` code also reaches into `PatchCoreModel` internals (e.g. accesses `.memory_bank` directly), the wrapper will need to be richer. Not yet confirmed.
- **`PipelineService` is 546 lines of code we are leaving unrunnable.** Routes that call it (fixes #17) will 500 until removed. Acceptable for demo if the frontend doesn't hit them.
- **No auth.** Intentional for the sprint; flag before any external deployment.
- **Postgres-free mode** means `anomaly_records` / historical analytics do not exist. The frontend's results page is file-less; may show empty state.
- **Schema drift re-opens** if we skip fix #14. The 11 new `InferenceResult` fields we added in step 1 exist only in a schema the FastAPI backend does not emit.

Stop — no code changes yet.
