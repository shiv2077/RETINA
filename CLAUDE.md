# CLAUDE.md — RETINA Project
# Operating contract for Claude Code.
# Read sections 0 and 1 before every session. Read the rest when relevant.

---

> ## ⚠ STALENESS BANNER — 2026-09-09
>
> **Every Rust/Axum instruction in this file was stale.** `backend/` (Rust/Axum)
> and `legacy/fastapi_backend/` were both **deleted** in commit `4ff54ba`.
> There is no Rust in this repo, no Cargo.toml, no `cargo check`, no JWT auth,
> and no Postgres. See `docs/DECISIONS.md` #14 for the rationale.
>
> The backend is now **`api/main.py`** — a single-file FastAPI wrapper over
> Redis. Its complete route list is in §2.3.
>
> Sections 0.2, 0.3, 1.2, 1.3, 1.5, 2.x, 3, 7.1 and 11 have been rewritten
> against the current tree. Sections 4, 5, 6 and 8 were audited but describe
> the worker/frontend/ML layers, which still exist.
>
> **Where this file and `docs/DECISIONS.md` disagree, DECISIONS.md wins** — it
> is written against the running code and cites file:line evidence.

---

## 0. PROJECT STATE — UPDATE THIS EVERY SESSION

> This section must be kept current. If it is stale, update it before
> doing anything else. A stale state summary is worse than no summary.

### 0.1 What layer you are in right now
- [ ] API (Python/FastAPI) — `api/main.py`
- [ ] ML Worker (Python) — `worker/`
- [ ] Frontend (Next.js) — `frontend/`
- [ ] Infra (Docker/config) — root

### 0.2 Current implementation status
Last verified: 2026-09-09

| Component | Status | Notes |
|---|---|---|
| API (`api/main.py`) | DONE | FastAPI over Redis. 8 routes, all listed in §2.3 |
| Rust/Axum backend | **DELETED** | Removed in `4ff54ba`. See DECISIONS.md #14 |
| Auth (JWT + Argon2) | **DELETED** | No auth on the API at all. DECISIONS.md #13 |
| Postgres | **DELETED** | Redis with TTL is the only store. DECISIONS.md #10 |
| Worker poll loop | DONE | XREADGROUP on `retina:jobs:queue`, group `workers` |
| PatchCore (Stage 1) | DONE | Per-category checkpoints, 2-model LRU cache (#11) |
| VLM router | DONE | `vlm_router.py` — identify / zero-shot / describe / refine |
| Stage 2 | DONE | GPT-4o in-context refiner, **not** BGAD. DECISIONS.md #3 |
| Multi-class classifier | NOT BUILT | Deliberately. DECISIONS.md #5 |
| BGAD / Push-Pull | RESEARCH ONLY | `research/supervised/` — zero imports from runtime |
| Active learning pool | DONE | `retina:al:pool`, sorted by score (known flaw, §6.3) |
| Expert Review page | DONE | `/label` runs on `/api/labels/pool` + `/api/labels/submit` |
| Model Performance page | DONE | Static benchmark reference |

### 0.3 Known unfixed bugs — mark FIXED when resolved
- [x] FIXED 2026-09-09 (F6): six frontend clients called routes no backend
      ever served (`/api/predict/cascade`, the four `/api/labeling/cascade/*`
      routes, `/api/system/status`). Deleted, call sites rebuilt on real routes.
- [x] FIXED 2026-09-09 (F24): UI claimed BGAD was the Stage 2 model.
- [x] FIXED 2026-04-20: `api.ts` fallback URL is `localhost:3001`.
- [ ] CORS is wide open on `api/main.py` — local dev only.
- [ ] No auth on any route (deliberate for the demo — DECISIONS.md #13, but it
      is a blocker for any deployment).
- [ ] Labels expire after 7 days with no export path (DECISIONS.md #10). This
      is the largest gap between this build and a deployable system.
- [ ] `results/page.tsx` still carries a per-category evaluation table, a
      confusion matrix and an "Evaluate" button wired to nothing. No endpoint
      supplies evaluation data; either build one or delete the sections.

### 0.4 What is in progress right now
Session: 2026-09-09
Completed:
  - F6: removed the six dead API clients and their types from `api.ts`;
    rebuilt the dashboard on `/health` + `/api/labels/pool`, rewired
    `/demo` onto `/api/submit` + `/api/result/{id}`.
  - F24: replaced every BGAD label in the UI with the GPT-4o refiner.
  - F5: this file, de-Rusted against the current tree.
  - F27: corrected the `worker/main.py` module docstring defaults.
Next session should start with:
  1. Decide the fate of the dead evaluation UI in `results/page.tsx`.
  2. Label persistence beyond the 7-day Redis TTL (DECISIONS.md #10).
  3. Lock down CORS before anything leaves localhost.

### 0.5 Repo structure
```
api/main.py       FastAPI wrapper over Redis — THE backend (production)
worker/           Python ML inference worker (production)
frontend/         Next.js 14 UI (production)
shared/schemas/   JSON Schema contracts (source of truth)
research/         Standalone ML research code — NOT wired into the pipeline
scripts/          Training, evaluation and run scripts (run_api.sh etc.)
docs/             Guides, figures, DECISIONS.md, session archives
docker-compose.yml
.env              Never commit — secrets live here
.env.example      Template — copy to .env to start
```

Deleted, do not look for them: `backend/` (Rust/Axum) and
`legacy/fastapi_backend/`, both removed in `4ff54ba` (DECISIONS.md #14).

---

## 1. ARCHITECTURE — THE INVARIANTS

### 1.1 The two-stage pipeline

```
Camera -> [Stage 1: Unsupervised] -> PASS or ANOMALY_FLAGGED
                                            |
                            [Stage 2: Supervised] -> Defect type + confidence
                                            |
                            [Active Learning] -> Expert labels unknown only
                                            |
                            [Retrain loop] -> Stage 2 gets smarter over time
```

**Stage 1 — Unsupervised anomaly detection:**
- Primary model: PatchCore (trained on normal images only)
- Fallback when PatchCore has no checkpoint: GPT-4V zero-shot
- GPT-4V is NOT a co-equal Stage 1 model — it is the cold-start fallback
  only. Once PatchCore has a memory bank, PatchCore runs first.
- Stage 1 can generate false positives — that is expected and acceptable.
- Stage 1 output: {anomaly_score: float, is_anomaly: bool, heatmap: array}

**Stage 2 — GPT-4o in-context refiner (NOT a trained model):**
- Only runs on scores in the uncertainty band `[0.5, 0.9)` — see §6.4
- `stage2_refine` sends the image plus up to 5 operator-labelled examples
  from `retina:labels:*` to gpt-4o as few-shot context. No training step.
- BGAD and Push-Pull were evaluated and rejected (DECISIONS.md #3). They live
  in `research/supervised/` with zero imports from the running system.
- Stage 2 output: `{verdict, defect_class, confidence}` where verdict is
  `confirmed_anomaly` | `rejected_false_positive` | `uncertain`
- Stage 2 runs before `describe_defect`, so a rejection skips the description
  call entirely (DECISIONS.md #15)

**The routing rule — lives in worker.py only:**
```
if stage2_is_active and image_is_flagged:
    run Stage 2 -> get defect_class
else if image_is_flagged:
    send to expert review queue
    optionally run GPT-4V for suggested label (non-blocking)
```

Do not implement this routing logic anywhere except worker.py.

### 1.2 Schema contracts — the ground truth

shared/schemas/job.json, result.json, label.json define the wire format
between `api/main.py` and the worker. They are the authoritative source.

**Update protocol — follow this order exactly:**
1. Edit the JSON Schema file first
2. Update `worker/src/retina_worker/schemas.py` (Pydantic models)
3. Update the matching Pydantic model in `api/main.py`
4. Update `frontend/src/lib/api.ts` if the shape crosses the HTTP boundary
5. Run: `docker compose build worker`

Breaking this order causes silent data corruption: both sides read the same
Redis hash, so a drifted field is written by one and ignored by the other with
no error anywhere.

### 1.3 Redis key namespacing — never invent keys without documenting here

```
retina:jobs:queue                Stream     Pending jobs (XADD by api, XREADGROUP by worker)
retina:jobs:{job_id}             Hash       Job status + data
retina:results:{job_id}          Hash       Field `result_data` = InferenceResult JSON
retina:images:{image_id}         Hash       Fields: image_path, latest_job_id
retina:labels:{image_id}         Hash       Expert label + polygons/boxes. TTL 7 days
retina:al:pool                   SortedSet  Labeling pool, score = anomaly_score (see §6.3)
retina:al:samples:{image_id}     String     UnlabeledSample metadata JSON
retina:system:stats              Hash       Counters, incl. the label count
retina:alerts                    List       Recent alerts (LPUSH, LTRIM to 100)
retina:taxonomy:{product_class}  String     Operator-added defect categories JSON
```

Consumer group name: `workers` (`redis_client.py:51`)
Stream name: `retina:jobs:queue`

Verified 2026-09-09 against `worker/src/retina_worker/redis_client.py` and
`api/main.py`. Two keys previously documented here **do not exist** anywhere in
the codebase and have been removed: `retina:mismatches` and `retina:system:stage`.
(The Stage 2 activation flag is obsolete regardless — see §6.4.)

> **Follow-up needed.** Work is landing in parallel that adds a dead-letter
> stream and a label index. This table documents only what existed at the time
> of writing; re-verify against `redis_client.py` after those merge.

If you add a Redis key, add it to this table with type, pattern, and purpose.

### 1.4 Docker volume contract — do not break this

Backend writes images here. Worker reads from the same path.
If this volume is not shared, the worker sees no images.

```yaml
# In docker-compose.yml — both services must have this exact mount:
volumes:
  - image-data:/data/images
```

This is the root cause of the image_id-only bug in the worker.

### 1.5 Port contract — these are fixed

```
frontend:  3000        Next.js dev server / container
api:       3001        uvicorn api.main:app  (scripts/run_api.sh)
                       <- NEXT_PUBLIC_API_URL must point here
redis:     6379        bound to 127.0.0.1 only, NOT published to the network.
                       Runs with --requirepass; REDIS_PASSWORD is required.
```

Postgres is gone — there is no 5432. The API runs on the host, not in
docker-compose; `scripts/run_api.sh` is the entry point.

---

## 2. BEFORE WRITING ANY CODE

### 2.1 Pre-flight gate — answer all before touching files

Do not proceed until you can answer YES to every applicable item:

```
[ ] I have read the file I am about to change (not assumed its contents)
[ ] If changing a shared schema: JSON -> worker schemas.py -> api/main.py -> api.ts
[ ] If adding a Redis key: I have documented it in section 1.3
[ ] If adding an API route: I have added the function to api.ts in the same commit
[ ] If removing an API route: I have removed its api.ts client and every call site
[ ] If adding a Python dep: added to pyproject.toml [project.dependencies]
[ ] If changing Docker config: verified volumes, ports, and env vars are consistent
[ ] My change does not load an ML model per-request
```

### 2.2 After making a change — verification steps

Do not mark a task done without running these:

**API change (`api/main.py`):**
```bash
python -m ruff check api/
python -c "import api.main"        # import must succeed
```

**Worker change:**
```bash
cd worker && python -m mypy src/   # Must pass
cd worker && python -m pytest      # Must pass
cd worker && python -m ruff check src/  # Must pass
```

**Frontend change:**
```bash
cd frontend && npx tsc --noEmit    # Must pass
cd frontend && npm run build       # Must pass
```

**Full stack:**
```bash
docker compose up --build          # redis + worker + frontend
./scripts/run_api.sh               # API on the host, port 3001
curl http://localhost:3001/health  # {"status":"ok","redis":"up"}
```

### 2.3 The complete backend route list

`api/main.py` serves these eight routes and **nothing else**. Any frontend call
to a path not on this list is a 404. Verify against the file before adding a
client function to `api.ts`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/submit` | multipart image upload -> XADD to the job stream, returns `{job_id}` |
| GET | `/api/result/{job_id}` | InferenceResult, `?wait=N` to long-poll |
| GET | `/api/labels/pool` | active learning pool, `?limit=N` |
| POST | `/api/labels/submit` | persist an expert label + polygons/boxes |
| GET | `/api/images/{image_id}` | serve image bytes (5 lookup strategies) |
| GET | `/api/taxonomy/{product_class}` | operator-added defect categories |
| POST | `/api/taxonomy/{product_class}` | append a defect category |
| GET | `/health` | `{"status":"ok","redis":"up"}` |

The five "phantom routes" this section used to track were resolved in 2026-09-09
by deleting the frontend clients (F6), not by adding routes. There was never a
backend that served them.

---

## 3. THE API (`api/main.py`)

> The Rust/Axum backend this section used to describe was deleted in `4ff54ba`.
> There is no `src/error.rs`, no `AppError`, no SQLx, no `.unwrap()` to avoid,
> and no `src/services/`. See `docs/DECISIONS.md` #14.

### 3.1 Shape
One file, ~350 lines, no router modules and no service layer. It is a thin
HTTP surface over Redis: every handler reads or writes a key from §1.3 and
returns. Keep it that way — inference logic belongs in the worker, not here.

### 3.2 Error handling
- Raise `HTTPException(status, detail)`. FastAPI serialises it as
  `{"detail": "..."}`, which is what `apiFetch()` in `api.ts` parses.
- Never let a `redis.RedisError` escape as a 500 with a stack trace; catch it
  and return a 503 with a readable detail, as `/health` does.
- Log with `structlog`, bound to `job_id` or `image_id`.

### 3.3 Persistence
- Redis is the only store. There is no database, no migration, no ORM
  (DECISIONS.md #10).
- Anything written must carry a TTL or be explicitly unbounded by design.
  Labels and results are 7 days.
- Redis requires a password (`REDIS_PASSWORD`); connect via `REDIS_URL`.

### 3.4 Queue
- The job queue is a Redis Stream. `api/main.py` XADDs; the worker consumes
  with XREADGROUP under group `workers`. Never LPUSH to it.

### 3.5 Auth and CORS
- **There is no authentication.** This is deliberate for the demo
  (DECISIONS.md #13) and is a hard blocker for any deployment.
- CORS is currently wide open. Restrict it before anything leaves localhost.

---

## 4. PYTHON WORKER

### 4.1 Every model class must satisfy this contract

```python
class MyModel(AnomalyDetector):
    def load(self, checkpoint_path: str) -> None:
        # Load weights. Raise ModelNotLoadedError if file missing.
        # Called ONCE at worker startup, not per-image.

    def predict(self, image: PIL.Image.Image) -> AnomalyPrediction:
        # Must always return AnomalyPrediction — never raise.
        # On any internal error: return low-confidence prediction,
        # log the error with structlog including job_id.
        # heatmap field: numpy uint8 array, same spatial size as input.
```

ModelNotLoadedError must be caught in worker.py and trigger
a fallback to GPT-4V, not a crash.

### 4.2 GPT-4V — implementation rules

**Model:** gpt-4o (not gpt-4-turbo, not gpt-4-vision-preview)
gpt-4o has vision capability and costs roughly 10x less than gpt-4-turbo.

**Image preprocessing before API call:**
- Resize to max 1024px on longest side — preserves detail, cuts token cost
- Encode as JPEG at quality=85 — PNG is 3-5x larger for no benefit
- Base64 encode the JPEG bytes — NOT the raw PIL image

**Token budget:** max_tokens=500
(300 was too low — the JSON with reasoning fields overflows it
and the API returns a truncated unparseable response)

**Required JSON response format:**
```json
{
  "is_anomaly": true,
  "confidence": 0.87,
  "anomaly_score": 0.87,
  "defect_description": "diagonal scratch across grain, approx 3cm",
  "defect_location": "upper-left quadrant",
  "reasoning": "Linear mark inconsistent with natural wood grain pattern"
}
```

**Parsing — strip markdown fences before json.loads():**
```python
raw = response.choices[0].message.content
clean = raw.strip().removeprefix("```json").removesuffix("```").strip()
result = json.loads(clean)
```

**Rate limit handling (429):** Exponential backoff — 2^attempt seconds,
max 3 retries, then return fallback prediction with confidence=0.0
and defect_description="GPT-4V unavailable". Do not crash the worker.

**API down or timeout:** Return AnomalyPrediction(anomaly_score=0.5,
is_anomaly=False, confidence=0.0) and log WARNING with job_id.
Score of 0.5 is neutral — it will not trigger a false alert.

**Product type context:**
product_type field in InferenceJob.metadata must be injected into
the GPT-4V prompt. This is the primary generalisation mechanism.
If missing, default to "manufactured product".

**Never log:** image bytes, base64 strings, or API keys. Log job_id only.

### 4.3 Worker loop

- Poll interval: config.poll_interval_ms (default 500ms)
- On job failure: XACK the message, write status=failed to result hash.
  Never leave a message unacknowledged — it will be redelivered forever.
- Graceful shutdown: catch SIGTERM -> finish current job -> exit cleanly.
- PatchCore memory bank: load ONCE at startup. It is 200-500MB.
  Reloading per-image will OOM the container and kill the worker.
- Worker must log at startup: model loaded, checkpoint path,
  Redis connected, stream group joined.

### 4.4 Dependencies — pyproject.toml only

```toml
[project.dependencies]
pydantic>=2.0.0
pydantic-settings>=2.0.0    # REQUIRED — was missing, caused import error
structlog>=24.0.0
redis>=5.0.0
pillow>=10.0.0
numpy>=1.24.0
openai>=1.0.0               # Required for GPT-4V
torch>=2.0.0,<3.0.0         # Required — not optional
torchvision>=0.15.0,<1.0.0  # Required — not optional
anomalib>=1.0.0,<2.0.0      # Required — not optional
```

Never add to requirements.txt. If you see one, delete it.

### 4.5 Logging with structlog

```python
import structlog
log = structlog.get_logger().bind(job_id=job.id)

log.debug("preprocessing_image", size=image.size)
log.info("prediction_complete", score=result.anomaly_score, model="patchcore")
log.warning("model_fallback", reason="checkpoint_missing", fallback="gpt4v")
log.error("prediction_failed", exc_info=True)
```

Every log entry must have job_id bound. No bare print() statements.

---

## 5. FRONTEND

### 5.1 API client — strict rules

src/lib/api.ts is the only file that knows backend URLs. No exceptions.

- All HTTP calls go through apiFetch() — no fetch() directly in components
- URL base: process.env.NEXT_PUBLIC_API_URL — never hardcode localhost:3001
- No any types on function return values — everything must be typed
- When adding a backend route: update api.ts in the same commit

### 5.2 Expert Review page — human-in-the-loop, treat it carefully

src/app/label/page.tsx — this is where active learning happens.
Every interaction here directly affects model quality.

Required behavior:
- Display flagged image with GPT-4V bounding box suggestion overlaid
  (purple dashed) and GPT-4V description + reasoning visible
- Operator can: accept suggestion / modify box / reject and reclassify
- Submit payload to POST /api/labels/submit — this is the `LabelSubmission`
  model in api/main.py, keyed on image_id, not job_id:
  {image_id, product_class, label, defect_class?, polygons?, boxes?,
   operator_id?, notes?}
  A successful submit removes the image from `retina:al:pool`.
- Keyboard shortcuts (non-negotiable for operator efficiency):
  S = skip, Enter = submit, Z = undo, 1-9 = quick class select
- Show running count: "12 reviewed this session / 47 remaining"

### 5.3 Design system

```
Background:      #0A0F1E with teal radial glow at top-center
Glass card:      bg-white/5 backdrop-blur-xl border border-white/12 rounded-2xl
Primary accent:  #00D4AA (teal)
Secondary:       #7C3AED (purple)
Defect/alert:    #FF4D6D with CSS pulse animation on border
Pass/normal:     #10B981 (green)
Font:            Inter
```

Never hardcode hex values in components — use CSS variables or Tailwind config.
Dark mode is the only mode. Do not add a light mode toggle.

### 5.4 Image and heatmap display

- Heatmaps: always overlaid with mix-blend-mode: multiply, opacity: 0.6
  Never show a raw heatmap — it looks broken and confuses operators.
- Anomaly score: always show as both a decimal number (0.847) AND a
  colored gauge bar (green to yellow to red). One without the other is insufficient.
- Bounding boxes: teal solid = confirmed, purple dashed = GPT-4V suggested
- Never show a loading spinner for more than 3 seconds without a status message

---

## 6. ML AND RESEARCH

### 6.1 Model benchmarks — the reference table

Any new model must beat the relevant baseline by AUC > 0.02 to justify addition.

| Model | Dataset | AUC | Type | Notes |
|---|---|---|---|---|
| PatchCore | MVTec AD | 0.895 | One-class | Memory bank, domain specific |
| PaDiM | MVTec AD | 0.884 | One-class | Fastest inference |
| WinCLIP | MVTec AD | 0.856 | Zero-shot | Baseline for VLM comparison |
| GPT-4V | MVTec AD | TBD | Zero-shot | Benchmark when integrated |
| BGAD | MVTec AD | 0.930 | Supervised | Requires masks |
| Push-Pull | Decospan | 0.860 | Supervised | No masks, 100-200 samples |

### 6.2 Decospan defect taxonomy — do not rename these

These are dataset labels. The Dutch names are canonical.
English names are for display only — never use them as code identifiers.

| Dutch (canonical) | English (display only) | Category |
|---|---|---|
| deuk | dent | structural |
| krassen | scratches | surface |
| vlekken | stains | surface |
| open voeg | open joint | structural |
| open fout | open defect | structural |
| open knop | open knot | structural |
| snijfout | cutting error | process |
| barst | crack | structural |
| scheef | skewed | process |
| stuk fineer | broken veneer | structural |

### 6.3 Active learning — current state and known flaw

**Current implementation:** Images added to retina:labeling:pool sorted set
with their anomaly_score as the sort key.

**Known flaw:** High anomaly score is NOT the same as high model uncertainty.
A score of 0.99 means "very anomalous" — the model may be highly confident
about that, making it a poor labelling candidate. True uncertainty sampling
uses prediction entropy or margin between top-2 class probabilities.

**Current behavior is acceptable for Stage 1** (less than 200 labels) because
any labelled defect is valuable when starting from zero.

**Required improvement for Stage 2** (when label count > 50):
Replace score-based sorting with entropy-based uncertainty:
```python
# Entropy of classifier output probabilities
uncertainty = -sum(p * log(p) for p in class_probs)
```
Do not implement this until the multi-class classifier exists.
When you implement it, update this section.

### 6.4 Stage 2 gating — there is no activation threshold

> **Obsolete.** This section used to describe a label-count threshold that
> "activated" a trained BGAD Stage 2, writing `retina:system:stage`. None of
> that exists. There is no threshold, no retraining trigger, and no stage key.

Stage 2 is always available, because it is a GPT-4o call rather than a trained
model (DECISIONS.md #3). It is gated on the **score**, not on a label count:

- `should_run_stage2` (`vlm_router.py`) returns true for
  `anomaly_score` in `[0.5, 0.9)` — the uncertainty band (DECISIONS.md #4).
- Above 0.9, PatchCore is confident and the refiner has nothing to add.
- Below 0.5, nothing runs — a confidently-wrong PatchCore score is never
  caught. That is a known and accepted consequence.
- Stage 2 runs *before* `describe_defect`, so a rejected false positive never
  pays for a description (DECISIONS.md #15).

More operator labels make Stage 2 better by enriching the few-shot examples
pulled from `retina:labels:*`, not by crossing any threshold.

### 6.5 Dataset locations

- MVTec AD: Public. Download from mvtec.com/company/research/datasets/mvtec-ad
- Decospan dataset: Private. KU Leuven HPC:
  /scratch/leuven/369/vsc36963/Vakantiejob/Decospan/Dataset
  The dataset itself is not in this repo and must be transferred separately
  for training. Note: 78 Decospan-derived image crops from an AdaCLIP run
  were previously committed under
  research/unsupervised/AdaCLIP/custom_adaclip/results/v21/ — they are now
  untracked and gitignored, but they still exist in the working tree on disk
  and remain in git history pending a separate decision on whether to purge
  them with filter-repo.
- Pre-trained AdaCLIP weights: Expected at weights/pretrained_all.pth
  Not in this repo. AdaCLIP code in Unsupervised_Models/AdaCLIP/ is real
  but not wired into the RETINA pipeline yet.

---

## 7. DOCKER AND INFRA

### 7.1 Required environment variables

These must be in .env at repo root — never in docker-compose.yml values:

```bash
# .env — never commit this file
REDIS_PASSWORD=<strong-random-password>   # REQUIRED — redis runs --requirepass
OPENAI_API_KEY=sk-...                     # REQUIRED — Stage 1 fallback + Stage 2

# Worker ML config
PATCHCORE_CHECKPOINT_PATH=/data/checkpoints/patchcore.pt
GPT4V_PRODUCT_TYPE=manufactured product

# Connection strings
REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:3001
```

`POSTGRES_PASSWORD`, `DATABASE_URL` and `JWT_SECRET` are **gone**. Postgres was
removed with the Rust backend (DECISIONS.md #10, #14) and there is no auth to
sign tokens for (#13). If you find them in a `.env`, they are dead entries.

docker-compose.yml references these as ${VAR_NAME} only.
Never paste secret values directly into the compose file.

### 7.2 Health checks — every service must have one

```yaml
# worker
healthcheck:
  test: ["CMD", "python", "-c", "import os,redis; redis.Redis.from_url(os.environ['REDIS_URL']).ping()"]
  interval: 15s
  timeout: 5s
  retries: 3

# frontend
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
  interval: 10s
  timeout: 5s
  retries: 3
```

There is no `backend` service to health-check. The API runs on the host via
`scripts/run_api.sh`; check it with `curl -f http://localhost:3001/health`.
Redis needs `REDIS_PASSWORD` in its healthcheck too now that it runs
`--requirepass` — a bare `redis-cli ping` returns NOAUTH.

### 7.3 Dockerfile layer ordering — do not break build cache

In the worker Dockerfile, dependency installation must come before
source code copy. This preserves the pip install cache layer:

```dockerfile
COPY pyproject.toml .
RUN pip install -e ".[ml]"   # this layer is cached across source changes
COPY src/ src/               # this layer invalidates on source changes only
```

If you swap this order: every source change triggers a full pip install.
torch + anomalib takes 10+ minutes. This has happened before.

---

## 8. WHAT BREAKS IF YOU DO X

Cause-and-effect rules derived from confirmed bugs in the audit.

| If you do this | This breaks |
|---|---|
| Change a JSON schema without updating worker schemas.py AND api/main.py | Silent data corruption in Redis — jobs process but results are wrong |
| Load ML model per-request | Worker OOMs after ~3 images, container restarts, jobs lost |
| LPUSH/LPOP on the job queue instead of XREADGROUP | At-least-once delivery broken — jobs silently dropped under load |
| Add a backend route without updating api.ts | Frontend cannot reach it; the route is dead weight |
| Add an api.ts client for a route not in §2.3 | Page 404s silently — users see blank data, no error. This is finding F6 |
| Leave a message unacknowledged on worker failure | Same job redelivered forever, blocking all new jobs |
| Write to Redis without a TTL | Unbounded growth — Redis is the only store and nothing evicts it |
| Set max_tokens=300 for GPT-4V | JSON truncated mid-field — json.loads() throws, job fails silently |
| Log image bytes or API keys | Security incident and 10MB log entries per image |
| Rename Dutch defect class labels | Dataset labels no longer match — all model metrics become invalid |
| Name BGAD or Push-Pull as the Stage 2 that runs | Misrepresents the system; neither is wired. This is finding F24 |
| Use requirements.txt | Two dep files diverge — wrong versions installed silently |
| Mount the image volume at different paths in api vs worker | Worker has image_id but cannot load pixels — inference fails silently |
| Resize images to >1024px before the GPT-4V call | Token cost spikes 4x — OpenAI bill grows unexpectedly |
| Publish Redis beyond 127.0.0.1 | Unauthenticated-by-default datastore exposed; it holds every image path and label |

---

## 9. WHEN YOU ARE UNSURE

**About architecture:** Section 1.1 is the law. Ask: does this change
preserve the Stage 1 -> Stage 2 flow?

**About schema:** shared/schemas/*.json is ground truth. Not the Pydantic
models in the worker or the API. Not the TypeScript interfaces. The JSON files.

**About model performance:** Section 6.1 benchmark table is the reference.
Do not claim a model improvement without AUC numbers.

**About whether a bug is fixed:** Check section 0.3. If not marked FIXED,
assume it is not fixed. Verify before declaring resolved.

**About an ambiguous task:** Do not stop. Make a decision, implement it,
leave a comment: DECISION(date): chose X over Y because Z.
Bias toward action. Document the choice for review.

---

## 10. SESSION HANDOFF TEMPLATE

At the end of every session, update section 0.4 with this format:

```
Session: 2026-MM-DD
Completed:
  - Added gpt4v_detector.py with base64 image encoding and retry logic
  - Fixed port mismatch in docker-compose.yml
In progress:
  - worker.py run() loop — XREADGROUP polling implemented,
    error recovery not done yet
    Last file edited: worker/src/retina_worker/worker.py line 87
Blocked:
  - BGAD integration — Anomalib 1.0 does not include BGAD yet
Next session should start with:
  1. Finish worker.py error recovery (section 4.3)
  2. Add /api/predict/cascade route to backend (section 2.3)
```

This takes 2 minutes and saves 20 minutes of re-orientation next session.

---

## 11. QUICK REFERENCE — KEY FILE LOCATIONS

```
API (Python/FastAPI) — the whole backend
api/main.py                                   All 8 routes (§2.3), Redis key constants
scripts/run_api.sh                            uvicorn on port 3001

WORKER (Python)
worker/src/retina_worker/main.py              Entry point
worker/src/retina_worker/config.py            Settings + defaults
worker/src/retina_worker/schemas.py           Pydantic wire models
worker/src/retina_worker/worker.py            run() loop, Stage 1/2 routing
worker/src/retina_worker/redis_client.py      All Redis access, key constants
worker/src/retina_worker/vlm_router.py        identify / zero_shot / describe / stage2_refine
worker/src/retina_worker/patchcore_registry.py  2-model LRU checkpoint cache
worker/src/retina_worker/models/base.py       AnomalyDetector ABC

FRONTEND (Next.js)
frontend/src/lib/api.ts                       ALL backend calls — central
frontend/src/app/page.tsx                     Dashboard — /health + /api/labels/pool
frontend/src/app/submit/page.tsx              Upload + predict
frontend/src/app/demo/page.tsx                Upload + predict, annotated routing
frontend/src/app/label/page.tsx               Expert review — the active learning loop
frontend/src/app/results/page.tsx             Evaluation dashboard (partly dead, see §0.3)
frontend/src/app/model-performance/page.tsx   Static benchmark reference

SHARED CONTRACTS
shared/schemas/job.json                       InferenceJob — ground truth
shared/schemas/result.json                    InferenceResult — ground truth
shared/schemas/label.json                     Label — ground truth

DOCS
docs/DECISIONS.md                             15 architecture decisions with evidence.
                                              Wins over this file on any disagreement.

RESEARCH (not wired into the pipeline — zero runtime imports)
research/unsupervised/AdaCLIP/                CLIP-based VLM, standalone only
research/supervised/BGAD/                     BGAD. NOT the Stage 2 that runs (§6.4)
research/supervised/Custom_Model_Push_Pull/   Push-Pull contrastive learning

INFRA
docker-compose.yml                            redis + worker + frontend. No backend, no postgres
.env                                          Secrets — never commit
```

DELETED — do not look for these:
```
backend/                                      Rust/Axum. Removed in 4ff54ba
legacy/fastapi_backend/                       Removed in 4ff54ba
```

---

*Last updated: 2026-09-09 — de-Rusted against the current tree (F5)*
*Maintainer: dries.vandaele@kuleuven.be / Flanders Make RETINA*
