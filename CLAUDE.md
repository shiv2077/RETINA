# CLAUDE.md — RETINA Project
# Operating contract for Claude Code.
# Read sections 0 and 1 before every session. Read the rest when relevant.

---

## 0. PROJECT STATE — UPDATE THIS EVERY SESSION

> This section must be kept current. If it is stale, update it before
> doing anything else. A stale state summary is worse than no summary.

### 0.1 What layer you are in right now
- [ ] Backend (Rust/Axum) — `backend/`
- [ ] ML Worker (Python) — `worker/`
- [ ] Frontend (Next.js) — `frontend/`
- [ ] Infra (Docker/config) — root

### 0.2 Current implementation status
Last verified: 2026-04-14

| Component | Status | Notes |
|---|---|---|
| Backend API (Axum) | ~70% | Routes for images/labels incomplete |
| Auth (JWT + Argon2) | DONE | Working |
| Image storage service | DONE | O(n) scan known perf issue |
| Redis service layer | ~50% | Job submission methods cut off |
| Worker poll loop | ~50% | Core run() loop unverified |
| PatchCore (real) | STUB ONLY | patchcore_stub.py is fake |
| GPT-4V detector | NOT BUILT | gpt4v_detector.py does not exist yet |
| BGAD supervisor | STUB ONLY | pushpull_stub.py is fake |
| Multi-class classifier | MISSING | No file exists |
| Active learning module | PARTIAL | Redis pool defined, sampling not built |
| Expert Review page | BROKEN | label/page.tsx — endpoints missing |
| Model Performance page | MISSING | Not built |
| Stage 2 activation logic | MISSING | Threshold in config, routing not wired |
| Cascade inference route | MISSING | Frontend calls it, backend has nothing |

### 0.3 Known unfixed bugs — mark FIXED when resolved
- [ ] Port mismatch: NEXT_PUBLIC_API_URL points to 8000, backend is on 3001
- [ ] pydantic-settings not in pyproject.toml dependencies
- [ ] Worker receives image_id only — cannot load actual image bytes
- [ ] 5 phantom frontend routes that 404 on every call (see section 2.3)
- [ ] CORS wildcard allow_origin(Any) — unsafe for any non-local deploy
- [ ] JWT_SECRET hardcoded in docker-compose.yml

### 0.4 What is in progress right now
<!-- Update this at the start and end of every session -->
Nothing in progress. Starting fresh.

### 0.5 Repo structure
```
backend/          Rust/Axum API server
worker/           Python ML inference worker
frontend/         Next.js 14 UI
shared/schemas/   JSON Schema contracts (source of truth)
docker-compose.yml
.env              Never commit — secrets live here
```

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

**Stage 2 — Supervised detection + classification:**
- Only runs on images that Stage 1 flags as anomalous
- Primary model: BGAD (AUC 0.93, requires labelled defects + masks)
- Fallback: Custom Push-Pull (AUC 0.86, works with 100-200 samples, no masks)
- Stage 2 activates automatically when label count >= active_learning_stage2_threshold
- Before Stage 2 is active: all Stage 1 flags go directly to expert review queue
- Stage 2 output: {defect_class: str, confidence: float, bounding_box: BBox | null}

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
between Rust and Python. They are the authoritative source.

**Update protocol — follow this order exactly:**
1. Edit the JSON Schema file first
2. Update backend/src/models/ (Rust structs) — cargo check must pass
3. Update worker/src/retina_worker/schemas.py (Pydantic models)
4. Run: docker compose build worker backend

Breaking this order causes silent data corruption. The audit found
schemas.py and the Rust models had already drifted once.

### 1.3 Redis key namespacing — never invent keys without documenting here

```
retina:jobs:{job_id}          Hash      InferenceJob fields
retina:queue:inference        Stream    Pending jobs (XREADGROUP, NOT LPUSH)
retina:results:{job_id}       Hash      InferenceResult fields
retina:alerts                 List      Recent anomaly alerts (LPUSH/LRANGE)
retina:labeling:pool          SortedSet score = anomaly_score (not entropy yet — see 6.3)
retina:labeling:{id}          Hash      Label assignment per image
retina:system:stage           String    "1" or "2" — current active pipeline stage
```

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
frontend:  3000 -> 3000
backend:   3001 -> 3001   <- NEXT_PUBLIC_API_URL must point here, not 8000
postgres:  5432 (internal only — never expose externally)
redis:     6379 (internal only — never expose externally)
```

---

## 2. BEFORE WRITING ANY CODE

### 2.1 Pre-flight gate — answer all before touching files

Do not proceed until you can answer YES to every applicable item:

```
[ ] I have read the file I am about to change (not assumed its contents)
[ ] If changing a shared schema: I will update JSON -> Rust -> Python in that order
[ ] If adding a Redis key: I have documented it in section 1.3
[ ] If adding an API route: I have added the function to api.ts in the same commit
[ ] If adding a Python dep: added to pyproject.toml [project.dependencies]
[ ] If adding a Rust dep: added to Cargo.toml with a pinned major version
[ ] If changing Docker config: verified volumes, ports, and env vars are consistent
[ ] My change does not load an ML model per-request
[ ] My change does not use .unwrap() or .expect() in a Rust route handler
```

### 2.2 After making a change — verification steps

Do not mark a task done without running these:

**Backend change:**
```bash
cd backend && cargo check          # Must pass with 0 errors
cd backend && cargo test           # Must pass
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

**Full stack change:**
```bash
docker compose up --build          # All services must reach healthy state
curl http://localhost:3001/health  # Must return 200
```

### 2.3 The five phantom frontend routes — fix before adding more

These frontend API calls currently 404 on every request.

| Frontend calls | Backend has | Action needed |
|---|---|---|
| POST /api/predict/cascade | Nothing | Add Rust route or remove cascade mode |
| POST /inference/predict | Nothing | Add route or map to /api/images/submit |
| GET /inference/history | Nothing | Add route |
| GET /pipeline/stage2/samples | Nothing | Add route or map to /labels/pool |
| GET /status | /api/system/status | Fix frontend URL |

---

## 3. RUST BACKEND

### 3.1 Error handling
- All errors go through AppError in src/error.rs. Use ? operator.
- Never use .unwrap() or .expect() in route handlers.
  If you do: the worker crashes on first unexpected input and takes
  down the entire inference pipeline.
- Log at the point of origin with tracing::error!, not at the boundary.
- Return structured JSON: {"error": "message", "code": "ERROR_CODE"}
  Never return plain text — the frontend cannot parse it.

### 3.2 Database
- Raw SQL via SQLx — no ORM. Queries live in src/services/ only.
  Never inline SQL in route handlers.
- All migrations in src/db/pool.rs. Use ADD COLUMN IF NOT EXISTS,
  never DROP COLUMN. We have no rollback mechanism yet.
- TODO: migrate to sqlx::migrate! for compile-time checked migrations.
  Until then, every SQL statement must be idempotent.
- Known perf issue: read_image() in image_storage.rs does an O(n)
  directory scan. Do not make it worse. Do not call it in a loop.

### 3.3 Redis
- All Redis calls go through src/services/redis.rs — never call
  the Redis client directly from a route handler.
- Job queue is a Redis Stream. Use XREADGROUP / XACK.
  Using LPUSH/LPOP breaks at-least-once delivery guarantees.
  Stream consumer group name: retina-workers

### 3.4 Auth
- JWT secret from config.jwt_secret — never hardcode.
- CORS: currently allow_origin(Any) — acceptable for local dev only.
  Restrict to config.cors_allowed_origins before any external deploy.

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
- Submit payload to POST /labels/submit:
  {job_id, label, defect_class, bounding_box?, operator_id, accepted_gpt4v_suggestion}
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

### 6.4 Stage 2 activation sequence

Trigger: COUNT(labeling_pool WHERE status='labelled') >= active_learning_stage2_threshold

Execute in this order:
1. Log: INFO stage2_activated label_count=N threshold=N
2. Start BGAD retraining as background task (non-blocking)
3. Set retina:system:stage key in Redis to "2"
4. Worker reads this key at start of each job to choose pipeline
5. Stage 1 continues running — it provides the heatmap even in Stage 2
6. Notify frontend via the alerts channel

The config key is active_learning_stage2_threshold (default: 200).
Never hardcode 200 anywhere in the codebase.

### 6.5 Dataset locations

- MVTec AD: Public. Download from mvtec.com/company/research/datasets/mvtec-ad
- Decospan dataset: Private. KU Leuven HPC:
  /scratch/leuven/369/vsc36963/Vakantiejob/Decospan/Dataset
  Not in this repo. Must be transferred separately for training.
- Pre-trained AdaCLIP weights: Expected at weights/pretrained_all.pth
  Not in this repo. AdaCLIP code in Unsupervised_Models/AdaCLIP/ is real
  but not wired into the RETINA pipeline yet.

---

## 7. DOCKER AND INFRA

### 7.1 Required environment variables

These must be in .env at repo root — never in docker-compose.yml values:

```bash
# .env — never commit this file
POSTGRES_PASSWORD=<strong-random-password>
JWT_SECRET=<min-32-char-random-string>
OPENAI_API_KEY=sk-...

# Worker ML config
PATCHCORE_CHECKPOINT_PATH=/data/checkpoints/patchcore.pt
GPT4V_PRODUCT_TYPE=manufactured product

# Connection strings
DATABASE_URL=postgresql://retina:${POSTGRES_PASSWORD}@postgres:5432/retina
REDIS_URL=redis://redis:6379

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:3001
```

docker-compose.yml references these as ${VAR_NAME} only.
Never paste secret values directly into the compose file.

### 7.2 Health checks — every service must have one

```yaml
# backend
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:3001/health"]
  interval: 10s
  timeout: 5s
  retries: 3

# worker
healthcheck:
  test: ["CMD", "python", "-c", "import redis; redis.Redis.from_url('redis://redis:6379').ping()"]
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
| Change JSON schema without updating Rust + Python | Silent data corruption in Redis — jobs process but results are wrong |
| Use .unwrap() in a Rust route handler | Worker crashes on first malformed input, takes down entire pipeline |
| Load ML model per-request | Worker OOMs after ~3 images, container restarts, jobs lost |
| LPUSH/LPOP on job queue instead of XREADGROUP | At-least-once delivery broken — jobs silently dropped under load |
| Add backend route without updating api.ts | Frontend 404s silently — users see blank data, no error message |
| Leave message unacknowledged on worker failure | Same job redelivered forever, blocking all new jobs |
| Store absolute host paths in DB | Path breaks when container is recreated — images become inaccessible |
| Set max_tokens=300 for GPT-4V | JSON truncated mid-field — json.loads() throws, job fails silently |
| Log image bytes or API keys | Security incident and 10MB log entries per image |
| Rename Dutch defect class labels | Dataset labels no longer match — all model metrics become invalid |
| Hardcode Stage 2 threshold | Different factory deployments cannot tune their label budget |
| Use requirements.txt | Two dep files diverge — wrong versions installed silently |
| Skip cargo check after Rust change | Compilation error only found at docker build time (minutes later) |
| Mount image volume at different paths in backend vs worker | Worker has image_id but cannot load pixels — inference fails silently |
| Resize images to >1024px before GPT-4V call | Token cost spikes 4x — OpenAI bill grows unexpectedly |

---

## 9. WHEN YOU ARE UNSURE

**About architecture:** Section 1.1 is the law. Ask: does this change
preserve the Stage 1 -> Stage 2 flow?

**About schema:** shared/schemas/*.json is ground truth. Not the Rust
structs. Not the Pydantic models. The JSON files.

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
BACKEND (Rust)
backend/src/main.rs                           App startup, router wiring
backend/src/config.rs                         All env vars with defaults
backend/src/error.rs                          AppError enum — always use this
backend/src/routes/health.rs                  DONE — working
backend/src/routes/auth.rs                    ~85% done
backend/src/routes/images.rs                  ~60% — queue submission cut off
backend/src/routes/labels.rs                  ~50% — body unverified
backend/src/routes/anomaly.rs                 Unknown — not fully read
backend/src/routes/system.rs                  ~50% — implementation cut off
backend/src/services/redis.rs                 ~50% — methods cut off
backend/src/services/image_storage.rs         DONE (O(n) scan known issue)
backend/src/services/alerts.rs                DONE
backend/src/db/models.rs                      ~90% — optional fields not populated
backend/src/db/pool.rs                        85% — raw SQL, not sqlx::migrate!

WORKER (Python)
worker/src/retina_worker/config.py            Settings — missing openai_api_key
worker/src/retina_worker/schemas.py           Pydantic mirror of Rust models
worker/src/retina_worker/worker.py            ~50% — run() loop unverified
worker/src/retina_worker/redis_client.py      DONE
worker/src/retina_worker/models/base.py       DONE — AnomalyDetector ABC
worker/src/retina_worker/models/factory.py    DONE — registry (points to stubs)
worker/src/retina_worker/models/patchcore_stub.py    STUB — hash-based fake
worker/src/retina_worker/models/winclip_stub.py      STUB — hash-based fake
worker/src/retina_worker/models/pushpull_stub.py     STUB — hash-based fake
worker/src/retina_worker/models/patchcore_real.py    MISSING — does not exist
worker/src/retina_worker/models/gpt4v_detector.py    MISSING — does not exist

FRONTEND (Next.js)
frontend/src/lib/api.ts                       ALL backend calls — central
frontend/src/app/page.tsx                     Dashboard (~95%)
frontend/src/app/submit/page.tsx              Upload + predict (~80%)
frontend/src/app/label/page.tsx               Expert review — BROKEN
frontend/src/app/results/page.tsx             Exists — not fully read
frontend/src/app/demo/page.tsx                Exists — not fully read
frontend/src/app/model-performance/page.tsx   MISSING — does not exist

SHARED CONTRACTS
shared/schemas/job.json                       InferenceJob — ground truth
shared/schemas/result.json                    InferenceResult — ground truth
shared/schemas/label.json                     Label — ground truth

RESEARCH (not wired into pipeline)
Unsupervised_Models/AdaCLIP/                  Real CLIP-based VLM — standalone only
Unsupervised_Models/AdaCLIP/config_decospan.yaml  ViT-L/14 config used for Decospan run

INFRA
docker-compose.yml                            Service wiring — port 3001 is backend
.env                                          Secrets — never commit
```

---

*Last updated: 2026-04-14*
*Maintainer: dries.vandaele@kuleuven.be / Flanders Make RETINA*
