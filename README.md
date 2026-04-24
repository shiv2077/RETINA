# RETINA — Multi-Stage Industrial Anomaly Detection

RETINA is a defect-detection pipeline developed at KU Leuven in partnership
with Flanders Make, targeted at industrial quality-control deployments across
multiple product lines (wood veneer, plastics, electrical components). It
combines a per-category PatchCore memory-bank detector with a GPT-4o router
that handles product identification, defect description, and supervised
refinement on ambiguous cases. Stage 1 covers all 15 MVTec AD categories with
a measured mean image AUROC of **0.9829** (13 of 15 categories ≥ 0.95). Stage
2 is a zero-training supervised refiner that uses operator-labeled examples
as in-context supervision, making it viable across customers without
per-customer retraining.

## Pipeline at a Glance

```
        ┌──────────────────────────────────────────────────────────────┐
        │                         image bytes                          │
        └──────────────────────────────┬───────────────────────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │ identify_product (gpt-4o-mini)│
                        │   session-cached 1h TTL       │
                        └──────────────┬──────────────┘
                                       │
                    ┌──────known?──────┴──────unknown?──────┐
                    │                                       │
     ┌──────────────▼──────────────┐          ┌─────────────▼────────────┐
     │  PatchCore (per-category)   │          │ zero_shot_detect (gpt-4o) │
     │  registry of 15 checkpoints │          │  reasoning + score        │
     └──────────────┬──────────────┘          └─────────────┬────────────┘
                    │ score                                 │
       ┌────────────┼────────────┐                          │
       │ <0.5       │ 0.5–0.9    │ ≥0.9                     │
       ▼            ▼            ▼                          │
    normal   stage2_refine   describe_defect                │
             (gpt-4o +       (gpt-4o)                       │
              in-context                                    │
              labels)                                       │
       │            │            │                          │
       └────────────┴────────────┴──────────┬───────────────┘
                                            │
                        ┌───────────────────▼────────────────────┐
                        │  InferenceResult → retina:results:{id} │
                        │  add to retina:al:pool if uncertain    │
                        └────────────────────────────────────────┘
```

- **identify_product** routes to the right specialist. Session-cached to keep
  per-image cost at ~$0.00015 only on cache miss.
- **PatchCore** is the numeric verdict for any of the 15 known product
  categories. Memory bank + k-NN; no GPT call unless the score is interesting.
- **describe_defect** turns a confirmed anomaly score into an operator-facing
  sentence. Gated by a low-score guard so it cannot hallucinate defects on
  clean images.
- **stage2_refine** runs only when Stage 1 confidence is ambiguous (score in
  [0.5, 0.9)). Confirms or rejects the flag and names the defect class,
  using operator-labeled examples as in-context supervision.
- **zero_shot_detect** is the fallback for products without a trained
  PatchCore checkpoint. Best-effort; flagged images go to expert review.

## Measured Performance

### Stage 1 — PatchCore on MVTec AD (15 categories)

Trained with identical pipeline: anomalib 2.3.3, 1-epoch memory bank
construction, RTX 3060 Laptop (6 GB). Source data in
[results/mvtec_stage1_results.csv](results/mvtec_stage1_results.csv).

| Category    | Image AUROC | Pixel AUROC | Wall clock |
|-------------|-------------|-------------|------------|
| bottle      | 1.0000      | 0.9856      | 132 s      |
| hazelnut    | 1.0000      | 0.9884      | 284 s      |
| leather     | 1.0000      | 0.9922      | 129 s      |
| tile        | 1.0000      | 0.9555      | 109 s      |
| metal_nut   | 0.9971      | 0.9870      | 101 s      |
| transistor  | 0.9950      | 0.9731      | 100 s      |
| capsule     | 0.9928      | 0.9900      | 110 s      |
| grid        | 0.9891      | 0.9815      | 128 s      |
| wood        | 0.9877      | 0.9317      | 120 s      |
| carpet      | 0.9872      | 0.9907      | 157 s      |
| cable       | 0.9867      | 0.9852      | 114 s      |
| zipper      | 0.9758      | 0.9817      | 125 s      |
| screw       | 0.9639      | 0.9891      | 197 s      |
| pill        | 0.9487      | 0.9817      | 150 s      |
| toothbrush  | 0.9194      | 0.9888      | 25 s       |

**Mean image AUROC: 0.9829.** 13 of 15 categories clear 0.95. Toothbrush is
the weakest category (small test set, small-extent defects — a known weak
spot for memory-bank methods). All 15 checkpoints live in `checkpoints/`
(~230 MB each, 7.4 GB total, `.ckpt` format — gitignored).

### Stage 2 — GPT-4o Supervised Refiner

Triggers only when the Stage 1 score falls in the uncertainty band
`[0.5, 0.9)`. Outside that band PatchCore is confident and no refiner is
needed.

- **Input:** image bytes, product_class, stage1_score, up to 5 operator-
  labeled examples pulled from `retina:labels:*` for the same product.
- **Model:** `gpt-4o`, temperature 0.1, max_tokens 300, JSON-mode response.
- **Output (Stage2Verdict):** `verdict ∈ {confirmed_anomaly,
  rejected_false_positive, uncertain}`, `defect_class`, `confidence`,
  `reasoning`.
- **Label fetch:** the worker reads recent `retina:labels:*` hashes and
  filters by product_class field at request time. Text-only context (label +
  defect_class per example) — images are not embedded in the few-shot prompt
  to keep cost bounded.

Verified end-to-end on `mvtec/pill/test/crack/000.png`: Stage 1 scored
0.5958, Stage 2 fired with `examples_used=5`, returned
`verdict=confirmed_anomaly, defect_class=contamination, confidence=0.85`.

### Cost per Inference

Per-image API cost depends on which branches fire:

| Path                                            | Calls                          | USD     |
|-------------------------------------------------|--------------------------------|---------|
| Normal path (score < 0.5), session-cache hit    | none                           | $0      |
| Normal path, cache miss                         | identify                       | $0.00015|
| High-confidence anomaly (score ≥ 0.9)           | identify + describe            | $0.00515|
| Uncertain (0.5 ≤ score < 0.9)                   | identify + describe + stage2   | $0.01015|
| Unknown product (zero-shot)                     | identify + zero_shot_detect    | $0.00515|

At 10,000 images/day with 5 % anomaly rate and typical session-cache hits,
projected spend is around **$45/day**. Identification cost dominates only
when the session cache is cold (product changeover).

## Architecture

### Components

- **[worker/](worker/)** — Python worker that consumes jobs from Redis,
  runs PatchCore + VLM router, writes `InferenceResult` back to Redis.
  Long-running process, single-threaded main loop, SIGTERM-safe.
- **[api/](api/)** — FastAPI wrapper on port 3001. Thin HTTP surface: upload
  endpoint pushes jobs to Redis, result endpoint polls Redis. No inference
  in-process.
- **[frontend/](frontend/)** — Next.js 14 dark-themed operator UI: submit
  page, label/annotation studio (polygon tool, runtime taxonomy extension),
  per-category performance dashboard.
- **[shared/schemas/](shared/schemas/)** — JSON schemas for `InferenceJob`,
  `InferenceResult`, `Label`. Source of truth for wire format.
- **[checkpoints/](checkpoints/)** — per-category PatchCore `.ckpt` files,
  lazy-loaded with a 2-model LRU (~1.7 GB resident per hot model).
  Gitignored.
- **[scripts/](scripts/)** — training, evaluation, CLI submit/watch,
  isolation tests.
- **[research/](research/)** — BGAD and Custom Push-Pull implementations.
  Not wired into the demo worker.

### Data Flow

1. Client POSTs an image to `POST /api/submit`. File is saved to
   `data/uploads/<job_id>.png`.
2. API writes `retina:images:<image_id>` with the file path (reverse lookup
   for later image serving) and XADDs an `InferenceJob` JSON to
   `retina:jobs:queue`.
3. Worker `XREADGROUP`s from the stream (consumer group `workers`), sets job
   status to `processing`.
4. Worker loads the image, runs `VLMRouter.identify_product` (gpt-4o-mini,
   cached via `retina:session:product_class` with 1-hour TTL).
5. If the product is known and has a checkpoint: `PatchCoreRegistry.get(...)`
   returns a hot model, worker runs inference, pulls `anomaly_score` and
   `anomaly_map`.
6. Routing:
   - `score < 0.5` → mark normal, no further VLM call.
   - `0.5 ≤ score < 0.9` → `stage2_refine` with labeled examples.
   - `score ≥ 0.9` → `describe_defect`.
7. If the product is unknown or has no checkpoint → `zero_shot_detect` via
   gpt-4o.
8. Worker builds `InferenceResult`, writes to `retina:results:{job_id}`
   (7-day TTL), adds to `retina:al:pool` if `uncertainty_score >
   threshold`, XACKs the stream entry.
9. Client polls `GET /api/result/{job_id}`; the endpoint blocks up to 30 s
   waiting for the Redis hash to appear.
10. Label UI reads `GET /api/labels/pool`, renders the queue; operator
    annotates and POSTs to `/api/labels/submit`. Label lands in
    `retina:labels:{image_id}` with 7-day TTL and feeds the next Stage 2
    call for that product.

### Design Rationale

**VLM as explainer, not primary detector.** Stage 1 ownership of the
numeric verdict stays with PatchCore because it produces a calibrated,
reproducible score grounded in the product's normal distribution. GPT-4o is
prone to priming bias ("you said there might be a defect, so I found one")
and is unsuitable as the verdict source. It is invoked only when it can add
value: identifying the product, describing a confirmed defect, or refining
ambiguous scores with labeled examples in context.

**Stage 2 as in-context supervised refiner, not BGAD.** BGAD works well on
MVTec but requires per-category labeled masks and a training run. For a
multi-customer deployment the training budget is hostile — every new
customer would need per-category labels and a fresh model. The GPT-4o
refiner instead consumes whatever labels the operator has produced so far,
as in-context examples. This traded AUROC for portability: no new model
files, no training pipeline, no per-customer mask budget. BGAD remains in
`research/` as a reference implementation if a deployment eventually
justifies the training cost.

**FastAPI instead of the Rust backend.** A Rust/Axum backend exists under
`backend/` but is unused in the current demo path. After the Phase 3
reorganization it accumulated compile errors unrelated to the demo, and
fixing them would not change any user-visible behavior. The FastAPI wrapper
at `api/main.py` is ~350 lines, maps 1:1 onto the Redis contract, and is
easy for a university reviewer to read top-to-bottom. The Rust backend can
be repaired later if production requires tower middleware, SQLx, or higher
per-request throughput.

## Quick Start

### Prerequisites

- Python 3.11 (conda environment recommended — the sprint used a `ml` env)
- Node.js 18+
- Redis 7+ (native install; the demo does not run Docker)
- NVIDIA GPU with ≥ 6 GB VRAM (Stage 1 loads up to two PatchCore models
  concurrently, peak ~1.7 GB each)
- OpenAI API key with gpt-4o and gpt-4o-mini access

### Install

```bash
# Python worker + shared deps
pip install -e worker/

# API wrapper deps
pip install -r api/requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### Configure

Copy the template and set at minimum `OPENAI_API_KEY` and `REDIS_URL`:

```bash
cp .env.example .env
$EDITOR .env
```

The Postgres / JWT entries in `.env.example` are vestigial (the demo does
not use Postgres). `REDIS_URL` should point at your local Redis, e.g.
`redis://localhost:6379`.

### Train PatchCore Checkpoints

```bash
# One category:
python scripts/train_anomalib.py --category bottle --model patchcore \
    --max-epochs 1 --output-dir ./checkpoints

# All 15 MVTec categories, writes results/mvtec_stage1_results.csv:
bash scripts/train_all_mvtec.sh
```

Memory bank construction runs in ~2 minutes per category on RTX 3060. The
15-category total completed in ~30 minutes during the sprint.

### Run the System

Four processes, each in its own terminal (or `nohup ... &`):

```bash
# Terminal 1 — Redis (skip if already running as a service)
redis-server

# Terminal 2 — ML worker
PYTHONPATH=worker/src REDIS_URL=redis://localhost:6379 \
    python -m retina_worker.main

# Terminal 3 — FastAPI on port 3001
bash scripts/run_api.sh

# Terminal 4 — Next.js frontend on port 3000
cd frontend && npm run dev
```

Health check:

```bash
redis-cli ping                          # PONG
curl -sf http://localhost:3001/health   # {"status":"ok","redis":"up"}
curl -sI http://localhost:3000          # HTTP/1.1 200 OK
```

Open [http://localhost:3000](http://localhost:3000). The Submit page runs
a full round-trip, the Label page serves the active learning queue, the
Model Performance page renders the measured results.

### CLI Demo (No Frontend)

```bash
JOB=$(python scripts/submit_job.py --image mvtec/bottle/test/broken_large/000.png --quiet)
python scripts/watch_results.py --job $JOB
```

`submit_job.py` enqueues the job directly to Redis using the same wire
format as the FastAPI submit endpoint. `watch_results.py` polls the result
hash and pretty-prints verdict, VLM description, routing reason, and cost.

## Repository Layout

```
RETINA/
├── api/                        FastAPI wrapper — the demo HTTP surface
│   ├── main.py                 8 endpoints, ~350 lines
│   └── requirements.txt
├── backend/                    Rust/Axum backend — not used for the demo
├── checkpoints/                PatchCore .ckpt files (gitignored, 7.4 GB)
├── data/uploads/               Image files uploaded via the API
├── docs/
│   ├── integration_plan.md     Worker rewrite blueprint
│   ├── vlm_router_design.md    Specialist-first / explainer-second pattern
│   ├── fastapi_revival_plan.md Why FastAPI was chosen over the Rust backend
│   ├── archive/                Session records + prior audits
│   └── guides/, figures/       Supplementary docs
├── frontend/                   Next.js 14 operator UI
│   └── src/
│       ├── app/                page.tsx, submit/, label/, results/,
│       │                       demo/, model-performance/
│       └── lib/
│           ├── api.ts          All frontend→API calls; single base URL
│           └── taxonomies.ts   Per-product defect taxonomies (MVTec + Decospan)
├── mvtec/                      MVTec AD dataset (gitignored)
├── research/
│   ├── supervised/BGAD/        Reference implementation — not wired
│   └── supervised/Custom_Model_Push_Pull/
├── results/
│   └── mvtec_stage1_results.csv  15-category training output
├── scripts/
│   ├── train_anomalib.py       Train one MVTec category
│   ├── train_all_mvtec.sh      Loop over 15 categories
│   ├── eval_anomalib.py        Evaluate a checkpoint
│   ├── submit_job.py           CLI submit
│   ├── watch_results.py        CLI result printer
│   ├── run_api.sh              Uvicorn launcher
│   ├── test_patchcore_registry.py  Isolation test for LRU registry
│   └── test_vlm_router.py      Isolation test for the VLM router
├── shared/schemas/
│   ├── job.json                InferenceJob wire format
│   ├── result.json             InferenceResult wire format
│   └── label.json              Label wire format
├── worker/
│   ├── pyproject.toml
│   └── src/retina_worker/
│       ├── main.py             Entry point + logging setup
│       ├── worker.py           Main loop + _run_inference routing
│       ├── redis_client.py     Stream consumer group + result writes
│       ├── schemas.py          Pydantic mirror of shared/schemas
│       ├── config.py           Settings (OpenAI key, thresholds, TTLs)
│       └── models/
│           ├── vlm_router.py   identify / describe / zero_shot / stage2_refine
│           ├── patchcore_registry.py  LRU-cached per-category loader
│           ├── patchcore_real.py      Anomalib / ResNet50 fallback
│           └── *_stub.py              Legacy stubs — retained for backward compat
├── .env.example
├── CLAUDE.md                   Operating contract (invariants + protocols)
├── docker-compose.yml          Wires Postgres/Redis/services — not the demo path
└── README.md                   (this file)
```

## Known Limitations

1. **Toothbrush image AUROC 0.9194.** The weakest of the 15 categories.
   Small test set (60 images) and small-extent defects hurt memory-bank
   methods. Candidate for Stage 2 override in practice.
2. **GPT-4o-mini over-classifies unknowns.** When presented with an object
   outside the 15 trained categories, the identifier still returns a known
   category with high confidence rather than falling back to `unknown`.
   Documented in the VLM router design note; mitigation is prompt-side
   pressure and a future confidence gate.
3. **Reference-dependent defects are an intrinsic blind spot.** Classes
   like `cable/cable_swap` (wires physically intact, just in the wrong
   order) cannot be caught by zero-shot VLM reasoning. Locked in as test
   `T6_reference_dependent_defect_limitation` so prompt changes do not
   accidentally regress other cases while trying to "fix" this one.
4. **BGAD exists but is not wired.** `research/supervised/BGAD/` contains
   a working reference implementation with published AUROC 0.930 on MVTec.
   It is not plumbed into the worker. Decision: multi-customer portability
   wins over per-category AUROC (see Design Rationale).
5. **Rust backend does not compile.** `backend/` has drift from the Phase 3
   reorganization. The demo runs entirely through `api/main.py`. The Rust
   code is preserved rather than deleted because production may eventually
   need what it was written for.
6. **No durable label storage.** Labels land in `retina:labels:{id}` with a
   7-day TTL. There is no Postgres persistence, no export pipeline, no
   versioning. Good enough for the active learning demo; insufficient for
   audit or retraining on historical labels.
7. **Session cache bleed across product changes.** `identify_product` is
   cached under `retina:session:product_class` with a 1-hour TTL to avoid
   per-image cost. If a demo switches product mid-session without clearing
   the cache, the next image routes through the previous product's
   PatchCore and scores badly. Operator mitigation:
   `redis-cli DEL retina:session:product_class retina:session:product_confidence`.

## Roadmap

1. **Durable label store.** Back `retina:labels:*` with a sqlite or
   Postgres table so labels survive TTL and power retraining.
2. **Accumulate per-customer labels.** Fill the Stage 2 in-context window
   with the N most informative recent labels per product (not just most
   recent).
3. **Confidence-gated identify.** Return `unknown` when gpt-4o-mini
   confidence is below a threshold rather than forcing a known category.
4. **Wire BGAD as an optional pixel-level Stage 2** for customers where
   training cost is acceptable.
5. **Retraining pipeline.** Nightly memory-bank refresh from labeled
   normals; diff coreset, keep rollback.
6. **Repair Rust backend + Docker packaging** for production throughput
   and deployment reproducibility.
7. **Multi-customer deployment.** One-config-per-customer, per-tenant
   taxonomy isolation, per-tenant label storage.

## Acknowledgments

This work is a KU Leuven research project supervised by
Prof. dr. Mathias Verbeke, developed in partnership with Flanders Make and
with industrial context from BMSvision, Deceuninck, and Vandemoortele.

- **PatchCore:** Roth et al., *Towards Total Recall in Industrial Anomaly
  Detection*, CVPR 2022.
- **anomalib:** Akcay et al., Intel Corporation. RETINA uses anomalib 2.3.3.
- **MVTec AD:** Bergmann et al., *MVTec AD — A Comprehensive Real-World
  Dataset for Unsupervised Anomaly Detection*, CVPR 2019.
- **BGAD:** Yao et al., *Explicit Boundary Guided Semi-Push-Pull
  Contrastive Learning for Supervised Anomaly Detection*, CVPR 2023
  (reference implementation in `research/supervised/BGAD/`).
