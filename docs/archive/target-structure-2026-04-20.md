# RETINA Target Repository Structure — 2026-04-20

**Phase 2 of 3 — DESIGN ONLY. No files moved until Phase 3 is approved.**

This document describes the target structure, the moves required, and the rationale.
It also includes a prioritised list of code bugs to fix alongside the reorganisation.

---

## 0. Guiding Principles

1. **Do not break the production stack.** Every `docker-compose up --build` must
   continue to work after each service is reorganised. Verify with health check.

2. **Absolute rules (from CLAUDE.md §1.2 and the audit):**
   - Never modify `shared/schemas/*.json` without updating Rust + Python in order
   - Never modify canvas draw handlers in `label/page.tsx`
   - Never rename Dutch defect class labels
   - Never delete business logic — only move it
   - Never proceed to Phase 3 without the word "APPROVED" or "proceed"

3. **The dual-backend decision defaults to Rust.** CLAUDE.md, docker-compose, and all
   deployment infrastructure target the Rust backend. The FastAPI backend (`src/backend/`)
   is treated as legacy/archive unless explicitly reversed.

---

## 1. Target Directory Tree

```
RETINA/
├── CLAUDE.md                    (stays at root — operating contract)
├── README.md                    (REWRITE — currently documents wrong backend)
├── docker-compose.yml           (stays — fix secrets + add .env.example)
├── .env.example                 (NEW — template from CLAUDE.md §7.1)
├── .gitignore                   (stays)
│
├── backend/                     (Rust/Axum — production backend, unchanged)
│   └── src/
│       ├── main.rs
│       ├── config.rs            (fix: AL_STAGE2_THRESHOLD default 100→200)
│       ├── error.rs
│       ├── auth/
│       ├── routes/
│       │   ├── health.rs
│       │   ├── auth.rs          (fix: use config.jwt_secret, not env::var)
│       │   ├── images.rs        (add: /upload endpoint; move routing logic to worker)
│       │   ├── labels.rs
│       │   ├── system.rs
│       │   └── mod.rs
│       ├── services/
│       │   ├── redis.rs         (fix: document actual key names in CLAUDE.md)
│       │   ├── image_storage.rs
│       │   ├── alerts.rs
│       │   └── mod.rs
│       ├── db/
│       └── models/
│
├── worker/                      (Python ML worker — production, unchanged)
│   └── src/retina_worker/
│       ├── config.py            (fix: debug_mode default False, delay default 0)
│       ├── schemas.py
│       ├── worker.py            (stays — authoritative worker loop)
│       ├── worker_dual.py       (REMOVE — dead code with broken enum)
│       ├── redis_client.py
│       ├── main.py
│       └── models/
│           ├── base.py
│           ├── factory.py
│           ├── patchcore_real.py
│           ├── patchcore_stub.py
│           ├── gpt4v_detector.py
│           ├── winclip_stub.py
│           ├── pushpull_stub.py  (fix: DEFECT_CATEGORIES → Dutch canonical names)
│           └── __init__.py
│
├── frontend/                    (Next.js — production, unchanged)
│   └── src/
│       ├── lib/
│       │   └── api.ts           (fix: fallback URL localhost:3001; add Rust routes)
│       ├── app/
│       │   ├── page.tsx
│       │   ├── submit/
│       │   ├── label/
│       │   ├── results/
│       │   ├── demo/
│       │   └── model-performance/
│       │       └── page.tsx     (CREATE — linked in NavHeader, currently 404)
│       └── components/
│
├── shared/                      (JSON Schema contracts — do not modify contents)
│   └── schemas/
│       ├── job.json             (fix: add "gpt4v" to model_type enum)
│       ├── result.json
│       └── label.json
│
├── research/                    (NEW — was Unsupervised_Models/ + Supervised_Models/)
│   ├── README.md                (NEW — explains what is here and what is wired)
│   ├── unsupervised/
│   │   ├── AdaCLIP/             (was Unsupervised_Models/AdaCLIP/)
│   │   ├── PatchCore/           (was Unsupervised_Models/PatchCore/)
│   │   ├── PaDiM/               (was Unsupervised_Models/PaDiM/)
│   │   ├── WinCLIP/             (was Unsupervised_Models/WinCLIP/)
│   │   └── unsupervisedAnomalyService.py
│   └── supervised/
│       ├── BGAD/                (was Supervised_Models/BGAD/)
│       └── Custom_Model_Push_Pull/  (was Supervised_Models/Custom_Model_Push_Pull/)
│
├── notebooks/                   (NEW — Jupyter notebooks)
│   ├── demo.ipynb
│   ├── efficientad.ipynb
│   ├── efficientad_fixed.ipynb
│   └── patchcore.ipynb
│
├── scripts/                     (stays — add root-level scripts)
│   ├── nightly_retrain.py
│   ├── verify_bgad_weights.py
│   ├── evaluate_model.py        (was root-level)
│   ├── fast_local_train.py      (was root-level)
│   ├── merge_datasets.py        (was root-level)
│   ├── preflight_check.py       (was root-level — update for Rust stack)
│   └── validate_dataset.py      (was root-level)
│
├── docs/
│   ├── archive/                 (session handoff notes — keep for history)
│   │   ├── audit-2026-04-20.md  (this audit)
│   │   ├── target-structure-2026-04-20.md (this file)
│   │   ├── BGAD_CODE_REFERENCE.md
│   │   ├── BGAD_COMPLETION_SUMMARY.md
│   │   ├── BGAD_IMPLEMENTATION.md
│   │   ├── BGAD_QUICK_START.md
│   │   ├── CASCADE_IMPLEMENTATION_COMPLETE.md
│   │   ├── CASCADE_INTEGRATION_TESTS.md
│   │   ├── CASCADE_ROUTER_GUIDE.md
│   │   ├── CASCADE_TO_ANNOTATION_GUIDE.md
│   │   ├── CRON_SETUP.md
│   │   ├── GPT4V_ARCHITECTURE_DIAGRAMS.md
│   │   ├── GPT4V_DELIVERY_CHECKLIST.md
│   │   ├── GPT4V_IMPLEMENTATION_SUMMARY.md
│   │   ├── GPT4V_INTEGRATION.md
│   │   ├── GPT4V_QUICK_REFERENCE.md
│   │   ├── NIGHTLY_RETRAIN_README.md
│   │   ├── QUICKSTART_GPT4V.md
│   │   ├── STEP3_FINAL_SUMMARY.md
│   │   └── STEP3_INTEGRATION_COMPLETE.md
│   ├── guides/
│   │   └── DATASET_GUIDE.md     (was root-level)
│   └── figures/
│       ├── bgad_results.png
│       ├── pipeline_demo_results.png
│       ├── sample_predictions.png
│       └── training_history.png
│
├── legacy/                      (NEW — was src/; FastAPI backend preserved here)
│   └── fastapi_backend/         (was src/backend/ — not deleted, just isolated)
│       ├── app.py
│       ├── config.py
│       ├── __init__.py
│       └── services/
│
└── mvtec/                       (stays — dataset directory, if present)
```

---

## 2. Files to DELETE (not move)

These files are dead code, violate CLAUDE.md rules, or document an obsolete system.

| File | Reason |
|------|--------|
| `requirements.txt` | CLAUDE.md §4.4: "Never add to requirements.txt. If you see one, delete it." |
| `worker_dual.py` | Dead code; broken enum crashes on import; `main.py` never uses it |
| `test_cascade_router.py` | Tests FastAPI cascade router which is not deployed |
| `FASTAPI_INTEGRATION.md` | Documents integration with `src/backend/` — now legacy |
| `GPT4V_FASTAPI_INTEGRATION.md` | Documents GPT-4V wiring into FastAPI — now obsolete |

---

## 3. Code Fixes Bundled with Reorganisation

These are bug fixes, not just moves. Execute in the order listed to avoid cascade failures.

### 3.1 shared/schemas/job.json (FIRST — schema is ground truth)

Add `"gpt4v"` to the `model_type` enum:
```json
"enum": ["patchcore", "padim", "winclip", "gpt4v", "pushpull"]
```

### 3.2 backend/src/config.rs

Change default from `100` to `200`:
```rust
active_learning_stage2_threshold: env::var("AL_STAGE2_THRESHOLD")
    .unwrap_or_else(|_| "200".to_string())
```

### 3.3 backend/src/routes/auth.rs

Replace direct `env::var` with config:
- Thread `config.jwt_secret` through `AppState` (it already has a `config` field)
- Replace the inline `std::env::var("JWT_SECRET").unwrap_or_else(...)` call

### 3.4 frontend/src/lib/api.ts

Fix fallback URL:
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
```

### 3.5 worker/src/retina_worker/config.py

Fix production-unsafe defaults:
```python
debug_mode: bool = False
mock_inference_delay_ms: int = 0
```

### 3.6 worker/src/retina_worker/models/pushpull_stub.py

Fix DEFECT_CATEGORIES to use Dutch canonical names:
```python
DEFECT_CATEGORIES = [
    "krassen",   # scratches
    "deuk",      # dent
    "vlekken",   # stains
    "barst",     # crack
    "open voeg", # open joint
]
```

### 3.7 docker-compose.yml

Remove hardcoded secrets (replace with env var references):
```yaml
# Line 45
POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
# Line 109
JWT_SECRET: ${JWT_SECRET}
```

### 3.8 .env.example (NEW FILE)

Create from CLAUDE.md §7.1 template so developers can onboard without reading CLAUDE.md.

### 3.9 frontend/src/app/model-performance/page.tsx (NEW FILE)

Create minimal page — NavHeader links to it, it currently 404s.
Content: show benchmark table from CLAUDE.md §6.1 as a static reference page.
(Full dynamic model performance page is a separate feature.)

### 3.10 CLAUDE.md §1.3 Redis Key Table

Update to match actual code:
```
retina:jobs:queue          Stream    Pending jobs (XREADGROUP)  ← was: retina:queue:inference
retina:al:pool             SortedSet Active learning pool       ← was: retina:labeling:pool
```
Consumer group name: `workers` ← was: `retina-workers`

---

## 4. What This Reorganisation Does NOT Do

To be explicit about scope:

- Does **not** implement cascade routes in the Rust backend (separate feature)
- Does **not** wire BGAD or Push-Pull real models into production worker (separate feature)
- Does **not** implement the `/model-performance` page with live metrics (only a stub)
- Does **not** implement the image upload endpoint (separate feature)
- Does **not** implement `retina:system:stage` key lifecycle (separate feature)
- Does **not** change the active learning uncertainty sampling from score-based to entropy-based
- Does **not** migrate database queries to `sqlx::migrate!`

---

## 5. Service-by-Service Execution Order for Phase 3

When approved, execute in this order. Each step ends with a verification command.

```
Step 1: shared/schemas/job.json — add gpt4v enum value
        Verify: diff shared/schemas/job.json | grep gpt4v

Step 2: backend/src/config.rs — threshold fix
        Verify: cd backend && cargo check

Step 3: backend/src/routes/auth.rs — JWT secret via config
        Verify: cd backend && cargo check

Step 4: worker/src/retina_worker/config.py — debug defaults
        Verify: cd worker && python -m mypy src/

Step 5: worker/src/retina_worker/models/pushpull_stub.py — Dutch names
        Verify: cd worker && python -m mypy src/

Step 6: Delete worker_dual.py
        Verify: cd worker && python -m mypy src/

Step 7: frontend/src/lib/api.ts — fallback URL fix
        Verify: cd frontend && npx tsc --noEmit

Step 8: Create frontend/src/app/model-performance/page.tsx
        Verify: cd frontend && npx tsc --noEmit && npm run build

Step 9: docker-compose.yml — remove hardcoded secrets, create .env.example
        Verify: docker compose config (with .env present)

Step 10: Move markdown files to docs/archive/
         Verify: ls *.md at root — only CLAUDE.md and README.md should remain

Step 11: Move Unsupervised_Models/ → research/unsupervised/
         Move Supervised_Models/ → research/supervised/
         Verify: ls research/

Step 12: Move notebooks (*.ipynb) → notebooks/
         Verify: ls notebooks/

Step 13: Move root scripts → scripts/
         Verify: ls scripts/

Step 14: Move src/backend/ → legacy/fastapi_backend/
         Verify: docker compose up --build (all services healthy)

Step 15: Move figures (*.png) → docs/figures/
         Verify: git status clean

Step 16: Delete requirements.txt, test_cascade_router.py, FASTAPI_INTEGRATION.md,
         GPT4V_FASTAPI_INTEGRATION.md
         Verify: git status

Step 17: Update CLAUDE.md §1.3 Redis key table, §0.2 status table, §0.3 bug list
         Verify: review diff

Step 18: Rewrite README.md (documents Rust backend, correct ports, correct start commands)
         Verify: read README.md

Step 19: Full stack smoke test
         docker compose up --build
         curl http://localhost:3001/health
```

---

## 6. Risk Assessment

| Step | Risk | Mitigation |
|------|------|-----------|
| Delete worker_dual.py | Low — never imported | Confirm with grep before delete |
| Move src/backend/ | Medium — api.ts still calls its routes | api.ts fix in Step 7 is prerequisite |
| docker-compose.yml secret removal | Medium — breaks deployment without .env | Create .env.example in same step |
| Rename Unsupervised_Models/ | Low — not imported by any production code | Verify with grep |
| shared/schemas/job.json change | Low — adds field, no removal | Update Rust + Python after |

---

*Phase 2 design complete.*
*Awaiting APPROVED or "proceed" to execute Phase 3.*
