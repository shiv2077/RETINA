# RETINA — Multi-Stage Industrial Anomaly Detection

> Two-stage anomaly detection pipeline for factory inspection.
> Stage 1: PatchCore (unsupervised) — Stage 2: Push-Pull/BGAD (supervised via active learning).
> KU Leuven / Flanders Make research project.

![Rust](https://img.shields.io/badge/backend-Rust%2FAxum-orange)
![Python 3.10+](https://img.shields.io/badge/worker-Python%203.10-blue)
![Next.js 14](https://img.shields.io/badge/frontend-Next.js%2014-black)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## Architecture

```
Camera → [Stage 1: PatchCore/GPT-4o] → PASS or ANOMALY_FLAGGED
                                               │
                           [Stage 2: Push-Pull/BGAD] → Defect class + confidence
                                               │
                           [Active Learning] → Expert labels uncertain samples
                                               │
                           [Retrain loop] → Stage 2 improves over time
```

**Services:**

| Service | Language | Port | Purpose |
|---------|----------|------|---------|
| Backend | Rust/Axum | 3001 | REST API, job queuing, active learning state |
| Worker | Python | — | ML inference, consumes Redis jobs |
| Frontend | Next.js | 3000 | Dashboard, submission, expert review |
| Redis | — | 6379 (internal) | Job queue (Streams), results, AL pool |
| PostgreSQL | — | 5432 (internal) | Users, anomaly records, labels |

---

## Quick Start

**Prerequisites:** Docker, Docker Compose, `.env` file (see `.env.example`)

```bash
# 1. Copy environment template
cp .env.example .env
# Edit .env — set POSTGRES_PASSWORD, JWT_SECRET, OPENAI_API_KEY

# 2. Start all services
docker compose up --build

# 3. Verify backend is healthy
curl http://localhost:3001/health

# 4. Open the UI
open http://localhost:3000
```

---

## Repository Structure

```
backend/          Rust/Axum REST API (production)
worker/           Python ML inference worker (production)
frontend/         Next.js 14 UI (production)
shared/schemas/   JSON Schema contracts — source of truth for all wire formats
research/         Standalone research models (NOT wired into pipeline)
notebooks/        Jupyter notebooks for experimentation
scripts/          Training and evaluation scripts
docs/             Guides, figures, session archive
legacy/           Archived FastAPI backend (superseded by Rust backend)
```

---

## Pipeline Details

### Stage 1 — Unsupervised Detection

- **Primary:** PatchCore — memory-bank k-NN with WideResNet-50 features
- **Cold-start fallback:** GPT-4o vision API (zero-shot, activates when no memory bank exists)
- Output: `anomaly_score ∈ [0,1]`, `is_anomaly: bool`, `heatmap`

### Stage 2 — Supervised Classification

- Activates when labeled sample count ≥ `AL_STAGE2_THRESHOLD` (default: 200)
- **Target model:** Push-Pull contrastive learning (no masks needed, 100–200 samples)
- **Better model:** BGAD (AUC 0.930, requires segmentation masks)
- Output: `defect_class` (Dutch canonical name), `confidence`, `bounding_box`

### Active Learning Loop

1. Stage 1 flags uncertain samples → adds to `retina:al:pool` sorted set
2. Expert reviews via `/label` page → submits label to `POST /labels/submit`
3. Label count increments → Stage 2 activates at threshold
4. Stage 2 retrains in background → improves classification

---

## Model Benchmarks

| Model | Dataset | Image AUROC | Type |
|-------|---------|-------------|------|
| PatchCore | MVTec AD | 0.895 | Memory bank |
| PaDiM | MVTec AD | 0.884 | Gaussian |
| WinCLIP | MVTec AD | 0.856 | Zero-shot |
| BGAD | MVTec AD | 0.930 | Supervised |
| Push-Pull | Decospan | 0.860 | Supervised |

Any new model must exceed the relevant baseline by AUC > 0.02 before integration.

---

## Development

### Backend (Rust)
```bash
cd backend
cargo check      # Verify compilation
cargo test       # Run tests
```

### Worker (Python)
```bash
cd worker
pip install -e ".[dev]"
python -m mypy src/
python -m pytest
python -m ruff check src/
```

### Frontend (Next.js)
```bash
cd frontend
npm install
npx tsc --noEmit   # Type check
npm run build      # Production build
npm run dev        # Dev server on :3000
```

### Full Stack
```bash
docker compose up --build
curl http://localhost:3001/health   # Must return 200
```

---

## Key Operational Notes

- **Secrets:** Never hardcode — always use `.env` (see `.env.example`)
- **Schema changes:** Edit `shared/schemas/*.json` first, then Rust, then Python (CLAUDE.md §1.2)
- **Redis keys:** All documented in CLAUDE.md §1.3 — never invent undocumented keys
- **ML model loading:** Load once at worker startup — never per-request (OOM risk)
- **Defect categories:** Dutch names are canonical (`krassen`, `deuk`, etc.) — see CLAUDE.md §6.2

---

*Flanders Make / KU Leuven — MPro + ProductionS Research Groups*
*Contact: dries.vandaele@kuleuven.be*
