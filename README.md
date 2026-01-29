# RETINA

## Multi-Stage Visual Anomaly Detection with Active Learning

> **Research Project** | KU Leuven | Department of Computer Science
>
> **Timeline**: October 2024 – Present
>
> **Status**: Early-stage development (internship prototype)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [System Architecture](#system-architecture)
- [Implementation Status](#implementation-status)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [ML Models](#ml-models)
- [Active Learning Workflow](#active-learning-workflow)
- [Development](#development)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

RETINA (**R**obust **E**valuation **T**echniques for **I**ndustrial a**N**omaly **A**nalysis) is a multi-stage anomaly detection system designed for industrial visual inspection. The system combines unsupervised learning approaches with human-in-the-loop active learning to progressively improve detection accuracy.

### Key Features

- **Two-stage detection pipeline**: Start with zero-shot methods, refine with labeled data
- **Multiple model architectures**: PatchCore, PaDiM, WinCLIP, and PushPull networks
- **Active learning integration**: Uncertainty-based sample selection for efficient labeling
- **Distributed processing**: Redis-based job queue for scalable inference
- **Research-friendly**: Comprehensive logging, experiment tracking, reproducible results

---

## Research Motivation

Industrial anomaly detection presents unique challenges:

1. **Label scarcity**: Defects are rare; collecting labeled examples is expensive
2. **Distribution shift**: Production conditions change over time
3. **Real-time requirements**: Inspection must keep pace with manufacturing

Our approach addresses these through a **progressive learning framework**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETINA Pipeline                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Stage 1: Zero-shot Detection                                               │
│   ┌─────────────────────────────────────────────────────────────┐           │
│   │  • WinCLIP: Vision-language model, no training required      │           │
│   │  • PatchCore: Memory bank from ImageNet features             │           │
│   │  • Output: Anomaly scores + uncertainty estimates            │           │
│   └─────────────────────────────────────────────────────────────┘           │
│                              │                                               │
│                              ▼                                               │
│   Active Learning: Uncertainty Sampling                                      │
│   ┌─────────────────────────────────────────────────────────────┐           │
│   │  • Select high-uncertainty samples for human review          │           │
│   │  • Prioritize samples near decision boundary                 │           │
│   │  • Track labeling progress and update pool                   │           │
│   └─────────────────────────────────────────────────────────────┘           │
│                              │                                               │
│                              ▼                                               │
│   Stage 2: Supervised Refinement                                             │
│   ┌─────────────────────────────────────────────────────────────┐           │
│   │  • PushPull: Contrastive learning with labeled examples      │           │
│   │  • Fine-tuned PatchCore: Memory bank from actual defects     │           │
│   │  • Calibrated thresholds based on labeled data               │           │
│   └─────────────────────────────────────────────────────────────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

RETINA follows a microservices architecture for flexibility and scalability:

```
                                    ┌─────────────────────────────────────┐
                                    │           Next.js Frontend          │
                                    │         (React 18 + Tailwind)       │
                                    │                                     │
                                    │  • Dashboard & monitoring           │
                                    │  • Image submission                 │
                                    │  • Result visualization             │
                                    │  • Active learning interface        │
                                    └───────────────┬─────────────────────┘
                                                    │ HTTP
                                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              Rust Backend (Axum)                               │
│                                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   /health    │  │  /api/images │  │  /api/labels │  │  /api/system │       │
│  │              │  │              │  │              │  │              │       │
│  │  Liveness    │  │  Submit      │  │  Add label   │  │  Stats       │       │
│  │  Readiness   │  │  Status      │  │  Query pool  │  │  Metrics     │       │
│  │              │  │  Results     │  │  Progress    │  │  Config      │       │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                                                │
└─────────────────────────────────────────────�──────────────────────────────────┘
                                          │ Redis Protocol
                                          ▼
                    ┌─────────────────────────────────────────────────┐
                    │                   Redis 7                        │
                    │                                                  │
                    │  Streams:  retina:jobs         (job queue)       │
                    │  Hashes:   retina:result:{id}  (inference out)   │
                    │  Hashes:   retina:job:{id}     (job metadata)    │
                    │  Sorted:   retina:al:pool      (AL candidates)   │
                    │  Hashes:   retina:labels       (human labels)    │
                    │  Strings:  retina:stats:*      (system metrics)  │
                    │                                                  │
                    └─────────────────────────────────────────┬────────┘
                                                              │ Redis Protocol
                                                              ▼
                    ┌─────────────────────────────────────────────────────────┐
                    │                  Python ML Worker                        │
                    │                                                          │
                    │  ┌────────────────────────────────────────────────────┐ │
                    │  │  Consumer Group: "workers"                          │ │
                    │  │  Consumes: XREADGROUP from retina:jobs              │ │
                    │  │  Produces: HSET to retina:result:{id}               │ │
                    │  └────────────────────────────────────────────────────┘ │
                    │                                                          │
                    │  Models (pluggable):                                     │
                    │  ┌────────────┐ ┌────────────┐ ┌────────────┐           │
                    │  │ PatchCore  │ │  WinCLIP   │ │  PushPull  │           │
                    │  │  (stub)    │ │   (stub)   │ │   (stub)   │           │
                    │  └────────────┘ └────────────┘ └────────────┘           │
                    │                                                          │
                    └──────────────────────────────────────────────────────────┘
```

### Component Communication

All inter-service communication uses **Redis Streams** with consumer groups:

```
Job Submission Flow:
────────────────────
1. Frontend → POST /api/images/submit → Backend
2. Backend → XADD retina:jobs → Redis
3. Worker → XREADGROUP workers → Redis (blocking read)
4. Worker processes image, runs inference
5. Worker → HSET retina:result:{job_id} → Redis
6. Worker → XACK retina:jobs → Redis
7. Frontend polls GET /api/images/{id}/result → Backend
```

---

## Implementation Status

> **Important**: This project is in early development. The table below distinguishes
> between completed work and planned features.

### ✅ Implemented

| Component | Feature | Status | Notes |
|-----------|---------|--------|-------|
| **Backend** | API scaffold | ✅ Complete | Axum 0.7 with async routes |
| **Backend** | Redis integration | ✅ Complete | Streams, Hashes, Sorted Sets |
| **Backend** | Health endpoints | ✅ Complete | Liveness + readiness checks |
| **Backend** | Image submission | ✅ Complete | Stage auto-selection |
| **Backend** | Result retrieval | ✅ Complete | Poll-based with caching |
| **Backend** | Label management | ✅ Complete | CRUD for human annotations |
| **Worker** | Redis consumer | ✅ Complete | Consumer group pattern |
| **Worker** | Model interface | ✅ Complete | Abstract base + factory |
| **Worker** | Model stubs | ✅ Complete | Deterministic test outputs |
| **Worker** | Worker loop | ✅ Complete | Graceful shutdown |
| **Frontend** | Dashboard | ✅ Complete | System status overview |
| **Frontend** | Image upload | ✅ Complete | Drag-drop + model selection |
| **Frontend** | Result viewer | ✅ Complete | Score visualization |
| **Frontend** | Labeling UI | ✅ Complete | Active learning interface |
| **Infra** | Docker configs | ✅ Complete | Multi-stage builds |
| **Infra** | Compose setup | ✅ Complete | Local development |

### 🚧 In Progress / Planned

| Component | Feature | Status | Target |
|-----------|---------|--------|--------|
| **ML** | PatchCore implementation | 🚧 Planned | v0.2 |
| **ML** | WinCLIP implementation | 🚧 Planned | v0.2 |
| **ML** | PaDiM implementation | 🚧 Planned | v0.3 |
| **ML** | PushPull implementation | 🚧 Planned | v0.4 |
| **ML** | Anomalib integration | 🚧 Planned | v0.2 |
| **AL** | Uncertainty estimation | 🚧 Planned | v0.2 |
| **AL** | Sample selection strategies | 🚧 Planned | v0.3 |
| **Backend** | Authentication | 🚧 Planned | v0.3 |
| **Backend** | Experiment tracking | 🚧 Planned | v0.4 |
| **Frontend** | Anomaly heatmaps | 🚧 Planned | v0.2 |
| **Frontend** | Batch operations | 🚧 Planned | v0.3 |
| **Infra** | Kubernetes manifests | 🚧 Planned | v0.5 |
| **Infra** | GPU scheduling | 🚧 Planned | v0.5 |
| **Eval** | Benchmark suite | 🚧 Planned | v0.3 |
| **Eval** | MVTec AD integration | 🚧 Planned | v0.2 |

---

## Getting Started

### Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Rust** 1.75+ (for backend development)
- **Python** 3.10+ (for worker development)
- **Node.js** 20+ (for frontend development)

### Quick Start with Docker Compose

```bash
# Clone the repository
git clone https://github.com/kuleuven/retina.git
cd retina

# Copy environment template
cp .env.example .env

# Build and start all services
docker compose up --build

# Services will be available at:
# - Frontend:  http://localhost:3000
# - Backend:   http://localhost:3001
# - Redis:     localhost:6379
```

### Local Development Setup

#### Backend (Rust)

```bash
cd backend

# Install dependencies and build
cargo build

# Run with hot-reload (requires cargo-watch)
cargo install cargo-watch
cargo watch -x run

# Run tests
cargo test
```

#### Worker (Python)

```bash
cd worker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run the worker
python -m retina_worker.main

# For ML development (with PyTorch):
pip install -e ".[ml]"
```

#### Frontend (Next.js)

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

---

## API Reference

### Health Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic liveness check |
| `/health/ready` | GET | Readiness with dependency checks |

### Image Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/images/submit` | POST | Submit image for inference |
| `/api/images/{id}/status` | GET | Get job status |
| `/api/images/{id}/result` | GET | Get inference result |

#### Submit Image Request

```json
{
  "image_data": "base64-encoded-image-data",
  "model_type": "patchcore",
  "metadata": {
    "filename": "sample_001.png",
    "product_type": "transistor"
  }
}
```

#### Result Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "anomaly_score": 0.73,
  "is_anomalous": true,
  "confidence": 0.89,
  "model_used": "patchcore",
  "pipeline_stage": 1,
  "processing_time_ms": 156,
  "anomaly_map": "base64-encoded-heatmap"
}
```

### Label Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/labels` | POST | Add human annotation |
| `/api/labels/pool` | GET | Get active learning pool |
| `/api/labels/stats` | GET | Get labeling statistics |
| `/api/labels/{job_id}` | GET | Get label for specific job |

#### Add Label Request

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_defect": true,
  "defect_category": "scratch",
  "notes": "Visible scratch on surface"
}
```

### System Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/stats` | GET | System statistics |
| `/api/system/config` | GET | Current configuration |

---

## ML Models

### Supported Architectures

#### PatchCore (Roth et al., 2022)

Memory-bank approach using pretrained CNN features.

- **Pros**: State-of-the-art on MVTec AD, no training required
- **Cons**: Memory-intensive, slow inference at scale
- **Stage**: 1 (unsupervised) and 2 (with defect bank)

#### PaDiM (Defard et al., 2021)

Probabilistic approach using multivariate Gaussian modeling.

- **Pros**: Fast inference, interpretable scores
- **Cons**: Assumes Gaussian distribution of features
- **Stage**: 1 (unsupervised)

#### WinCLIP (Jeong et al., 2023)

Vision-language model using CLIP for zero-shot detection.

- **Pros**: True zero-shot, natural language descriptions
- **Cons**: Lower accuracy than feature-based methods
- **Stage**: 1 (unsupervised, zero-shot)

#### PushPull (Work in Progress)

Contrastive learning approach for Stage 2 refinement.

- **Pros**: Leverages labeled data effectively
- **Cons**: Requires labeled examples
- **Stage**: 2 (supervised)

### Model Implementation Status

Current implementations are **stubs** that return deterministic outputs based on image ID hashing. This allows end-to-end testing of the pipeline without ML dependencies.

```python
# Example stub behavior (for testing)
def predict(self, image_data: bytes, image_id: str) -> InferenceResult:
    # Deterministic score from image ID (for reproducible tests)
    hash_val = int(hashlib.sha256(image_id.encode()).hexdigest(), 16)
    anomaly_score = (hash_val % 1000) / 1000.0
    return InferenceResult(anomaly_score=anomaly_score, ...)
```

Real implementations will use [anomalib](https://github.com/openvinotoolkit/anomalib) as the backend.

---

## Active Learning Workflow

### Uncertainty Sampling Strategy

The system uses uncertainty sampling to select the most informative samples for human labeling:

```
Uncertainty Score Calculation:
──────────────────────────────

For a sample with anomaly score p ∈ [0, 1]:

    uncertainty = 1 - |2p - 1|

This peaks at p = 0.5 (maximum uncertainty) and approaches 0
at p = 0 or p = 1 (high confidence).
```

### Labeling Process

1. **Pool Population**: After Stage 1 inference, uncertain samples are added to the active learning pool
2. **Queue Presentation**: Samples are sorted by uncertainty (highest first)
3. **Human Labeling**: Annotator provides binary label + optional category
4. **Threshold Activation**: When labels reach threshold (default: 100), Stage 2 becomes available

### Data Flow

```
Stage 1 Inference
       │
       ▼
┌──────────────────┐
│ Uncertainty      │──────────────────────────────────┐
│ Calculation      │                                  │
└──────────────────┘                                  │
       │                                              │
       ▼                                              │
┌──────────────────┐     ┌──────────────────┐        │
│ High Uncertainty │────▶│ AL Pool (Redis)  │        │
│ (p ≈ 0.5)        │     │ Sorted Set       │        │
└──────────────────┘     └────────┬─────────┘        │
                                  │                   │
                                  ▼                   │
                         ┌──────────────────┐        │
                         │ Labeling Queue   │        │
                         │ (Frontend UI)    │        │
                         └────────┬─────────┘        │
                                  │                   │
                                  ▼                   │
                         ┌──────────────────┐        │
                         │ Human Annotation │        │
                         └────────┬─────────┘        │
                                  │                   │
                                  ▼                   │
                         ┌──────────────────┐        │
                         │ Label Store      │◀───────┘
                         │ (Redis Hash)     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Stage 2 Training │
                         │ (when n ≥ 100)   │
                         └──────────────────┘
```

---

## Development

### Project Structure

```
RETINA/
├── backend/                  # Rust/Axum API server
│   ├── src/
│   │   ├── main.rs          # Entry point
│   │   ├── config.rs        # Configuration management
│   │   ├── error.rs         # Error types
│   │   ├── models/          # Data structures
│   │   ├── routes/          # HTTP handlers
│   │   └── services/        # Business logic
│   ├── Cargo.toml
│   └── Dockerfile
│
├── worker/                   # Python ML worker
│   ├── src/retina_worker/
│   │   ├── config.py        # Configuration
│   │   ├── schemas.py       # Pydantic models
│   │   ├── models/          # ML model implementations
│   │   ├── redis_client.py  # Redis utilities
│   │   └── worker.py        # Main worker loop
│   ├── pyproject.toml
│   └── Dockerfile
│
├── frontend/                 # Next.js web application
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   └── lib/             # Utilities
│   ├── package.json
│   └── Dockerfile
│
├── shared/                   # Cross-component schemas
│   └── schemas/
│       ├── job.json
│       ├── result.json
│       └── label.json
│
├── docker-compose.yml        # Local development setup
├── .env.example             # Environment template
└── README.md                # This file
```

### Code Style

- **Rust**: Follow `rustfmt` defaults, use `clippy` for linting
- **Python**: Black + isort + ruff, type hints required
- **TypeScript**: Prettier + ESLint, strict mode enabled

### Testing

```bash
# Backend tests
cd backend && cargo test

# Worker tests
cd worker && pytest

# Frontend tests
cd frontend && npm test

# Integration tests (requires Docker)
docker compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Logging

All components use structured logging in JSON format:

```json
{
  "timestamp": "2024-10-15T10:23:45.123Z",
  "level": "info",
  "message": "Job completed",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "patchcore",
  "duration_ms": 156
}
```

---

## Roadmap

### v0.1 (Current) - Foundation

- [x] Project scaffold
- [x] Backend API with Redis integration
- [x] Worker framework with model stubs
- [x] Frontend for submission and visualization
- [x] Docker configuration

### v0.2 - Core ML

- [ ] PatchCore implementation via anomalib
- [ ] WinCLIP implementation
- [ ] MVTec AD benchmark integration
- [ ] Anomaly heatmap visualization

### v0.3 - Active Learning

- [ ] Multiple uncertainty sampling strategies
- [ ] Query-by-committee implementation
- [ ] Labeling efficiency metrics
- [ ] Batch labeling interface

### v0.4 - Stage 2 Pipeline

- [ ] PushPull contrastive learning
- [ ] Fine-tuned PatchCore with defect bank
- [ ] Automatic stage transition
- [ ] A/B comparison between stages

### v0.5 - Production Readiness

- [ ] Kubernetes deployment
- [ ] GPU scheduling
- [ ] Monitoring and alerting
- [ ] Performance optimization

---

## References

### Papers

1. Roth, K., et al. (2022). "Towards Total Recall in Industrial Anomaly Detection." *CVPR 2022*. [arXiv:2106.08265](https://arxiv.org/abs/2106.08265)

2. Defard, T., et al. (2021). "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization." *ICPR 2021*. [arXiv:2011.08785](https://arxiv.org/abs/2011.08785)

3. Jeong, J., et al. (2023). "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation." *CVPR 2023*. [arXiv:2303.14814](https://arxiv.org/abs/2303.14814)

4. Bergmann, P., et al. (2019). "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." *CVPR 2019*. [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Libraries

- [anomalib](https://github.com/openvinotoolkit/anomalib) - Anomaly detection library
- [Axum](https://github.com/tokio-rs/axum) - Rust web framework
- [Next.js](https://nextjs.org/) - React framework

---

## License

This project is developed as part of research at KU Leuven. License terms TBD.

---

## Acknowledgments

- KU Leuven Department of Computer Science
- Research advisors and collaborators
- Open-source community (anomalib, Axum, Next.js)