# Research Code

This directory contains standalone research implementations that are **not** wired into
the production pipeline. They were used for benchmarking, experimentation, and dataset
evaluation on KU Leuven hardware.

## Structure

```
unsupervised/
  AdaCLIP/          CLIP-based VLM for Decospan; results in v21/
  PatchCore/        Standalone PatchCore evaluation scripts
  PaDiM/            Standalone PaDiM evaluation scripts
  WinCLIP/          Standalone WinCLIP evaluation scripts

supervised/
  BGAD/             Real BGAD implementation (AUC 0.930 on MVTec)
  Custom_Model_Push_Pull/  Push-Pull contrastive learning
```

## What IS Wired into Production

The production worker (`worker/`) uses:
- `worker/src/retina_worker/models/patchcore_real.py` — real PatchCore via Anomalib
- `worker/src/retina_worker/models/gpt4v_detector.py` — real GPT-4o vision inference
- Stubs for all Stage 2 models (see `worker/src/retina_worker/models/pushpull_stub.py`)

## Integration Path

To wire a research model into the production worker:
1. Create a class inheriting `AnomalyDetector` in `worker/src/retina_worker/models/`
2. Register it in `worker/src/retina_worker/models/factory.py`
3. Add the model name to `shared/schemas/job.json` enum
4. Update the Rust `ModelType` enum in `backend/src/models/job.rs`
5. Follow CLAUDE.md §1.2 schema update protocol (JSON → Rust → Python)
