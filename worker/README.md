# retina-worker

The inference worker. Consumes jobs from the Redis stream
`retina:jobs:queue` (consumer group `workers`), runs the two-stage
pipeline, and writes results back to `retina:results:{job_id}`.

Per job:

1. `identify_product` (gpt-4o-mini) picks the product category, cached per
   session in Redis for an hour.
2. The per-category PatchCore checkpoint produces the numeric anomaly
   score. Products with no checkpoint fall back to `zero_shot_detect`.
3. Scores in `[anomaly_threshold, stage2_trigger_max)` go to
   `stage2_refine`, a GPT-4o in-context pass over recent operator labels.
4. Uncertain samples are added to the active-learning pool.

Images are addressed by the sha256 of their bytes, never by a filesystem
path: the submitter and this worker each resolve the same id against
their own image root, so the payload survives a container boundary.

## Running

```bash
# Natively, from the repo root
PYTHONPATH=worker/src python -m retina_worker.main

# Containerized (bind-mounts ./data/images and ./checkpoints)
docker compose up worker
```

Configuration is environment-driven; see `src/retina_worker/config.py`.
`REDIS_URL` must carry the password when Redis runs with `requirepass`.

## Tests

```bash
cd worker && python -m pytest        # offline unit suite
python -m pytest -m integration      # billed OpenAI calls / real checkpoints
```
