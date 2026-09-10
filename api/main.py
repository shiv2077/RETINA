"""Thin FastAPI wrapper that sits in front of the retina_worker.

Four endpoints:
  POST /api/submit          — upload an image, enqueue an inference job
  GET  /api/result/{job_id} — poll for a completed InferenceResult
  GET  /api/labels/pool     — list samples awaiting operator labels
  POST /api/labels/submit   — persist a label + polygons to retina:labels:*
  GET  /api/images/{image_id} — serve uploaded image bytes

This process does NO inference. Jobs are handed to the existing worker via
Redis streams. Wire format matches scripts/submit_job.py exactly.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Async client: every route handler is `async def` and this process runs as a
# single uvicorn worker, so a blocking Redis call stalls the whole event loop.
# redis.asyncio re-exports the same exception hierarchy (redis.RedisError).
import redis.asyncio as redis
import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = structlog.get_logger()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "worker" / "src"))

from retina_worker.config import Settings, image_relpath  # noqa: E402
from retina_worker.schemas import (  # noqa: E402
    InferenceJob,
    JobStatus,
    ModelType,
    PipelineStage,
)

# Shared with the worker on purpose: both sides must agree on where an image
# lives for a given id, and duplicating that rule in two codebases is how the
# two halves of this system have drifted apart before.
SETTINGS = Settings()
IMAGE_ROOT = SETTINGS.resolved_image_root()
IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

# Carries credentials when Redis runs with requirepass; falls back to a
# local unauthenticated instance for native dev runs.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
JOB_QUEUE_STREAM = "retina:jobs:queue"
RESULT_KEY = "retina:results:{job_id}"
AL_POOL_KEY = "retina:al:pool"
AL_SAMPLE_KEY = "retina:al:samples:{image_id}"
IMAGE_META_KEY = "retina:images:{image_id}"  # hash: image_relpath, latest_job_id
LABEL_KEY = "retina:labels:{image_id}"
LABEL_COUNT_FIELD = "labels_collected"
STATS_KEY = "retina:system:stats"
TAXONOMY_KEY = "retina:taxonomy:{product_class}"

# Cap on retina:jobs:queue. Redis Streams never evict on their own, so without
# this the queue grows without bound for the life of the deployment.
#
# Trimming a stream deletes entries even if they are still unacknowledged in a
# consumer group's Pending Entries List, which would silently drop jobs the
# worker had claimed but not finished. 100k is chosen to make that impossible
# in practice: the worker consumes one job at a time (worker.py), so the PEL
# holds a single entry in steady state and, after a crash, at most the handful
# of entries claimed before it died. For 100k entries to be dropped from under
# a live consumer, a worker would have to fall 100k jobs behind while still
# holding the oldest one pending. At ~1KB per job that is also only ~100MB of
# Redis, so the cap is cheap to keep generous.
JOB_STREAM_MAXLEN = int(os.getenv("RETINA_JOB_STREAM_MAXLEN", "100000"))

app = FastAPI(title="RETINA API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

redis_client = redis.from_url(REDIS_URL, decode_responses=True)


# ── POST /api/submit ─────────────────────────────────────────────────────
@app.post("/api/submit")
async def submit(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(400, "missing filename")
    job_id = uuid.uuid4().hex[:12]
    content = await file.read()

    # Content address: identical bytes get the same id and the same location,
    # so a resubmission overwrites itself instead of accumulating duplicates.
    image_id = hashlib.sha256(content).hexdigest()
    dest = SETTINGS.image_path(image_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)

    job = InferenceJob(
        job_id=job_id,
        image_id=image_id,
        model_type=ModelType.PATCHCORE,
        stage=PipelineStage.UNSUPERVISED,
        status=JobStatus.PENDING,
        submitted_at=datetime.utcnow(),
    )
    # Only the relative layout is recorded. The worker resolves the same id
    # against its own mount, so nothing here is tied to this host's paths.
    await redis_client.hset(
        IMAGE_META_KEY.format(image_id=image_id),
        mapping={"image_relpath": str(image_relpath(image_id))},
    )
    # maxlen + approximate emits `XADD ... MAXLEN ~ JOB_STREAM_MAXLEN`: Redis
    # trims lazily at macro-node boundaries, so this is the periodic trim, done
    # server-side with no extra round trip and no scheduler to keep alive.
    await redis_client.xadd(
        JOB_QUEUE_STREAM,
        {"job_data": job.model_dump_json()},
        maxlen=JOB_STREAM_MAXLEN,
        approximate=True,
    )
    return {"job_id": job_id, "image_id": image_id}


# ── GET /api/result/{job_id} ─────────────────────────────────────────────
@app.get("/api/result/{job_id}")
async def get_result(job_id: str, wait: int = 30) -> dict:
    """Return the InferenceResult for a job. If still processing, poll up
    to `wait` seconds before returning 404."""
    key = RESULT_KEY.format(job_id=job_id)
    deadline = time.time() + max(0, min(wait, 60))
    while True:
        raw = await redis_client.hget(key, "result_data")
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise HTTPException(500, f"corrupt result JSON: {e}") from None
        if time.time() >= deadline:
            raise HTTPException(404, f"no result for job_id={job_id}")
        await asyncio.sleep(0.5)


# ── GET /api/labels/pool ─────────────────────────────────────────────────
@app.get("/api/labels/pool")
async def labels_pool(limit: int = 20) -> dict:
    items = await redis_client.zrevrange(
        AL_POOL_KEY, 0, max(0, limit - 1), withscores=True,
    )
    pool = []
    for image_id, score in items:
        sample_json = await redis_client.get(AL_SAMPLE_KEY.format(image_id=image_id))
        meta: dict = {}
        if sample_json:
            try:
                meta = json.loads(sample_json)
            except json.JSONDecodeError:
                pass

        # Enrich with product_class by chasing retina:images → retina:results.
        product_class: Optional[str] = None
        image_hash = await redis_client.hgetall(IMAGE_META_KEY.format(image_id=image_id))
        latest_job_id = image_hash.get("latest_job_id") if image_hash else None
        if latest_job_id:
            result_json = await redis_client.hget(
                RESULT_KEY.format(job_id=latest_job_id), "result_data",
            )
            if result_json:
                try:
                    product_class = json.loads(result_json).get("product_class")
                except json.JSONDecodeError:
                    pass

        pool.append({
            "image_id": image_id,
            "score": float(score),
            "image_url": f"/api/images/{image_id}",
            "anomaly_score": meta.get("anomaly_score"),
            "uncertainty_score": meta.get("uncertainty_score"),
            "product_class": product_class,
        })
    return {"pool": pool, "count": len(pool)}


# ── POST /api/labels/submit ──────────────────────────────────────────────
class LabelSubmission(BaseModel):
    image_id: str
    product_class: str
    label: str  # "anomaly" | "normal"
    defect_class: Optional[str] = None
    polygons: Optional[list[dict]] = None  # [{"vertices": [{x,y},...], "class": str}, ...]
    boxes: Optional[list[dict]] = None  # [{x,y,w,h,class}, ...]
    operator_id: Optional[str] = None
    notes: Optional[str] = None


@app.post("/api/labels/submit")
async def labels_submit(body: LabelSubmission) -> dict:
    key = LABEL_KEY.format(image_id=body.image_id)
    hash_fields = {
        "image_id": body.image_id,
        "product_class": body.product_class,
        "label": body.label,
        "defect_class": body.defect_class or "",
        "polygons": json.dumps(body.polygons) if body.polygons else "",
        "boxes": json.dumps(body.boxes) if body.boxes else "",
        "operator_id": body.operator_id or "",
        "notes": body.notes or "",
        "labeled_at": datetime.utcnow().isoformat(),
    }
    await redis_client.hset(key, mapping=hash_fields)
    await redis_client.expire(key, 7 * 24 * 3600)
    await redis_client.zrem(AL_POOL_KEY, body.image_id)
    await redis_client.delete(AL_SAMPLE_KEY.format(image_id=body.image_id))
    labels_count = await redis_client.hincrby(STATS_KEY, LABEL_COUNT_FIELD, 1)
    return {"ok": True, "labels_count": int(labels_count)}


# ── GET /api/images/{image_id} ───────────────────────────────────────────
@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    """Serve an image by its content address.

    Content addressing collapses what used to be five lookup paths (uploads
    dir, AL sample metadata, result hash, reverse mapping, and a stream
    scan) into one derivation. Those paths existed only because the id
    alone did not tell you where the bytes were; now it does. The id is
    never joined raw onto a path — image_relpath rejects separators, so a
    traversal attempt is a 400 rather than a filesystem probe.
    """
    try:
        path = SETTINGS.image_path(image_id)
    except ValueError:
        raise HTTPException(400, f"malformed image_id: {image_id}") from None

    if not path.is_file():
        logger.warning("image_not_found", image_id=image_id, path=str(path))
        raise HTTPException(404, f"image {image_id} not found")

    logger.info("image_served", image_id=image_id, path=str(path))
    return FileResponse(path, media_type="image/png")


# ── /api/taxonomy/{product_class} ───────────────────────────────────────
# Base taxonomy lives in the frontend (lib/taxonomies.ts). This endpoint
# only stores operator-added custom categories so they persist across
# restarts.

class TaxonomyEntry(BaseModel):
    key: str
    name: str
    color: str
    shortcut: str


@app.get("/api/taxonomy/{product_class}")
async def get_taxonomy(product_class: str) -> dict:
    raw = await redis_client.get(TAXONOMY_KEY.format(product_class=product_class))
    if not raw:
        return {"product_class": product_class, "custom": []}
    try:
        custom = json.loads(raw)
    except json.JSONDecodeError:
        custom = []
    return {"product_class": product_class, "custom": custom}


@app.post("/api/taxonomy/{product_class}")
async def add_taxonomy_entry(product_class: str, entry: TaxonomyEntry) -> dict:
    # Validate against existing custom entries. (Base-taxonomy uniqueness is
    # enforced client-side since the base set lives in the frontend.)
    key = TAXONOMY_KEY.format(product_class=product_class)
    raw = await redis_client.get(key)
    existing: list[dict] = []
    if raw:
        try:
            existing = json.loads(raw)
        except json.JSONDecodeError:
            existing = []

    key_lower = entry.key.lower()
    if any(e.get("key", "").lower() == key_lower for e in existing):
        raise HTTPException(409, f"custom category '{entry.key}' already exists")

    if not entry.key or len(entry.key) < 2 or len(entry.key) > 30:
        raise HTTPException(400, "key must be 2–30 characters")

    import re
    if not re.match(r"^[a-z0-9_]+$", entry.key):
        raise HTTPException(400, "key must be snake_case (a–z, 0–9, _)")

    new_entry = entry.model_dump()
    new_entry["custom"] = True
    existing.append(new_entry)
    await redis_client.set(key, json.dumps(existing))
    return {"product_class": product_class, "custom": existing}


@app.get("/health")
async def health() -> dict:
    try:
        await redis_client.ping()
        return {"status": "ok", "redis": "up"}
    except redis.RedisError as e:
        raise HTTPException(503, f"redis down: {e}") from None
