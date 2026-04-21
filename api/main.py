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

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "worker" / "src"))

from retina_worker.schemas import (  # noqa: E402
    InferenceJob,
    JobStatus,
    ModelType,
    PipelineStage,
)

UPLOAD_DIR = REPO_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = "redis://localhost:6379"
JOB_QUEUE_STREAM = "retina:jobs:queue"
RESULT_KEY = "retina:results:{job_id}"
AL_POOL_KEY = "retina:al:pool"
AL_SAMPLE_KEY = "retina:al:samples:{image_id}"
LABEL_KEY = "retina:labels:{image_id}"
LABEL_COUNT_FIELD = "labels_collected"
STATS_KEY = "retina:system:stats"

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
    dest = UPLOAD_DIR / f"{job_id}.png"
    content = await file.read()
    dest.write_bytes(content)

    job = InferenceJob(
        job_id=job_id,
        image_id=job_id,
        model_type=ModelType.PATCHCORE,
        stage=PipelineStage.UNSUPERVISED,
        status=JobStatus.PENDING,
        submitted_at=datetime.utcnow(),
        image_path=str(dest.resolve()),
    )
    redis_client.xadd(JOB_QUEUE_STREAM, {"job_data": job.model_dump_json()})
    return {"job_id": job_id}


# ── GET /api/result/{job_id} ─────────────────────────────────────────────
@app.get("/api/result/{job_id}")
async def get_result(job_id: str, wait: int = 30) -> dict:
    """Return the InferenceResult for a job. If still processing, poll up
    to `wait` seconds before returning 404."""
    key = RESULT_KEY.format(job_id=job_id)
    deadline = time.time() + max(0, min(wait, 60))
    while True:
        raw = redis_client.hget(key, "result_data")
        if raw:
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise HTTPException(500, f"corrupt result JSON: {e}") from None
        if time.time() >= deadline:
            raise HTTPException(404, f"no result for job_id={job_id}")
        time.sleep(0.5)


# ── GET /api/labels/pool ─────────────────────────────────────────────────
@app.get("/api/labels/pool")
async def labels_pool(limit: int = 20) -> dict:
    items = redis_client.zrevrange(AL_POOL_KEY, 0, max(0, limit - 1), withscores=True)
    pool = []
    for image_id, score in items:
        sample_json = redis_client.get(AL_SAMPLE_KEY.format(image_id=image_id))
        meta = {}
        if sample_json:
            try:
                meta = json.loads(sample_json)
            except json.JSONDecodeError:
                pass
        pool.append({
            "image_id": image_id,
            "score": float(score),
            "image_url": f"/api/images/{image_id}",
            "anomaly_score": meta.get("anomaly_score"),
            "uncertainty_score": meta.get("uncertainty_score"),
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
    redis_client.hset(key, mapping=hash_fields)
    redis_client.expire(key, 7 * 24 * 3600)
    redis_client.zrem(AL_POOL_KEY, body.image_id)
    redis_client.delete(AL_SAMPLE_KEY.format(image_id=body.image_id))
    labels_count = redis_client.hincrby(STATS_KEY, LABEL_COUNT_FIELD, 1)
    return {"ok": True, "labels_count": int(labels_count)}


# ── GET /api/images/{image_id} ───────────────────────────────────────────
@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    path = UPLOAD_DIR / f"{image_id}.png"
    if not path.is_file():
        raise HTTPException(404, f"no image for {image_id}")
    return FileResponse(path, media_type="image/png")


@app.get("/health")
async def health() -> dict:
    try:
        redis_client.ping()
        return {"status": "ok", "redis": "up"}
    except redis.RedisError as e:
        raise HTTPException(503, f"redis down: {e}") from None
