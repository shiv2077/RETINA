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
import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

logger = structlog.get_logger()

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
IMAGE_META_KEY = "retina:images:{image_id}"  # hash: image_path, latest_job_id
LABEL_KEY = "retina:labels:{image_id}"
LABEL_COUNT_FIELD = "labels_collected"
STATS_KEY = "retina:system:stats"
TAXONOMY_KEY = "retina:taxonomy:{product_class}"

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
    # Record reverse image_id → image_path mapping so /api/images/ can
    # serve the file even after the worker processes the job.
    redis_client.hset(
        IMAGE_META_KEY.format(image_id=job_id),
        mapping={"image_path": str(dest.resolve())},
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
        meta: dict = {}
        if sample_json:
            try:
                meta = json.loads(sample_json)
            except json.JSONDecodeError:
                pass

        # Enrich with product_class by chasing retina:images → retina:results.
        product_class: Optional[str] = None
        image_hash = redis_client.hgetall(IMAGE_META_KEY.format(image_id=image_id))
        latest_job_id = image_hash.get("latest_job_id") if image_hash else None
        if latest_job_id:
            result_json = redis_client.hget(
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
    redis_client.hset(key, mapping=hash_fields)
    redis_client.expire(key, 7 * 24 * 3600)
    redis_client.zrem(AL_POOL_KEY, body.image_id)
    redis_client.delete(AL_SAMPLE_KEY.format(image_id=body.image_id))
    labels_count = redis_client.hincrby(STATS_KEY, LABEL_COUNT_FIELD, 1)
    return {"ok": True, "labels_count": int(labels_count)}


# ── GET /api/images/{image_id} ───────────────────────────────────────────
def _serve_if_exists(path_str: str, source: str, image_id: str):
    """Return a FileResponse if the path exists on disk, else None.

    Logs which lookup path succeeded so we can trace 404s in production.
    """
    if not path_str:
        return None
    cand = Path(path_str)
    if not cand.is_file():
        return None
    mime = "image/png" if cand.suffix.lower() == ".png" else "image/jpeg"
    logger.info("image_served", image_id=image_id, source=source, path=str(cand))
    return FileResponse(cand, media_type=mime)


@app.get("/api/images/{image_id}")
async def get_image(image_id: str):
    # 1. uploads directory (FastAPI /api/submit multipart path)
    upload_path = UPLOAD_DIR / f"{image_id}.png"
    if upload_path.is_file():
        return FileResponse(upload_path, media_type="image/png")

    # 2. AL sample metadata — retina:al:samples:{id}
    sample_json = redis_client.get(AL_SAMPLE_KEY.format(image_id=image_id))
    if sample_json:
        try:
            sample = json.loads(sample_json)
            r = _serve_if_exists(sample.get("image_path", ""), "al_sample", image_id)
            if r:
                return r
        except json.JSONDecodeError:
            pass

    # 3. Result hash — retina:results:{image_id} (if an InferenceResult
    #    ever carries image_path, pick it up here)
    result_json = redis_client.hget(RESULT_KEY.format(job_id=image_id), "result_data")
    if result_json:
        try:
            result = json.loads(result_json)
            r = _serve_if_exists(result.get("image_path", ""), "result", image_id)
            if r:
                return r
        except json.JSONDecodeError:
            pass

    # 4. Reverse mapping — retina:images:{image_id}
    meta = redis_client.hgetall(IMAGE_META_KEY.format(image_id=image_id))
    if meta:
        r = _serve_if_exists(meta.get("image_path", ""), "image_meta", image_id)
        if r:
            return r

    # 5. Stream scan — last resort. Walk retina:jobs:queue looking for a job
    #    whose image_id matches, then try its image_path.
    try:
        entries = redis_client.xrange(JOB_QUEUE_STREAM, "-", "+", count=1000)
        for _entry_id, fields in entries:
            try:
                job = json.loads(fields.get("job_data", "{}"))
            except json.JSONDecodeError:
                continue
            if job.get("image_id") == image_id:
                r = _serve_if_exists(job.get("image_path", ""), "stream_scan", image_id)
                if r:
                    return r
                break
    except redis.RedisError:
        pass

    logger.warning("image_not_found", image_id=image_id)
    raise HTTPException(
        404, f"image {image_id} not found in uploads, al_samples, results, "
             "image_meta, or job stream",
    )


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
    raw = redis_client.get(TAXONOMY_KEY.format(product_class=product_class))
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
    raw = redis_client.get(key)
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
    redis_client.set(key, json.dumps(existing))
    return {"product_class": product_class, "custom": existing}


@app.get("/health")
async def health() -> dict:
    try:
        redis_client.ping()
        return {"status": "ok", "redis": "up"}
    except redis.RedisError as e:
        raise HTTPException(503, f"redis down: {e}") from None
