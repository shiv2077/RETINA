"""Submit a single inference job to retina:jobs:queue.

Wire format matches worker/src/retina_worker/redis_client.py read_job():
  XADD retina:jobs:queue * job_data <InferenceJob as JSON>
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

import redis

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker" / "src"))

from retina_worker.schemas import (  # noqa: E402
    InferenceJob,
    JobStatus,
    ModelType,
    PipelineStage,
)

JOB_QUEUE_STREAM = "retina:jobs:queue"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", type=Path, required=True, help="Path to PNG/JPG image")
    p.add_argument(
        "--model",
        default="patchcore",
        choices=["patchcore", "auto"],
        help="'patchcore' = hint to skip VLM product id; 'auto' = let router decide.",
    )
    p.add_argument("--redis-url", default="redis://localhost:6379")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def validate_image(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"[error] image not found: {path}")
    if not path.is_file():
        raise SystemExit(f"[error] not a regular file: {path}")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise SystemExit(
            f"[error] unsupported extension {path.suffix!r}; "
            f"allowed: {sorted(ALLOWED_SUFFIXES)}"
        )
    return path.resolve()


def build_job(image_abs: Path) -> InferenceJob:
    job_id = uuid.uuid4().hex[:12]
    return InferenceJob(
        job_id=job_id,
        image_id=job_id,
        model_type=ModelType.PATCHCORE,
        stage=PipelineStage.UNSUPERVISED,
        status=JobStatus.PENDING,
        submitted_at=datetime.utcnow(),
        image_path=str(image_abs),
    )


def main() -> None:
    args = parse_args()
    image_abs = validate_image(args.image)
    job = build_job(image_abs)

    try:
        client = redis.from_url(args.redis_url, decode_responses=True)
        client.ping()
    except redis.RedisError as e:
        raise SystemExit(
            f"[error] Redis not reachable at {args.redis_url}: {e}\n"
            "        Start it with: redis-server --port 6379"
        ) from None

    payload = job.model_dump_json()
    entry_id = client.xadd(JOB_QUEUE_STREAM, {"job_data": payload})

    if args.quiet:
        print(job.job_id)
    else:
        print(
            f"[submitted] job_id={job.job_id} image={image_abs}\n"
            f"            stream_entry={entry_id}\n"
            f"            watch with: python scripts/watch_results.py --job {job.job_id}"
        )


if __name__ == "__main__":
    main()
