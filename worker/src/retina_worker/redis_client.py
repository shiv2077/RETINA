"""
Redis Client Utilities
======================

Provides Redis connection management and helper functions for the worker.

Key Schema
----------

The worker interacts with the following Redis keys:

**Job Queue (Stream)**
- ``retina:jobs:queue`` - Inference job queue (consumed via XREADGROUP)
- ``retina:jobs:dlq`` - Dead-letter stream for entries that cannot be parsed
  or that exceeded the redelivery cap. Fields: entry_id, reason, failed_at,
  job_data (the raw payload). Never consumed by the worker; drained by hand.

**Job Metadata (Hashes)**
- ``retina:jobs:{job_id}`` - Job data and status

**Results (Hashes)**
- ``retina:results:{job_id}`` - Inference results

**Active Learning Pool (Sorted Set)**
- ``retina:al:pool`` - Samples awaiting labeling (scored by uncertainty)

**Active Learning Samples (Strings)**
- ``retina:al:samples:{image_id}`` - Sample metadata JSON

**Labels (Hashes) and their index (Sorted Set)**
- ``retina:labels:{image_id}`` - Operator label, written by the API process
- ``retina:labels_index`` - image_id scored by the label's ``labeled_at``
  epoch seconds, so the worker can read the N most recent labels in order.
  Maintained by the worker; labels written before it exists are backfilled
  on the next read.

**System Stats (Hash)**
- ``retina:system:stats`` - Counters and metrics
"""

import json
from datetime import datetime

import redis
import structlog
from pydantic import ValidationError

from .config import Settings
from .schemas import (
    InferenceJob,
    InferenceResult,
    JobStatus,
    UnlabeledSample,
)

logger = structlog.get_logger()


# Redis key constants
KEY_PREFIX = "retina"
JOB_QUEUE_STREAM = f"{KEY_PREFIX}:jobs:queue"
JOB_DLQ_STREAM = f"{KEY_PREFIX}:jobs:dlq"
WORKER_GROUP = "workers"

# Retention for per-job keys (results and job status hashes). Both describe
# the same job, so they expire together — otherwise retina:jobs:* accumulates
# one hash per job forever.
JOB_TTL_S = 7 * 24 * 60 * 60

# Insertion-ordered index over retina:labels:{image_id}. Named outside the
# `retina:labels:*` pattern so it is not itself matched by the label scan.
LABEL_INDEX_KEY = f"{KEY_PREFIX}:labels_index"


def _parse_timestamp(labeled_at: str | None) -> float:
    """ISO-8601 label timestamp as a sort score; 0.0 when absent or unparseable."""
    if not labeled_at:
        return 0.0
    try:
        return datetime.fromisoformat(labeled_at).timestamp()
    except ValueError:
        return 0.0


class RedisClient:
    """
    Redis client wrapper for the ML worker.

    Provides methods for:
    - Consuming jobs from the stream
    - Storing inference results
    - Managing the active learning pool
    - Updating system statistics

    Attributes
    ----------
    client : redis.Redis
        Underlying Redis client
    consumer_name : str
        Unique identifier for this worker in the consumer group
    """

    def __init__(self, settings: Settings):
        """
        Initialize Redis client.

        Parameters
        ----------
        settings : Settings
            Worker configuration
        """
        self.client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self.consumer_name = settings.consumer_name
        self.settings = settings
        self._ensure_consumer_group()

    def _ensure_consumer_group(self) -> None:
        """Create the worker consumer group if it does not yet exist.

        Idempotent: on BUSYGROUP (group already exists) we log debug and move
        on. Any other ResponseError propagates.
        """
        try:
            self.client.xgroup_create(
                name=JOB_QUEUE_STREAM,
                groupname=WORKER_GROUP,
                id="0",
                mkstream=True,
            )
            logger.info(
                "consumer_group_created",
                stream=JOB_QUEUE_STREAM,
                group=WORKER_GROUP,
            )
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(
                    "consumer_group_exists",
                    stream=JOB_QUEUE_STREAM,
                    group=WORKER_GROUP,
                )
            else:
                raise

    def health_check(self) -> bool:
        """
        Check Redis connectivity.

        Returns
        -------
        bool
            True if connected, False otherwise
        """
        try:
            return self.client.ping()
        except redis.RedisError:
            return False

    # -------------------------------------------------------------------------
    # Job Queue Operations
    # -------------------------------------------------------------------------

    def read_job(self, block_ms: int = 5000) -> tuple[str, InferenceJob] | None:
        """
        Read the next job from the queue.

        Uses XREADGROUP for consumer group semantics:
        - Jobs are distributed among workers
        - Jobs must be acknowledged after processing

        Parameters
        ----------
        block_ms : int
            Milliseconds to block waiting for new jobs

        Returns
        -------
        tuple[str, InferenceJob] | None
            (stream_entry_id, job) if available, None otherwise
        """
        try:
            result = self.client.xreadgroup(
                groupname=WORKER_GROUP,
                consumername=self.consumer_name,
                streams={JOB_QUEUE_STREAM: ">"},  # Only new messages
                count=1,
                block=block_ms,
            )
        except redis.RedisError as e:
            logger.error("Failed to read job from queue", error=str(e))
            return None

        if not result:
            return None

        # Parse result: [[stream_name, [[entry_id, {fields}]]]]
        _stream_name, messages = result[0]
        entry_id, fields = messages[0]

        job = self._parse_entry(entry_id, fields)
        if job is None:
            return None

        logger.debug(
            "Read job from queue",
            entry_id=entry_id,
            job_id=job.job_id,
            model_type=job.model_type.value,
        )
        return entry_id, job

    def _parse_entry(
        self, entry_id: str, fields: dict[str, str],
    ) -> InferenceJob | None:
        """Decode one stream entry, dead-lettering it if it cannot be parsed.

        A payload the worker cannot decode will never become decodable on a
        retry, so leaving it in the pending entries list only blocks the slot
        forever. Every failure path here XACKs and copies the raw payload to
        the DLQ, where it can be inspected without stalling the queue.
        """
        job_json = fields.get("job_data")
        if not job_json:
            self._dead_letter(entry_id, fields, "missing_job_data_field")
            return None

        try:
            job_data = json.loads(job_json)
        except json.JSONDecodeError as e:
            self._dead_letter(entry_id, fields, f"invalid_json: {e}")
            return None

        try:
            return InferenceJob(**job_data)
        except (ValidationError, TypeError) as e:
            self._dead_letter(entry_id, fields, f"schema_violation: {e}")
            return None

    def _dead_letter(
        self, entry_id: str, fields: dict[str, str], reason: str,
    ) -> None:
        """Move one poisoned entry off the pending list into the DLQ stream."""
        try:
            self.client.xadd(
                JOB_DLQ_STREAM,
                {
                    "entry_id": entry_id,
                    "reason": reason,
                    "failed_at": datetime.utcnow().isoformat(),
                    "job_data": fields.get("job_data", ""),
                },
            )
            self.client.xack(JOB_QUEUE_STREAM, WORKER_GROUP, entry_id)
        except redis.RedisError as e:
            logger.error("dead_letter_failed", entry_id=entry_id, error=str(e))
            return
        logger.error("job_dead_lettered", entry_id=entry_id, reason=reason)

    def reclaim_stale_jobs(
        self,
        min_idle_ms: int,
        max_deliveries: int,
        count: int = 10,
    ) -> list[tuple[str, InferenceJob]]:
        """Claim entries abandoned by a crashed worker and return them.

        XREADGROUP with ">" never revisits the pending entries list, so an
        entry read by a worker that then died is invisible to every other
        replica until something claims it. Entries redelivered more than
        `max_deliveries` times are dead-lettered instead — they are the ones
        that keep killing whichever worker picks them up.
        """
        try:
            response = self.client.xautoclaim(
                name=JOB_QUEUE_STREAM,
                groupname=WORKER_GROUP,
                consumername=self.consumer_name,
                min_idle_time=min_idle_ms,
                count=count,
            )
        except redis.RedisError as e:
            logger.error("reclaim_failed", error=str(e))
            return []

        # Redis >= 7 returns (next_cursor, entries, deleted); 6.2 omits deleted.
        claimed = response[1] if len(response) > 1 else []
        if not claimed:
            return []

        try:
            delivered = {
                e["message_id"]: e["times_delivered"]
                for e in self.client.xpending_range(
                    JOB_QUEUE_STREAM, WORKER_GROUP, min="-", max="+", count=count * 10,
                )
            }
        except redis.RedisError as e:
            logger.error("reclaim_pending_lookup_failed", error=str(e))
            delivered = {}

        jobs: list[tuple[str, InferenceJob]] = []
        for entry_id, fields in claimed:
            if delivered.get(entry_id, 1) > max_deliveries:
                self._dead_letter(entry_id, fields, "max_deliveries_exceeded")
                continue
            job = self._parse_entry(entry_id, fields)
            if job is not None:
                logger.warning(
                    "job_reclaimed",
                    entry_id=entry_id,
                    job_id=job.job_id,
                    deliveries=delivered.get(entry_id),
                )
                jobs.append((entry_id, job))
        return jobs

    def acknowledge_job(self, entry_id: str) -> bool:
        """
        Acknowledge that a job has been processed.

        This removes the job from the pending entries list,
        ensuring it won't be redelivered.

        Parameters
        ----------
        entry_id : str
            Stream entry ID to acknowledge

        Returns
        -------
        bool
            True if acknowledged successfully
        """
        try:
            self.client.xack(JOB_QUEUE_STREAM, WORKER_GROUP, entry_id)
            logger.debug("Job acknowledged", entry_id=entry_id)
            return True
        except redis.RedisError as e:
            logger.error("Failed to acknowledge job", entry_id=entry_id, error=str(e))
            return False

    def update_job_status(self, job_id: str, status: JobStatus) -> None:
        """
        Update job status in the metadata hash.

        Parameters
        ----------
        job_id : str
            Job identifier
        status : JobStatus
            New status
        """
        key = f"{KEY_PREFIX}:jobs:{job_id}"
        self.client.hset(key, "status", status.value)
        self.client.expire(key, JOB_TTL_S)

    # -------------------------------------------------------------------------
    # Result Operations
    # -------------------------------------------------------------------------

    def store_result(self, result: InferenceResult) -> None:
        """
        Store inference result in Redis.

        Parameters
        ----------
        result : InferenceResult
            Inference result to store
        """
        key = f"{KEY_PREFIX}:results:{result.job_id}"
        result_json = result.model_dump_json()

        # Store in hash
        self.client.hset(key, "result_data", result_json)

        # Set TTL (7 days)
        self.client.expire(key, JOB_TTL_S)

        # Also update the image -> job mapping
        image_key = f"{KEY_PREFIX}:images:{result.image_id}"
        self.client.hset(image_key, "latest_job_id", result.job_id)

        # Update job status
        self.update_job_status(result.job_id, result.status)

        logger.debug(
            "Stored inference result",
            job_id=result.job_id,
            status=result.status.value,
        )

    # -------------------------------------------------------------------------
    # Active Learning Operations
    # -------------------------------------------------------------------------

    def add_to_labeling_pool(
        self,
        image_id: str,
        anomaly_score: float,
        uncertainty_score: float,
    ) -> None:
        """
        Add a sample to the active learning labeling pool.

        Samples are stored in a sorted set, scored by uncertainty.
        This allows efficient retrieval of the most uncertain samples.

        Parameters
        ----------
        image_id : str
            Image identifier
        anomaly_score : float
            Model's anomaly prediction
        uncertainty_score : float
            Uncertainty measure (used as score)
        """
        pool_key = f"{KEY_PREFIX}:al:pool"
        sample_key = f"{KEY_PREFIX}:al:samples:{image_id}"

        # Create sample metadata
        sample = UnlabeledSample(
            image_id=image_id,
            anomaly_score=anomaly_score,
            uncertainty_score=uncertainty_score,
            added_at=datetime.utcnow(),
        )

        # Add to sorted set (score = uncertainty)
        self.client.zadd(pool_key, {image_id: uncertainty_score})

        # Store sample metadata
        self.client.set(sample_key, sample.model_dump_json())

        # Trim pool to max size (keep highest uncertainty)
        pool_size = self.client.zcard(pool_key)
        max_size = self.settings.al_pool_max_size
        if pool_size > max_size:
            # Remove lowest uncertainty samples, and their metadata with them:
            # dropping the sorted-set member alone orphans the matching
            # retina:al:samples:* string, which then has no TTL and no owner.
            last_rank = pool_size - max_size - 1
            evicted = self.client.zrange(pool_key, 0, last_rank)
            self.client.zremrangebyrank(pool_key, 0, last_rank)
            if evicted:
                self.client.delete(
                    *(f"{KEY_PREFIX}:al:samples:{i}" for i in evicted)
                )

        logger.debug(
            "Added sample to labeling pool",
            image_id=image_id,
            uncertainty=f"{uncertainty_score:.3f}",
        )

    # -------------------------------------------------------------------------
    # Label Operations
    # -------------------------------------------------------------------------

    def recent_labels(self, limit: int = 20) -> list[dict[str, str]]:
        """Return the most recently submitted labels, newest first.

        Ordering comes from LABEL_INDEX_KEY, a sorted set scored by the
        label's own ``labeled_at`` timestamp. SCAN returns keys in no defined
        order, so slicing its output was never "most recent" — it was an
        arbitrary 20.
        """
        try:
            self._sync_label_index()
            image_ids = self.client.zrevrange(LABEL_INDEX_KEY, 0, limit - 1)
            labels: list[dict[str, str]] = []
            for image_id in image_ids:
                data = self.client.hgetall(f"{KEY_PREFIX}:labels:{image_id}")
                if not data:
                    # The label hash expired (7d TTL); drop the stale pointer.
                    self.client.zrem(LABEL_INDEX_KEY, image_id)
                    continue
                labels.append(data)
            return labels
        except redis.RedisError as e:
            logger.warning("recent_labels_failed", error=str(e))
            return []

    def _sync_label_index(self) -> None:
        """Index any label the API wrote without touching the index.

        Labels are written by the API process, which does not maintain this
        index, so labels that predate it (or that any API version writes)
        are picked up here rather than being invisible. A label with no
        parseable ``labeled_at`` is scored 0 — it still appears, it just
        sorts oldest.
        """
        # ponytail: one SCAN per call, same O(n) as the code it replaces. The
        # upgrade path is a ZADD to retina:labels_index in the API's label
        # handler, after which this sweep can go.
        for key in self.client.scan_iter(
            match=f"{KEY_PREFIX}:labels:*", count=100,
        ):
            image_id = key.rsplit(":", 1)[-1]
            if self.client.zscore(LABEL_INDEX_KEY, image_id) is not None:
                continue
            self.client.zadd(
                LABEL_INDEX_KEY,
                {image_id: _parse_timestamp(self.client.hget(key, "labeled_at"))},
            )

    # -------------------------------------------------------------------------
    # Statistics Operations
    # -------------------------------------------------------------------------

    def increment_completed_jobs(self) -> int:
        """
        Increment the completed jobs counter.

        Returns
        -------
        int
            New count of completed jobs
        """
        key = f"{KEY_PREFIX}:system:stats"
        return self.client.hincrby(key, "jobs_completed", 1)

    # -------------------------------------------------------------------------
    # Alert Operations (Matching Professor's Framework)
    # -------------------------------------------------------------------------

    def send_alert(self, alert: dict) -> None:
        """
        Send a real-time alert for detected anomaly.

        Uses LPUSH to Redis list for real-time notification.
        Matches professor's framework alert pattern.

        Parameters
        ----------
        alert : dict
            Alert data containing job_id, user, label, timestamp
        """
        alerts_key = f"{KEY_PREFIX}:alerts"

        self.client.lpush(alerts_key, json.dumps(alert))

        # Trim to keep only last 100 alerts
        self.client.ltrim(alerts_key, 0, 99)

        logger.info("Alert sent", job_id=alert.get("job_id"))

    def get_result(self, job_id: str) -> dict | None:
        """
        Get stored result for a job.

        Parameters
        ----------
        job_id : str
            Job identifier

        Returns
        -------
        dict | None
            Result data if exists
        """
        key = f"{KEY_PREFIX}:results:{job_id}"
        result_json = self.client.hget(key, "result_data")

        if result_json:
            return json.loads(result_json)
        return None

    def update_result_unsupervised(
        self,
        job_id: str,
        unsupervised_label: bool,
        mismatch: bool,
    ) -> None:
        """
        Update result with unsupervised model output.

        Matches professor's framework pattern where unsupervised
        model runs in batch and updates existing records.

        Parameters
        ----------
        job_id : str
            Job identifier
        unsupervised_label : bool
            Unsupervised model's anomaly prediction
        mismatch : bool
            Whether supervised and unsupervised disagree
        """
        key = f"{KEY_PREFIX}:results:{job_id}"

        # Get existing result
        result_json = self.client.hget(key, "result_data")
        if not result_json:
            logger.warning("Result not found for unsupervised update", job_id=job_id)
            return

        # Update result
        result = json.loads(result_json)
        result["unsupervised_label"] = unsupervised_label
        result["mismatch"] = mismatch

        # Store updated result
        self.client.hset(key, "result_data", json.dumps(result))

        logger.debug(
            "Updated result with unsupervised",
            job_id=job_id,
            unsupervised=unsupervised_label,
            mismatch=mismatch,
        )
