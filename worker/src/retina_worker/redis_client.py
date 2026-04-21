"""
Redis Client Utilities
======================

Provides Redis connection management and helper functions for the worker.

Key Schema
----------

The worker interacts with the following Redis keys:

**Job Queue (Stream)**
- ``retina:jobs:queue`` - Inference job queue (consumed via XREADGROUP)

**Job Metadata (Hashes)**
- ``retina:jobs:{job_id}`` - Job data and status

**Results (Hashes)**
- ``retina:results:{job_id}`` - Inference results

**Active Learning Pool (Sorted Set)**
- ``retina:al:pool`` - Samples awaiting labeling (scored by uncertainty)

**Active Learning Samples (Strings)**
- ``retina:al:samples:{image_id}`` - Sample metadata JSON

**System Stats (Hash)**
- ``retina:system:stats`` - Counters and metrics
"""

from datetime import datetime
from typing import Any

import redis
import structlog

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
WORKER_GROUP = "workers"


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
            
            if not result:
                return None
            
            # Parse result: [[stream_name, [[entry_id, {fields}]]]]
            stream_name, messages = result[0]
            entry_id, fields = messages[0]
            
            # Parse job data
            job_json = fields.get("job_data")
            if not job_json:
                logger.error("Job missing job_data field", entry_id=entry_id)
                return None
            
            import json
            job_data = json.loads(job_json)
            job = InferenceJob(**job_data)
            
            logger.debug(
                "Read job from queue",
                entry_id=entry_id,
                job_id=job.job_id,
                model_type=job.model_type.value,
            )
            
            return entry_id, job
            
        except redis.RedisError as e:
            logger.error("Failed to read job from queue", error=str(e))
            return None
    
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
        import json
        
        key = f"{KEY_PREFIX}:results:{result.job_id}"
        result_json = result.model_dump_json()
        
        # Store in hash
        self.client.hset(key, "result_data", result_json)
        
        # Set TTL (7 days)
        self.client.expire(key, 7 * 24 * 60 * 60)
        
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
        import json
        
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
            # Remove lowest uncertainty samples
            self.client.zremrangebyrank(pool_key, 0, pool_size - max_size - 1)
        
        logger.debug(
            "Added sample to labeling pool",
            image_id=image_id,
            uncertainty=f"{uncertainty_score:.3f}",
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
        import json
        
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
        import json
        
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
        import json
        
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
