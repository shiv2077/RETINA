"""
Worker Loop
===========

Main worker loop that consumes jobs from Redis and runs inference.

Architecture Overview
---------------------

The worker operates as follows:

1. Connect to Redis and join the consumer group
2. Poll for new jobs from the stream
3. For each job:
   a. Update status to "processing"
   b. Load the appropriate model
   c. Run inference
   d. Store results
   e. Add to labeling pool if uncertain
   f. Acknowledge the job
4. Repeat

Error Handling
--------------

Jobs that fail are marked with status "failed" and an error message.
The job is still acknowledged to prevent infinite retry loops.
A dead-letter queue pattern could be implemented for persistent failures.

Graceful Shutdown
-----------------

The worker handles SIGTERM/SIGINT signals for graceful shutdown,
completing any in-progress job before exiting.
"""

import signal
import time
from datetime import datetime
from typing import Optional

import structlog

from .config import Settings
from .models import get_model
from .redis_client import RedisClient
from .schemas import (
    ActiveLearningMeta,
    InferenceError,
    InferenceJob,
    InferenceResult,
    JobStatus,
    PipelineStage,
    Stage1Output,
    Stage2Output,
)

logger = structlog.get_logger()


class Worker:
    """
    ML inference worker that processes jobs from Redis.
    
    The worker continuously polls for new jobs and runs anomaly detection
    inference using the appropriate model (Stage 1 or Stage 2).
    
    Attributes
    ----------
    settings : Settings
        Worker configuration
    redis : RedisClient
        Redis client for job queue and results
    running : bool
        Whether the worker is currently running
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the worker.
        
        Parameters
        ----------
        settings : Optional[Settings]
            Worker configuration. If None, loads from environment.
        """
        self.settings = settings or Settings()
        self.redis = RedisClient(self.settings)
        self.running = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received", signal=signum)
        self.running = False
    
    def run(self) -> None:
        """
        Start the worker loop.
        
        The worker will continuously poll for jobs until a shutdown
        signal is received or an unrecoverable error occurs.
        """
        logger.info(
            "Starting RETINA ML Worker",
            consumer_name=self.settings.consumer_name,
            default_model=self.settings.default_unsupervised_model,
        )
        
        # Check Redis connection
        if not self.redis.health_check():
            logger.error("Cannot connect to Redis")
            return
        
        logger.info("Connected to Redis")
        self.running = True
        
        while self.running:
            try:
                self._process_next_job()
            except Exception as e:
                logger.exception("Unexpected error in worker loop", error=str(e))
                # Brief pause before retrying to avoid tight loop on persistent errors
                time.sleep(1)
        
        logger.info("Worker shutdown complete")
    
    def _process_next_job(self) -> None:
        """
        Process the next job from the queue.
        
        Blocks waiting for a job, then processes it.
        """
        # Read next job (blocking)
        job_data = self.redis.read_job(block_ms=5000)
        
        if job_data is None:
            # No job available, continue polling
            return
        
        entry_id, job = job_data
        
        logger.info(
            "Processing job",
            job_id=job.job_id,
            image_id=job.image_id,
            model=job.model_type.value,
            stage=job.stage.value,
        )
        
        # Update status to processing
        self.redis.update_job_status(job.job_id, JobStatus.PROCESSING)
        
        start_time = time.time()
        
        try:
            # Run inference
            result = self._run_inference(job)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time_ms
            
            # Store result
            self.redis.store_result(result)
            
            # Add to labeling pool if uncertain
            if result.active_learning.uncertainty_score > self.settings.uncertainty_threshold:
                self.redis.add_to_labeling_pool(
                    image_id=job.image_id,
                    anomaly_score=result.anomaly_score or 0.0,
                    uncertainty_score=result.active_learning.uncertainty_score,
                )
            
            # Increment completed counter
            self.redis.increment_completed_jobs()
            
            logger.info(
                "Job completed",
                job_id=job.job_id,
                anomaly_score=f"{result.anomaly_score:.3f}" if result.anomaly_score else "N/A",
                is_anomaly=result.is_anomaly,
                processing_time_ms=processing_time_ms,
            )
            
        except Exception as e:
            logger.exception("Job failed", job_id=job.job_id, error=str(e))
            
            # Store failure result
            result = InferenceResult(
                job_id=job.job_id,
                image_id=job.image_id,
                status=JobStatus.FAILED,
                stage=job.stage,
                error=InferenceError(
                    code="INFERENCE_ERROR",
                    message=str(e),
                ),
            )
            self.redis.store_result(result)
        
        finally:
            # Always acknowledge the job
            self.redis.acknowledge_job(entry_id)
    
    def _run_inference(self, job: InferenceJob) -> InferenceResult:
        """
        Run inference on a job.
        
        Parameters
        ----------
        job : InferenceJob
            Job to process
            
        Returns
        -------
        InferenceResult
            Inference result
        """
        # Simulate inference delay in debug mode
        if self.settings.debug_mode and self.settings.mock_inference_delay_ms > 0:
            delay_sec = self.settings.mock_inference_delay_ms / 1000.0
            logger.debug("Simulating inference delay", delay_ms=self.settings.mock_inference_delay_ms)
            time.sleep(delay_sec)
        
        # Get the appropriate model.
        # Cold-start fallback: if PatchCore is requested but has no memory bank yet,
        # route to GPT-4V so every image still gets scored from day one.
        from .models.patchcore_real import PatchCoreReal
        from .schemas import ModelType

        model = get_model(job.model_type)
        if job.model_type == ModelType.PATCHCORE and isinstance(model, PatchCoreReal):
            if not model.has_memory_bank:
                logger.info(
                    "PatchCore cold-start: no memory bank — routing to GPT-4V",
                    image_id=job.image_id,
                )
                model = get_model(ModelType.GPT4V)
        
        # Load image bytes from shared volume path if available.
        # The backend sets job.image_path when it saves the uploaded file.
        image_data: bytes | None = None
        if job.image_path:
            try:
                with open(job.image_path, "rb") as f:
                    image_data = f.read()
                logger.debug("Image loaded from path", path=job.image_path, bytes=len(image_data))
            except OSError as exc:
                logger.warning(
                    "Could not read image file — proceeding without image bytes",
                    path=job.image_path,
                    error=str(exc),
                )

        # Run prediction
        prediction = model.predict(
            image_id=job.image_id,
            image_data=image_data,
        )
        
        # Build result based on stage
        stage1_output = None
        stage2_output = None
        
        if job.stage == PipelineStage.UNSUPERVISED:
            stage1_output = Stage1Output(
                heatmap_available=prediction.heatmap is not None,
                heatmap_key=None,  # Would store heatmap and get key
                feature_distance=prediction.feature_distance,
                clip_similarity=prediction.clip_similarity,
            )
        else:
            stage2_output = Stage2Output(
                defect_category=prediction.defect_category,
                category_probabilities=prediction.category_probabilities,
                embedding_distance=prediction.embedding_distance,
            )
        
        # Build active learning metadata
        active_learning = ActiveLearningMeta(
            uncertainty_score=prediction.uncertainty,
            in_labeling_pool=False,  # Will be updated if added to pool
            labeled=False,
        )
        
        return InferenceResult(
            job_id=job.job_id,
            image_id=job.image_id,
            status=JobStatus.COMPLETED,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            model_used=job.model_type,
            stage=job.stage,
            anomaly_score=prediction.anomaly_score,
            is_anomaly=prediction.is_anomaly,
            confidence=prediction.confidence,
            stage1_output=stage1_output,
            stage2_output=stage2_output,
            active_learning=active_learning,
            # GPT-4V / VLM fields — None for non-VLM models
            defect_description=prediction.defect_description,
            defect_location=prediction.defect_location,
            gpt4v_reasoning=prediction.gpt4v_reasoning,
        )
