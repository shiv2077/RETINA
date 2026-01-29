"""
Worker Loop - Dual Model Flow
==============================

Main worker loop implementing professor's framework pattern with:
- Supervised model for immediate classification
- Unsupervised model for batched verification 
- Mismatch detection between the two models

Architecture (Matching Professor's Framework)
---------------------------------------------

Two concurrent loops:

1. Job Consumer Loop:
   - Reads jobs from Redis stream
   - Runs supervised model immediately (is_anomaly, label)
   - Stores result in database
   - If anomaly detected: sends real-time alert

2. Unsupervised Checker Loop:
   - Runs every BATCH_INTERVAL seconds (default 60)
   - Collects records that haven't been checked by unsupervised
   - Runs unsupervised model in batch
   - Updates database with unsupervised_label
   - Detects mismatch (supervised != unsupervised)

This dual-model approach:
- Provides fast initial classification
- Validates with unsupervised to catch edge cases
- Flags mismatches for expert review
"""

import asyncio
import signal
import time
from datetime import datetime
from typing import Optional, List, Tuple

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
    ModelType,
)

logger = structlog.get_logger()


class DualModelWorker:
    """
    ML inference worker with dual-model flow matching professor's framework.
    
    Implements:
    - Supervised model for immediate prediction
    - Unsupervised model for batch verification
    - Mismatch detection and alerting
    """
    
    # Batch interval for unsupervised checking (seconds)
    UNSUPERVISED_BATCH_INTERVAL = 60
    
    # Minimum batch size to run unsupervised
    MIN_BATCH_SIZE = 5
    
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
        
        # Buffer for unsupervised checking
        self.unsupervised_buffer: List[Tuple[str, str]] = []  # (job_id, image_path)
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
    
    def _handle_shutdown(self, signum: int, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Shutdown signal received", signal=signum)
        self.running = False
    
    def run(self) -> None:
        """
        Start the dual worker loops.
        
        Runs:
        1. Job consumer loop (continuous)
        2. Unsupervised checker loop (periodic)
        """
        logger.info(
            "Starting RETINA Dual Model Worker",
            consumer_name=self.settings.consumer_name,
            supervised_model=self.settings.default_supervised_model,
            unsupervised_model=self.settings.default_unsupervised_model,
        )
        
        # Check Redis connection
        if not self.redis.health_check():
            logger.error("Cannot connect to Redis")
            return
        
        logger.info("Connected to Redis")
        self.running = True
        
        # Run both loops
        # In a real implementation, these would be asyncio tasks
        self._job_consumer_loop()
    
    def _job_consumer_loop(self) -> None:
        """
        Main job consumer loop.
        
        Processes jobs immediately with supervised model.
        """
        last_unsupervised_check = time.time()
        
        while self.running:
            try:
                self._process_next_job()
                
                # Check if it's time to run unsupervised batch
                if time.time() - last_unsupervised_check > self.UNSUPERVISED_BATCH_INTERVAL:
                    self._run_unsupervised_batch()
                    last_unsupervised_check = time.time()
                    
            except Exception as e:
                logger.exception("Unexpected error in worker loop", error=str(e))
                time.sleep(1)
        
        # Final unsupervised check on shutdown
        if self.unsupervised_buffer:
            self._run_unsupervised_batch()
        
        logger.info("Worker shutdown complete")
    
    def _process_next_job(self) -> None:
        """
        Process the next job from the queue.
        
        Uses supervised model for immediate classification.
        Sends alert if anomaly detected.
        """
        # Read next job (blocking)
        job_data = self.redis.read_job(block_ms=5000)
        
        if job_data is None:
            return
        
        entry_id, job = job_data
        
        logger.info(
            "Processing job (supervised)",
            job_id=job.job_id,
            image_id=job.image_id,
        )
        
        # Update status to processing
        self.redis.update_job_status(job.job_id, JobStatus.PROCESSING)
        
        start_time = time.time()
        
        try:
            # Run supervised model
            result = self._run_supervised_inference(job)
            
            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time_ms
            
            # Store result
            self.redis.store_result(result)
            
            # If anomaly detected, send alert
            if result.is_anomaly:
                self._send_alert(job, result)
            
            # Add to unsupervised buffer for later verification
            self.unsupervised_buffer.append((job.job_id, job.image_id))
            
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
                "Job completed (supervised)",
                job_id=job.job_id,
                is_anomaly=result.is_anomaly,
                label=result.stage2_output.defect_category if result.stage2_output else None,
                processing_time_ms=processing_time_ms,
            )
            
        except Exception as e:
            logger.exception("Job failed", job_id=job.job_id, error=str(e))
            
            result = InferenceResult(
                job_id=job.job_id,
                image_id=job.image_id,
                status=JobStatus.FAILED,
                stage=PipelineStage.SUPERVISED,
                error=InferenceError(
                    code="INFERENCE_ERROR",
                    message=str(e),
                ),
            )
            self.redis.store_result(result)
        
        finally:
            self.redis.acknowledge_job(entry_id)
    
    def _run_supervised_inference(self, job: InferenceJob) -> InferenceResult:
        """
        Run supervised model inference.
        
        Returns immediate classification result.
        """
        # Simulate inference delay in debug mode
        if self.settings.debug_mode and self.settings.mock_inference_delay_ms > 0:
            time.sleep(self.settings.mock_inference_delay_ms / 1000.0)
        
        # Get supervised model
        model = get_model(ModelType.PUSH_PULL)
        
        # Run prediction
        prediction = model.predict(
            image_id=job.image_id,
            image_data=None,  # TODO: Fetch from storage
        )
        
        stage2_output = Stage2Output(
            defect_category=prediction.defect_category,
            category_probabilities=prediction.category_probabilities,
            embedding_distance=prediction.embedding_distance,
        )
        
        active_learning = ActiveLearningMeta(
            uncertainty_score=prediction.uncertainty,
            in_labeling_pool=False,
            labeled=False,
        )
        
        return InferenceResult(
            job_id=job.job_id,
            image_id=job.image_id,
            status=JobStatus.COMPLETED,
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            model_used=ModelType.PUSH_PULL,
            stage=PipelineStage.SUPERVISED,
            anomaly_score=prediction.anomaly_score,
            is_anomaly=prediction.is_anomaly,
            confidence=prediction.confidence,
            stage2_output=stage2_output,
            active_learning=active_learning,
        )
    
    def _run_unsupervised_batch(self) -> None:
        """
        Run unsupervised model on buffered samples.
        
        Matches professor's unsupervised_checker_loop pattern:
        - Batch processing for efficiency
        - Update unsupervised_label in results
        - Detect mismatch between supervised and unsupervised
        """
        if len(self.unsupervised_buffer) < self.MIN_BATCH_SIZE:
            logger.debug(
                "Skipping unsupervised batch (not enough samples)",
                buffer_size=len(self.unsupervised_buffer),
            )
            return
        
        logger.info(
            "Running unsupervised batch",
            batch_size=len(self.unsupervised_buffer),
        )
        
        # Get unsupervised model
        model = get_model(ModelType.PATCHCORE)
        
        mismatches = []
        
        for job_id, image_id in self.unsupervised_buffer:
            try:
                # Run unsupervised inference
                prediction = model.predict(
                    image_id=image_id,
                    image_data=None,  # TODO: Fetch from storage
                )
                
                unsupervised_label = prediction.is_anomaly
                
                # Get supervised result to check for mismatch
                supervised_result = self.redis.get_result(job_id)
                
                if supervised_result:
                    supervised_label = supervised_result.get("is_anomaly", False)
                    
                    # Detect mismatch
                    mismatch = supervised_label != unsupervised_label
                    
                    if mismatch:
                        mismatches.append({
                            "job_id": job_id,
                            "image_id": image_id,
                            "supervised": supervised_label,
                            "unsupervised": unsupervised_label,
                        })
                        
                        logger.warning(
                            "Mismatch detected",
                            job_id=job_id,
                            supervised=supervised_label,
                            unsupervised=unsupervised_label,
                        )
                    
                    # Update result with unsupervised info
                    self.redis.update_result_unsupervised(
                        job_id=job_id,
                        unsupervised_label=unsupervised_label,
                        mismatch=mismatch,
                    )
                
            except Exception as e:
                logger.exception(
                    "Failed unsupervised check",
                    job_id=job_id,
                    error=str(e),
                )
        
        # Clear buffer
        self.unsupervised_buffer.clear()
        
        if mismatches:
            logger.warning(
                "Batch complete with mismatches",
                total=len(mismatches),
                mismatches=mismatches,
            )
        else:
            logger.info("Batch complete, no mismatches")
    
    def _send_alert(self, job: InferenceJob, result: InferenceResult) -> None:
        """
        Send real-time alert for detected anomaly.
        
        Matches professor's framework alert pattern.
        """
        label = "unknown"
        if result.stage2_output and result.stage2_output.defect_category:
            label = result.stage2_output.defect_category
        
        alert = {
            "job_id": job.job_id,
            "user": getattr(job, "user", "system"),
            "label": label,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        self.redis.send_alert(alert)
        
        logger.info(
            "Alert sent",
            job_id=job.job_id,
            label=label,
        )


# Keep the original Worker class for backward compatibility
class Worker(DualModelWorker):
    """
    Backward compatible worker class.
    
    Now uses dual-model flow by default.
    """
    pass
