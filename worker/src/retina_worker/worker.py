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

import io
import signal
import time
from datetime import datetime
from typing import Optional

import numpy as np
import structlog
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as TVF

from .config import Settings
from .models.patchcore_registry import get_default_registry
from .models.vlm_router import VLMRouter
from .redis_client import RedisClient
from .schemas import (
    ActiveLearningMeta,
    InferenceError,
    InferenceJob,
    InferenceResult,
    JobStatus,
    ModelType,
    PipelineStage,
    Stage1Output,
)

logger = structlog.get_logger()


# ── VLM cost constants (per-call USD estimates) ──────────────────────────
COST_IDENTIFY = 0.00015       # gpt-4o-mini, product identification
COST_DESCRIBE = 0.00500       # gpt-4o, describe_defect on anomalous image
COST_ZERO_SHOT = 0.00500      # gpt-4o, zero_shot_detect for unknown products
COST_STAGE2_REFINE = 0.00500  # gpt-4o, stage2_refine on uncertainty-zone scores

# ── Shared-session product cache keys ────────────────────────────────────
SESSION_PRODUCT_KEY = "retina:session:product_class"
SESSION_CONFIDENCE_KEY = "retina:session:product_confidence"
SESSION_TTL_S = 3600


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

        # Router + registry — loaded lazily on first inference.
        self.registry = get_default_registry()
        self.vlm_router = VLMRouter(api_key=self.settings.openai_api_key)
        self._session_product_class: Optional[str] = None
        self._session_product_confidence: Optional[float] = None
        self._score_clamp_warned: bool = False

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
        Run inference on a job using the VLM router + per-category PatchCore
        registry. Flow:
          1. Identify product via gpt-4o-mini (session-cached).
          2. If no PatchCore checkpoint for the product → zero-shot via GPT-4o.
          3. Else run PatchCore; if anomalous, describe the defect via GPT-4o.
        """
        t_start = time.time()

        # Optional debug delay.
        if self.settings.debug_mode and self.settings.mock_inference_delay_ms > 0:
            time.sleep(self.settings.mock_inference_delay_ms / 1000.0)

        image_bytes = self._load_image_bytes(job)
        if image_bytes is None:
            raise RuntimeError(
                f"Cannot run inference without image bytes (job_id={job.job_id}, "
                f"image_path={job.image_path!r})"
            )

        vlm_cost_usd = 0.0

        # ── Step 1: product identification (session-cached) ─────────────
        product_class = self._get_cached_product_class()
        product_confidence = self._session_product_confidence if product_class else None

        if product_class is None:
            logger.info("identifying_product", image_id=job.image_id)
            pid = self.vlm_router.identify_product(image_bytes)
            product_class = pid.product_class
            product_confidence = pid.confidence
            self._set_cached_product_class(product_class, product_confidence)
            vlm_cost_usd += COST_IDENTIFY
        else:
            logger.debug("using_cached_product", product_class=product_class)

        # ── Step 2: zero-shot branch for unknown or untrained products ──
        if product_class == "unknown" or not self.registry.has_checkpoint(product_class):
            logger.info(
                "zero_shot_detection",
                product_class=product_class,
                reason="no_checkpoint_available",
            )
            zs = self.vlm_router.zero_shot_detect(
                image_bytes,
                product_description=(
                    product_class if product_class != "unknown" else "industrial product"
                ),
            )
            vlm_cost_usd += COST_ZERO_SHOT
            return self._build_result(
                job=job,
                anomaly_score=zs.anomaly_score,
                is_anomaly=zs.is_anomaly,
                product_class=product_class,
                product_confidence=product_confidence,
                natural_description=zs.reasoning,
                defect_type=zs.suggested_defect_type,
                defect_location=None,
                defect_severity=None,
                routing_reason="unknown_product_zero_shot",
                vlm_model_used="gpt-4o",
                vlm_api_cost_estimate_usd=vlm_cost_usd,
                heatmap=None,
                model_used=ModelType.GPT4V,
                t_start=t_start,
            )

        # ── Step 3: PatchCore on the identified category ───────────────
        model = self.registry.get(product_class)
        anomaly_score, heatmap = self._run_patchcore(model, image_bytes)
        is_anomaly = anomaly_score > self.settings.anomaly_threshold

        # ── Step 4: if anomalous, VLM describes it ─────────────────────
        natural_description = None
        defect_type = None
        defect_location = None
        defect_severity = None
        vlm_model_used: Optional[str] = None
        routing_reason = "patchcore_normal"

        if is_anomaly:
            routing_reason = "patchcore_confirmed_anomaly"
            logger.info(
                "describing_defect",
                product_class=product_class,
                score=round(anomaly_score, 4),
            )
            desc = self.vlm_router.describe_defect(
                image_bytes,
                product_class=product_class,
                anomaly_score=anomaly_score,
            )
            natural_description = desc.natural_description
            defect_type = desc.defect_type
            defect_location = desc.location
            defect_severity = desc.severity
            vlm_model_used = "gpt-4o"
            vlm_cost_usd += COST_DESCRIBE

        # ── Stage 2: supervised refinement in the uncertainty zone ──────
        stage2_verdict: Optional[str] = None
        stage2_defect_class: Optional[str] = None
        stage2_confidence: Optional[float] = None

        if is_anomaly and self.vlm_router.should_run_stage2(anomaly_score):
            logger.info(
                "stage2_running",
                product_class=product_class,
                stage1_score=round(anomaly_score, 4),
            )
            labeled = self._fetch_labeled_examples(product_class, k=5)
            s2 = self.vlm_router.stage2_refine(
                image_bytes=image_bytes,
                product_class=product_class,
                stage1_score=anomaly_score,
                labeled_examples=labeled or None,
            )
            stage2_verdict = s2.verdict
            stage2_defect_class = s2.defect_class
            stage2_confidence = s2.confidence
            vlm_cost_usd += COST_STAGE2_REFINE
            vlm_model_used = "gpt-4o"

            if s2.verdict == "rejected_false_positive" and s2.confidence >= 0.7:
                logger.info(
                    "stage2_overrode_stage1",
                    was_anomaly=True,
                    now_normal=True,
                    stage2_confidence=s2.confidence,
                )
                is_anomaly = False
                routing_reason = "stage2_rejected"
            elif s2.verdict == "confirmed_anomaly":
                routing_reason = "stage2_confirmed"
            else:
                routing_reason = "stage2_uncertain_kept"

        return self._build_result(
            job=job,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            product_class=product_class,
            product_confidence=product_confidence,
            natural_description=natural_description,
            defect_type=defect_type,
            defect_location=defect_location,
            defect_severity=defect_severity,
            routing_reason=routing_reason,
            vlm_model_used=vlm_model_used,
            vlm_api_cost_estimate_usd=vlm_cost_usd,
            heatmap=heatmap,
            model_used=ModelType.PATCHCORE,
            t_start=t_start,
            stage2_verdict=stage2_verdict,
            stage2_defect_class=stage2_defect_class,
            stage2_confidence=stage2_confidence,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _fetch_labeled_examples(
        self, product_class: str, k: int = 5,
    ) -> list[dict]:
        """Return up to k operator-labeled examples matching product_class.

        Text-only metadata — image bytes are not loaded (Stage 2 keeps its
        few-shot context text to cap per-call cost).
        """
        try:
            label_keys = list(self.redis.client.scan_iter(match="retina:labels:*"))
            examples: list[dict] = []
            for key in label_keys[-20:]:
                data = self.redis.client.hgetall(key)
                if not data:
                    continue
                if data.get("product_class") == product_class:
                    examples.append({
                        "label": data.get("label", "unknown"),
                        "defect_class": data.get("defect_class") or None,
                    })
                    if len(examples) >= k:
                        break
            return examples
        except Exception as e:
            logger.warning("stage2_label_fetch_failed", error=str(e))
            return []

    def _load_image_bytes(self, job: InferenceJob) -> Optional[bytes]:
        """Read the image file written by the submitter, or return None."""
        if not job.image_path:
            return None
        try:
            with open(job.image_path, "rb") as f:
                data = f.read()
            logger.debug("image_loaded", path=job.image_path, bytes=len(data))
            return data
        except OSError as exc:
            logger.warning(
                "image_read_failed",
                path=job.image_path,
                error=str(exc),
            )
            return None

    def _get_cached_product_class(self) -> Optional[str]:
        """Return the current session product class from Redis (or None)."""
        pc = self.redis.client.get(SESSION_PRODUCT_KEY)
        if pc is None:
            self._session_product_class = None
            self._session_product_confidence = None
            return None
        self._session_product_class = pc
        conf = self.redis.client.get(SESSION_CONFIDENCE_KEY)
        try:
            self._session_product_confidence = float(conf) if conf is not None else None
        except (TypeError, ValueError):
            self._session_product_confidence = None
        return pc

    def _set_cached_product_class(self, product_class: str, confidence: float) -> None:
        """Store the session product class in Redis with a 1-hour TTL."""
        self.redis.client.set(SESSION_PRODUCT_KEY, product_class, ex=SESSION_TTL_S)
        self.redis.client.set(
            SESSION_CONFIDENCE_KEY, f"{confidence:.4f}", ex=SESSION_TTL_S,
        )
        self._session_product_class = product_class
        self._session_product_confidence = confidence

    def _run_patchcore(
        self, model, image_bytes: bytes,
    ) -> tuple[float, Optional[np.ndarray]]:
        """Run Patchcore on raw image bytes; return (score, heatmap)."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = TVF.to_image(img)
        tensor = TVF.to_dtype(tensor, torch.float32, scale=True)
        batch = tensor.unsqueeze(0)
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            out = model(batch)
        score = float(out.pred_score.item())
        if not (0.0 <= score <= 1.0):
            if not self._score_clamp_warned:
                logger.warning("patchcore_score_out_of_range", raw_score=score)
                self._score_clamp_warned = True
            score = max(0.0, min(1.0, score))
        heatmap: Optional[np.ndarray] = None
        if out.anomaly_map is not None:
            heatmap = out.anomaly_map.squeeze().cpu().numpy()
        return score, heatmap

    def _build_result(
        self,
        *,
        job: InferenceJob,
        anomaly_score: float,
        is_anomaly: bool,
        product_class: Optional[str],
        product_confidence: Optional[float],
        natural_description: Optional[str],
        defect_type: Optional[str],
        defect_location: Optional[str],
        defect_severity: Optional[str],
        routing_reason: str,
        vlm_model_used: Optional[str],
        vlm_api_cost_estimate_usd: float,
        heatmap: Optional[np.ndarray],
        model_used: ModelType,
        t_start: float,
        stage2_verdict: Optional[str] = None,
        stage2_defect_class: Optional[str] = None,
        stage2_confidence: Optional[float] = None,
    ) -> InferenceResult:
        """Assemble the final InferenceResult. Stage2Output stays None."""
        uncertainty = 1.0 - abs(2.0 * anomaly_score - 1.0)
        confidence = 1.0 - uncertainty
        now = datetime.utcnow()
        cost = vlm_api_cost_estimate_usd if vlm_api_cost_estimate_usd > 0 else None

        return InferenceResult(
            job_id=job.job_id,
            image_id=job.image_id,
            status=JobStatus.COMPLETED,
            created_at=now,
            completed_at=now,
            model_used=model_used,
            stage=job.stage,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            confidence=confidence,
            stage1_output=Stage1Output(
                heatmap_available=heatmap is not None,
                heatmap_key=None,  # blob storage deferred
            ),
            stage2_output=None,
            active_learning=ActiveLearningMeta(
                uncertainty_score=uncertainty,
                in_labeling_pool=False,
                labeled=False,
            ),
            processing_time_ms=int((time.time() - t_start) * 1000),
            product_class=product_class,
            product_confidence=product_confidence,
            natural_description=natural_description,
            defect_type=defect_type,
            defect_location=defect_location,
            defect_severity=defect_severity,
            routing_reason=routing_reason,
            vlm_model_used=vlm_model_used,
            vlm_api_cost_estimate_usd=cost,
            stage2_verdict=stage2_verdict,
            stage2_defect_class=stage2_defect_class,
            stage2_confidence=stage2_confidence,
        )
