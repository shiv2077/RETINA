"""
Data Schemas
============

Pydantic models for job and result data structures.
These match the JSON schemas in /shared/schemas/ and the Rust backend models.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """
    Anomaly detection model types.

    Stage 1 (Unsupervised/Zero-shot):
    - PATCHCORE: Memory-bank with pretrained features + k-NN
    - PADIM: Gaussian modeling of patch distributions
    - WINCLIP: CLIP-based zero-shot detection (legacy stub)
    - GPT4V: GPT-4o Vision API — zero-shot VLM detector (replaces WinCLIP)

    Stage 2 (Supervised):
    - PUSHPULL: Contrastive learning with labeled samples
    """
    PATCHCORE = "patchcore"
    PADIM = "padim"
    WINCLIP = "winclip"
    GPT4V = "gpt4v"
    PUSHPULL = "pushpull"


class PipelineStage(int, Enum):
    """Pipeline stage indicator."""
    UNSUPERVISED = 1
    SUPERVISED = 2


class JobStatus(str, Enum):
    """Job lifecycle status."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobMetadata(BaseModel):
    """Optional metadata for job tracking."""
    source: str | None = None
    batch_id: str | None = None
    callback_url: str | None = None


class InferenceJob(BaseModel):
    """
    An inference job received from the Redis queue.

    This structure matches the Rust backend's InferenceJob struct
    to ensure seamless JSON serialization across language boundaries.
    """
    job_id: str = Field(..., description="Unique job identifier (UUID)")
    # Content address: sha256 of the image bytes. The submitter and the worker
    # each resolve it against their OWN image root via Settings.image_path(),
    # so the job payload stays portable across a container boundary. It must
    # never carry a resolved filesystem path again — a host-absolute path in
    # here is unreadable from inside the worker container (CLAUDE.md §8).
    image_id: str = Field(..., description="sha256 content address of the image")
    model_type: ModelType = Field(default=ModelType.PATCHCORE)
    stage: PipelineStage = Field(default=PipelineStage.UNSUPERVISED)
    priority: int = Field(default=5, ge=0, le=10)
    status: JobStatus = Field(default=JobStatus.PENDING)
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: JobMetadata = Field(default_factory=JobMetadata)
    # Operator-declared product category. When present the worker routes
    # straight to that PatchCore checkpoint and never calls identify_product.
    # The API validates it against the checkpoints on disk before enqueueing,
    # so the worker can trust it. Absent means "infer it" — the cold-start
    # path, not the normal one (DECISIONS.md 17).
    product_class: str | None = Field(
        None, description="Declared product category; skips VLM identification"
    )


class Stage1Output(BaseModel):
    """Stage 1 (unsupervised) specific outputs."""
    heatmap_available: bool = False
    heatmap_key: str | None = None
    feature_distance: float | None = None
    clip_similarity: float | None = None


class Stage2Output(BaseModel):
    """Stage 2 (supervised) specific outputs."""
    defect_category: str | None = None
    category_probabilities: dict[str, float] | None = None
    embedding_distance: float | None = None


class ActiveLearningMeta(BaseModel):
    """Active learning metadata for sample selection."""
    uncertainty_score: float = 0.0
    in_labeling_pool: bool = False
    labeled: bool = False


class InferenceError(BaseModel):
    """Error information for failed jobs."""
    code: str
    message: str


class InferenceResult(BaseModel):
    """
    Complete inference result to be stored in Redis.

    This structure matches the Rust backend's InferenceResult struct.
    """
    job_id: str
    image_id: str
    status: JobStatus = JobStatus.COMPLETED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    model_used: ModelType | None = None
    stage: PipelineStage = PipelineStage.UNSUPERVISED

    # Core prediction outputs
    anomaly_score: float | None = Field(None, ge=0.0, le=1.0)
    is_anomaly: bool | None = None
    confidence: float | None = Field(None, ge=0.0, le=1.0)

    # Stage-specific outputs
    stage1_output: Stage1Output | None = None
    stage2_output: Stage2Output | None = None

    # Active learning metadata
    active_learning: ActiveLearningMeta = Field(default_factory=ActiveLearningMeta)

    # Error handling
    error: InferenceError | None = None

    # Performance metrics
    processing_time_ms: int | None = None

    # ── VLM / router outputs (top-level) ───────────────────────────────────
    # Legacy GPT-4V fields (now populated by the VLM router as well; kept
    # for backward compatibility with existing records).
    defect_description: str | None = None
    defect_location: str | None = None
    gpt4v_reasoning: str | None = None

    # New VLM-router fields — see docs/vlm_router_design.md for usage.
    product_class: str | None = None
    product_confidence: float | None = Field(None, ge=0.0, le=1.0)
    natural_description: str | None = None
    defect_severity: str | None = None   # "minor" | "moderate" | "severe"
    defect_type: str | None = None
    routing_reason: str | None = None    # see routing_reason enum in result.json
    vlm_model_used: str | None = None    # "gpt-4o" | "gpt-4o-mini"
    vlm_api_cost_estimate_usd: float | None = Field(None, ge=0.0)

    # Stage 2 supervised refiner — populated only for stage1 scores inside the
    # configured Stage 2 band (settings.anomaly_threshold..stage2_trigger_max).
    # verdict: confirmed_anomaly | rejected_false_positive | uncertain
    stage2_verdict: str | None = None
    stage2_defect_class: str | None = None
    stage2_confidence: float | None = Field(None, ge=0.0, le=1.0)


class UnlabeledSample(BaseModel):
    """A sample in the active learning pool awaiting labeling."""
    image_id: str
    anomaly_score: float
    uncertainty_score: float
    added_at: datetime = Field(default_factory=datetime.utcnow)
