"""
Data Schemas
============

Pydantic models for job and result data structures.
These match the JSON schemas in /shared/schemas/ and the Rust backend models.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """
    Anomaly detection model types.
    
    Stage 1 (Unsupervised/Zero-shot):
    - PATCHCORE: Memory-bank with pretrained features + k-NN
    - PADIM: Gaussian modeling of patch distributions
    - WINCLIP: CLIP-based zero-shot detection
    
    Stage 2 (Supervised):
    - PUSHPULL: Contrastive learning with labeled samples
    """
    PATCHCORE = "patchcore"
    PADIM = "padim"
    WINCLIP = "winclip"
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
    source: Optional[str] = None
    batch_id: Optional[str] = None
    callback_url: Optional[str] = None


class InferenceJob(BaseModel):
    """
    An inference job received from the Redis queue.
    
    This structure matches the Rust backend's InferenceJob struct
    to ensure seamless JSON serialization across language boundaries.
    """
    job_id: str = Field(..., description="Unique job identifier (UUID)")
    image_id: str = Field(..., description="Reference to the image")
    model_type: ModelType = Field(default=ModelType.PATCHCORE)
    stage: PipelineStage = Field(default=PipelineStage.UNSUPERVISED)
    priority: int = Field(default=5, ge=0, le=10)
    status: JobStatus = Field(default=JobStatus.PENDING)
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: JobMetadata = Field(default_factory=JobMetadata)


class Stage1Output(BaseModel):
    """Stage 1 (unsupervised) specific outputs."""
    heatmap_available: bool = False
    heatmap_key: Optional[str] = None
    feature_distance: Optional[float] = None
    clip_similarity: Optional[float] = None


class Stage2Output(BaseModel):
    """Stage 2 (supervised) specific outputs."""
    defect_category: Optional[str] = None
    category_probabilities: Optional[dict[str, float]] = None
    embedding_distance: Optional[float] = None


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
    completed_at: Optional[datetime] = None
    model_used: Optional[ModelType] = None
    stage: PipelineStage = PipelineStage.UNSUPERVISED
    
    # Core prediction outputs
    anomaly_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    is_anomaly: Optional[bool] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Stage-specific outputs
    stage1_output: Optional[Stage1Output] = None
    stage2_output: Optional[Stage2Output] = None
    
    # Active learning metadata
    active_learning: ActiveLearningMeta = Field(default_factory=ActiveLearningMeta)
    
    # Error handling
    error: Optional[InferenceError] = None
    
    # Performance metrics
    processing_time_ms: Optional[int] = None


class UnlabeledSample(BaseModel):
    """A sample in the active learning pool awaiting labeling."""
    image_id: str
    anomaly_score: float
    uncertainty_score: float
    added_at: datetime = Field(default_factory=datetime.utcnow)
