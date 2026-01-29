//! # Inference Job Models
//!
//! Defines the structure of inference jobs that flow through the system.
//!
//! ## Job Lifecycle
//!
//! 1. Client submits image via API → Job created with status `Pending`
//! 2. Backend pushes job to Redis Stream → Status becomes `Queued`
//! 3. Worker picks up job → Status becomes `Processing`
//! 4. Worker completes inference → Status becomes `Completed` or `Failed`
//!
//! ## Research Note
//!
//! The `model_type` field determines which anomaly detection approach is used:
//!
//! | Model | Type | Stage | Description |
//! |-------|------|-------|-------------|
//! | PatchCore | Memory-bank | 1 | Feature embedding + k-NN |
//! | PaDiM | Gaussian | 1 | Multivariate Gaussian modeling |
//! | WinCLIP | Zero-shot | 1 | CLIP-based text-image matching |
//! | Push-Pull | Contrastive | 2 | Supervised with labeled data |

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Type of anomaly detection model to use.
///
/// Stage 1 models (unsupervised/zero-shot):
/// - `PatchCore`: Memory-bank approach with feature embeddings
/// - `PaDiM`: Probabilistic approach with Gaussian modeling
/// - `WinCLIP`: Zero-shot detection using CLIP
///
/// Stage 2 models (supervised):
/// - `PushPull`: Contrastive learning with labeled samples
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    /// PatchCore: Memory-bank based anomaly detection
    /// Uses pretrained CNN features + k-NN for anomaly scoring
    /// Reference: Roth et al., "Towards Total Recall in Industrial Anomaly Detection"
    PatchCore,

    /// PaDiM: Patch Distribution Modeling
    /// Fits multivariate Gaussian to patch embeddings
    /// Reference: Defard et al., "PaDiM: A Patch Distribution Modeling Framework"
    PaDiM,

    /// WinCLIP: Zero-shot anomaly detection using CLIP
    /// Compares images against text prompts for normal/anomaly
    /// Reference: Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification"
    WinCLIP,

    /// Push-Pull: Contrastive learning for supervised classification
    /// Requires labeled samples from active learning
    /// Reference: Simplified version of BGAD approach
    PushPull,
}

impl Default for ModelType {
    fn default() -> Self {
        Self::PatchCore
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PatchCore => write!(f, "patchcore"),
            Self::PaDiM => write!(f, "padim"),
            Self::WinCLIP => write!(f, "winclip"),
            Self::PushPull => write!(f, "pushpull"),
        }
    }
}

/// Pipeline stage indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PipelineStage {
    /// Stage 1: Unsupervised/Zero-shot detection
    /// Used when no labeled defect data exists
    #[serde(rename = "1")]
    Unsupervised = 1,

    /// Stage 2: Supervised classification
    /// Activated after collecting labeled samples via active learning
    #[serde(rename = "2")]
    Supervised = 2,
}

impl Default for PipelineStage {
    fn default() -> Self {
        Self::Unsupervised
    }
}

/// Current status of an inference job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// Job has been created but not yet queued
    Pending,

    /// Job has been added to the Redis queue
    Queued,

    /// Worker is currently processing the job
    Processing,

    /// Inference completed successfully
    Completed,

    /// Inference failed (see error field for details)
    Failed,
}

impl Default for JobStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// Optional metadata for job tracking and debugging.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JobMetadata {
    /// Origin of the submission
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Batch identifier (if part of batch submission)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_id: Option<String>,

    /// Webhook URL for result notification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callback_url: Option<String>,
}

/// An inference job to be processed by the ML worker.
///
/// This structure is serialized to JSON and pushed to a Redis Stream
/// for consumption by Python workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceJob {
    /// Unique identifier for this job
    pub job_id: Uuid,

    /// Reference to the submitted image
    pub image_id: String,

    /// Model to use for inference
    pub model_type: ModelType,

    /// Pipeline stage (1 = unsupervised, 2 = supervised)
    pub stage: PipelineStage,

    /// Job priority (0-10, higher = more urgent)
    #[serde(default = "default_priority")]
    pub priority: u8,

    /// Current job status
    #[serde(default)]
    pub status: JobStatus,

    /// Timestamp when job was submitted
    pub submitted_at: DateTime<Utc>,

    /// Optional metadata for tracking
    #[serde(default)]
    pub metadata: JobMetadata,
}

fn default_priority() -> u8 {
    5
}

impl InferenceJob {
    /// Create a new inference job with default settings.
    pub fn new(image_id: String, model_type: ModelType) -> Self {
        Self {
            job_id: Uuid::new_v4(),
            image_id,
            model_type,
            stage: PipelineStage::Unsupervised,
            priority: 5,
            status: JobStatus::Pending,
            submitted_at: Utc::now(),
            metadata: JobMetadata::default(),
        }
    }

    /// Set the pipeline stage for this job.
    pub fn with_stage(mut self, stage: PipelineStage) -> Self {
        self.stage = stage;
        self
    }

    /// Set the priority for this job.
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }

    /// Add metadata to this job.
    pub fn with_metadata(mut self, metadata: JobMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Request body for submitting a new inference job.
#[derive(Debug, Clone, Deserialize)]
pub struct SubmitJobRequest {
    /// Image identifier (in current implementation, this is a simulated ID)
    pub image_id: String,

    /// Model to use (optional, defaults to PatchCore)
    #[serde(default)]
    pub model_type: Option<ModelType>,

    /// Priority (optional, defaults to 5)
    pub priority: Option<u8>,

    /// Submission source (optional)
    pub source: Option<String>,
}

/// Response after submitting a new inference job.
#[derive(Debug, Clone, Serialize)]
pub struct SubmitJobResponse {
    /// The created job ID
    pub job_id: Uuid,

    /// Current status
    pub status: JobStatus,

    /// Estimated position in queue
    pub queue_position: Option<u64>,

    /// Message for the client
    pub message: String,
}
