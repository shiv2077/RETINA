//! # Anomaly Detection Result Models
//!
//! Defines the structure of inference results returned by the ML worker.
//!
//! ## Result Structure
//!
//! Results contain both Stage 1 and Stage 2 specific outputs:
//!
//! - **Stage 1 Output**: Anomaly scores, heatmaps, feature distances
//! - **Stage 2 Output**: Defect categories, classification probabilities
//!
//! Additionally, active learning metadata is attached to help select
//! the most informative samples for labeling.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{JobStatus, ModelType, PipelineStage};

/// Stage 1 (unsupervised) specific output fields.
///
/// These fields are populated when running PatchCore, PaDiM, WinCLIP, or GPT-4V.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stage1Output {
    /// Whether a spatial anomaly heatmap was generated
    pub heatmap_available: bool,

    /// Redis key or S3 path to heatmap data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heatmap_key: Option<String>,

    /// Distance in feature space from normal distribution
    /// (PatchCore: k-NN distance, PaDiM: Mahalanobis distance)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_distance: Option<f64>,

    /// CLIP text-image similarity score (WinCLIP only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clip_similarity: Option<f64>,
}

/// Stage 2 (supervised) specific output fields.
///
/// These fields are populated when running the push-pull contrastive model
/// after sufficient labeled samples have been collected.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Stage2Output {
    /// Predicted defect category (if multi-class classification)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defect_category: Option<String>,

    /// Probability distribution over defect categories
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category_probabilities: Option<std::collections::HashMap<String, f64>>,

    /// Distance from decision boundary in contrastive embedding space
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_distance: Option<f64>,
}

/// Active learning metadata for sample selection.
///
/// This information helps the system identify which samples would be
/// most valuable for human labeling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActiveLearningMeta {
    /// Uncertainty measure for sampling
    /// High uncertainty = valuable for labeling
    /// Calculation depends on strategy: entropy, margin, or least confidence
    pub uncertainty_score: f64,

    /// Whether this sample is currently in the active learning pool
    pub in_labeling_pool: bool,

    /// Whether this sample has been labeled by an expert
    pub labeled: bool,
}

/// Error information when inference fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceError {
    /// Error code for programmatic handling
    pub code: String,

    /// Human-readable error description
    pub message: String,
}

/// Complete inference result from the ML worker.
///
/// This structure is stored in Redis after the worker completes processing
/// and returned to clients via the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    /// Reference to the originating job
    pub job_id: Uuid,

    /// Reference to the processed image
    pub image_id: String,

    /// Current status
    pub status: JobStatus,

    /// Timestamp when result was created
    pub created_at: DateTime<Utc>,

    /// Timestamp when inference completed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,

    /// Model that performed inference
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_used: Option<ModelType>,

    /// Pipeline stage that produced this result
    pub stage: PipelineStage,

    /// Overall anomaly score (0.0 = normal, 1.0 = anomalous)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anomaly_score: Option<f64>,

    /// Binary classification result (anomaly_score > threshold)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_anomaly: Option<bool>,

    /// Model confidence in the prediction
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,

    /// Stage 1 specific outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage1_output: Option<Stage1Output>,

    /// Stage 2 specific outputs
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stage2_output: Option<Stage2Output>,

    /// Active learning metadata
    #[serde(default)]
    pub active_learning: ActiveLearningMeta,

    /// Error information (if status is Failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<InferenceError>,

    /// Processing time in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<u64>,

    // ── VLM / router outputs (top-level) ──────────────────────────────────
    /// Human-readable defect description produced by the VLM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defect_description: Option<String>,

    /// Spatial description of the defect location from the VLM.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defect_location: Option<String>,

    /// Legacy one-sentence reasoning from the GPT-4V detector.
    /// Kept for backward compat; new code should use `natural_description`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpt4v_reasoning: Option<String>,

    /// Routed product category (one of the trained PatchCore categories
    /// or "unknown" for zero-shot fallback).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product_class: Option<String>,

    /// Confidence of the VLM product identification (0.0-1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub product_confidence: Option<f64>,

    /// Operator-facing single-sentence defect description from the VLM router.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub natural_description: Option<String>,

    /// Severity bucket assigned by the VLM ("minor" | "moderate" | "severe").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defect_severity: Option<String>,

    /// Defect-type label from the VLM (e.g. "scratch", "crack").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defect_type: Option<String>,

    /// Which branch of the router fired.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_reason: Option<String>,

    /// Which OpenAI model was invoked for this result, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vlm_model_used: Option<String>,

    /// Rough USD cost estimate for the VLM calls that produced this result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vlm_api_cost_estimate_usd: Option<f64>,
}

impl InferenceResult {
    /// Create a new pending result for a job.
    pub fn pending(job_id: Uuid, image_id: String) -> Self {
        Self {
            job_id,
            image_id,
            status: JobStatus::Pending,
            created_at: Utc::now(),
            completed_at: None,
            model_used: None,
            stage: PipelineStage::Unsupervised,
            anomaly_score: None,
            is_anomaly: None,
            confidence: None,
            stage1_output: None,
            stage2_output: None,
            active_learning: ActiveLearningMeta::default(),
            error: None,
            processing_time_ms: None,
            defect_description: None,
            defect_location: None,
            gpt4v_reasoning: None,
            product_class: None,
            product_confidence: None,
            natural_description: None,
            defect_severity: None,
            defect_type: None,
            routing_reason: None,
            vlm_model_used: None,
            vlm_api_cost_estimate_usd: None,
        }
    }
}

/// API response for result queries.
#[derive(Debug, Clone, Serialize)]
pub struct ResultResponse {
    /// Whether the result was found
    pub found: bool,

    /// The inference result (if found)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<InferenceResult>,

    /// Message for the client
    pub message: String,
}
