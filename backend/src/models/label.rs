//! # Active Learning Label Models
//!
//! Defines structures for the active learning labeling workflow.
//!
//! ## Active Learning Flow
//!
//! 1. Stage 1 inference produces anomaly scores + uncertainty estimates
//! 2. High-uncertainty samples are added to the labeling pool
//! 3. Domain experts review samples and provide labels
//! 4. Labels accumulate until threshold (~100-200 samples)
//! 5. Stage 2 model training is triggered
//!
//! ## Research Motivation
//!
//! Active learning reduces labeling effort by selecting the most informative
//! samples. We use uncertainty sampling with configurable strategies:
//! - Entropy: Select samples where model is most uncertain
//! - Margin: Select samples with smallest margin between top predictions
//! - Least Confidence: Select samples with lowest top-class probability

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Labeler's confidence in their assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LabelConfidence {
    Low,
    Medium,
    High,
}

impl Default for LabelConfidence {
    fn default() -> Self {
        Self::Medium
    }
}

/// Quality control status for a label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReviewStatus {
    Pending,
    Approved,
    Rejected,
    NeedsReview,
}

impl Default for ReviewStatus {
    fn default() -> Self {
        Self::Pending
    }
}

/// Optional bounding box for defect localization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// A label submission from a domain expert.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Label {
    /// Reference to the labeled image
    pub image_id: String,

    /// Binary label: true = anomaly/defect, false = normal
    pub is_anomaly: bool,

    /// Optional defect category for multi-class classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defect_category: Option<String>,

    /// Labeler's confidence in their assessment
    #[serde(default)]
    pub confidence: LabelConfidence,

    /// Identifier of the labeler
    pub labeled_by: String,

    /// Timestamp of submission
    pub labeled_at: DateTime<Utc>,

    /// Optional notes from the labeler
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,

    /// Optional defect localization
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounding_box: Option<BoundingBox>,

    /// Quality control status
    #[serde(default)]
    pub review_status: ReviewStatus,
}

/// Request body for submitting a label.
#[derive(Debug, Clone, Deserialize)]
pub struct SubmitLabelRequest {
    /// Image ID being labeled
    pub image_id: String,

    /// Binary classification
    pub is_anomaly: bool,

    /// Optional defect category
    pub defect_category: Option<String>,

    /// Labeler confidence
    #[serde(default)]
    pub confidence: LabelConfidence,

    /// Who is providing the label
    pub labeled_by: String,

    /// Optional notes
    pub notes: Option<String>,

    /// Optional bounding box
    pub bounding_box: Option<BoundingBox>,
}

impl SubmitLabelRequest {
    /// Convert request to a Label with timestamp.
    pub fn into_label(self) -> Label {
        Label {
            image_id: self.image_id,
            is_anomaly: self.is_anomaly,
            defect_category: self.defect_category,
            confidence: self.confidence,
            labeled_by: self.labeled_by,
            labeled_at: Utc::now(),
            notes: self.notes,
            bounding_box: self.bounding_box,
            review_status: ReviewStatus::Pending,
        }
    }
}

/// Response after submitting a label.
#[derive(Debug, Clone, Serialize)]
pub struct SubmitLabelResponse {
    /// Whether the label was accepted
    pub success: bool,

    /// Image that was labeled
    pub image_id: String,

    /// Current total number of labels
    pub total_labels: u64,

    /// Number of labels needed for Stage 2
    pub labels_for_stage2: u32,

    /// Whether Stage 2 is now available
    pub stage2_available: bool,

    /// Message for the client
    pub message: String,
}

/// A sample in the active learning pool awaiting labeling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlabeledSample {
    /// Image ID
    pub image_id: String,

    /// Model's anomaly score
    pub anomaly_score: f64,

    /// Uncertainty score (higher = more valuable to label)
    pub uncertainty_score: f64,

    /// When this sample was added to the pool
    pub added_at: DateTime<Utc>,
}

/// Response containing samples for labeling.
#[derive(Debug, Clone, Serialize)]
pub struct LabelingPoolResponse {
    /// Samples to label, ordered by uncertainty
    pub samples: Vec<UnlabeledSample>,

    /// Total samples in pool
    pub pool_size: u64,

    /// Number of labels already collected
    pub labels_collected: u64,

    /// Threshold needed for Stage 2
    pub stage2_threshold: u32,
}
