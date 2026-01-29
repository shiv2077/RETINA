//! # Database Models
//!
//! SQLx-compatible models for PostgreSQL tables.
//! Matches the professor's framework schema for compatibility.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

/// User account for authentication.
///
/// Matches professor's `users` table schema.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct User {
    /// Auto-increment primary key
    pub id: i32,

    /// Unique username
    pub username: String,

    /// Argon2 hashed password
    pub hashed_password: String,
}

/// Anomaly detection record with full lifecycle tracking.
///
/// This table stores:
/// - Initial inference results (supervised + unsupervised)
/// - Expert review decisions
/// - Mismatch flags for active learning
///
/// Matches professor's `anomaly_records` table schema.
#[derive(Debug, Clone, FromRow, Serialize, Deserialize)]
pub struct AnomalyRecord {
    /// Job ID (UUID as string)
    pub id: String,

    /// Username who submitted the image
    pub user_id: String,

    /// Path to stored image file
    pub file_path: String,

    /// Submission timestamp
    pub timestamp: DateTime<Utc>,

    /// Stage 1 unsupervised model result (true = anomaly)
    pub unsupervised_label: bool,

    /// Supervised model result (true = anomaly)
    pub supervised_label: bool,

    /// Whether supervised and unsupervised disagree
    /// Used to prioritize samples for expert review
    pub mismatch: bool,

    /// Whether an expert has reviewed this sample
    pub reviewed: bool,

    /// Expert's final label (null until reviewed)
    /// Values: "anomalous", "normal", or null
    #[sqlx(default)]
    pub expert_label: Option<String>,

    /// Expert's defect classification (if anomalous)
    /// e.g., "scratch", "dent", "contamination"
    #[sqlx(default)]
    pub final_classification: Option<String>,

    /// Anomaly score from the model (0.0 - 1.0)
    #[sqlx(default)]
    pub anomaly_score: Option<f64>,

    /// Model used for inference
    #[sqlx(default)]
    pub model_used: Option<String>,

    /// Pipeline stage (1 or 2)
    #[sqlx(default)]
    pub pipeline_stage: Option<i32>,
}

impl AnomalyRecord {
    /// Create a new anomaly record from inference results.
    pub fn new(
        id: String,
        user_id: String,
        file_path: String,
        unsupervised_label: bool,
        supervised_label: bool,
        anomaly_score: f64,
        model_used: String,
    ) -> Self {
        Self {
            id,
            user_id,
            file_path,
            timestamp: Utc::now(),
            unsupervised_label,
            supervised_label,
            mismatch: unsupervised_label != supervised_label,
            reviewed: false,
            expert_label: None,
            final_classification: None,
            anomaly_score: Some(anomaly_score),
            model_used: Some(model_used),
            pipeline_stage: Some(1),
        }
    }

    /// Convert to API response format.
    pub fn to_response(&self) -> AnomalyRecordResponse {
        AnomalyRecordResponse {
            id: self.id.clone(),
            user_id: self.user_id.clone(),
            file_path: self.file_path.clone(),
            timestamp: self.timestamp.to_rfc3339(),
            unsupervised_label: self.unsupervised_label,
            supervised_label: self.supervised_label,
            mismatch: self.mismatch,
            reviewed: self.reviewed,
            expert_label: self.expert_label.clone(),
            final_classification: self.final_classification.clone(),
            anomaly_score: self.anomaly_score,
        }
    }
}

/// API response format for anomaly records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyRecordResponse {
    pub id: String,
    pub user_id: String,
    pub file_path: String,
    pub timestamp: String,
    pub unsupervised_label: bool,
    pub supervised_label: bool,
    pub mismatch: bool,
    pub reviewed: bool,
    pub expert_label: Option<String>,
    pub final_classification: Option<String>,
    pub anomaly_score: Option<f64>,
}

/// Request to create a new user.
#[derive(Debug, Clone, Deserialize)]
pub struct CreateUserRequest {
    pub username: String,
    pub password: String,
}

/// Request to update expert feedback.
#[derive(Debug, Clone, Deserialize)]
pub struct ExpertFeedbackRequest {
    /// "anomalous" or "normal"
    pub label: String,

    /// Optional defect classification (only for anomalous)
    #[serde(default)]
    pub classification: Option<String>,
}
