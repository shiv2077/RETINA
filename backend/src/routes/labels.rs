//! # Active Learning Label Endpoints
//!
//! Handles label submission and labeling pool management for active learning.
//! Matches professor's framework learningController pattern.
//!
//! ## Endpoints
//!
//! - POST /labels/submit - Submit a label (original)
//! - GET /labels/pool - Get samples for labeling
//! - GET /labels/review - Get records pending expert review (mismatch cases)
//! - POST /labels/review - Submit expert review with final classification
//!
//! ## Active Learning Overview
//!
//! The system uses uncertainty sampling to select the most informative
//! samples for human labeling:
//!
//! 1. Stage 1 models produce anomaly scores + uncertainty estimates
//! 2. High-uncertainty samples are added to the labeling pool
//! 3. Domain experts label samples via this API
//! 4. Labels accumulate until Stage 2 threshold is reached
//! 5. Stage 2 supervised model is trained on labeled data
//!
//! ## Mismatch Review (Professor's Framework)
//!
//! When supervised and unsupervised models disagree:
//! 1. Record is flagged with mismatch=true
//! 2. Expert reviews via /labels/review endpoint
//! 3. Expert provides final classification
//! 4. Record marked as reviewed=true

use axum::{extract::State, routing::{get, post}, Json, Router};

use crate::error::AppResult;
use crate::models::{LabelingPoolResponse, SubmitLabelRequest, SubmitLabelResponse};
use crate::AppState;

/// Build the labels router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/submit", post(submit_label))
        .route("/pool", get(get_labeling_pool))
        .route("/review", get(get_pending_reviews))
        .route("/review", post(submit_review))
}

/// Submit a label for an image.
///
/// ## Request Body
///
/// ```json
/// {
///   "image_id": "img_2024_001_front_panel",
///   "is_anomaly": true,
///   "defect_category": "scratch",     // optional
///   "confidence": "high",             // low, medium, high
///   "labeled_by": "user1",
///   "notes": "Diagonal scratch visible" // optional
/// }
/// ```
///
/// ## Response
///
/// ```json
/// {
///   "success": true,
///   "image_id": "img_2024_001_front_panel",
///   "total_labels": 87,
///   "labels_for_stage2": 100,
///   "stage2_available": false,
///   "message": "Label submitted successfully. 13 more labels needed for Stage 2."
/// }
/// ```
///
/// ## Side Effects
///
/// - Removes sample from active learning pool
/// - Increments label counter
/// - May trigger Stage 2 availability notification
async fn submit_label(
    State(state): State<AppState>,
    Json(request): Json<SubmitLabelRequest>,
) -> AppResult<Json<SubmitLabelResponse>> {
    // Convert request to label with timestamp
    let label = request.into_label();
    let image_id = label.image_id.clone();

    // Submit label to Redis
    let total_labels = state.redis.submit_label(&label).await?;

    // Check if Stage 2 is now available
    let threshold = state.config.active_learning_stage2_threshold;
    let stage2_available = total_labels >= threshold as u64;

    // Build response message
    let message = if stage2_available {
        format!(
            "Label submitted successfully. Stage 2 is now available with {} labels!",
            total_labels
        )
    } else {
        let remaining = threshold as u64 - total_labels;
        format!(
            "Label submitted successfully. {} more labels needed for Stage 2.",
            remaining
        )
    };

    if stage2_available && total_labels == threshold as u64 {
        tracing::info!(
            total_labels = total_labels,
            "Stage 2 threshold reached! Supervised learning now available."
        );
    }

    Ok(Json(SubmitLabelResponse {
        success: true,
        image_id,
        total_labels,
        labels_for_stage2: threshold,
        stage2_available,
        message,
    }))
}

/// Get samples from the active learning pool for labeling.
///
/// Returns samples ordered by uncertainty (highest first), as these
/// are the most valuable for improving model performance.
///
/// ## Query Parameters
///
/// - `limit`: Maximum number of samples to return (default: 10)
///
/// ## Response
///
/// ```json
/// {
///   "samples": [
///     {
///       "image_id": "img_2024_042_panel_b",
///       "anomaly_score": 0.52,
///       "uncertainty_score": 0.89,
///       "added_at": "2025-01-15T10:30:00Z"
///     }
///   ],
///   "pool_size": 47,
///   "labels_collected": 53,
///   "stage2_threshold": 100
/// }
/// ```
async fn get_labeling_pool(
    State(state): State<AppState>,
) -> AppResult<Json<LabelingPoolResponse>> {
    // Get samples from pool (ordered by uncertainty)
    let limit = state.config.active_learning_pool_size as usize;
    let samples = state.redis.get_labeling_pool(limit).await?;

    // Get counts
    let pool_size = state.redis.get_pool_size().await?;
    let labels_collected = state.redis.get_label_count().await?;

    Ok(Json(LabelingPoolResponse {
        samples,
        pool_size,
        labels_collected,
        stage2_threshold: state.config.active_learning_stage2_threshold,
    }))
}

// =============================================================================
// Expert Review Endpoints (Matching Professor's Framework)
// =============================================================================

/// Request to submit expert review.
#[derive(serde::Deserialize)]
pub struct ReviewRequest {
    /// Job ID to review
    pub job_id: String,
    /// Expert's label (the ground truth)
    pub expert_label: String,
    /// Final classification category (e.g., "scratch", "dent", "ok")
    pub final_classification: String,
    /// Optional notes from expert
    pub notes: Option<String>,
}

/// Response for pending reviews.
#[derive(serde::Serialize)]
pub struct PendingReviewsResponse {
    /// Records pending review (mismatch cases)
    pub records: Vec<PendingReviewRecord>,
    /// Total count of pending reviews
    pub count: usize,
}

/// A record pending expert review.
#[derive(serde::Serialize)]
pub struct PendingReviewRecord {
    /// Job ID
    pub job_id: String,
    /// Image ID
    pub image_id: String,
    /// Supervised model's prediction
    pub supervised_label: bool,
    /// Unsupervised model's prediction
    pub unsupervised_label: Option<bool>,
    /// Whether there was mismatch
    pub mismatch: bool,
    /// When the record was created
    pub timestamp: String,
}

/// Get records pending expert review.
///
/// Returns records where supervised and unsupervised models disagreed.
/// Matching professor's framework `/review` GET endpoint.
///
/// ## Response
///
/// ```json
/// {
///   "records": [
///     {
///       "job_id": "job-123",
///       "image_id": "img_001",
///       "supervised_label": true,
///       "unsupervised_label": false,
///       "mismatch": true,
///       "timestamp": "2024-01-15T10:30:00Z"
///     }
///   ],
///   "count": 5
/// }
/// ```
async fn get_pending_reviews(
    State(state): State<AppState>,
) -> AppResult<Json<PendingReviewsResponse>> {
    // Get mismatch records from Redis
    let records = state.redis.get_mismatch_records().await?;
    let count = records.len();

    Ok(Json(PendingReviewsResponse { records, count }))
}

/// Submit expert review for a mismatch record.
///
/// Matching professor's framework `/review` POST endpoint.
///
/// ## Request Body
///
/// ```json
/// {
///   "job_id": "job-123",
///   "expert_label": "anomaly",
///   "final_classification": "scratch"
/// }
/// ```
///
/// ## Response
///
/// ```json
/// {
///   "success": true,
///   "message": "Review submitted"
/// }
/// ```
async fn submit_review(
    State(state): State<AppState>,
    Json(request): Json<ReviewRequest>,
) -> AppResult<Json<serde_json::Value>> {
    // Update the record with expert's review
    state.redis.submit_review(
        &request.job_id,
        &request.expert_label,
        &request.final_classification,
    ).await?;

    tracing::info!(
        job_id = %request.job_id,
        expert_label = %request.expert_label,
        final_classification = %request.final_classification,
        "Expert review submitted"
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "message": "Review submitted"
    })))
}
