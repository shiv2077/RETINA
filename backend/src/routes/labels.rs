//! # Active Learning Label Endpoints
//!
//! Handles label submission and labeling pool management for active learning.
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
//! ## Research Note
//!
//! Active learning significantly reduces annotation effort by focusing
//! on samples where the model is most uncertain. This is especially
//! valuable in industrial inspection where labeled defect data is scarce.

use axum::{extract::State, routing::{get, post}, Json, Router};

use crate::error::AppResult;
use crate::models::{LabelingPoolResponse, SubmitLabelRequest, SubmitLabelResponse};
use crate::AppState;

/// Build the labels router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/submit", post(submit_label))
        .route("/pool", get(get_labeling_pool))
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
///   "labeled_by": "expert_001",
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
///       "added_at": "2026-01-15T10:30:00Z"
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
