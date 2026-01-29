//! # System Status Endpoint
//!
//! Provides system-wide status and metrics for monitoring and debugging.

use axum::{extract::State, routing::get, Json, Router};
use serde::Serialize;

use crate::error::AppResult;
use crate::services::redis::SystemStats;
use crate::AppState;

/// Build the system router.
pub fn router() -> Router<AppState> {
    Router::new().route("/status", get(get_system_status))
}

/// System status response.
#[derive(Debug, Serialize)]
pub struct SystemStatusResponse {
    /// Service health status
    pub status: String,

    /// Current pipeline stage
    pub current_stage: u8,

    /// Stage 2 availability
    pub stage2_available: bool,

    /// Labels needed for Stage 2
    pub labels_for_stage2: u32,

    /// Progress towards Stage 2 (percentage)
    pub stage2_progress: f64,

    /// System statistics
    pub stats: SystemStats,

    /// Active models
    pub active_models: ActiveModels,
}

/// Information about active models.
#[derive(Debug, Serialize)]
pub struct ActiveModels {
    /// Stage 1 model currently in use
    pub stage1_model: String,

    /// Stage 2 model (if available)
    pub stage2_model: Option<String>,

    /// Whether models are loaded (in real implementation)
    pub models_loaded: bool,
}

/// Get comprehensive system status.
///
/// ## Response
///
/// ```json
/// {
///   "status": "healthy",
///   "current_stage": 1,
///   "stage2_available": false,
///   "labels_for_stage2": 100,
///   "stage2_progress": 53.0,
///   "stats": {
///     "jobs_submitted": 247,
///     "jobs_completed": 243,
///     "queue_length": 4,
///     "labels_collected": 53,
///     "labeling_pool_size": 47
///   },
///   "active_models": {
///     "stage1_model": "patchcore",
///     "stage2_model": null,
///     "models_loaded": true
///   }
/// }
/// ```
async fn get_system_status(State(state): State<AppState>) -> AppResult<Json<SystemStatusResponse>> {
    // Get system stats from Redis
    let stats = state.redis.get_system_stats().await?;

    // Calculate Stage 2 availability
    let threshold = state.config.active_learning_stage2_threshold;
    let stage2_available = stats.labels_collected >= threshold as u64;
    let current_stage = if stage2_available { 2 } else { 1 };

    // Calculate progress percentage
    let stage2_progress = if stage2_available {
        100.0
    } else {
        (stats.labels_collected as f64 / threshold as f64) * 100.0
    };

    // Build active models info
    // In the current stub implementation, models are "loaded" if the worker is running
    let active_models = ActiveModels {
        stage1_model: "patchcore".to_string(), // Default model
        stage2_model: if stage2_available {
            Some("pushpull".to_string())
        } else {
            None
        },
        models_loaded: true, // Stub: always true for now
    };

    Ok(Json(SystemStatusResponse {
        status: "healthy".to_string(),
        current_stage,
        stage2_available,
        labels_for_stage2: threshold,
        stage2_progress,
        stats,
        active_models,
    }))
}
