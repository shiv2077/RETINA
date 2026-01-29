//! # Anomaly Routes
//!
//! Real-time alert endpoints matching professor's framework.
//!
//! ## Endpoints
//!
//! - GET /anomaly/since - Get recent alerts (for polling)
//! - POST /anomaly/reset - Clear all alerts

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};

use crate::error::AppResult;
use crate::services::Alert;
use crate::AppState;

/// Build the anomaly router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/since", get(get_alerts))
        .route("/reset", post(reset_alerts))
        .route("/count", get(get_alert_count))
}

/// Response for alerts endpoint.
#[derive(serde::Serialize)]
pub struct AlertsResponse {
    /// List of active alerts
    pub alerts: Vec<Alert>,
    /// Number of alerts
    pub count: usize,
}

/// Get all current alerts.
///
/// ## Response
///
/// ```json
/// {
///   "alerts": [
///     {
///       "job_id": "abc-123",
///       "user": "admin",
///       "label": "scratch",
///       "timestamp": "2024-01-15T10:30:00Z"
///     }
///   ],
///   "count": 1
/// }
/// ```
async fn get_alerts(State(state): State<AppState>) -> AppResult<Json<AlertsResponse>> {
    let alerts = state.redis.get_alerts().await?;
    let count = alerts.len();

    Ok(Json(AlertsResponse { alerts, count }))
}

/// Reset/clear all alerts (user acknowledged them).
async fn reset_alerts(State(state): State<AppState>) -> AppResult<Json<serde_json::Value>> {
    state.redis.reset_alerts().await?;

    Ok(Json(serde_json::json!({
        "message": "Alerts reset",
        "success": true
    })))
}

/// Get just the alert count (for badge in UI).
async fn get_alert_count(State(state): State<AppState>) -> AppResult<Json<serde_json::Value>> {
    let count = state.redis.get_alert_count().await?;

    Ok(Json(serde_json::json!({
        "count": count
    })))
}
