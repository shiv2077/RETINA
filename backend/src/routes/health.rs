//! # Health Check Endpoint
//!
//! Provides a simple health check for container orchestration (Docker, Kubernetes).

use axum::{extract::State, http::StatusCode, routing::get, Json, Router};
use serde::Serialize;

use crate::AppState;

/// Build the health check router.
pub fn router() -> Router<AppState> {
    Router::new().route("/health", get(health_check))
}

/// Health check response.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,

    /// Service name
    pub service: String,

    /// Version
    pub version: String,

    /// Redis connectivity
    pub redis_connected: bool,
}

/// Health check handler.
///
/// Returns 200 if the service is healthy, 503 if Redis is down.
///
/// ## Example Response
///
/// ```json
/// {
///   "status": "healthy",
///   "service": "retina-backend",
///   "version": "0.1.0",
///   "redis_connected": true
/// }
/// ```
async fn health_check(State(state): State<AppState>) -> (StatusCode, Json<HealthResponse>) {
    let redis_ok = state.redis.health_check().await.unwrap_or(false);

    let status_code = if redis_ok {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    let response = HealthResponse {
        status: if redis_ok {
            "healthy".to_string()
        } else {
            "unhealthy".to_string()
        },
        service: "retina-backend".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        redis_connected: redis_ok,
    };

    (status_code, Json(response))
}
