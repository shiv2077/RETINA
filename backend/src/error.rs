//! # Error Handling Module
//!
//! Defines custom error types for the RETINA backend.
//!
//! We use a combination of:
//! - `thiserror` for ergonomic error type definitions
//! - `anyhow` for error propagation in application code
//!
//! The `AppError` type implements `IntoResponse` to convert errors
//! into appropriate HTTP responses.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Application-level errors that can occur during request handling.
#[derive(Error, Debug)]
pub enum AppError {
    /// Redis operation failed
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    /// Resource not found (404)
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Invalid request data (400)
    #[error("Invalid request: {0}")]
    BadRequest(String),

    /// Internal server error (500)
    #[error("Internal error: {0}")]
    Internal(String),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Job queue error
    #[error("Queue error: {0}")]
    Queue(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match &self {
            AppError::Redis(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "REDIS_ERROR",
                format!("Database error: {}", e),
            ),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg.clone()),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "BAD_REQUEST", msg.clone()),
            AppError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                msg.clone(),
            ),
            AppError::Json(e) => (
                StatusCode::BAD_REQUEST,
                "JSON_ERROR",
                format!("Invalid JSON: {}", e),
            ),
            AppError::Queue(msg) => (
                StatusCode::SERVICE_UNAVAILABLE,
                "QUEUE_ERROR",
                msg.clone(),
            ),
        };

        // Log the error for debugging
        tracing::error!(
            error_code = error_code,
            message = %message,
            "Request failed"
        );

        let body = Json(json!({
            "error": {
                "code": error_code,
                "message": message
            }
        }));

        (status, body).into_response()
    }
}

/// Result type alias using our AppError
pub type AppResult<T> = Result<T, AppError>;
