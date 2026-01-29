//! # API Routes
//!
//! This module defines the REST API endpoints for the RETINA backend.
//!
//! ## Endpoint Overview
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/health` | GET | Health check |
//! | `/api/images/submit` | POST | Submit image for inference |
//! | `/api/images/{id}/result` | GET | Get inference result |
//! | `/api/labels/submit` | POST | Submit active learning label |
//! | `/api/labels/pool` | GET | Get samples for labeling |
//! | `/api/system/status` | GET | Get system status |

use axum::Router;

use crate::AppState;

pub mod health;
pub mod images;
pub mod labels;
pub mod system;

/// Build the main API router with all sub-routes.
pub fn api_router() -> Router<AppState> {
    Router::new()
        .nest("/images", images::router())
        .nest("/labels", labels::router())
        .nest("/system", system::router())
}
