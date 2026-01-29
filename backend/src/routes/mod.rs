//! # API Routes
//!
//! This module defines the REST API endpoints for the RETINA backend.
//!
//! ## Endpoint Overview
//!
//! | Endpoint | Method | Description |
//! |----------|--------|-------------|
//! | `/health` | GET | Health check |
//! | `/auth/token` | POST | Login and get JWT |
//! | `/auth/register` | POST | Register new user |
//! | `/api/images/submit` | POST | Submit image for inference |
//! | `/api/images/{id}/result` | GET | Get inference result |
//! | `/api/labels/submit` | POST | Submit active learning label |
//! | `/api/labels/pool` | GET | Get samples for labeling |
//! | `/api/anomaly/since` | GET | Get recent alerts |
//! | `/api/anomaly/reset` | POST | Reset alerts |
//! | `/api/system/status` | GET | Get system status |

use axum::Router;

use crate::AppState;

pub mod anomaly;
pub mod auth;
pub mod health;
pub mod images;
pub mod labels;
pub mod system;

/// Build the main API router with all sub-routes.
pub fn api_router() -> Router<AppState> {
    Router::new()
        .nest("/images", images::router())
        .nest("/labels", labels::router())
        .nest("/anomaly", anomaly::router())
        .nest("/system", system::router())
}

/// Build the auth router (not under /api prefix).
pub fn auth_router() -> Router<AppState> {
    auth::router()
}
