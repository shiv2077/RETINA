//! # RETINA Backend - Main Entry Point
//!
//! This is the main entry point for the RETINA backend API server.
//!
//! ## Architecture Overview
//!
//! The backend serves as the orchestration layer for the multi-stage anomaly
//! detection pipeline. It does NOT run ML models directly - instead, it:
//!
//! 1. Accepts image submissions via REST API
//! 2. Queues inference jobs to Redis Streams
//! 3. Retrieves results from Redis (populated by Python workers)
//! 4. Manages active learning state and label collection
//!
//! ## Research Context
//!
//! This system implements a two-stage anomaly detection approach:
//!
//! - **Stage 1**: Unsupervised/zero-shot detection (PatchCore, PaDiM, WinCLIP)
//!   - Used when no labeled defect data exists
//!   - Outputs anomaly scores and uncertainty estimates
//!
//! - **Stage 2**: Supervised classification (Push-Pull Contrastive Learning)
//!   - Activated after ~100-200 labeled samples via active learning
//!   - Provides high-precision defect classification
//!
//! The active learning loop bridges these stages by collecting expert labels
//! on the most uncertain/informative samples from Stage 1.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use sqlx::PgPool;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod auth;
mod config;
mod db;
mod error;
mod models;
mod routes;
mod services;

use config::Config;
use services::redis::RedisService;
use services::ImageStorage;

/// Application state shared across all request handlers.
///
/// This struct holds shared resources like database connections and
/// configuration. It's wrapped in Arc and cloned into each request handler.
#[derive(Clone)]
pub struct AppState {
    /// PostgreSQL connection pool
    pub db: Arc<PgPool>,

    /// Redis service for job queuing and result storage
    pub redis: Arc<RedisService>,

    /// Image storage service
    pub images: Arc<ImageStorage>,

    /// Application configuration
    pub config: Arc<Config>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // -------------------------------------------------------------------------
    // Initialize Tracing (Logging)
    // -------------------------------------------------------------------------
    // We use the tracing crate for structured, async-aware logging.
    // Log levels can be configured via RUST_LOG environment variable.
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=debug,retina_backend=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting RETINA Backend API Server...");

    // -------------------------------------------------------------------------
    // Load Configuration
    // -------------------------------------------------------------------------
    // Configuration is loaded from environment variables with sensible defaults.
    // In development, we use dotenv for convenience.
    dotenvy::dotenv().ok(); // Ignore if .env doesn't exist
    let config = Config::from_env()?;
    info!(?config, "Configuration loaded");

    // -------------------------------------------------------------------------
    // Initialize Redis Connection
    // -------------------------------------------------------------------------
    // Redis is used for:
    // - Job queues (Redis Streams)
    // - Result caching (Hashes)
    // - Active learning pool (Sorted Sets)
    // - System metrics (Hashes)
    let redis_service = RedisService::new(&config.redis_url).await?;
    info!("Redis connection established");

    // Initialize Redis data structures (streams, consumer groups, etc.)
    redis_service.initialize_data_structures().await?;
    info!("Redis data structures initialized");

    // -------------------------------------------------------------------------
    // Initialize PostgreSQL Database
    // -------------------------------------------------------------------------
    // PostgreSQL stores user accounts and anomaly records for persistence.
    // Matching professor's framework pattern.
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://retina:retina@localhost:5432/retina".to_string());
    
    let db_pool = db::pool::create_pool(&database_url).await?;
    info!("PostgreSQL connection established");

    // Run database migrations
    db::pool::run_migrations(&db_pool).await?;
    info!("Database migrations complete");

    // -------------------------------------------------------------------------
    // Initialize Image Storage
    // -------------------------------------------------------------------------
    let image_storage = ImageStorage::new(None).await?;
    info!("Image storage initialized");

    // -------------------------------------------------------------------------
    // Build Application State
    // -------------------------------------------------------------------------
    let state = AppState {
        db: Arc::new(db_pool),
        redis: Arc::new(redis_service),
        images: Arc::new(image_storage),
        config: Arc::new(config.clone()),
    };

    // -------------------------------------------------------------------------
    // Configure CORS
    // -------------------------------------------------------------------------
    // Allow requests from the frontend origin.
    // In production, this should be more restrictive.
    let cors = CorsLayer::new()
        .allow_origin(Any) // TODO: Restrict to config.cors_allowed_origins in production
        .allow_methods(Any)
        .allow_headers(Any);

    // -------------------------------------------------------------------------
    // Build Router
    // -------------------------------------------------------------------------
    // The router is organized by domain:
    // - /health: Health check (for Docker/K8s)
    // - /auth: Authentication (login, register)
    // - /api/images: Image submission and result retrieval
    // - /api/labels: Active learning label submission
    // - /api/anomaly: Real-time alerts
    // - /api/system: System status and metrics
    let app = Router::new()
        .merge(routes::health::router())
        .nest("/auth", routes::auth_router())
        .nest("/api", routes::api_router())
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    // -------------------------------------------------------------------------
    // Start Server
    // -------------------------------------------------------------------------
    let addr = SocketAddr::from(([0, 0, 0, 0], config.backend_port));
    info!("Listening on {}", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
