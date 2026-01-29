//! # Configuration Module
//!
//! Handles loading and validation of application configuration from
//! environment variables.
//!
//! ## Environment Variables
//!
//! | Variable | Description | Default |
//! |----------|-------------|---------|
//! | `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
//! | `BACKEND_HOST` | Server bind address | `0.0.0.0` |
//! | `BACKEND_PORT` | Server port | `3001` |
//! | `CORS_ALLOWED_ORIGINS` | Allowed CORS origins | `*` |
//! | `AL_STAGE2_THRESHOLD` | Labels needed for Stage 2 | `100` |

use std::env;

/// Application configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub struct Config {
    /// Redis connection URL
    pub redis_url: String,

    /// Server bind address
    pub backend_host: String,

    /// Server port
    pub backend_port: u16,

    /// Comma-separated list of allowed CORS origins
    pub cors_allowed_origins: String,

    /// Number of labeled samples required to activate Stage 2
    /// (supervised learning with push-pull contrastive model)
    pub active_learning_stage2_threshold: u32,

    /// Active learning pool size (number of uncertain samples to present for labeling)
    pub active_learning_pool_size: u32,
}

impl Config {
    /// Load configuration from environment variables with defaults.
    ///
    /// # Errors
    ///
    /// Returns an error if required environment variables are missing
    /// or have invalid values.
    pub fn from_env() -> anyhow::Result<Self> {
        Ok(Self {
            redis_url: env::var("REDIS_URL")
                .unwrap_or_else(|_| "redis://localhost:6379".to_string()),

            backend_host: env::var("BACKEND_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),

            backend_port: env::var("BACKEND_PORT")
                .unwrap_or_else(|_| "3001".to_string())
                .parse()?,

            cors_allowed_origins: env::var("CORS_ALLOWED_ORIGINS")
                .unwrap_or_else(|_| "*".to_string()),

            active_learning_stage2_threshold: env::var("AL_STAGE2_THRESHOLD")
                .unwrap_or_else(|_| "100".to_string())
                .parse()?,

            active_learning_pool_size: env::var("AL_POOL_SIZE")
                .unwrap_or_else(|_| "50".to_string())
                .parse()?,
        })
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            redis_url: "redis://localhost:6379".to_string(),
            backend_host: "0.0.0.0".to_string(),
            backend_port: 3001,
            cors_allowed_origins: "*".to_string(),
            active_learning_stage2_threshold: 100,
            active_learning_pool_size: 50,
        }
    }
}
