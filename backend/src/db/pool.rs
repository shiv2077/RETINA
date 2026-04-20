//! # Database Connection Pool
//!
//! Manages PostgreSQL connection pool using SQLx.

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::error::{AppError, AppResult};

/// Create a new PostgreSQL connection pool.
///
/// # Arguments
///
/// * `database_url` - PostgreSQL connection URL
///
/// # Example
///
/// ```text
/// postgresql://user:password@localhost:5432/database
/// ```
pub async fn create_pool(database_url: &str) -> AppResult<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await
        .map_err(|e| AppError::Internal(format!("Database connection failed: {}", e)))?;

    tracing::info!("PostgreSQL connection pool created");
    Ok(pool)
}

/// Run database migrations.
///
/// Creates tables if they don't exist.
pub async fn run_migrations(pool: &PgPool) -> AppResult<()> {
    // Create users table
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL,
            hashed_password VARCHAR(255) NOT NULL
        )
        "#,
    )
    .execute(pool)
    .await
    .map_err(|e| AppError::Internal(format!("Migration failed: {}", e)))?;

    // Create anomaly_records table (matches professor's schema)
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS anomaly_records (
            id VARCHAR(255) PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            file_path VARCHAR(500) NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            unsupervised_label BOOLEAN NOT NULL,
            supervised_label BOOLEAN NOT NULL,
            mismatch BOOLEAN DEFAULT FALSE,
            reviewed BOOLEAN DEFAULT FALSE,
            expert_label VARCHAR(50),
            final_classification VARCHAR(100),
            anomaly_score DOUBLE PRECISION,
            model_used VARCHAR(50),
            pipeline_stage INTEGER,
            defect_description TEXT,
            gpt4v_reasoning TEXT
        )
        "#,
    )
    .execute(pool)
    .await
    .map_err(|e| AppError::Internal(format!("Migration failed: {}", e)))?;

    // Idempotent: add new columns to existing tables (safe on re-run)
    for alter_sql in [
        "ALTER TABLE anomaly_records ADD COLUMN IF NOT EXISTS defect_description TEXT",
        "ALTER TABLE anomaly_records ADD COLUMN IF NOT EXISTS gpt4v_reasoning TEXT",
    ] {
        sqlx::query(alter_sql)
            .execute(pool)
            .await
            .map_err(|e| AppError::Internal(format!("Column migration failed: {}", e)))?;
    }

    // Create index for pending reviews (active learning query)
    sqlx::query(
        r#"
        CREATE INDEX IF NOT EXISTS idx_anomaly_records_pending 
        ON anomaly_records (reviewed, timestamp DESC) 
        WHERE reviewed = FALSE
        "#,
    )
    .execute(pool)
    .await
    .map_err(|e| AppError::Internal(format!("Index creation failed: {}", e)))?;

    tracing::info!("Database migrations completed");
    Ok(())
}
