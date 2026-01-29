//! # Database Module
//!
//! PostgreSQL database integration for persistent storage of:
//! - User accounts (authentication)
//! - Anomaly records (inference results + expert labels)
//!
//! ## Schema Alignment
//!
//! This module aligns with the professor's framework schema:
//! - `users` table: id, username, hashed_password
//! - `anomaly_records` table: full inference lifecycle
//!
//! ## Research Note
//!
//! PostgreSQL is used for:
//! - Persistent storage of labeled data (survives restarts)
//! - Active learning queries (fetch unreviewed samples)
//! - Audit trail of expert decisions
//!
//! Redis continues to handle:
//! - Job queues (transient)
//! - Real-time alerts
//! - Caching

pub mod models;
pub mod pool;
