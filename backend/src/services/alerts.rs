//! # Real-time Alerts Service
//!
//! Redis-based alert system for immediate anomaly notifications.
//! Matches professor's framework alert pattern.
//!
//! ## How It Works
//!
//! 1. Worker detects anomaly via supervised model
//! 2. Worker pushes alert to Redis list
//! 3. Frontend polls /anomaly/since endpoint
//! 4. User acknowledges → alerts cleared

use chrono::{DateTime, Utc};
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};

use crate::error::{AppError, AppResult};
use crate::services::redis::RedisService;

/// Redis key for alerts list.
const ALERTS_KEY: &str = "retina:alerts";

/// An anomaly alert for real-time notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Job ID that triggered the alert
    pub job_id: String,

    /// User who submitted the image
    pub user: String,

    /// Classification label (defect type)
    pub label: String,

    /// When the alert was created
    pub timestamp: String,
}

impl Alert {
    /// Create a new alert.
    pub fn new(job_id: String, user: String, label: String) -> Self {
        Self {
            job_id,
            user,
            label,
            timestamp: Utc::now().to_rfc3339(),
        }
    }
}

impl RedisService {
    /// Add an alert to the alerts list.
    ///
    /// Called by the worker when a supervised model detects an anomaly.
    pub async fn add_alert(&self, alert: &Alert) -> AppResult<()> {
        let mut conn = self.connection.clone();
        let alert_json = serde_json::to_string(alert)?;

        // LPUSH to add at the front (newest first)
        let _: () = conn
            .lpush(ALERTS_KEY, &alert_json)
            .await
            .map_err(AppError::Redis)?;

        // Trim to keep only last 100 alerts
        let _: () = conn
            .ltrim(ALERTS_KEY, 0, 99)
            .await
            .map_err(AppError::Redis)?;

        tracing::info!(job_id = %alert.job_id, label = %alert.label, "Alert added");
        Ok(())
    }

    /// Get all current alerts.
    pub async fn get_alerts(&self) -> AppResult<Vec<Alert>> {
        let mut conn = self.connection.clone();

        let alerts_json: Vec<String> = conn
            .lrange(ALERTS_KEY, 0, -1)
            .await
            .map_err(AppError::Redis)?;

        let mut alerts = Vec::new();
        for json in alerts_json {
            if let Ok(alert) = serde_json::from_str::<Alert>(&json) {
                alerts.push(alert);
            }
        }

        Ok(alerts)
    }

    /// Clear all alerts (user acknowledged them).
    pub async fn reset_alerts(&self) -> AppResult<()> {
        let mut conn = self.connection.clone();
        let _: () = conn.del(ALERTS_KEY).await.map_err(AppError::Redis)?;

        tracing::info!("Alerts reset");
        Ok(())
    }

    /// Get alert count.
    pub async fn get_alert_count(&self) -> AppResult<u64> {
        let mut conn = self.connection.clone();
        let count: u64 = conn.llen(ALERTS_KEY).await.map_err(AppError::Redis)?;
        Ok(count)
    }
}
