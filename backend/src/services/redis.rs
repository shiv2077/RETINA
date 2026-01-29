//! # Redis Service
//!
//! Provides async Redis operations for job queuing, result storage,
//! and active learning state management.
//!
//! ## Redis Key Schema
//!
//! We use a consistent key naming convention for organization:
//!
//! ```text
//! retina:jobs:queue          - Stream for inference job queue
//! retina:jobs:{job_id}       - Hash with job metadata
//! retina:results:{job_id}    - Hash with inference results
//! retina:images:{image_id}   - Hash with image metadata
//! retina:labels:{image_id}   - Hash with label data
//! retina:labels:count        - Counter for total labels
//! retina:al:pool             - Sorted set for active learning candidates (by uncertainty)
//! retina:system:stats        - Hash with system statistics
//! ```
//!
//! ## Redis Streams for Job Queue
//!
//! We use Redis Streams with consumer groups for reliable job distribution:
//!
//! - Producer (backend): `XADD retina:jobs:queue * job_data {json}`
//! - Consumer (worker): `XREADGROUP GROUP workers worker-1 ... STREAMS retina:jobs:queue >`
//! - Acknowledgment: `XACK retina:jobs:queue workers {message_id}`
//!
//! This pattern ensures at-least-once delivery and allows scaling workers.

use redis::aio::MultiplexedConnection;
use redis::{AsyncCommands, Client, RedisResult};
use uuid::Uuid;

use crate::error::{AppError, AppResult};
use crate::models::{InferenceJob, InferenceResult, JobStatus, Label, UnlabeledSample};

/// Key prefix for all RETINA Redis keys
const KEY_PREFIX: &str = "retina";

/// Stream name for the inference job queue
const JOB_QUEUE_STREAM: &str = "retina:jobs:queue";

/// Consumer group name for workers
const WORKER_GROUP: &str = "workers";

/// Redis service for RETINA backend.
///
/// Manages all Redis interactions including:
/// - Job queue operations (Streams)
/// - Result storage (Hashes)
/// - Active learning pool (Sorted Sets)
/// - System statistics
pub struct RedisService {
    /// Multiplexed connection for concurrent operations
    connection: MultiplexedConnection,
}

impl RedisService {
    /// Create a new Redis service with the given connection URL.
    ///
    /// # Arguments
    ///
    /// * `url` - Redis connection URL (e.g., `redis://localhost:6379`)
    ///
    /// # Errors
    ///
    /// Returns an error if connection fails.
    pub async fn new(url: &str) -> AppResult<Self> {
        let client = Client::open(url).map_err(AppError::Redis)?;
        let connection = client
            .get_multiplexed_async_connection()
            .await
            .map_err(AppError::Redis)?;

        Ok(Self { connection })
    }

    /// Initialize required Redis data structures.
    ///
    /// Creates:
    /// - Consumer group for the job queue stream
    /// - Initial system statistics hash
    ///
    /// This is idempotent - safe to call multiple times.
    pub async fn initialize_data_structures(&self) -> AppResult<()> {
        let mut conn = self.connection.clone();

        // Create the stream and consumer group
        // XGROUP CREATE creates both if they don't exist
        let result: RedisResult<()> = redis::cmd("XGROUP")
            .arg("CREATE")
            .arg(JOB_QUEUE_STREAM)
            .arg(WORKER_GROUP)
            .arg("$") // Start from new messages
            .arg("MKSTREAM") // Create stream if doesn't exist
            .query_async(&mut conn)
            .await;

        // Ignore "BUSYGROUP" error (group already exists)
        match result {
            Ok(_) => tracing::info!("Created consumer group '{}'", WORKER_GROUP),
            Err(e) if e.to_string().contains("BUSYGROUP") => {
                tracing::debug!("Consumer group '{}' already exists", WORKER_GROUP);
            }
            Err(e) => return Err(AppError::Redis(e)),
        }

        // Initialize system stats if not exists
        let stats_key = format!("{}:system:stats", KEY_PREFIX);
        let _: () = conn
            .hset_nx(&stats_key, "jobs_submitted", 0i64)
            .await
            .map_err(AppError::Redis)?;
        let _: () = conn
            .hset_nx(&stats_key, "jobs_completed", 0i64)
            .await
            .map_err(AppError::Redis)?;
        let _: () = conn
            .hset_nx(&stats_key, "labels_collected", 0i64)
            .await
            .map_err(AppError::Redis)?;

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Job Queue Operations
    // -------------------------------------------------------------------------

    /// Submit an inference job to the queue.
    ///
    /// The job is:
    /// 1. Serialized to JSON
    /// 2. Added to the Redis Stream
    /// 3. Stored in a hash for status lookups
    ///
    /// Returns the stream entry ID.
    pub async fn submit_job(&self, job: &InferenceJob) -> AppResult<String> {
        let mut conn = self.connection.clone();

        // Serialize job to JSON
        let job_json = serde_json::to_string(job)?;

        // Add to stream
        // XADD retina:jobs:queue * job_data {json}
        let entry_id: String = redis::cmd("XADD")
            .arg(JOB_QUEUE_STREAM)
            .arg("*") // Auto-generate ID
            .arg("job_data")
            .arg(&job_json)
            .query_async(&mut conn)
            .await
            .map_err(AppError::Redis)?;

        tracing::debug!(job_id = %job.job_id, entry_id = %entry_id, "Job added to queue");

        // Store job metadata in hash for lookups
        let job_key = format!("{}:jobs:{}", KEY_PREFIX, job.job_id);
        let _: () = conn
            .hset_multiple(
                &job_key,
                &[
                    ("job_data", &job_json),
                    ("status", &"queued".to_string()),
                    ("stream_id", &entry_id),
                ],
            )
            .await
            .map_err(AppError::Redis)?;

        // Set TTL (7 days)
        let _: () = conn
            .expire(&job_key, 7 * 24 * 60 * 60)
            .await
            .map_err(AppError::Redis)?;

        // Increment submitted counter
        let stats_key = format!("{}:system:stats", KEY_PREFIX);
        let _: () = conn
            .hincr(&stats_key, "jobs_submitted", 1i64)
            .await
            .map_err(AppError::Redis)?;

        Ok(entry_id)
    }

    /// Get the current queue length (pending jobs).
    pub async fn get_queue_length(&self) -> AppResult<u64> {
        let mut conn = self.connection.clone();

        let length: u64 = redis::cmd("XLEN")
            .arg(JOB_QUEUE_STREAM)
            .query_async(&mut conn)
            .await
            .map_err(AppError::Redis)?;

        Ok(length)
    }

    /// Get job status by job ID.
    pub async fn get_job_status(&self, job_id: Uuid) -> AppResult<Option<JobStatus>> {
        let mut conn = self.connection.clone();
        let job_key = format!("{}:jobs:{}", KEY_PREFIX, job_id);

        let status: Option<String> = conn
            .hget(&job_key, "status")
            .await
            .map_err(AppError::Redis)?;

        Ok(status.map(|s| match s.as_str() {
            "pending" => JobStatus::Pending,
            "queued" => JobStatus::Queued,
            "processing" => JobStatus::Processing,
            "completed" => JobStatus::Completed,
            "failed" => JobStatus::Failed,
            _ => JobStatus::Pending,
        }))
    }

    // -------------------------------------------------------------------------
    // Result Operations
    // -------------------------------------------------------------------------

    /// Get inference result by job ID.
    pub async fn get_result(&self, job_id: Uuid) -> AppResult<Option<InferenceResult>> {
        let mut conn = self.connection.clone();
        let result_key = format!("{}:results:{}", KEY_PREFIX, job_id);

        let result_json: Option<String> = conn
            .hget(&result_key, "result_data")
            .await
            .map_err(AppError::Redis)?;

        match result_json {
            Some(json) => {
                let result: InferenceResult = serde_json::from_str(&json)?;
                Ok(Some(result))
            }
            None => Ok(None),
        }
    }

    /// Get result by image ID (looks up the most recent job for that image).
    pub async fn get_result_by_image(&self, image_id: &str) -> AppResult<Option<InferenceResult>> {
        let mut conn = self.connection.clone();
        let image_key = format!("{}:images:{}", KEY_PREFIX, image_id);

        // Get the latest job ID for this image
        let job_id: Option<String> = conn
            .hget(&image_key, "latest_job_id")
            .await
            .map_err(AppError::Redis)?;

        match job_id {
            Some(id) => {
                let uuid = Uuid::parse_str(&id)
                    .map_err(|e| AppError::Internal(format!("Invalid job ID: {}", e)))?;
                self.get_result(uuid).await
            }
            None => Ok(None),
        }
    }

    // -------------------------------------------------------------------------
    // Active Learning Operations
    // -------------------------------------------------------------------------

    /// Submit a label for an image.
    ///
    /// This:
    /// 1. Stores the label
    /// 2. Removes the sample from the labeling pool
    /// 3. Increments the label counter
    pub async fn submit_label(&self, label: &Label) -> AppResult<u64> {
        let mut conn = self.connection.clone();

        // Serialize and store label
        let label_key = format!("{}:labels:{}", KEY_PREFIX, label.image_id);
        let label_json = serde_json::to_string(label)?;

        let _: () = conn
            .hset(&label_key, "label_data", &label_json)
            .await
            .map_err(AppError::Redis)?;

        // Remove from active learning pool
        let pool_key = format!("{}:al:pool", KEY_PREFIX);
        let _: () = conn
            .zrem(&pool_key, &label.image_id)
            .await
            .map_err(AppError::Redis)?;

        // Increment label counter
        let stats_key = format!("{}:system:stats", KEY_PREFIX);
        let total: i64 = conn
            .hincr(&stats_key, "labels_collected", 1i64)
            .await
            .map_err(AppError::Redis)?;

        tracing::info!(
            image_id = %label.image_id,
            total_labels = total,
            "Label submitted"
        );

        Ok(total as u64)
    }

    /// Get total number of labels collected.
    pub async fn get_label_count(&self) -> AppResult<u64> {
        let mut conn = self.connection.clone();
        let stats_key = format!("{}:system:stats", KEY_PREFIX);

        let count: i64 = conn
            .hget(&stats_key, "labels_collected")
            .await
            .map_err(AppError::Redis)?;

        Ok(count as u64)
    }

    /// Get samples from the active learning pool for labeling.
    ///
    /// Returns samples ordered by uncertainty (highest first).
    pub async fn get_labeling_pool(&self, limit: usize) -> AppResult<Vec<UnlabeledSample>> {
        let mut conn = self.connection.clone();
        let pool_key = format!("{}:al:pool", KEY_PREFIX);

        // Get top samples by uncertainty score (descending)
        let samples: Vec<(String, f64)> = conn
            .zrevrange_withscores(&pool_key, 0, (limit - 1) as isize)
            .await
            .map_err(AppError::Redis)?;

        // Fetch full sample data
        let mut result = Vec::with_capacity(samples.len());
        for (image_id, uncertainty_score) in samples {
            let sample_key = format!("{}:al:samples:{}", KEY_PREFIX, image_id);
            let sample_json: Option<String> = conn
                .get(&sample_key)
                .await
                .map_err(AppError::Redis)?;

            if let Some(json) = sample_json {
                if let Ok(mut sample) = serde_json::from_str::<UnlabeledSample>(&json) {
                    sample.uncertainty_score = uncertainty_score;
                    result.push(sample);
                }
            }
        }

        Ok(result)
    }

    /// Get the size of the active learning pool.
    pub async fn get_pool_size(&self) -> AppResult<u64> {
        let mut conn = self.connection.clone();
        let pool_key = format!("{}:al:pool", KEY_PREFIX);

        let size: u64 = conn.zcard(&pool_key).await.map_err(AppError::Redis)?;

        Ok(size)
    }

    // -------------------------------------------------------------------------
    // System Status Operations
    // -------------------------------------------------------------------------

    /// Get system statistics.
    pub async fn get_system_stats(&self) -> AppResult<SystemStats> {
        let mut conn = self.connection.clone();
        let stats_key = format!("{}:system:stats", KEY_PREFIX);

        let stats: Vec<(String, i64)> = conn
            .hgetall(&stats_key)
            .await
            .map_err(AppError::Redis)?;

        let mut result = SystemStats::default();
        for (key, value) in stats {
            match key.as_str() {
                "jobs_submitted" => result.jobs_submitted = value as u64,
                "jobs_completed" => result.jobs_completed = value as u64,
                "labels_collected" => result.labels_collected = value as u64,
                _ => {}
            }
        }

        // Get queue length
        result.queue_length = self.get_queue_length().await?;

        // Get pool size
        result.labeling_pool_size = self.get_pool_size().await?;

        Ok(result)
    }

    /// Health check - verifies Redis connectivity.
    pub async fn health_check(&self) -> AppResult<bool> {
        let mut conn = self.connection.clone();
        let pong: String = redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(AppError::Redis)?;

        Ok(pong == "PONG")
    }

    // -------------------------------------------------------------------------
    // Mismatch Review Operations (Matching Professor's Framework)
    // -------------------------------------------------------------------------

    /// Get all records with mismatch between supervised and unsupervised.
    pub async fn get_mismatch_records(&self) -> AppResult<Vec<crate::routes::labels::PendingReviewRecord>> {
        let mut conn = self.connection.clone();
        
        // Get all results and filter for mismatch
        // In production, use a dedicated index or sorted set
        let mismatch_key = format!("{}:mismatches", KEY_PREFIX);
        
        let job_ids: Vec<String> = conn
            .smembers(&mismatch_key)
            .await
            .map_err(AppError::Redis)?;
        
        let mut records = Vec::new();
        for job_id in job_ids {
            let result_key = format!("{}:results:{}", KEY_PREFIX, job_id);
            let result_data: Option<String> = conn
                .hget(&result_key, "result_data")
                .await
                .map_err(AppError::Redis)?;
            
            if let Some(data) = result_data {
                if let Ok(result) = serde_json::from_str::<serde_json::Value>(&data) {
                    // Only include if not yet reviewed
                    if result.get("reviewed").and_then(|v| v.as_bool()).unwrap_or(false) {
                        continue;
                    }
                    
                    records.push(crate::routes::labels::PendingReviewRecord {
                        job_id: job_id.clone(),
                        image_id: result.get("image_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        supervised_label: result.get("is_anomaly")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        unsupervised_label: result.get("unsupervised_label")
                            .and_then(|v| v.as_bool()),
                        mismatch: result.get("mismatch")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false),
                        timestamp: result.get("created_at")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                    });
                }
            }
        }
        
        Ok(records)
    }

    /// Submit expert review for a record.
    pub async fn submit_review(
        &self,
        job_id: &str,
        expert_label: &str,
        final_classification: &str,
    ) -> AppResult<()> {
        let mut conn = self.connection.clone();
        
        // Get existing result
        let result_key = format!("{}:results:{}", KEY_PREFIX, job_id);
        let result_data: Option<String> = conn
            .hget(&result_key, "result_data")
            .await
            .map_err(AppError::Redis)?;
        
        if let Some(data) = result_data {
            let mut result: serde_json::Value = serde_json::from_str(&data)?;
            
            // Update with review data
            result["reviewed"] = serde_json::json!(true);
            result["expert_label"] = serde_json::json!(expert_label);
            result["final_classification"] = serde_json::json!(final_classification);
            
            // Store updated result
            let updated_data = serde_json::to_string(&result)?;
            let _: () = conn
                .hset(&result_key, "result_data", &updated_data)
                .await
                .map_err(AppError::Redis)?;
            
            // Remove from mismatch set
            let mismatch_key = format!("{}:mismatches", KEY_PREFIX);
            let _: () = conn
                .srem(&mismatch_key, job_id)
                .await
                .map_err(AppError::Redis)?;
            
            tracing::info!(job_id = %job_id, "Review submitted");
        }
        
        Ok(())
    }

    /// Mark a record as having mismatch.
    pub async fn mark_mismatch(&self, job_id: &str) -> AppResult<()> {
        let mut conn = self.connection.clone();
        let mismatch_key = format!("{}:mismatches", KEY_PREFIX);
        
        let _: () = conn
            .sadd(&mismatch_key, job_id)
            .await
            .map_err(AppError::Redis)?;
        
        Ok(())
    }
}

/// System statistics from Redis.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SystemStats {
    /// Total jobs submitted
    pub jobs_submitted: u64,

    /// Jobs completed successfully
    pub jobs_completed: u64,

    /// Current queue length
    pub queue_length: u64,

    /// Labels collected via active learning
    pub labels_collected: u64,

    /// Samples in labeling pool
    pub labeling_pool_size: u64,
}
