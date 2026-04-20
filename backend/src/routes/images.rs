//! # Image Submission and Result Endpoints
//!
//! Handles image submission for inference and result retrieval.
//!
//! ## Workflow
//!
//! 1. Client submits image ID via `POST /api/images/submit`
//! 2. Backend creates job and pushes to Redis queue
//! 3. Python worker processes job (via Redis Streams)
//! 4. Client polls `GET /api/images/{id}/result` for results
//!
//! ## Note on Image Storage
//!
//! Currently, images are referenced by ID only (no blob storage).
//! This allows us to focus on the pipeline architecture first.
//! Future implementation will integrate MinIO (S3-compatible) storage.

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use uuid::Uuid;

use crate::error::{AppError, AppResult};
use crate::models::{
    ImageMeta, InferenceJob, InferenceResult, JobMetadata, JobStatus, ModelType, PipelineStage,
    ResultResponse, SubmitJobRequest, SubmitJobResponse,
};
use crate::AppState;

/// Build the images router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/submit", post(submit_image))
        .route("/{id}/result", get(get_result))
        .route("/{id}/status", get(get_status))
}

/// Submit an image for anomaly detection.
///
/// ## Request Body
///
/// ```json
/// {
///   "image_id": "img_2024_001_front_panel",
///   "model_type": "patchcore",  // optional, defaults to patchcore
///   "priority": 5,              // optional, 0-10
///   "source": "api"             // optional
/// }
/// ```
///
/// ## Response
///
/// ```json
/// {
///   "job_id": "550e8400-e29b-41d4-a716-446655440000",
///   "status": "queued",
///   "queue_position": 3,
///   "message": "Image submitted for analysis"
/// }
/// ```
///
/// ## Pipeline Stage Selection
///
/// The endpoint automatically selects the appropriate pipeline stage:
/// - Stage 1 (unsupervised) if fewer than `AL_STAGE2_THRESHOLD` labels exist
/// - Stage 2 (supervised) if sufficient labels have been collected
async fn submit_image(
    State(state): State<AppState>,
    Json(request): Json<SubmitJobRequest>,
) -> AppResult<Json<SubmitJobResponse>> {
    // Validate image_id
    if request.image_id.is_empty() {
        return Err(AppError::BadRequest("image_id cannot be empty".to_string()));
    }

    // Determine pipeline stage based on label count
    let label_count = state.redis.get_label_count().await?;
    let stage = if label_count >= state.config.active_learning_stage2_threshold as u64 {
        tracing::info!(
            label_count = label_count,
            threshold = state.config.active_learning_stage2_threshold,
            "Stage 2 activated - using supervised model"
        );
        PipelineStage::Supervised
    } else {
        PipelineStage::Unsupervised
    };

    // Select model based on stage
    let model_type = match stage {
        PipelineStage::Supervised => ModelType::PushPull,
        // Default Stage 1 model is PatchCore (memory-bank k-NN).
        // The worker falls back to GPT-4V automatically when no memory bank exists (cold start).
        PipelineStage::Unsupervised => request.model_type.unwrap_or(ModelType::PatchCore),
    };

    // Resolve image path for the worker.
    // The client should set this after uploading the file via POST /api/images/upload.
    // If not provided, the worker will run in stub/no-image mode.
    let image_path = request.image_path.clone();
    if image_path.is_some() {
        tracing::debug!(image_id = %request.image_id, "Image path provided for worker");
    } else {
        tracing::debug!(
            image_id = %request.image_id,
            "No image_path in request — worker will run without pixel data. \
             Upload via POST /api/images/upload first."
        );
    }

    // Create inference job
    let mut job = InferenceJob::new(request.image_id.clone(), model_type)
        .with_stage(stage)
        .with_priority(request.priority.unwrap_or(5))
        .with_metadata(JobMetadata {
            source: request.source,
            batch_id: None,
            callback_url: None,
        });

    if let Some(path) = image_path {
        job = job.with_image_path(path);
    }

    let job_id = job.job_id;

    // Store image metadata reference
    let _image_meta = ImageMeta::new(request.image_id);

    // Submit job to queue
    let _stream_id = state.redis.submit_job(&job).await?;

    // Get queue position
    let queue_length = state.redis.get_queue_length().await?;

    tracing::info!(
        job_id = %job_id,
        model = %model_type,
        stage = ?stage,
        "Image submitted for analysis"
    );

    Ok(Json(SubmitJobResponse {
        job_id,
        status: JobStatus::Queued,
        queue_position: Some(queue_length),
        message: format!(
            "Image submitted for {} analysis (Stage {})",
            model_type,
            if stage == PipelineStage::Supervised {
                2
            } else {
                1
            }
        ),
    }))
}

/// Get inference result for an image.
///
/// ## Path Parameters
///
/// - `id`: Can be either a job_id (UUID) or image_id (string)
///
/// ## Response
///
/// ```json
/// {
///   "found": true,
///   "result": {
///     "job_id": "550e8400-e29b-41d4-a716-446655440000",
///     "image_id": "img_2024_001_front_panel",
///     "status": "completed",
///     "anomaly_score": 0.73,
///     "is_anomaly": true,
///     ...
///   },
///   "message": "Result found"
/// }
/// ```
async fn get_result(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<ResultResponse>> {
    // Try parsing as UUID first (job_id)
    let result = if let Ok(job_id) = Uuid::parse_str(&id) {
        state.redis.get_result(job_id).await?
    } else {
        // Otherwise treat as image_id
        state.redis.get_result_by_image(&id).await?
    };

    match result {
        Some(r) => Ok(Json(ResultResponse {
            found: true,
            result: Some(r),
            message: "Result found".to_string(),
        })),
        None => {
            // Check if job exists but isn't complete yet
            if let Ok(job_id) = Uuid::parse_str(&id) {
                if let Some(status) = state.redis.get_job_status(job_id).await? {
                    return Ok(Json(ResultResponse {
                        found: true,
                        result: Some(InferenceResult::pending(job_id, id.clone())),
                        message: format!("Job is {:?}, result not yet available", status),
                    }));
                }
            }

            Ok(Json(ResultResponse {
                found: false,
                result: None,
                message: "No result found for this ID".to_string(),
            }))
        }
    }
}

/// Get job status without full result.
///
/// ## Path Parameters
///
/// - `id`: Job ID (UUID)
///
/// ## Response
///
/// ```json
/// {
///   "job_id": "550e8400-e29b-41d4-a716-446655440000",
///   "status": "processing"
/// }
/// ```
async fn get_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> AppResult<Json<StatusResponse>> {
    let job_id = Uuid::parse_str(&id)
        .map_err(|_| AppError::BadRequest("Invalid job ID format".to_string()))?;

    let status = state.redis.get_job_status(job_id).await?;

    match status {
        Some(s) => Ok(Json(StatusResponse {
            job_id,
            status: s,
            found: true,
        })),
        None => Err(AppError::NotFound(format!("Job {} not found", job_id))),
    }
}

/// Status-only response.
#[derive(Debug, serde::Serialize)]
struct StatusResponse {
    job_id: Uuid,
    status: JobStatus,
    found: bool,
}
