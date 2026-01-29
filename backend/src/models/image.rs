//! # Image Submission Models
//!
//! In the current implementation, images are referenced by ID rather than
//! stored directly. This allows us to focus on the pipeline architecture
//! while deferring blob storage implementation.
//!
//! ## Future Storage Strategy
//!
//! When real image storage is implemented, we plan to use:
//! - MinIO (S3-compatible) for image blobs
//! - Signed URLs for secure upload/download
//! - Redis for metadata caching

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Metadata about a submitted image.
///
/// In the current stub implementation, only the ID is required.
/// Future versions will include actual image storage references.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMeta {
    /// Unique identifier for the image
    pub image_id: String,

    /// Timestamp when the image was submitted
    pub submitted_at: DateTime<Utc>,

    /// Source of the submission (api, batch, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,

    /// Original filename (if provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,

    /// MIME type of the image
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,

    /// Image dimensions (width, height)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<(u32, u32)>,

    /// Storage location (S3 key, file path, etc.)
    /// Currently None as we're using ID-only references
    #[serde(skip_serializing_if = "Option::is_none")]
    pub storage_key: Option<String>,
}

impl ImageMeta {
    /// Create new image metadata with just an ID.
    pub fn new(image_id: String) -> Self {
        Self {
            image_id,
            submitted_at: Utc::now(),
            source: None,
            filename: None,
            content_type: None,
            dimensions: None,
            storage_key: None,
        }
    }
}
