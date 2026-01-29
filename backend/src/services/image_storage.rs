//! # Image Storage Service
//!
//! Filesystem-based image storage matching professor's framework.
//!
//! ## Storage Pattern
//!
//! Images are stored as:
//! ```
//! data/images/{job_id}_{original_filename}
//! ```
//!
//! This allows:
//! - Easy retrieval by job ID
//! - Preserving original filenames for debugging
//! - Simple filesystem backup

use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::error::{AppError, AppResult};

/// Default storage directory.
const STORAGE_DIR: &str = "data/images";

/// Image storage service.
#[derive(Clone)]
pub struct ImageStorage {
    /// Base directory for images.
    base_dir: PathBuf,
}

impl ImageStorage {
    /// Create a new image storage service.
    pub async fn new(base_dir: Option<&str>) -> AppResult<Self> {
        let base_dir = PathBuf::from(base_dir.unwrap_or(STORAGE_DIR));

        // Create directory if it doesn't exist
        fs::create_dir_all(&base_dir).await.map_err(|e| {
            AppError::Internal(format!("Failed to create storage directory: {}", e))
        })?;

        tracing::info!(path = %base_dir.display(), "Image storage initialized");

        Ok(Self { base_dir })
    }

    /// Store an image and return its file path.
    ///
    /// ## Arguments
    ///
    /// - `job_id` - Unique job identifier
    /// - `filename` - Original filename (used for extension)
    /// - `data` - Image bytes
    ///
    /// ## Returns
    ///
    /// The full file path where the image was stored.
    pub async fn store_image(
        &self,
        job_id: &str,
        filename: &str,
        data: &[u8],
    ) -> AppResult<String> {
        // Extract extension from original filename
        let extension = Path::new(filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("jpg");

        // Sanitize filename
        let safe_filename = filename
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '.' || *c == '_' || *c == '-')
            .collect::<String>();

        // Create storage path: {job_id}_{original_filename}
        let storage_name = format!("{}_{}", job_id, safe_filename);
        let file_path = self.base_dir.join(&storage_name);

        // Write file
        let mut file = fs::File::create(&file_path).await.map_err(|e| {
            AppError::Internal(format!("Failed to create image file: {}", e))
        })?;

        file.write_all(data).await.map_err(|e| {
            AppError::Internal(format!("Failed to write image data: {}", e))
        })?;

        file.flush().await.map_err(|e| {
            AppError::Internal(format!("Failed to flush image file: {}", e))
        })?;

        let path_str = file_path.to_string_lossy().to_string();
        tracing::debug!(job_id = %job_id, path = %path_str, "Image stored");

        Ok(path_str)
    }

    /// Read an image by job ID.
    ///
    /// Scans the storage directory for files matching the job ID prefix.
    pub async fn read_image(&self, job_id: &str) -> AppResult<Option<Vec<u8>>> {
        let mut entries = fs::read_dir(&self.base_dir).await.map_err(|e| {
            AppError::Internal(format!("Failed to read storage directory: {}", e))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            AppError::Internal(format!("Failed to iterate storage: {}", e))
        })? {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(job_id) {
                let data = fs::read(entry.path()).await.map_err(|e| {
                    AppError::Internal(format!("Failed to read image: {}", e))
                })?;
                return Ok(Some(data));
            }
        }

        Ok(None)
    }

    /// Delete an image by job ID.
    pub async fn delete_image(&self, job_id: &str) -> AppResult<bool> {
        let mut entries = fs::read_dir(&self.base_dir).await.map_err(|e| {
            AppError::Internal(format!("Failed to read storage directory: {}", e))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            AppError::Internal(format!("Failed to iterate storage: {}", e))
        })? {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(job_id) {
                fs::remove_file(entry.path()).await.map_err(|e| {
                    AppError::Internal(format!("Failed to delete image: {}", e))
                })?;
                tracing::debug!(job_id = %job_id, "Image deleted");
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get the file path for a job ID (without reading).
    pub async fn get_image_path(&self, job_id: &str) -> AppResult<Option<String>> {
        let mut entries = fs::read_dir(&self.base_dir).await.map_err(|e| {
            AppError::Internal(format!("Failed to read storage directory: {}", e))
        })?;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            AppError::Internal(format!("Failed to iterate storage: {}", e))
        })? {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with(job_id) {
                return Ok(Some(entry.path().to_string_lossy().to_string()));
            }
        }

        Ok(None)
    }

    /// Get storage statistics.
    pub async fn get_stats(&self) -> AppResult<StorageStats> {
        let mut entries = fs::read_dir(&self.base_dir).await.map_err(|e| {
            AppError::Internal(format!("Failed to read storage directory: {}", e))
        })?;

        let mut count = 0u64;
        let mut total_bytes = 0u64;

        while let Some(entry) = entries.next_entry().await.map_err(|e| {
            AppError::Internal(format!("Failed to iterate storage: {}", e))
        })? {
            if let Ok(metadata) = entry.metadata().await {
                if metadata.is_file() {
                    count += 1;
                    total_bytes += metadata.len();
                }
            }
        }

        Ok(StorageStats { count, total_bytes })
    }
}

/// Storage statistics.
#[derive(Debug, serde::Serialize)]
pub struct StorageStats {
    /// Number of stored images.
    pub count: u64,
    /// Total bytes used.
    pub total_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_store_and_read() {
        let dir = tempdir().unwrap();
        let storage = ImageStorage::new(Some(dir.path().to_str().unwrap()))
            .await
            .unwrap();

        let job_id = "test-job-123";
        let data = b"fake image data";

        let path = storage
            .store_image(job_id, "test.jpg", data)
            .await
            .unwrap();

        assert!(path.contains(job_id));

        let read_data = storage.read_image(job_id).await.unwrap();
        assert_eq!(read_data, Some(data.to_vec()));
    }

    #[tokio::test]
    async fn test_delete() {
        let dir = tempdir().unwrap();
        let storage = ImageStorage::new(Some(dir.path().to_str().unwrap()))
            .await
            .unwrap();

        let job_id = "delete-test";
        storage
            .store_image(job_id, "test.png", b"data")
            .await
            .unwrap();

        let deleted = storage.delete_image(job_id).await.unwrap();
        assert!(deleted);

        let data = storage.read_image(job_id).await.unwrap();
        assert!(data.is_none());
    }
}
