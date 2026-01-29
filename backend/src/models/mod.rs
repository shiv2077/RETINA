//! # Domain Models
//!
//! This module defines the core data structures used throughout the RETINA
//! backend. These models are aligned with the JSON schemas in `/shared/schemas/`
//! to ensure consistency with the Python worker.

pub mod image;
pub mod job;
pub mod label;
pub mod result;

pub use image::*;
pub use job::*;
pub use label::*;
pub use result::*;
