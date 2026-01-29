//! # Services Module
//!
//! Contains business logic and external service integrations.

pub mod redis;

pub use self::redis::RedisService;
