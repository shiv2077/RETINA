//! # Authentication Module
//!
//! JWT-based authentication matching professor's OAuth2 password flow.
//!
//! ## Flow
//!
//! 1. User registers: POST /auth/register
//! 2. User logs in: POST /auth/token → receives JWT
//! 3. User includes JWT in Authorization header
//! 4. Protected routes verify JWT via middleware

use axum::{
    async_trait,
    extract::FromRequestParts,
    http::{header::AUTHORIZATION, request::Parts, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use chrono::{Duration, Utc};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// JWT claims structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (username)
    pub sub: String,

    /// Expiration time (Unix timestamp)
    pub exp: i64,

    /// Issued at (Unix timestamp)
    pub iat: i64,
}

impl Claims {
    /// Create new claims for a user.
    pub fn new(username: &str, expires_in_hours: i64) -> Self {
        let now = Utc::now();
        Self {
            sub: username.to_string(),
            iat: now.timestamp(),
            exp: (now + Duration::hours(expires_in_hours)).timestamp(),
        }
    }
}

/// Create a JWT token for a user.
pub fn create_token(username: &str, secret: &str) -> Result<String, jsonwebtoken::errors::Error> {
    let claims = Claims::new(username, 24); // 24 hour expiry
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
}

/// Verify and decode a JWT token.
pub fn verify_token(token: &str, secret: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )?;
    Ok(token_data.claims)
}

/// Hash a password using Argon2.
pub fn hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
    use argon2::{
        password_hash::{rand_core::OsRng, PasswordHasher, SaltString},
        Argon2,
    };

    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let hash = argon2.hash_password(password.as_bytes(), &salt)?;
    Ok(hash.to_string())
}

/// Verify a password against a hash.
pub fn verify_password(password: &str, hash: &str) -> bool {
    use argon2::{password_hash::PasswordVerifier, Argon2, PasswordHash};

    let parsed_hash = match PasswordHash::new(hash) {
        Ok(h) => h,
        Err(_) => return false,
    };

    Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .is_ok()
}

/// Authenticated user extracted from JWT.
#[derive(Debug, Clone)]
pub struct AuthenticatedUser {
    pub username: String,
}

/// Authentication error.
#[derive(Debug)]
pub struct AuthError {
    pub message: String,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let body = Json(json!({
            "error": {
                "code": "UNAUTHORIZED",
                "message": self.message
            }
        }));
        (StatusCode::UNAUTHORIZED, body).into_response()
    }
}

/// Extractor for authenticated users.
///
/// Use this in route handlers to require authentication:
///
/// ```rust
/// async fn protected_route(user: AuthenticatedUser) -> impl IntoResponse {
///     format!("Hello, {}!", user.username)
/// }
/// ```
#[async_trait]
impl<S> FromRequestParts<S> for AuthenticatedUser
where
    S: Send + Sync,
{
    type Rejection = AuthError;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        // Get Authorization header
        let auth_header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| AuthError {
                message: "Missing Authorization header".to_string(),
            })?;

        // Extract Bearer token
        let token = auth_header
            .strip_prefix("Bearer ")
            .ok_or_else(|| AuthError {
                message: "Invalid Authorization header format".to_string(),
            })?;

        // Get secret from environment (in production, use proper config)
        let secret = std::env::var("JWT_SECRET").unwrap_or_else(|_| "retina-dev-secret".to_string());

        // Verify token
        let claims = verify_token(token, &secret).map_err(|e| AuthError {
            message: format!("Invalid token: {}", e),
        })?;

        Ok(AuthenticatedUser {
            username: claims.sub,
        })
    }
}

// =============================================================================
// Request/Response Types
// =============================================================================

/// Login request (OAuth2 password grant style).
#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

/// Token response (matches professor's format).
#[derive(Debug, Serialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
}

/// Register request.
#[derive(Debug, Deserialize)]
pub struct RegisterRequest {
    pub username: String,
    pub password: String,
}
