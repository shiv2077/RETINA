//! # Authentication Routes
//!
//! JWT-based authentication endpoints matching professor's framework.
//!
//! ## Endpoints
//!
//! - POST /auth/register - Create new user account
//! - POST /auth/token - Login and receive JWT

use axum::{
    extract::State,
    routing::post,
    Json, Router,
};
use sqlx::PgPool;

use crate::auth::{
    create_token, hash_password, verify_password, LoginRequest, RegisterRequest, TokenResponse,
};
use crate::error::{AppError, AppResult};
use crate::AppState;

/// Build the auth router.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/token", post(login))
        .route("/register", post(register))
}

/// Login and receive JWT token.
///
/// ## Request Body (form data or JSON)
///
/// ```json
/// {
///   "username": "admin",
///   "password": "yourpassword"
/// }
/// ```
///
/// ## Response
///
/// ```json
/// {
///   "access_token": "eyJ...",
///   "token_type": "bearer"
/// }
/// ```
async fn login(
    State(state): State<AppState>,
    Json(request): Json<LoginRequest>,
) -> AppResult<Json<TokenResponse>> {
    // Query user from database
    let user: Option<crate::db::models::User> = sqlx::query_as(
        "SELECT id, username, hashed_password FROM users WHERE username = $1",
    )
    .bind(&request.username)
    .fetch_optional(&*state.db)
    .await
    .map_err(|e| AppError::Internal(format!("Database error: {}", e)))?;

    let user = user.ok_or_else(|| AppError::BadRequest("Invalid username or password".to_string()))?;

    // Verify password
    if !verify_password(&request.password, &user.hashed_password) {
        return Err(AppError::BadRequest("Invalid username or password".to_string()));
    }

    // Create JWT — secret from config, never from env directly
    let token = create_token(&user.username, &state.config.jwt_secret)
        .map_err(|e| AppError::Internal(format!("Token creation failed: {}", e)))?;

    tracing::info!(username = %user.username, "User logged in");

    Ok(Json(TokenResponse {
        access_token: token,
        token_type: "bearer".to_string(),
    }))
}

/// Register a new user account.
///
/// ## Request Body
///
/// ```json
/// {
///   "username": "newuser",
///   "password": "securepass"
/// }
/// ```
///
/// ## Response
///
/// ```json
/// {
///   "message": "User created successfully"
/// }
/// ```
async fn register(
    State(state): State<AppState>,
    Json(request): Json<RegisterRequest>,
) -> AppResult<Json<serde_json::Value>> {
    // Validate input
    if request.username.is_empty() || request.password.is_empty() {
        return Err(AppError::BadRequest("Username and password required".to_string()));
    }

    if request.password.len() < 6 {
        return Err(AppError::BadRequest("Password must be at least 6 characters".to_string()));
    }

    // Check if username exists
    let existing: Option<(i32,)> = sqlx::query_as("SELECT id FROM users WHERE username = $1")
        .bind(&request.username)
        .fetch_optional(&*state.db)
        .await
        .map_err(|e| AppError::Internal(format!("Database error: {}", e)))?;

    if existing.is_some() {
        return Err(AppError::BadRequest("Username already exists".to_string()));
    }

    // Hash password
    let hashed = hash_password(&request.password)
        .map_err(|e| AppError::Internal(format!("Password hashing failed: {}", e)))?;

    // Insert user
    sqlx::query("INSERT INTO users (username, hashed_password) VALUES ($1, $2)")
        .bind(&request.username)
        .bind(&hashed)
        .execute(&*state.db)
        .await
        .map_err(|e| AppError::Internal(format!("User creation failed: {}", e)))?;

    tracing::info!(username = %request.username, "New user registered");

    Ok(Json(serde_json::json!({
        "message": "User created successfully"
    })))
}
