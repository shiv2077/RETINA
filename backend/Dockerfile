# =============================================================================
# RETINA Backend - Dockerfile
# =============================================================================
# Multi-stage build for the Rust/Axum backend API server.
#
# Build: docker build -t retina-backend .
# Run:   docker run -p 3001:3001 retina-backend
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build
# -----------------------------------------------------------------------------
# Use the official Rust image for building
FROM rust:1.75-slim-bookworm AS builder

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a new empty project for caching dependencies
WORKDIR /app

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock* ./

# Create dummy source to build dependencies
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Now copy the actual source code
COPY src ./src

# Build the actual application
# Touch main.rs to ensure it's rebuilt
RUN touch src/main.rs && \
    cargo build --release

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
# Use minimal Debian image for runtime
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 retina

# Set working directory
WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/retina-backend /app/retina-backend

# Change ownership to non-root user
RUN chown -R retina:retina /app

# Switch to non-root user
USER retina

# Expose the API port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3001/health || exit 1

# Run the backend
CMD ["./retina-backend"]
