# =============================================================================
# RETINA ML Worker - Dockerfile
# =============================================================================
# Python-based ML worker for running anomaly detection inference.
#
# Build: docker build -t retina-worker .
# Run:   docker run retina-worker
#
# NOTE: For production with actual ML models, extend this Dockerfile to include
# PyTorch, anomalib, and GPU support. See the "GPU Support" section below.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency specification
COPY pyproject.toml ./

# Install dependencies
# Note: Using pip install with pyproject.toml
RUN pip install --upgrade pip && \
    pip install .

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 retina

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY src/retina_worker ./retina_worker

# Change ownership to non-root user
RUN chown -R retina:retina /app

# Switch to non-root user
USER retina

# Health check - verifies the worker can import successfully
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "from retina_worker.worker import main; print('ok')" || exit 1

# Run the worker
CMD ["python", "-m", "retina_worker.main"]

# =============================================================================
# GPU Support (Future)
# =============================================================================
# For GPU-enabled inference with PyTorch and CUDA, use the following base image
# and install additional dependencies:
#
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS gpu-runtime
#
# And install anomalib:
# RUN pip install anomalib torch torchvision
#
# The Kubernetes deployment should request GPU resources:
# resources:
#   limits:
#     nvidia.com/gpu: 1
# =============================================================================
