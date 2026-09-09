"""
Configuration Management
========================

Loads configuration from environment variables with sensible defaults.
Uses pydantic-settings for validation and type coercion.
"""

import secrets
import socket

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_consumer_name() -> str:
    """One consumer identity per worker process.

    ``docker compose up --scale worker=4`` gives every replica its own
    container hostname, so the hostname alone separates the four consumers
    in the ``workers`` group. The random suffix only covers the degenerate
    case where gethostname() is empty or fails.
    """
    try:
        host = socket.gethostname()
    except OSError:
        host = ""
    return host or f"worker-{secrets.token_hex(4)}"


class Settings(BaseSettings):
    """
    Worker configuration loaded from environment variables.
    
    All settings can be overridden via environment variables.
    The prefix 'RETINA_' is not used to maintain compatibility
    with the docker-compose configuration.
    
    Attributes
    ----------
    redis_url : str
        Redis connection URL for job queue and result storage.
    worker_concurrency : int
        Number of concurrent jobs to process (currently single-threaded).
    default_unsupervised_model : str
        Default model for Stage 1 inference.
    debug_mode : bool
        Enable debug logging and mock delays.
    mock_inference_delay_ms : int
        Simulated inference delay in milliseconds (for testing).
    anomaly_threshold : float
        Threshold for binary anomaly classification.
    uncertainty_threshold : float
        Minimum uncertainty to add sample to active learning pool.
    consumer_name : str
        Unique identifier for this worker in the consumer group. Defaults to
        the container hostname; override with WORKER_CONSUMER_NAME.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
    
    # -------------------------------------------------------------------------
    # Redis Configuration
    # -------------------------------------------------------------------------
    redis_url: str = "redis://localhost:6379"
    
    # -------------------------------------------------------------------------
    # Worker Configuration
    # -------------------------------------------------------------------------
    worker_concurrency: int = 1
    default_unsupervised_model: str = "patchcore"
    # Consumer identity inside the `workers` group. Must differ per replica
    # or every replica competes for the same pending-entries list.
    # Override with WORKER_CONSUMER_NAME.
    consumer_name: str = Field(
        default_factory=_default_consumer_name,
        validation_alias="WORKER_CONSUMER_NAME",
    )
    
    # -------------------------------------------------------------------------
    # Development/Debug
    # -------------------------------------------------------------------------
    debug_mode: bool = False
    mock_inference_delay_ms: int = 0
    
    # -------------------------------------------------------------------------
    # Model Configuration
    # -------------------------------------------------------------------------
    # Threshold for binary anomaly classification (score > threshold = anomaly)
    anomaly_threshold: float = 0.5
    
    # Minimum uncertainty score to add sample to active learning pool
    # Samples with uncertainty > this value are candidates for labeling
    uncertainty_threshold: float = 0.3
    
    # -------------------------------------------------------------------------
    # Active Learning Configuration
    # -------------------------------------------------------------------------
    # Maximum samples to keep in the labeling pool
    al_pool_max_size: int = 100

    # -------------------------------------------------------------------------
    # GPT-4V / OpenAI Configuration
    # -------------------------------------------------------------------------
    # OpenAI API key for GPT-4o vision inference (Stage 1 VLM detector)
    openai_api_key: str = ""

    # Product type description passed to the GPT-4o inspection prompt
    # Change per deployment: "PCB board", "wood panel", "car door", etc.
    gpt4v_product_type: str = "manufactured product"

    # Retry attempts on rate-limit or timeout errors
    gpt4v_max_retries: int = 3

    # -------------------------------------------------------------------------
    # PatchCore Configuration
    # -------------------------------------------------------------------------
    # Directory where the memory bank checkpoint is stored.
    # Leave empty to start without a checkpoint (call train() to build it).
    patchcore_checkpoint_path: str = ""


def get_settings() -> Settings:
    """Get the application settings singleton."""
    return Settings()
