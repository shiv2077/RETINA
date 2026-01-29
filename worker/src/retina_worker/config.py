"""
Configuration Management
========================

Loads configuration from environment variables with sensible defaults.
Uses pydantic-settings for validation and type coercion.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
        Unique identifier for this worker in the consumer group.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
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
    consumer_name: str = "worker-1"
    
    # -------------------------------------------------------------------------
    # Development/Debug
    # -------------------------------------------------------------------------
    debug_mode: bool = True
    mock_inference_delay_ms: int = 500
    
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


def get_settings() -> Settings:
    """Get the application settings singleton."""
    return Settings()
