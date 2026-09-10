"""
Configuration Management
========================

Loads configuration from environment variables with sensible defaults.
Uses pydantic-settings for validation and type coercion.
"""

import re
import secrets
import socket
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at worker/src/retina_worker/config.py, so the repo root is
# four levels up. Deriving the default from the module's own location rather
# than the process working directory is the point: the worker is launched
# from the repo root natively and from / in the container, and a CWD-relative
# default silently resolves to a different place in each.
_MODULE = Path(__file__).resolve()
# parents[3] is the repo root for the native layout
# (worker/src/retina_worker/config.py). The container installs the package at
# /app/retina_worker/, which has fewer parents than that — indexing blindly
# raised IndexError at import and killed the container on startup. The
# defaults below are unused there anyway, because compose sets both paths
# explicitly, but the module still has to import.
_REPO_ROOT = _MODULE.parents[3] if len(_MODULE.parents) > 3 else _MODULE.parent
DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints"
DEFAULT_IMAGE_ROOT = _REPO_ROOT / "data" / "images"

# Content addresses are hex digests; this also excludes "..", separators
# and anything else that could climb out of the image root.
_SAFE_IMAGE_ID = re.compile(r"[A-Za-z0-9_-]{4,128}")


def image_relpath(image_id: str) -> Path:
    """Storage path for an image, RELATIVE to whatever root resolves to.

    The submitter and the worker each join this onto their own root, so the
    layout below the root is byte-identical on both sides of a container
    boundary and neither process ever serializes an absolute path.

    Sharded on the first two characters of the (content-addressed) id: a
    flat directory would accumulate every image ever submitted, and this
    repo already has a documented O(n) directory-scan problem.
    """
    # Allowlist rather than blocklist. Rejecting only separators is not
    # enough: ".." carries neither, is exactly long enough to shard on, and
    # would resolve one level ABOVE the root.
    if not _SAFE_IMAGE_ID.fullmatch(image_id):
        raise ValueError(f"unusable image_id: {image_id!r}")
    return Path(image_id[:2]) / f"{image_id}.png"


CHECKPOINT_NAMING = "patchcore_{category}.ckpt"


def available_categories(checkpoint_dir: Path) -> list[str]:
    """Categories with a PatchCore checkpoint on disk.

    Lives here rather than on PatchCoreRegistry so the API can validate a
    caller-supplied product_class without importing torch and anomalib into
    a web process. The registry calls this too, so the two cannot disagree
    about what "trained" means.
    """
    if not checkpoint_dir.is_dir():
        return []
    return sorted(
        f.stem.removeprefix("patchcore_")
        for f in checkpoint_dir.glob("patchcore_*.ckpt")
    )


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

    # An entry pending longer than this is assumed to belong to a worker that
    # died mid-job, and is reclaimed by the next XAUTOCLAIM sweep.
    job_reclaim_idle_ms: int = 300_000

    # After this many deliveries an entry is dead-lettered rather than
    # reclaimed again — past this point it is poison, not bad luck.
    job_max_deliveries: int = 3

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

    # Upper edge of the Stage 2 band. Scores at or above this are ones
    # PatchCore is already confident about, so a VLM call adds nothing.
    # The lower edge is anomaly_threshold — anything below was never flagged.
    stage2_trigger_max: float = 0.9

    # Minimum uncertainty score to add sample to active learning pool
    # Samples with uncertainty > this value are candidates for labeling
    uncertainty_threshold: float = 0.3

    # -------------------------------------------------------------------------
    # Active Learning Configuration
    # -------------------------------------------------------------------------
    # Maximum samples to keep in the labeling pool
    al_pool_max_size: int = 100

    # A Stage 2 verdict at or above this confidence counts as resolved, and
    # the sample is kept out of the labeling pool entirely.
    stage2_resolved_confidence: float = 0.8

    # A score whose distance-from-boundary uncertainty reaches this is an
    # abstention: the pipeline ran but has no verdict worth acting on, so the
    # job terminates as NEEDS_REVIEW rather than COMPLETED. 0.5 corresponds to
    # scores in roughly [0.25, 0.75]. See DECISIONS.md 18.
    abstain_uncertainty: float = 0.5

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

    # Per-request timeout for a single OpenAI call, in seconds. The worker is
    # single-threaded (DECISIONS.md 12), so an untimed call blocks the whole
    # poll loop for as long as the SDK's own default allows — minutes.
    openai_timeout_s: float = 30.0

    # Ceiling on total wall time for one logical call including retries and
    # backoff. Bounds the worst case a single job can cost the loop, which
    # per-attempt timeouts alone do not.
    openai_total_deadline_s: float = 90.0

    # Consecutive VLM failures before the breaker opens. Low on purpose:
    # every job past the first is re-proving an outage already diagnosed,
    # at the full retry budget above. See DECISIONS.md 21.
    vlm_breaker_failure_threshold: int = 3

    # How long the breaker stays open before letting one probe through.
    vlm_breaker_cooldown_s: float = 60.0

    # -------------------------------------------------------------------------
    # PatchCore Configuration
    # -------------------------------------------------------------------------
    # Directory holding per-category patchcore_{category}.ckpt files.
    # Leave empty to use DEFAULT_CHECKPOINT_DIR (repo-root/checkpoints,
    # derived from this module's location, not the working directory).
    # Set an absolute path in containers, where the repo layout differs.
    patchcore_checkpoint_path: str = ""

    # Root of the shared image store. Both the submitting API and the worker
    # resolve this independently — natively it is the repo's data/images, in
    # a container it is the mount point set by RETINA_IMAGE_ROOT — and join
    # image_relpath() onto it. Same relative layout, different roots.
    image_root: str = Field(default="", validation_alias="RETINA_IMAGE_ROOT")

    def resolved_image_root(self) -> Path:
        """Absolute root of the image store. Same resolution order as
        resolved_checkpoint_dir: explicit config, then env, then an anchor
        derived from this module's location — never the working directory."""
        if self.image_root:
            return Path(self.image_root).expanduser().resolve()
        return DEFAULT_IMAGE_ROOT

    def image_path(self, image_id: str) -> Path:
        """Absolute on-disk location of an image for this process."""
        return self.resolved_image_root() / image_relpath(image_id)

    def resolved_checkpoint_dir(self) -> Path:
        """Absolute directory to load PatchCore checkpoints from.

        Always absolute, so a worker started from any working directory
        resolves the same checkpoints. An explicitly configured relative
        path is still honoured (resolved against the CWD) because that can
        only be a deliberate choice by whoever set the variable.
        """
        if self.patchcore_checkpoint_path:
            return Path(self.patchcore_checkpoint_path).expanduser().resolve()
        return DEFAULT_CHECKPOINT_DIR


def get_settings() -> Settings:
    """Get the application settings singleton."""
    return Settings()
