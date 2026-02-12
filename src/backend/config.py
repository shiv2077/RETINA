"""
RETINA Configuration Module
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "weights"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
UPLOADS_DIR = DATA_DIR / "uploads"
EXPORTS_DIR = DATA_DIR / "exports"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, ANNOTATIONS_DIR, UPLOADS_DIR, EXPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class PatchCoreConfig:
    """PatchCore model configuration."""
    backbone: str = "wide_resnet50_2"
    layers: Tuple[str, ...] = ("layer2", "layer3")
    img_size: int = 224
    coreset_ratio: float = 0.01
    num_neighbors: int = 9
    anomaly_threshold: float = 0.5


@dataclass
class BGADConfig:
    """BGAD (Boundary-Guided Anomaly Detection) configuration."""
    backbone: str = "resnet18"
    img_size: int = 224
    feature_dim: int = 256
    center_lr: float = 0.01
    margin: float = 1.0
    pull_weight: float = 1.0
    push_weight: float = 0.1


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:3001"])


@dataclass
class Config:
    """Main configuration."""
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    bgad: BGADConfig = field(default_factory=BGADConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    
    mvtec_path: str = str(BASE_DIR / "mvtec_anomaly_detection")
    device: str = "cuda"
    batch_size: int = 8
    num_workers: int = 4


config = Config()
