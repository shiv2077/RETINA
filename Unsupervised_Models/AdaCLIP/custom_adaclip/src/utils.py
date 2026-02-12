# Utils for AdaCLIP evaluation
import yaml
import os
import time
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device():
    """Setup CUDA device"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    return device

def setup_paths(config):
    """Setup and validate all paths"""
    paths = config['paths']
    
    # Check if paths exist
    dataset_path = Path(paths['dataset'])
    adaclip_repo = Path(paths['adaclip_repo'])
    weights_path = Path(paths['weights'])
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not adaclip_repo.exists():
        raise FileNotFoundError(f"AdaCLIP repo not found: {adaclip_repo}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    # Setup cache directory
    cache_dir = Path(paths['cache_dir'])
    cache_dir.mkdir(exist_ok=True)
    os.environ['TORCH_HOME'] = str(cache_dir)
    
    print(f"Dataset: {dataset_path}")
    print(f"Weights: {weights_path}")
    print(f"Cache: {cache_dir}")
    
    return paths

def create_results_folder(script_dir=None):
    """Create versioned results folder"""
    if script_dir is None:
        script_dir = Path(".")
    
    results_base = script_dir / "results"
    results_base.mkdir(exist_ok=True)
    
    # Find next version number
    existing_versions = [d.name for d in results_base.iterdir() if d.is_dir() and d.name.startswith('v')]
    if existing_versions:
        version_numbers = [int(v[1:]) for v in existing_versions if v[1:].isdigit()]
        next_version = max(version_numbers) + 1 if version_numbers else 0
    else:
        next_version = 0
    
    version_folder = results_base / f"v{next_version}"
    version_folder.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Results folder: {version_folder}")
    print(f"Timestamp: {timestamp}")
    
    return version_folder, timestamp

def save_config_backup(config, results_folder, timestamp):
    """Save config backup to results folder"""
    config_backup = results_folder / f"config_backup_{timestamp}.yaml"
    with open(config_backup, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config backup: {config_backup}")

class Timer:
    """Simple timer for measuring execution time"""
    def __init__(self):
        self.start_time = None
        self.total_time = 0
        self.count = 0
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.total_time += elapsed
            self.count += 1
            return elapsed
        return 0
    
    def average_time(self):
        return self.total_time / self.count if self.count > 0 else 0
    
    def total_time_str(self):
        minutes = int(self.total_time // 60)
        seconds = int(self.total_time % 60)
        return f"{minutes}m {seconds}s"