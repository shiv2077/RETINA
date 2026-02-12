"""
Utilities for Advanced Push-Pull Wood Defect Detection
=====================================================

Includes:
- Configuration management
- Logging setup
- Visualization utilities
- Results saving and analysis
"""

import yaml
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import torch


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        config: Configuration dictionary
    """
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure all paths exist
    for path_key in ['outputs', 'models', 'data']:
        if path_key in config.get('paths', {}):
            Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)
    
    return config


def setup_logging(log_dir, level=logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        logger: Configured logger
    """
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"push_pull_training_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('push_pull')
    logger.info(f"📝 Logging initialized. Log file: {log_file}")
    
    return logger


def save_results(results, output_path, config=None):
    """
    Save comprehensive results to files
    
    Args:
        results: Results dictionary
        output_path: Output directory path
        config: Optional configuration for additional context
    """
    
    output_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main results as JSON
    results_file = output_path / f"results_{timestamp}.json"
    
    # Make results JSON serializable
    serializable_results = make_json_serializable(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Create human-readable summary
    summary_file = output_path / f"summary_{timestamp}.txt"
    create_summary_report(results, summary_file, config)
    
    logging.getLogger('push_pull').info(f"📊 Results saved: {results_file}")
    logging.getLogger('push_pull').info(f"📋 Summary saved: {summary_file}")


def create_summary_report(results, output_file, config=None):
    """Create human-readable summary report"""
    
    with open(output_file, 'w') as f:
        f.write("🎯 PUSH-PULL WOOD DEFECT DETECTION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        # Timestamp
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Information
        if config:
            f.write("🏗️  MODEL CONFIGURATION:\n")
            f.write(f"   Model Type: {config.get('training', {}).get('model_type', 'simple')}\n")
            f.write(f"   Backbone: {config.get('model', {}).get('backbone', 'resnet18')}\n")
            f.write(f"   Feature Dim: {config.get('model', {}).get('feature_dim', 256)}\n")
            f.write(f"   Epochs: {config.get('training', {}).get('epochs', 30)}\n")
            f.write(f"   Learning Rate: {config.get('training', {}).get('learning_rate', 0.001)}\n\n")
        
        # Training Results
        f.write("📈 TRAINING RESULTS:\n")
        final_results = results.get('final_results', {})
        f.write(f"   Final AUC: {final_results.get('auc', 0):.4f}\n")
        f.write(f"   Total Epochs: {results.get('total_epochs', 0)}\n")
        f.write(f"   Training Time: {results.get('training_time', 0):.1f}s\n\n")
        
        # Final Test Results
        f.write("🧪 FINAL TEST RESULTS:\n")
        f.write(f"   AUC: {final_results.get('auc', 0):.4f}\n")
        f.write(f"   Accuracy: {final_results.get('accuracy', 0):.4f}\n")
        f.write(f"   Precision: {final_results.get('precision', 0):.4f}\n")
        f.write(f"   Recall: {final_results.get('recall', 0):.4f}\n")
        f.write(f"   F1-Score: {final_results.get('f1', 0):.4f}\n")
        f.write(f"   Test Samples: {final_results.get('num_samples', 0)}\n\n")
        
        # Assessment
        auc = final_results.get('auc', 0)
        f.write("📊 PERFORMANCE ASSESSMENT:\n")
        if auc >= 0.85:
            f.write("   🎉 EXCELLENT! Outstanding performance!\n")
        elif auc >= 0.75:
            f.write("   ✅ VERY GOOD! Strong performance!\n")
        elif auc >= 0.65:
            f.write("   👍 GOOD! Decent performance!\n")
        else:
            f.write("   ⚠️  FAIR! Room for improvement!\n")


def make_json_serializable(obj):
    """Convert object to JSON serializable format"""
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return