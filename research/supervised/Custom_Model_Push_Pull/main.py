#!/usr/bin/env python3
"""
Advanced Push-Pull Wood Defect Detection - Main Script
=====================================================

Features:
- Support for both simple and advanced push-pull models
- Attention-based learning for better boundary detection
- Multiple normal centers for different wood types
- Comprehensive evaluation and visualization

Usage:
    python main.py --config config/config.yaml --model advanced
    python main.py --config config/config.yaml --model simple
"""

import argparse
import torch
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

try:
    from utils import load_config, setup_logging, save_results
    from data import load_expert_samples, load_all_test_samples, create_dataloaders, save_dataset_info
    from training import AdvancedPushPullTrainer
    
    # Try to import visuals
    try:
        from visuals import create_all_visualizations
        VISUALS_AVAILABLE = True
    except ImportError:
        VISUALS_AVAILABLE = False
        print("⚠️  Visuals module not available")
    
    # Optional advanced visualization imports
    try:
        from utils import create_comprehensive_visualizations, create_attention_visualizations
        VISUALIZATIONS_AVAILABLE = True
    except ImportError:
        VISUALIZATIONS_AVAILABLE = False
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all src files are in place")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Advanced Push-Pull Wood Defect Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['simple', 'advanced'],
        help='Override model type from config'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create comprehensive visualizations'
    )
    
    parser.add_argument(
        '--attention-vis',
        action='store_true',
        help='Create attention visualizations (advanced model only)'
    )
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the appropriate device"""
    
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔧 Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("🔧 Using CPU")
    
    return device


def set_seed(seed):
    """Set random seeds for reproducibility"""
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to: {seed}")


def main():
    """Main execution function"""
    
    print("🎯 ADVANCED PUSH-PULL WOOD DEFECT DETECTION")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"📋 Configuration loaded from: {args.config}")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Override config with command line arguments
    if args.model:
        config['training']['model_type'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Setup device and seed
    device = setup_device(args.device)
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(config['paths']['outputs'], level=getattr(logging, config.get('logging', {}).get('level', 'INFO')))
    
    # Log experiment info
    experiment = config.get('experiment', {})
    logger.info(f"🧪 Experiment: {experiment.get('name', 'unnamed')}")
    logger.info(f"📝 Description: {experiment.get('description', 'No description')}")
    logger.info(f"🤖 Model Type: {config['training']['model_type']}")
    logger.info(f"📊 Config: {args.config}")
    
    try:
        # Step 1: Load and prepare data
        logger.info("📂 Loading dataset...")
        
        expert_config = config['data']['expert_samples']
        train_samples, expert_paths = load_expert_samples(
            config['paths']['dataset'],
            n_normal=expert_config['n_normal'],
            n_anomaly=expert_config['n_anomaly'],
            seed=expert_config['seed']
        )
        
        test_samples = load_all_test_samples(
            config['paths']['dataset'],
            expert_paths
        )
        
        # Save dataset info
        dataset_info_path = Path(config['paths']['outputs']) / 'dataset_info.json'
        save_dataset_info(train_samples, test_samples, dataset_info_path)
        logger.info(f"📊 Dataset info saved: {dataset_info_path}")
        
        # Step 2: Create data loaders
        logger.info("📦 Creating data loaders...")
        train_loader, test_loader = create_dataloaders(train_samples, test_samples, config)
        
        # Step 3: Initialize trainer
        logger.info("🏗️  Initializing trainer...")
        trainer = AdvancedPushPullTrainer(config, device)
        
        # Step 4: Train model
        logger.info("🚀 Starting training...")
        results = trainer.train(train_loader, test_loader)
        
        # Step 5: Save results
        logger.info("💾 Saving results...")
        save_results(results, config['paths']['outputs'], config)
        
        # Step 6: Create visualizations (from config or argument)
        create_visuals = args.visualize or config.get('visualization', {}).get('create_plots', False)
        
        if create_visuals and VISUALS_AVAILABLE:
            logger.info("🎨 Creating visualizations...")
            create_all_visualizations(trainer.model, test_loader, device, output_dir="visuals")
            logger.info("✅ Visualizations saved to visuals/")
        elif create_visuals:
            logger.info("⚠️  Visualizations requested but visuals module not available")
        
        # Step 7: Create attention visualizations (advanced model only)
        if args.attention_vis and config['training']['model_type'] == 'advanced' and VISUALIZATIONS_AVAILABLE:
            logger.info("🧠 Creating attention visualizations...")
            attention_path = create_attention_visualizations(
                trainer.model,
                test_loader,
                config['paths']['outputs'],
                num_samples=16
            )
            logger.info(f"🎨 Attention visualizations saved: {attention_path}")
        elif args.attention_vis:
            logger.info("⚠️  Attention visualizations only available for advanced model")
        
        # Final summary
        final_auc = results['final_results']['auc']
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"🏆 Final Test AUC: {final_auc:.4f}")
        
        if final_auc >= 0.90:
            logger.info("🌟 OUTSTANDING PERFORMANCE! Model is ready for deployment.")
        elif final_auc >= 0.80:
            logger.info("✅ EXCELLENT PERFORMANCE! Model works very well.")
        elif final_auc >= 0.70:
            logger.info("👍 GOOD PERFORMANCE! Model is promising.")
        else:
            logger.info("⚠️  PERFORMANCE NEEDS IMPROVEMENT. Consider tuning.")
        
        logger.info(f"📁 All outputs saved in: {config['paths']['outputs']}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)