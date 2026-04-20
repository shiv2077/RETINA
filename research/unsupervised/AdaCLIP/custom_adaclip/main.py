#!/usr/bin/env python3
"""
AdaCLIP Zero-Shot Evaluation Suite
Main entry point for evaluation and inference modes
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import (
    load_config, setup_device, setup_paths, 
    create_results_folder, save_config_backup, Timer
)
from src.model_loader import setup_adaclip_imports, load_adaclip_model, get_prompts
from src.data_loader import load_evaluation_dataset, load_inference_dataset
from src.evaluator import run_evaluation, calculate_metrics, create_error_analysis, print_results
from src.inference import run_inference, analyze_inference_results, save_detected_anomalies, print_inference_results
from src.visualizer import create_all_visualizations, save_error_analysis

def main():
    """Main function"""
    print("🚀 AdaCLIP Zero-Shot Evaluation Suite")
    print("="*50)
    
    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / "config" / "config_decospan.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config = load_config(config_path)
    mode = config['mode']
    print(f"Mode: {mode}")
    
    # Setup
    device = setup_device()
    paths = setup_paths(config)
    results_folder, timestamp = create_results_folder(script_dir)
    save_config_backup(config, results_folder, timestamp)
    
    # Setup AdaCLIP imports and load model
    setup_adaclip_imports(paths['adaclip_repo'])
    model = load_adaclip_model(config, device)
    normal_prompt, anomaly_prompt = get_prompts(config)
    
    # Initialize timer
    timer = Timer()
    
    try:
        if mode == "evaluation":
            run_evaluation_mode(config, model, normal_prompt, anomaly_prompt, 
                              device, timer, results_folder)
        elif mode == "inference":
            run_inference_mode(config, model, normal_prompt, anomaly_prompt, 
                             device, timer, results_folder)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'evaluation' or 'inference'")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    
    print("✅ Complete!")

def run_evaluation_mode(config, model, normal_prompt, anomaly_prompt, device, timer, results_folder):
    """Run evaluation mode with labeled data"""
    print("\n📊 EVALUATION MODE")
    print("-" * 30)
    
    # Get data settings
    data_config = config.get('data', {})
    max_samples = data_config.get('max_samples', None)
    batch_size = data_config.get('batch_size', 8)
    
    # Load dataset
    eval_data = load_evaluation_dataset(config['paths']['dataset'], max_samples)
    
    # Run evaluation
    results = run_evaluation(model, eval_data, normal_prompt, anomaly_prompt, device, timer, batch_size)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Create error analysis
    false_positives, false_negatives = create_error_analysis(results, metrics)
    
    # Print results
    print_results(results, metrics, timer)
    
    # Create visualizations if enabled
    if config['output']['create_visualizations']:
        print("\n🎨 Creating visualizations...")
        create_all_visualizations(results, metrics, model, anomaly_prompt, results_folder)
    
    # Save error folders if enabled
    if config['output']['create_error_folders']:
        print("\n📁 Saving error analysis...")
        save_error_analysis(false_positives, false_negatives, results_folder)
    
    print(f"\n📁 All results saved to: {results_folder}")

def run_inference_mode(config, model, normal_prompt, anomaly_prompt, device, timer, results_folder):
    """Run inference mode on unlabeled data"""
    print("\n🔍 INFERENCE MODE")
    print("-" * 30)
    
    # Get data settings
    data_config = config.get('data', {})
    max_samples = data_config.get('max_samples', None)
    
    # Load dataset
    inference_data = load_inference_dataset(config['paths']['dataset'], max_samples)
    
    # Run inference
    threshold = config['inference']['threshold']
    results = run_inference(model, inference_data, normal_prompt, anomaly_prompt, 
                          device, threshold, timer)
    
    # Analyze results (pass threshold!)
    analysis = analyze_inference_results(results, threshold)
    
    # Print results
    print_inference_results(analysis, timer)
    
    # Save detected anomalies if enabled
    if config['output']['save_detected_anomalies'] and analysis['detected_anomalies']:
        print("\n💾 Saving detected anomalies...")
        save_detected_anomalies(analysis['detected_anomalies'], results_folder)
    
    print(f"\n📁 All results saved to: {results_folder}")

if __name__ == "__main__":
    main()