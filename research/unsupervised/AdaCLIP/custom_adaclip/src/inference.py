# Inference for AdaCLIP zero-shot anomaly detection
import torch
import numpy as np
import shutil
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def evaluate_single_image(model, image_path, normal_prompt, anomaly_prompt, device):
    """Evaluate single image with AdaCLIP and normalize score to 0-1 range"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = model.preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get scores for both prompts
            normal_result = model.clip_model(image_tensor, [normal_prompt], aggregation=True)
            anomaly_result = model.clip_model(image_tensor, [anomaly_prompt], aggregation=True)
            
            # Extract scores
            _, normal_score = normal_result[:2]
            _, anomaly_score = anomaly_result[:2]
            
            normalized_score = anomaly_score.item()
            
        return normalized_score
        
    except Exception as e:
        print(f"Failed to evaluate {image_path}: {e}")
        return None

def run_inference(model, inference_data, normal_prompt, anomaly_prompt, device, threshold, timer):
    """Run zero-shot inference on unlabeled dataset"""
    print(f"🎯 Using prompts:")
    print(f"  Normal: '{normal_prompt}'")
    print(f"  Anomaly: '{anomaly_prompt}'")
    print(f"Running inference on {len(inference_data)} samples...")
    print(f"Using threshold: {threshold}")
    
    results = {
        'scores': [],
        'image_paths': []
    }
    
    failed_count = 0
    
    for (image_path,) in tqdm(inference_data, desc="Inference"):
        timer.start()
        score = evaluate_single_image(model, image_path, normal_prompt, anomaly_prompt, device)
        timer.stop()
        
        if score is not None:
            results['scores'].append(score)
            results['image_paths'].append(str(image_path))
        else:
            failed_count += 1
    
    if failed_count > 0:
        print(f"Failed to evaluate {failed_count} images")
    
    # Print score statistics
    scores = np.array(results['scores'])
    print(f"\n📈 Score Statistics (normalized 0-1):")
    print(f"Min: {scores.min():.4f}, Max: {scores.max():.4f}")
    print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
    print(f"Threshold: {threshold:.4f}")
    
    return results

def analyze_inference_results(results, threshold):
    """Analyze inference results with normalized scores"""
    scores = np.array(results['scores'])
    
    # Use provided threshold (should be 0.5 for sigmoid normalized scores)
    predictions = (scores >= threshold).astype(int)
    
    normal_count = np.sum(predictions == 0)
    anomaly_count = np.sum(predictions == 1)
    
    # Get detected anomalies sorted by score
    detected_anomalies = []
    for i, (score, pred, img_path) in enumerate(zip(
        scores, predictions, results['image_paths']
    )):
        if pred == 1:  # Detected as anomaly
            detected_anomalies.append({
                'index': i,
                'path': img_path,
                'score': score
            })
    
    # Sort by score (highest first)
    detected_anomalies.sort(key=lambda x: x['score'], reverse=True)
    
    analysis = {
        'total_images': len(scores),
        'normal_count': normal_count,
        'anomaly_count': anomaly_count,
        'anomaly_percentage': (anomaly_count / len(scores)) * 100,
        'detected_anomalies': detected_anomalies,
        'score_stats': {
            'min': float(scores.min()),
            'max': float(scores.max()),
            'mean': float(scores.mean()),
            'std': float(scores.std())
        }
    }
    
    return analysis

def save_detected_anomalies(detected_anomalies, results_folder):
    """Save ALL detected anomaly images to folder"""
    if not detected_anomalies:
        print("No anomalies detected to save")
        return None
        
    anomalies_folder = results_folder / "detected_anomalies"
    anomalies_folder.mkdir(exist_ok=True)
    
    # Save all anomalies
    save_count = len(detected_anomalies)
    
    # Create analysis file
    analysis_file = anomalies_folder / "analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write(f"=== DETECTED ANOMALIES ===\n\n")
        f.write(f"Total detected: {len(detected_anomalies)}\n")
        f.write(f"Saved: {save_count}\n\n")
        
        for i, anomaly in enumerate(detected_anomalies):
            source_path = Path(anomaly['path'])
            dest_path = anomalies_folder / source_path.name
            
            try:
                shutil.copy2(source_path, dest_path)
                f.write(f"{i+1:3d}. Score: {anomaly['score']:7.4f} | Image: {source_path.name}\n")
            except Exception as e:
                f.write(f"{i+1:3d}. Score: {anomaly['score']:7.4f} | COPY FAILED: {e} | {source_path.name}\n")
    
    print(f"Saved all {save_count} detected anomalies to: {anomalies_folder}")
    return anomalies_folder

def print_inference_results(analysis, timer):
    """Print inference results"""
    print(f"\n📊 Inference Results:")
    print(f"Total images: {analysis['total_images']}")
    print(f"Normal: {analysis['normal_count']}")
    print(f"Anomalies detected: {analysis['anomaly_count']} ({analysis['anomaly_percentage']:.1f}%)")
    
    if analysis['detected_anomalies']:
        top_5 = analysis['detected_anomalies'][:5]
        print(f"\n🔍 Top 5 Anomalies:")
        for i, anomaly in enumerate(top_5, 1):
            filename = Path(anomaly['path']).name
            print(f"{i}. {filename} (Score: {anomaly['score']:.4f})")
    
    print(f"\n⏱️ Timing:")
    print(f"Total time: {timer.total_time_str()}")
    print(f"Average time per image: {timer.average_time():.3f}s")