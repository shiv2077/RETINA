# Evaluator for AdaCLIP zero-shot evaluation
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix

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

def run_evaluation(model, eval_data, normal_prompt, anomaly_prompt, device, timer, batch_size=8):
    """Run zero-shot evaluation on labeled dataset"""
    print(f"🎯 Using prompts:")
    print(f"  Normal: '{normal_prompt}'")
    print(f"  Anomaly: '{anomaly_prompt}'")
    print(f"Running evaluation on {len(eval_data)} samples...")
    
    results = {
        'scores': [],
        'labels': [],
        'defect_types': [],
        'image_paths': []
    }
    
    failed_count = 0
    
    for image_path, label, defect_type in tqdm(eval_data, desc="Evaluating"):
        timer.start()
        score = evaluate_single_image(model, image_path, normal_prompt, anomaly_prompt, device)
        timer.stop()
        
        if score is not None:
            results['scores'].append(score)
            results['labels'].append(label)
            results['defect_types'].append(defect_type)
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
    
    return results

def calculate_metrics(results):
    """Calculate evaluation metrics"""
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])
    
    # Basic metrics
    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(labels, scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate confusion matrix with optimal threshold
    predictions = (scores >= optimal_threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    # Calculate additional metrics
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'auroc': auroc,
        'average_precision': ap,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'predictions': predictions
    }
    
    return metrics

def create_error_analysis(results, metrics):
    """Create false positive and false negative analysis"""
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])
    predictions = metrics['predictions']
    
    false_positives = []
    false_negatives = []
    
    for i, (score, label, defect_type, img_path) in enumerate(zip(
        scores, labels, results['defect_types'], results['image_paths']
    )):
        pred = predictions[i]
        
        if label == 0 and pred == 1:  # False Positive
            false_positives.append({
                'index': i,
                'path': img_path,
                'score': score,
                'defect_type': defect_type,
                'true_label': 'normal',
                'predicted': 'anomaly'
            })
        elif label == 1 and pred == 0:  # False Negative
            false_negatives.append({
                'index': i,
                'path': img_path,
                'score': score,
                'defect_type': defect_type,
                'true_label': 'anomaly', 
                'predicted': 'normal'
            })
    
    # Sort by score
    false_positives.sort(key=lambda x: x['score'], reverse=True)
    false_negatives.sort(key=lambda x: x['score'])
    
    return false_positives, false_negatives

def print_results(results, metrics, timer):
    """Print evaluation results"""
    normal_count = np.sum(np.array(results['labels']) == 0)
    anomaly_count = np.sum(np.array(results['labels']) == 1)
    
    print(f"\n📊 Evaluation Results:")
    print(f"Samples: {len(results['scores'])} (Normal: {normal_count}, Anomaly: {anomaly_count})")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    
    print(f"\n⏱️ Timing:")
    print(f"Total time: {timer.total_time_str()}")
    print(f"Average time per image: {timer.average_time():.3f}s")