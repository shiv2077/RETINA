"""
CLEAN Visualization Module - Just Works Edition
===============================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.decomposition import PCA
from pathlib import Path
import logging
from tqdm import tqdm

# Setup plotting
plt.style.use('default')
sns.set_palette("husl")


def create_all_visualizations(model, test_loader, device, output_dir="visuals"):
    """Create ALL visualizations - CLEAN version"""
    
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"🎨 Creating visualizations in {output_dir}")
    
    # Extract ALL data we need
    features, scores, labels, categories = extract_test_data(model, test_loader, device)
    
    # Create plots
    logger.info("📊 Creating score distribution...")
    create_score_distribution(scores, labels, output_dir)
    
    logger.info("🔍 Creating confusion matrix...")
    create_clean_confusion_matrix(scores, labels, output_dir)
    
    logger.info("📈 Creating ROC curve...")
    create_roc_curve(scores, labels, output_dir)
    
    logger.info("🎯 Creating circular push-pull...")
    create_circular_visualization(features, labels, output_dir)
    
    logger.info("📊 Creating comprehensive dashboard...")
    create_dashboard(features, scores, labels, categories, output_dir)
    
    logger.info(f"✅ All visualizations saved to {output_dir}")


def extract_test_data(model, test_loader, device):
    """Extract features, scores, labels from test set"""
    
    model.eval()
    all_features = []
    all_scores = []
    all_labels = []
    all_categories = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Extracting', leave=False, ncols=60):
            images = batch['image'].to(device)
            labels = batch['label']
            categories = batch['category']
            
            # Extract features and scores
            features = model(images)
            scores = model.compute_anomaly_scores(features)
            
            all_features.append(features.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_categories.extend(categories)
    
    features = np.vstack(all_features)
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    
    return features, scores, labels, all_categories


def create_score_distribution(scores, labels, output_dir):
    """Create clean score distribution plot"""
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    # Find threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else np.median(scores)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='green', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
    ax.axvline(optimal_threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold: {optimal_threshold:.3f}')
    
    ax.set_xlabel('Anomaly Score (Distance from Normal Center)')
    ax.set_ylabel('Density')
    ax.set_title('Push-Pull Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_clean_confusion_matrix(scores, labels, output_dir):
    """Create CLEAN confusion matrix - only counts"""
    
    # Find threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else np.median(scores)
    
    predictions = (scores > optimal_threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Only show counts - no normalized version
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'shrink': 0.8},
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    ax.set_title(f'Confusion Matrix (Threshold: {optimal_threshold:.3f})')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Calculate and show metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
    
    metrics_text = f'''Metrics:
Accuracy:  {accuracy:.3f}
Precision: {precision_val:.3f}
Recall:    {recall_val:.3f}
F1-Score:  {f1:.3f}'''
    
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_roc_curve(scores, labels, output_dir):
    """Create ROC curve"""
    
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Push-Pull Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_circular_visualization(features, labels, output_dir):
    """Create REAL circular visualization using actual model predictions"""
    
    # Calculate ACTUAL distances in feature space
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    # Use actual normal center from feature space
    normal_features = features[normal_mask]
    center_features = normal_features.mean(axis=0)
    
    # Calculate real distances from center
    all_distances = np.linalg.norm(features - center_features, axis=1)
    normal_distances = all_distances[normal_mask]
    anomaly_distances = all_distances[anomaly_mask]
    
    # Decision boundary (95th percentile of normal distances)
    radius = np.percentile(normal_distances, 95)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: REAL data visualization using polar coordinates
    ax1 = axes[0]
    
    # Sample data for cleaner visualization
    n_normal_show = min(800, len(normal_distances))
    n_anomaly_show = min(300, len(anomaly_distances))
    
    # Use REAL distances but convert to polar for circular display
    sampled_normal_idx = np.random.choice(len(normal_distances), n_normal_show, replace=False)
    sampled_anomaly_idx = np.random.choice(len(anomaly_distances), n_anomaly_show, replace=False)
    
    # Real normal distances and random angles
    real_normal_distances = normal_distances[sampled_normal_idx]
    normal_angles = np.random.uniform(0, 2*np.pi, n_normal_show)
    normal_x = real_normal_distances * np.cos(normal_angles)
    normal_y = real_normal_distances * np.sin(normal_angles)
    
    # Real anomaly distances and random angles  
    real_anomaly_distances = anomaly_distances[sampled_anomaly_idx]
    anomaly_angles = np.random.uniform(0, 2*np.pi, n_anomaly_show)
    anomaly_x = real_anomaly_distances * np.cos(anomaly_angles)
    anomaly_y = real_anomaly_distances * np.sin(anomaly_angles)
    
    # Plot with REAL distances
    ax1.scatter(normal_x, normal_y, c='lightgreen', alpha=0.6, s=25, 
               label=f'Normal', edgecolors='darkgreen', linewidth=0.5)
    ax1.scatter(anomaly_x, anomaly_y, c='lightcoral', alpha=0.8, s=35, 
               label=f'Anomaly', edgecolors='darkred', linewidth=0.5)
    
    # Center point
    ax1.scatter(0, 0, c='blue', s=150, marker='*', 
               edgecolors='black', linewidth=2, label='Normal Center', zorder=10)
    
    # Decision circle
    circle = plt.Circle((0, 0), radius, fill=False, color='black', 
                       linestyle='--', linewidth=2, label=f'Decision Boundary')
    ax1.add_patch(circle)
    
    ax1.set_title('Push-Pull Results (Real Distances)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distance * cos(angle)')
    ax1.set_ylabel('Distance * sin(angle)') 
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Calculate actual performance metrics for this visualization
    normal_correct = (real_normal_distances <= radius).sum()
    anomaly_correct = (real_anomaly_distances > radius).sum()
    total_correct = normal_correct + anomaly_correct
    total_samples = len(real_normal_distances) + len(real_anomaly_distances)
    accuracy = total_correct / total_samples
    
    # Right plot: Distance histogram showing real performance
    ax2 = axes[1]
    
    ax2.hist(normal_distances, bins=50, alpha=0.7, label=f'Normal (n={len(normal_distances)})', 
             color='green', density=True)
    ax2.hist(anomaly_distances, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_distances)})', 
             color='red', density=True)
    ax2.axvline(radius, color='black', linestyle='--', linewidth=2, label=f'Threshold={radius:.2f}')
    
    ax2.set_xlabel('Distance from Normal Center')
    ax2.set_ylabel('Density')
    ax2.set_title('Distance Distribution (All Data)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'circular_push_pull.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_dashboard(features, scores, labels, categories, output_dir):
    """Create comprehensive dashboard"""
    
    auc = roc_auc_score(labels, scores)
    
    # Find threshold
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    #optimal_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else np.median(scores)
    optimal_threshold = 0.5
    predictions = (scores > optimal_threshold).astype(int)
    accuracy = (predictions == labels).mean()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Push-Pull Wood Defect Detection Dashboard (AUC: {auc:.3f})', 
                 fontsize=16, fontweight='bold')
    
    # 1. Score distribution
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    axes[0, 0].hist(normal_scores, bins=40, alpha=0.7, label='Normal', color='green', density=True)
    axes[0, 0].hist(anomaly_scores, bins=40, alpha=0.7, label='Anomaly', color='red', density=True)
    axes[0, 0].axvline(optimal_threshold, color='black', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Score Distribution')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    axes[0, 1].plot(fpr, tpr, linewidth=3, label=f'AUC = {auc:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    axes[1, 0].set_title('Confusion Matrix')
    
    # 4. Summary
    axes[1, 1].axis('off')
    
    summary_text = f"""PUSH-PULL RESULTS
    
Performance:
• AUC: {auc:.3f}
• Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)
• Threshold: {optimal_threshold:.3f}

Dataset:
• Total Samples: {len(labels):,}
• Normal: {(labels==0).sum():,}
• Anomaly: {(labels==1).sum():,}
• Training: 300 expert samples

Status: {'Excellent' if auc >= 0.85 else 'Very Good' if auc >= 0.75 else 'Good'} Performance!

Method: Simple Push-Pull
• Normal samples pulled to center
• Anomaly samples pushed away
• Distance = Anomaly score"""
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()