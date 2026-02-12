# Visualizer for AdaCLIP evaluation results
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2
import shutil
from PIL import Image
from pathlib import Path
from sklearn.metrics import roc_curve
from scipy.ndimage import gaussian_filter
import torch 

def get_adaclip_heatmap(model, image_path, text_prompt):
    """Get heatmap from AdaCLIP"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        np_image = np.array(image)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        np_image = cv2.resize(np_image, (518, 518))
        
        # Preprocess for model
        img_input = model.preprocess(image).unsqueeze(0)
        img_input = img_input.to(model.device)
        
        with torch.no_grad():
            anomaly_map, anomaly_score = model.clip_model(img_input, [text_prompt], aggregation=True)
        
        # Process anomaly map
        anomaly_map = anomaly_map[0, :, :].cpu().numpy()
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        anomaly_map = (anomaly_map * 255).astype(np.uint8)
        
        # Resize to original image size
        orig_size = image.size
        if anomaly_map.shape != (orig_size[1], orig_size[0]):
            anomaly_map = cv2.resize(anomaly_map, orig_size, interpolation=cv2.INTER_CUBIC)
        
        return anomaly_map, anomaly_score
        
    except Exception as e:
        print(f"Heatmap generation failed for {image_path}: {e}")
        return None, None

def create_heatmap_overlay(image, heatmap, global_min, global_max, alpha=0.6):
    """Create heatmap overlay with global color scale"""
    if heatmap is None:
        return image
    
    # Convert image to numpy
    if isinstance(image, Image.Image):
        np_image = np.array(image)
    else:
        np_image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    
    # Global normalization
    if global_max > global_min:
        heatmap_norm = np.clip((heatmap - global_min) / (global_max - global_min), 0, 1)
    else:
        heatmap_norm = np.ones_like(heatmap) * 0.5
    
    heatmap_norm = (heatmap_norm * 255).astype(np.uint8)
    heat_map = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Convert RGB to BGR for OpenCV
    if len(np_image.shape) == 3:
        np_image_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    else:
        np_image_bgr = np_image
    
    # Blend
    vis_map = cv2.addWeighted(heat_map, alpha, np_image_bgr, 1-alpha, 0)
    vis_map_rgb = cv2.cvtColor(vis_map, cv2.COLOR_BGR2RGB)
    
    return vis_map_rgb

def calculate_global_heatmap_scale(model, results, metrics, anomaly_prompt, sample_size=150):
    """Calculate global min/max for consistent heatmap scaling"""
    print("Calculating global heatmap scale...")
    
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])
    predictions = metrics['predictions']
    
    # Get indices for each category
    tn_idx = [i for i in range(len(scores)) if labels[i] == 0 and predictions[i] == 0]
    tp_idx = [i for i in range(len(scores)) if labels[i] == 1 and predictions[i] == 1]
    fp_idx = [i for i in range(len(scores)) if labels[i] == 0 and predictions[i] == 1]
    fn_idx = [i for i in range(len(scores)) if labels[i] == 1 and predictions[i] == 0]
    
    categories = [
        ("TN", tn_idx[:sample_size]),
        ("TP", tp_idx[:sample_size]),
        ("FP", fp_idx[:min(sample_size, len(fp_idx))]),
        ("FN", fn_idx[:min(sample_size, len(fn_idx))])
    ]
    
    all_heatmaps = []
    failed_count = 0
    
    for cat_name, indices in categories:
        if len(indices) > 0:
            print(f"  Processing {cat_name}: {len(indices)} samples")
            for idx in indices:
                img_path = results['image_paths'][idx]
                heatmap, _ = get_adaclip_heatmap(model, img_path, anomaly_prompt)
                if heatmap is not None:
                    all_heatmaps.append(heatmap)
                else:
                    failed_count += 1
    
    if failed_count > 0:
        print(f"  Failed to generate {failed_count} heatmaps")
    
    if len(all_heatmaps) > 0:
        all_values = np.concatenate([h.flatten() for h in all_heatmaps])
        global_min = float(all_values.min())
        global_max = float(all_values.max())
        print(f"  Global range: {global_min:.1f} to {global_max:.1f}")
        return global_min, global_max
    else:
        print("  No heatmaps generated - using default range")
        return 0, 255

def create_performance_plot(results, metrics, results_folder):
    """Create performance analysis plot"""
    
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])
    predictions = metrics['predictions']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    axes[0,0].plot(fpr, tpr, linewidth=2, label=f'AUROC = {metrics["auroc"]:.4f}')
    axes[0,0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0,0].set_xlabel('False Positive Rate')
    axes[0,0].set_ylabel('True Positive Rate')
    axes[0,0].set_title('ROC Curve')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Score Distribution
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    axes[0,1].hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', color='green', density=True)
    axes[0,1].hist(anomaly_scores, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_scores)})', color='red', density=True)
    axes[0,1].axvline(metrics['optimal_threshold'], color='black', linestyle='--', label=f'Threshold: {metrics["optimal_threshold"]:.3f}')
    axes[0,1].set_xlabel('Score')
    axes[0,1].set_ylabel('Density')
    axes[0,1].set_title('Score Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = metrics['confusion_matrix']
    im = axes[0,2].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0,2].set_title(f'Confusion Matrix')
    axes[0,2].set_xticks([0, 1])
    axes[0,2].set_yticks([0, 1])
    axes[0,2].set_xticklabels(['Normal', 'Anomaly'])
    axes[0,2].set_yticklabels(['Normal', 'Anomaly'])
    
    for i in range(2):
        for j in range(2):
            axes[0,2].text(j, i, f'{cm[i, j]}', ha="center", va="center", color="black", fontsize=14)
    
    # Per-Defect Performance
    defect_performance = {}
    for defect_type in set(results['defect_types']):
        if defect_type == 'normal':
            continue
        defect_indices = [i for i, dt in enumerate(results['defect_types']) if dt == defect_type]
        if len(defect_indices) > 0:
            defect_labels = [labels[i] for i in defect_indices]
            defect_preds = [predictions[i] for i in defect_indices]
            
            correct = sum(1 for label, pred in zip(defect_labels, defect_preds) if label == pred)
            accuracy_def = correct / len(defect_labels)
            defect_performance[defect_type] = {'accuracy': accuracy_def, 'count': len(defect_indices)}
    
    if defect_performance:
        sorted_defects = sorted(defect_performance.items(), key=lambda x: x[1]['count'], reverse=True)[:10]
        defect_names = [item[0] for item in sorted_defects]
        defect_accuracies = [item[1]['accuracy'] for item in sorted_defects]
        defect_counts = [item[1]['count'] for item in sorted_defects]
        
        bars = axes[1,0].bar(range(len(defect_names)), defect_accuracies, color='skyblue')
        axes[1,0].set_xticks(range(len(defect_names)))
        axes[1,0].set_xticklabels(defect_names, rotation=45, ha='right')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_title('Per-Defect Performance')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim(0, 1)
        
        for i, (bar, count) in enumerate(zip(bars, defect_counts)):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Performance Summary
    axes[1,1].axis('off')
    axes[1,1].text(0.1, 0.8, "Performance Summary", transform=axes[1,1].transAxes, fontsize=14, fontweight='bold')
    axes[1,1].text(0.1, 0.6, f"AUROC: {metrics['auroc']:.4f}", transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].text(0.1, 0.5, f"Accuracy: {metrics['accuracy']:.4f}", transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].text(0.1, 0.4, f"Precision: {metrics['precision']:.4f}", transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].text(0.1, 0.3, f"Recall: {metrics['recall']:.4f}", transform=axes[1,1].transAxes, fontsize=12)
    axes[1,1].text(0.1, 0.2, f"F1-Score: {metrics['f1_score']:.4f}", transform=axes[1,1].transAxes, fontsize=12)
    
    # Clear unused subplots
    axes[1,2].axis('off')
    
    plt.tight_layout()
    save_path = results_folder / 'performance_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")  # Keep this one print

def create_category_grid(category_name, short_name, indices, results, model, anomaly_prompt, 
                        global_min, global_max, results_folder, color, samples=16):
    """Create 4x4 grid for one category"""
    if len(indices) == 0:
        return
        
    # Select samples
    if len(indices) >= samples:
        selected_idx = np.random.choice(indices, samples, replace=False)
    else:
        selected_idx = indices[:samples]
    
    # Create figure
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle(f'{short_name} - {category_name} (Range: {global_min:.1f}-{global_max:.1f})', 
                fontsize=14, fontweight='bold', color=color, y=0.95)
    
    # Plot samples
    for row in range(4):
        for col in range(4):
            sample_num = row * 4 + col
            ax_img = axes[row, col * 2]
            ax_heat = axes[row, col * 2 + 1]
            
            if sample_num < len(selected_idx):
                idx = selected_idx[sample_num]
                img_path = results['image_paths'][idx]
                score = results['scores'][idx]
                defect_type = results['defect_types'][idx]
                
                # Get heatmap
                heatmap, _ = get_adaclip_heatmap(model, img_path, anomaly_prompt)
                
                # Load and show image
                image = Image.open(img_path).convert('RGB')
                img_array = np.array(image)
                ax_img.imshow(img_array)
                
                filename = Path(img_path).name
                if len(filename) > 12:
                    filename = filename[:9] + "..."
                ax_img.set_title(f'{filename}\n{defect_type}\nScore: {score:.2f}', 
                                fontsize=8, color=color, pad=5)
                ax_img.axis('off')
                
                # Show heatmap
                if heatmap is not None:
                    overlay = create_heatmap_overlay(image, heatmap, global_min, global_max, alpha=0.7)
                    ax_heat.imshow(overlay)
                    local_min = float(heatmap.min())
                    local_max = float(heatmap.max())
                    ax_heat.set_title(f'{local_min:.0f}-{local_max:.0f}', 
                                     fontsize=8, color=color, pad=5)
                else:
                    ax_heat.text(0.5, 0.5, 'No heatmap', ha='center', va='center')
                ax_heat.axis('off')
            else:
                ax_img.axis('off')
                ax_heat.axis('off')
    
    # Add colorbar
    norm = Normalize(vmin=global_min, vmax=global_max)
    sm = ScalarMappable(norm=norm, cmap='jet')
    sm.set_array([])
    cbar_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Global Anomaly Score', rotation=270, labelpad=15)
    
    plt.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.94, hspace=0.3, wspace=0.1)
    
    # Save
    save_path = results_folder / f'{short_name.lower()}_grid.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")

def save_error_analysis(false_positives, false_negatives, results_folder):
    """Save error analysis folders"""
    
    # False Positives folder
    if false_positives:
        fp_folder = results_folder / "false_positives"
        fp_folder.mkdir(exist_ok=True)
        
        fp_file = fp_folder / "analysis.txt"
        with open(fp_file, 'w') as f:
            f.write(f"=== FALSE POSITIVES ===\n\n")
            f.write(f"Total: {len(false_positives)}\n\n")
            
            for i, fp in enumerate(false_positives):
                source_path = Path(fp['path'])
                dest_path = fp_folder / source_path.name
                try:
                    shutil.copy2(source_path, dest_path)
                    f.write(f"{i+1:3d}. Score: {fp['score']:7.4f} | Image: {source_path.name}\n")
                except Exception as e:
                    f.write(f"{i+1:3d}. Score: {fp['score']:7.4f} | COPY FAILED: {e} | {source_path.name}\n")
        
        print(f"Saved {len(false_positives)} false positives to: {fp_folder}")
    
    # False Negatives folder
    if false_negatives:
        fn_folder = results_folder / "false_negatives"
        fn_folder.mkdir(exist_ok=True)
        
        fn_file = fn_folder / "analysis.txt"
        with open(fn_file, 'w') as f:
            f.write(f"=== FALSE NEGATIVES ===\n\n")
            f.write(f"Total: {len(false_negatives)}\n\n")
            
            for i, fn in enumerate(false_negatives):
                source_path = Path(fn['path'])
                dest_path = fn_folder / source_path.name
                try:
                    shutil.copy2(source_path, dest_path)
                    f.write(f"{i+1:3d}. Score: {fn['score']:7.4f} | Type: {fn['defect_type']:15s} | Image: {source_path.name}\n")
                except Exception as e:
                    f.write(f"{i+1:3d}. Score: {fn['score']:7.4f} | Type: {fn['defect_type']:15s} | COPY FAILED: {e} | {source_path.name}\n")
        
        print(f"Saved {len(false_negatives)} false negatives to: {fn_folder}")

def create_all_visualizations(results, metrics, model, anomaly_prompt, results_folder):
    """Create all visualizations for evaluation"""
    print("Creating visualizations...")
    
    # Calculate global scale
    global_min, global_max = calculate_global_heatmap_scale(model, results, metrics, anomaly_prompt)
    
    # Performance plot
    create_performance_plot(results, metrics, results_folder)
    
    # Get category indices
    scores = np.array(results['scores'])
    labels = np.array(results['labels'])
    predictions = metrics['predictions']
    
    tn_idx = [i for i in range(len(scores)) if labels[i] == 0 and predictions[i] == 0]
    tp_idx = [i for i in range(len(scores)) if labels[i] == 1 and predictions[i] == 1]
    fp_idx = [i for i in range(len(scores)) if labels[i] == 0 and predictions[i] == 1]
    fn_idx = [i for i in range(len(scores)) if labels[i] == 1 and predictions[i] == 0]
    
    # Create category grids
    categories = [
        ("True Negatives", "TN", tn_idx, 'green'),
        ("True Positives", "TP", tp_idx, 'blue'),
        ("False Positives", "FP", fp_idx, 'red'),
        ("False Negatives", "FN", fn_idx, 'orange')
    ]
    
    for cat_name, short_name, indices, color in categories:
        if len(indices) > 0:
            create_category_grid(cat_name, short_name, indices, results, model, 
                               anomaly_prompt, global_min, global_max, results_folder, color)
    
    print(f"All visualizations saved to: {results_folder}")