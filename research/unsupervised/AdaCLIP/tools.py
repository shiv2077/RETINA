
# Tools module for AdaCLIP
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import matplotlib.pyplot as plt

def calculate_metric(scores, labels):
    """Calculate metrics for anomaly detection"""
    try:
        auroc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        return {'auroc': auroc, 'ap': ap}
    except:
        return {'auroc': 0.0, 'ap': 0.0}

def calculate_average_metric(metrics_list):
    """Calculate average metrics"""
    if not metrics_list:
        return {}

    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])

    return avg_metrics

def visualization(image, anomaly_map, save_path=None):
    """Visualize anomaly detection results"""
    if save_path:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('Original')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(anomaly_map, cmap='hot')
        plt.title('Anomaly Map')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(image)
        plt.imshow(anomaly_map, alpha=0.5, cmap='hot')
        plt.title('Overlay')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
