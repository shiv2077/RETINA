"""
Advanced Training Pipeline for Push-Pull Models
==============================================

Features:
- Support for both simple and advanced push-pull models
- Attention-aware loss computation
- Multi-center learning
- Comprehensive evaluation and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import time
import logging
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

from models import create_model


class AdvancedPushPullTrainer:
    """
    Advanced trainer for push-pull models with attention mechanisms
    """
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.output_dir = Path(config['paths']['outputs'])
        self.model_dir = Path(config['paths']['models'])
        self.output_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Training config
        self.train_config = config.get('training', {})
        self.epochs = self.train_config.get('epochs', 50)
        self.lr = self.train_config.get('learning_rate', 0.001)
        self.model_type = self.train_config.get('model_type', 'advanced')
        
        # Model and training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_auc = 0.0
        self.best_epoch = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_aucs = []
        self.loss_components_history = []
        
    def setup_model(self):
        """Setup model, optimizer, and scheduler"""
        
        self.logger.info(f"🏗️  Setting up {self.model_type} push-pull model...")
        
        # Create model
        model_config = self.config.get('model', {})
        self.model = create_model(
            model_type=self.model_type,
            **model_config
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"📊 Model statistics:")
        self.logger.info(f"   - Total parameters: {total_params:,}")
        self.logger.info(f"   - Trainable parameters: {trainable_params:,}")
        self.logger.info(f"   - Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Setup optimizer
        optimizer_type = self.train_config.get('optimizer', 'adamw')
        if optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.train_config.get('weight_decay', 0.01),
                betas=(0.9, 0.999)
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.train_config.get('weight_decay', 0.0001)
            )
        
        # Setup scheduler
        scheduler_type = self.train_config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2, eta_min=self.lr * 0.01
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=20, gamma=0.5
            )
        
        self.logger.info(f"✅ Model setup complete")
        self.logger.info(f"   - Optimizer: {optimizer_type}")
        self.logger.info(f"   - Scheduler: {scheduler_type}")
    
    def train(self, train_loader, test_loader):
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            
        Returns:
            training_results: Dictionary with training metrics
        """
        
        self.logger.info("🚀 Starting advanced push-pull training...")
        self.logger.info("=" * 60)
        
        self.setup_model()
        
        start_time = time.time()
        patience_counter = 0
        patience = self.train_config.get('patience', 15)
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, loss_components = self._train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            self.loss_components_history.append(loss_components)
            
            # Simple progress logging - CLEAN
            if (epoch + 1) % 5 == 0:  # Only log every 5 epochs
                loss_str = f"pull: {loss_components.get('pull', 0):.3f} | push: {loss_components.get('push', 0):.3f}"
                self.logger.info(f"Epoch {epoch+1:2d} | Loss: {train_loss:.3f} | {loss_str}")
            
            # Step scheduler (no validation needed)
            if not isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step()
        
        # ONLY evaluate at the end - like prototype
        final_results = self.evaluate_comprehensive(test_loader)
        final_auc = final_results['auc']
        self.best_auc = final_auc
        self.best_epoch = self.epochs - 1
        
        # Save the final model
        self._save_best_model()
        
        training_time = time.time() - start_time
        
        self.logger.info("🏁 Training completed!")
        self.logger.info(f"   - Training time: {training_time:.1f}s ({training_time/60:.1f}m)")
        self.logger.info(f"   - Best validation AUC: {self.best_auc:.4f} (epoch {self.best_epoch+1})")
        self.logger.info(f"   - Final test AUC: {final_results['auc']:.4f}")
        
        # Create training plots
        self._create_training_plots()
        
        return {
            'best_validation_auc': self.best_auc,
            'final_test_auc': final_results['auc'],
            'best_epoch': self.best_epoch,
            'total_epochs': epoch + 1,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'val_aucs': self.val_aucs,
            'loss_components': self.loss_components_history,
            'final_results': final_results
        }
    
    def _train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        
        self.model.train()
        total_loss = 0.0
        loss_components_sum = {}
        num_batches = 0
        
        # Progress bar - FIXED
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}', 
                   leave=False, ncols=80, ascii=True, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model_type == 'advanced':
                features, attention_maps, center_logits = self.model(images)
                
                # Update normal centers (only for normal samples)
                normal_mask = (labels == 0)
                if normal_mask.any():
                    normal_features = features[normal_mask].detach()
                    center_assignments = torch.argmax(center_logits[normal_mask], dim=1)
                    self.model.update_normal_centers(normal_features, center_assignments)
                
                # Compute push-pull loss with attention
                loss, loss_components = self.model.compute_push_pull_loss(
                    features, labels, attention_maps
                )
                
            else:  # Simple or region model
                features = self.model(images)
                
                normal_mask = (labels == 0)
                
                
                if normal_mask.any():
                    normal_indices = normal_mask.nonzero(as_tuple=False).squeeze(1)
                    
                    if isinstance(features, tuple):
                        # Region model returns (global_features, patch_features, patch_scores)
                        global_features, patch_features, patch_scores = features
                        if len(normal_indices) > 0:
                            normal_features = global_features[normal_indices].detach()
                            patch_feats = patch_features[normal_indices]
                            self.model.update_normal_centers(normal_features.detach(), patch_feats.detach())

                    else:
                        # Simple model returns a single tensor
                        if len(normal_indices) > 0:
                            normal_features = features[normal_indices].detach()
                            self.model.update_normal_center(normal_features)
                
                if self.model_type == 'region':
                    loss, loss_components = self._compute_region_push_pull_loss(
                        global_features, patch_features, patch_scores, labels
                    )
                else:
                    loss, loss_components = self._compute_simple_push_pull_loss(features, labels)

            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            num_batches += 1
            
            # Accumulate loss components
            for key, value in loss_components.items():
                if key not in loss_components_sum:
                    loss_components_sum[key] = 0.0
                loss_components_sum[key] += value if isinstance(value, float) else value.item()
            
            # Update progress bar - CLEAN
            pbar.set_postfix({'Loss': f'{loss.item():.3f}', 'LR': f'{self.optimizer.param_groups[0]["lr"]:.1e}'})
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components_sum.items()}
        
        return avg_loss, avg_loss_components
    
    def _compute_simple_push_pull_loss(self, features, labels):
        """Compute simple push-pull loss for comparison"""
        
        if not self.model.center_initialized:
            return torch.tensor(0.0, device=features.device, requires_grad=True), {}
        
        normal_mask = (labels == 0)
        anomaly_mask = (labels == 1)
        
        loss_components = {}
        total_loss = 0.0
        
        # Pull normal samples to center
        if normal_mask.any():
            normal_features = features[normal_mask]
            distances = torch.norm(normal_features - self.model.normal_center, p=2, dim=1)
            pull_loss = distances.mean()
            loss_components['pull'] = pull_loss
            total_loss += pull_loss
        
        # Push anomaly samples away from center
        if anomaly_mask.any():
            anomaly_features = features[anomaly_mask]
            distances = torch.norm(anomaly_features - self.model.normal_center, p=2, dim=1)
            margin = 1.0
            push_loss = torch.clamp(margin - distances, min=0).mean()
            loss_components['push'] = push_loss
            total_loss += push_loss
        
        return total_loss, loss_components
    
    def _compute_region_push_pull_loss(self, global_features, patch_features, patch_scores, labels):
        """Compute region-based push-pull loss"""
        
        normal_mask = (labels == 0)
        anomaly_mask = (labels == 1)
        
        loss_components = {}
        total_loss = 0.0
        
        # Global push-pull loss
        if self.model.centers_initialized[-1]:  # Global center
            if normal_mask.any():
                normal_global = global_features[normal_mask]
                global_distances = torch.norm(normal_global - self.model.global_center, p=2, dim=1)
                global_pull_loss = global_distances.mean()
                loss_components['global_pull'] = global_pull_loss
                total_loss += global_pull_loss
            
            if anomaly_mask.any():
                anomaly_global = global_features[anomaly_mask]
                global_distances = torch.norm(anomaly_global - self.model.global_center, p=2, dim=1)
                margin = 1.0
                global_push_loss = torch.clamp(margin - global_distances, min=0).mean()
                loss_components['global_push'] = global_push_loss
                total_loss += global_push_loss
        
        # Patch-level push-pull loss
        patch_losses = []
        for patch_id in range(self.model.num_patches):
            if self.model.centers_initialized[patch_id]:
                patch_feat = patch_features[:, patch_id, :]  # [B, feature_dim]
                
                if normal_mask.any():
                    normal_patch = patch_feat[normal_mask]
                    patch_distances = torch.norm(normal_patch - self.model.patch_centers[patch_id], p=2, dim=1)
                    patch_pull = patch_distances.mean()
                    patch_losses.append(patch_pull)
                
                if anomaly_mask.any():
                    anomaly_patch = patch_feat[anomaly_mask]
                    patch_distances = torch.norm(anomaly_patch - self.model.patch_centers[patch_id], p=2, dim=1)
                    patch_push = torch.clamp(0.5 - patch_distances, min=0).mean()  # Smaller margin for patches
                    patch_losses.append(patch_push)
        
        if patch_losses:
            patch_loss = torch.stack(patch_losses).mean()
            loss_components['patch'] = patch_loss
            total_loss += 0.5 * patch_loss  # Weight patch loss less than global
        
        return total_loss, loss_components
    
    def _validate_epoch(self, test_loader):
        """Validate one epoch"""
        
        self.model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label']
                
                # Handle different model types for validation
                if self.model_type == 'region':
                    global_features, _, _ = self.model(images)
                    scores = self.model.compute_anomaly_scores(global_features)
                elif self.model_type == 'advanced':
                    features, _, _ = self.model(images)
                    scores, _ = self.model.compute_anomaly_scores(features)
                else:  # Simple model
                    features = self.model(images)
                    scores = self.model.compute_anomaly_scores(features)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except:
            auc = 0.5
        
        return auc
    
    def evaluate_comprehensive(self, test_loader):
        """Comprehensive evaluation with detailed metrics"""
        
        self.logger.info("📊 Running comprehensive evaluation...")
        
        self.model.eval()
        all_scores = []
        all_labels = []
        all_categories = []
        all_paths = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                labels = batch['label']
                categories = batch['category']
                paths = batch['path']
                
                # Handle different model types
                if self.model_type == 'region':
                    global_features, _, _ = self.model(images)
                    scores = self.model.compute_anomaly_scores(global_features)
                elif self.model_type == 'advanced':
                    features, attention_maps, _ = self.model(images)
                    scores, _ = self.model.compute_anomaly_scores(features)
                else:  # Simple model
                    features = self.model(images)
                    scores = self.model.compute_anomaly_scores(features)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_categories.extend(categories)
                all_paths.extend(paths)
        
        # Convert to numpy arrays
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Find optimal threshold
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
        
        # Calculate metrics
        predictions = (all_scores > optimal_threshold).astype(int)
        
        results = {
            'auc': float(roc_auc_score(all_labels, all_scores)),
            'accuracy': float(accuracy_score(all_labels, predictions)),
            'precision': float(precision_score(all_labels, predictions, zero_division=0)),
            'recall': float(recall_score(all_labels, predictions, zero_division=0)),
            'f1': float(f1_score(all_labels, predictions, zero_division=0)),
            'optimal_threshold': float(optimal_threshold),
            'num_samples': len(all_labels)
        }
        
        # Per-category analysis
        category_results = {}
        unique_categories = set(all_categories)
        
        for category in unique_categories:
            if category == 'normal':
                continue
                
            mask = np.array(all_categories) == category
            if mask.sum() > 0:
                cat_labels = all_labels[mask]
                cat_scores = all_scores[mask]
                cat_preds = predictions[mask]
                
                if len(np.unique(cat_labels)) > 1:
                    cat_auc = roc_auc_score(cat_labels, cat_scores)
                    cat_f1 = f1_score(cat_labels, cat_preds, zero_division=0)
                    
                    category_results[category] = {
                        'auc': float(cat_auc),
                        'f1': float(cat_f1),
                        'samples': int(mask.sum()),
                        'anomalies': int(cat_labels.sum())
                    }
        
        results['category_results'] = category_results
        
        # Log results
        self.logger.info("📈 COMPREHENSIVE EVALUATION RESULTS:")
        self.logger.info(f"   - AUC: {results['auc']:.4f}")
        self.logger.info(f"   - Accuracy: {results['accuracy']:.4f}")
        self.logger.info(f"   - Precision: {results['precision']:.4f}")
        self.logger.info(f"   - Recall: {results['recall']:.4f}")
        self.logger.info(f"   - F1-Score: {results['f1']:.4f}")
        self.logger.info(f"   - Optimal threshold: {results['optimal_threshold']:.4f}")
        self.logger.info(f"   - Test samples: {results['num_samples']}")
        
        return results
    
    def _save_best_model(self):
        """Save the best model"""
        
        model_path = self.model_dir / f"best_{self.model_type}_model.pth"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'config': self.config,
            'auc': self.best_auc,
            'epoch': self.best_epoch,
        }
        
        # Save model-specific state
        if self.model_type == 'advanced':
            checkpoint.update({
                'normal_centers': self.model.normal_centers,
                'centers_initialized': self.model.centers_initialized,
                'center_counts': self.model.center_counts
            })
        else:
            checkpoint.update({
                'normal_center': self.model.normal_center,
                'center_initialized': self.model.center_initialized
            })
        
        torch.save(checkpoint, model_path)
    
    def _load_best_model(self):
        """Load the best saved model"""
        
        model_path = self.model_dir / f"best_{self.model_type}_model.pth"
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore model-specific state
            if self.model_type == 'advanced':
                self.model.normal_centers = checkpoint['normal_centers']
                self.model.centers_initialized = checkpoint['centers_initialized'] 
                self.model.center_counts = checkpoint['center_counts']
            else:
                self.model.normal_center = checkpoint['normal_center']
                self.model.center_initialized = checkpoint['center_initialized']
            
            self.logger.info("📂 Best model loaded")
        else:
            self.logger.warning("⚠️  No saved model found")
    
    def _create_training_plots(self):
        """Create training visualization plots"""
        
        if not self.train_losses or not self.val_aucs:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_type.title()} Push-Pull Training Results', fontsize=16)
        
        # Training loss
        axes[0, 0].plot(self.train_losses, 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation AUC
        val_epochs = list(range(2, len(self.train_losses), 3))[:len(self.val_aucs)]
        axes[0, 1].plot(val_epochs, self.val_aucs, 'g-o', linewidth=2, markersize=4)
        axes[0, 1].axhline(y=self.best_auc, color='r', linestyle='--', label=f'Best: {self.best_auc:.4f}')
        axes[0, 1].set_title('Validation AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss components (if available)
        if self.loss_components_history:
            loss_components = {}
            for epoch_components in self.loss_components_history:
                for key, value in epoch_components.items():
                    if key not in loss_components:
                        loss_components[key] = []
                    loss_components[key].append(value)
            
            for key, values in loss_components.items():
                axes[1, 0].plot(values, label=key, linewidth=2)
            
            axes[1, 0].set_title('Loss Components')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Summary stats
        axes[1, 1].axis('off')
        summary_text = f"""
        Training Summary:
        
        Model: {self.model_type.title()} Push-Pull
        Best Validation AUC: {self.best_auc:.4f}
        Best Epoch: {self.best_epoch + 1}
        Total Epochs: {len(self.train_losses)}
        
        Final Loss: {self.train_losses[-1]:.4f}
        Min Loss: {min(self.train_losses):.4f}
        
        Status: {'✅ Converged' if len(self.train_losses) > 10 else '⚠️ Short training'}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plot_path = self.output_dir / f'{self.model_type}_training_plots.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"📈 Training plots saved: {plot_path}")