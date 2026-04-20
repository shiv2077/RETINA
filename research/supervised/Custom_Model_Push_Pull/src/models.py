"""
Advanced Push-Pull Models for Wood Defect Detection
==================================================

Models that address the patch vs whole-image problem with:
- Attention mechanisms for smart focusing
- Multiple centers for different wood types
- Whole-image context preservation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import logging


class AttentionPushPullCNN(nn.Module):
    """
    Advanced Push-Pull with Attention Mechanism
    
    Key innovations:
    - Preserves whole-image context
    - Learns where to focus automatically  
    - Multiple normal centers for different wood types
    - Weighted push-pull based on attention
    """
    
    def __init__(self, feature_dim=512, num_centers=3, backbone='resnet18'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_centers = num_centers
        self.logger = logging.getLogger(__name__)
        self.is_vit = False
        
        # Backbone for feature extraction
        self.backbone = self._build_backbone(backbone)
        
        # Attention mechanism for smart focusing
        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),  # Single channel attention map
            nn.Sigmoid()
        )
        
        # Feature projector (global average pooled features)
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Multiple normal centers (different wood types/colors)
        self.register_buffer('normal_centers', torch.zeros(num_centers, feature_dim))
        self.register_buffer('centers_initialized', torch.zeros(num_centers, dtype=torch.bool))
        self.register_buffer('center_counts', torch.zeros(num_centers))
        
        # Center assignment network (which center for each sample)
        self.center_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_centers)
        )
        
        self.center_momentum = 0.9
        
    def _build_backbone(self, backbone):
        """Build feature extraction backbone - MULTIPLE OPTIONS"""
        
        if backbone == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            self.backbone_dim = 512
            self.backbone = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool + fc
            
        elif backbone == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')  
            self.backbone_dim = 2048
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            
        elif backbone == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V1')
            self.backbone_dim = 2048
            self.backbone = nn.Sequential(*list(model.children())[:-2])
            
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            self.backbone_dim = 1280
            # Remove classifier
            self.backbone = model.features
            
        elif backbone == 'efficientnet_b3':
            model = models.efficientnet_b3(weights='IMAGENET1K_V1')
            self.backbone_dim = 1536
            self.backbone = model.features
            
        elif backbone == 'vit_b_16':
            model = models.vit_b_16(weights='IMAGENET1K_V1')
            self.backbone_dim = 768
            # For ViT, we need special handling
            self.backbone = model
            self.is_vit = True
            
        elif backbone == 'convnext_tiny':
            model = models.convnext_tiny(weights='IMAGENET1K_V1')
            self.backbone_dim = 768
            self.backbone = model.features
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Options: resnet18, resnet50, resnet101, efficientnet_b0, efficientnet_b3, vit_b_16, convnext_tiny")
        
        return self.backbone
    
    def forward(self, x):
        """
        Forward pass with attention-weighted features
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            features: Global features [B, feature_dim]
            attention_maps: Attention maps [B, 1, H', W'] (None for ViT)
            center_logits: Center assignment logits [B, num_centers]
        """
        
        batch_size = x.size(0)
        
        if hasattr(self, 'is_vit') and self.is_vit:
            # Special handling for Vision Transformer
            features = self.backbone(x)  # [B, feature_dim]
            features = self.projector(features)
            features = F.normalize(features, p=2, dim=1)
            center_logits = self.center_classifier(features.detach())
            return features, None, center_logits  # No attention maps for ViT
        
        # Standard CNN handling
        feature_maps = self.backbone(x)  # [B, backbone_dim, H', W']
        
        # Generate attention maps
        attention_maps = self.attention_conv(feature_maps)  # [B, 1, H', W']
        
        # Apply attention weighting to features
        weighted_features = feature_maps * attention_maps  # [B, backbone_dim, H', W']
        
        # Global average pooling with attention weighting
        attention_weights = attention_maps.view(batch_size, -1)  # [B, H'*W']
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize
        
        flat_features = weighted_features.view(batch_size, self.backbone_dim, -1)  # [B, backbone_dim, H'*W']
        attention_weights = attention_weights.unsqueeze(1)  # [B, 1, H'*W']
        
        # Weighted global features
        global_features = torch.sum(flat_features * attention_weights, dim=2)  # [B, backbone_dim]
        
        # Project to final feature space
        features = self.projector(global_features)  # [B, feature_dim]
        features = F.normalize(features, p=2, dim=1)  # L2 normalize
        
        # Center assignment
        center_logits = self.center_classifier(features.detach())  # [B, num_centers]
        
        return features, attention_maps, center_logits
    
    def update_normal_centers(self, normal_features, center_assignments=None):
        """
        Update multiple normal centers
        
        Args:
            normal_features: Features from normal samples [N, feature_dim]
            center_assignments: Optional center assignments [N]
        """
        
        if len(normal_features) == 0:
            return
        
        # If no assignments provided, use closest center assignment
        if center_assignments is None:
            with torch.no_grad():
                center_logits = self.center_classifier(normal_features)
                center_assignments = torch.argmax(center_logits, dim=1)
        
        # Update each center
        for center_id in range(self.num_centers):
            mask = (center_assignments == center_id)
            if mask.any():
                center_features = normal_features[mask]
                batch_center = center_features.mean(dim=0)
                
                if not self.centers_initialized[center_id]:
                    self.normal_centers[center_id].copy_(batch_center)
                    self.centers_initialized[center_id] = True
                    self.center_counts[center_id] = mask.sum().float()
                    self.logger.info(f"🎯 Normal center {center_id} initialized")
                else:
                    # Exponential moving average
                    self.normal_centers[center_id].mul_(self.center_momentum).add_(
                        batch_center, alpha=1 - self.center_momentum
                    )
                    self.center_counts[center_id] += mask.sum().float()
    
    def compute_anomaly_scores(self, features):
        """
        Compute anomaly scores as distance to nearest normal center
        
        Args:
            features: Features [B, feature_dim]
            
        Returns:
            anomaly_scores: Distance to nearest center [B]
            center_assignments: Which center each sample is closest to [B]
        """
        
        if not self.centers_initialized.any():
            return torch.zeros(features.size(0), device=features.device), None
        
        batch_size = features.size(0)
        distances = torch.zeros(batch_size, self.num_centers, device=features.device)
        
        # Compute distance to each initialized center
        for center_id in range(self.num_centers):
            if self.centers_initialized[center_id]:
                center_distances = torch.norm(
                    features - self.normal_centers[center_id], p=2, dim=1
                )
                distances[:, center_id] = center_distances
            else:
                distances[:, center_id] = float('inf')
        
        # Use minimum distance as anomaly score
        anomaly_scores, center_assignments = torch.min(distances, dim=1)
        
        return anomaly_scores, center_assignments
    
    def compute_push_pull_loss(self, features, labels, attention_maps):
        """
        Compute attention-weighted push-pull loss
        
        Args:
            features: Features [B, feature_dim]
            labels: Labels [B] (0=normal, 1=anomaly)
            attention_maps: Attention maps [B, 1, H', W']
            
        Returns:
            loss: Total push-pull loss
            loss_components: Dictionary of loss components
        """
        
        if not self.centers_initialized.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True), {}
        
        normal_mask = (labels == 0)
        anomaly_mask = (labels == 1)
        
        loss_components = {}
        total_loss = 0.0
        
        # PULL: Normal samples toward their assigned centers
        if normal_mask.any():
            normal_features = features[normal_mask]
            normal_attention = attention_maps[normal_mask]
            
            # Get center assignments for normal samples
            with torch.no_grad():
                center_logits = self.center_classifier(normal_features)
                center_assignments = torch.argmax(center_logits, dim=1)
            
            pull_losses = []
            for center_id in range(self.num_centers):
                if self.centers_initialized[center_id]:
                    center_mask = (center_assignments == center_id)
                    if center_mask.any():
                        center_features = normal_features[center_mask]
                        center_attention = normal_attention[center_mask]
                        
                        # Compute distances
                        distances = torch.norm(
                            center_features - self.normal_centers[center_id], p=2, dim=1
                        )
                        
                        # Weight by attention (higher attention = more important)
                        attention_weights = center_attention.mean(dim=(1, 2, 3))  # [N]
                        weighted_distances = distances * (1 + attention_weights)
                        
                        pull_losses.append(weighted_distances.mean())
            
            if pull_losses:
                pull_loss = torch.stack(pull_losses).mean()
                loss_components['pull'] = pull_loss
                total_loss += pull_loss
        
        # PUSH: Anomaly samples away from nearest centers
        if anomaly_mask.any():
            anomaly_features = features[anomaly_mask]
            anomaly_attention = attention_maps[anomaly_mask]
            
            anomaly_scores, center_assignments = self.compute_anomaly_scores(anomaly_features)
            
            # Margin-based push loss
            margin = 1.0
            attention_weights = anomaly_attention.mean(dim=(1, 2, 3))  # [N] 
            weighted_margin = margin * (1 + attention_weights)  # Higher attention = larger margin
            
            push_distances = torch.clamp(weighted_margin - anomaly_scores, min=0)
            push_loss = push_distances.mean()
            
            loss_components['push'] = push_loss
            total_loss += push_loss
        
        # CENTER DIVERSITY: Encourage centers to be different
        if self.centers_initialized.sum() > 1:
            initialized_centers = self.normal_centers[self.centers_initialized]
            center_distances = torch.pdist(initialized_centers, p=2)
            diversity_loss = torch.clamp(1.0 - center_distances, min=0).mean()
            loss_components['diversity'] = diversity_loss
            total_loss += 0.1 * diversity_loss  # Small weight
        
        return total_loss, loss_components
    
    def get_attention_visualization(self, x):
        """Get attention maps for visualization"""
        
        self.eval()
        with torch.no_grad():
            _, attention_maps, _ = self.forward(x)
            
            # Upsample attention maps to input size
            attention_upsampled = F.interpolate(
                attention_maps, size=x.shape[-2:], mode='bilinear', align_corners=False
            )
            
            return attention_upsampled


class SimplePushPullCNN(nn.Module):
    """
    Simple Push-Pull model - EXACTLY like your working prototype
    """
    
    def __init__(self, feature_dim=256, backbone='resnet18'):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.logger = logging.getLogger(__name__)
        self.is_vit = False
        
        # Build backbone - MULTIPLE OPTIONS
        if backbone == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            backbone_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.backbone = model
        elif backbone == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            backbone_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.backbone = model
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            backbone_dim = 1280
            model.classifier = nn.Identity()
            self.backbone = model
        elif backbone == 'vit_b_16':
            model = models.vit_b_16(weights='IMAGENET1K_V1')
            backbone_dim = 768
            model.heads = nn.Identity()
            self.backbone = model
            self.is_vit = True
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature projector
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Single normal center
        self.register_buffer('normal_center', torch.zeros(feature_dim))
        self.register_buffer('center_initialized', torch.tensor(False))
        self.center_momentum = 0.9
    
    def forward(self, x):
        """Simple forward pass"""
        backbone_out = self.backbone(x)
        features = self.projector(backbone_out)
        return nn.functional.normalize(features, p=2, dim=1)
    
    def update_normal_center(self, normal_features):
        """Update normal center - EXACTLY like prototype"""
        if len(normal_features) == 0:
            return
            
        batch_center = normal_features.mean(dim=0)
        
        if not self.center_initialized:
            self.normal_center.copy_(batch_center)
            self.center_initialized.fill_(True)
            self.logger.info("🎯 Normal center initialized")
        else:
            self.normal_center.mul_(self.center_momentum).add_(
                batch_center, alpha=1 - self.center_momentum
            )
    
    def compute_anomaly_scores(self, features):
        """Compute distance-based anomaly scores"""
        if not self.center_initialized:
            return torch.zeros(features.size(0), device=features.device)
        
        distances = torch.norm(features - self.normal_center, p=2, dim=1)
        return distances


class RegionPushPullCNN(nn.Module):
    """
    Region-based Push-Pull - Process image patches separately
    
    Key idea: Instead of whole image, divide into regions and 
    apply push-pull to each region separately for local defect detection
    """
    
    def __init__(self, feature_dim=256, backbone='resnet18', patch_size=112, num_patches=4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.num_patches = num_patches  # 2x2 patches = 4 total
        self.logger = logging.getLogger(__name__)
        

        # Build backbone for patch processing
        if backbone == 'resnet18':
            model = models.resnet18(weights='IMAGENET1K_V1')
            backbone_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.backbone = model
        elif backbone == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V1')
            backbone_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.backbone = model
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature projector for each patch
        self.patch_projector = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Global feature aggregator
        self.global_projector = nn.Sequential(
            nn.Linear(feature_dim * num_patches, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # Multiple normal centers (one per patch + one global)
        self.register_buffer('patch_centers', torch.zeros(num_patches, feature_dim))
        self.register_buffer('global_center', torch.zeros(feature_dim))
        self.register_buffer('centers_initialized', torch.zeros(num_patches + 1, dtype=torch.bool))
        self.center_momentum = 0.9

        self.normal_center = None
        self.patch_normal_center = None
    
    def forward(self, x):
        """
        Forward pass with region-based processing
        
        Args:
            x: Input images [B, 3, 224, 224]
            
        Returns:
            global_features: Global features [B, feature_dim]
            patch_features: Patch features [B, num_patches, feature_dim]
            patch_scores: Anomaly scores per patch [B, num_patches]
        """
        
        batch_size = x.size(0)
        
        # Extract patches (2x2 = 4 patches)
        patches = self._extract_patches(x)  # [B*num_patches, 3, patch_size, patch_size]
        
        # Process each patch through backbone
        patch_features = self.backbone(patches)  # [B*num_patches, backbone_dim]
        patch_features = self.patch_projector(patch_features)  # [B*num_patches, feature_dim]
        patch_features = F.normalize(patch_features, p=2, dim=1)
        
        # Reshape to [B, num_patches, feature_dim]
        patch_features = patch_features.view(batch_size, self.num_patches, self.feature_dim)
        
        # Global features (concatenate all patches)
        global_features = patch_features.view(batch_size, -1)  # [B, num_patches * feature_dim]
        global_features = self.global_projector(global_features)  # [B, feature_dim]
        global_features = F.normalize(global_features, p=2, dim=1)
        
        # Compute patch-level anomaly scores
        patch_scores = self._compute_patch_scores(patch_features)  # [B, num_patches]
        
        return global_features, patch_features, patch_scores
    
    def _extract_patches(self, x):
        """Extract 2x2 patches from image"""
        
        batch_size = x.size(0)
        
        # Resize to ensure divisible patches
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract 2x2 patches
        patches = []
        for i in range(2):
            for j in range(2):
                patch = x_resized[:, :, i*112:(i+1)*112, j*112:(j+1)*112]
                patches.append(patch)
        
        # Stack and reshape
        patches = torch.stack(patches, dim=1)  # [B, 4, 3, 112, 112]
        patches = patches.view(-1, 3, 112, 112)  # [B*4, 3, 112, 112]
        
        return patches
    
    def update_normal_centers(self, normal_features, normal_patch_features):
        """Update both patch and global centers"""
        
        if len(normal_features) == 0:
            return
        
        # Update global center
        global_center = normal_features.mean(dim=0)
        if not self.centers_initialized[-1]:  # Global center is last
            self.global_center.copy_(global_center)
            self.centers_initialized[-1] = True
            self.logger.info("🎯 Global center initialized")
        else:
            self.global_center.mul_(self.center_momentum).add_(
                global_center, alpha=1 - self.center_momentum
            )
        
        # Update patch centers
        for patch_id in range(self.num_patches):
            patch_features = normal_patch_features[:, patch_id, :]  # [N, feature_dim]
            patch_center = patch_features.mean(dim=0)
            
            if not self.centers_initialized[patch_id]:
                self.patch_centers[patch_id].copy_(patch_center)
                self.centers_initialized[patch_id] = True
                self.logger.info(f"🎯 Patch center {patch_id} initialized")
            else:
                self.patch_centers[patch_id].mul_(self.center_momentum).add_(
                    patch_center, alpha=1 - self.center_momentum
                )
    
    def _compute_patch_scores(self, patch_features):
        """Compute anomaly scores for each patch"""
        
        batch_size = patch_features.size(0)
        patch_scores = torch.zeros(batch_size, self.num_patches, device=patch_features.device)
        
        for patch_id in range(self.num_patches):
            if self.centers_initialized[patch_id]:
                patch_feat = patch_features[:, patch_id, :]  # [B, feature_dim]
                distances = torch.norm(patch_feat - self.patch_centers[patch_id], p=2, dim=1)
                patch_scores[:, patch_id] = distances
        
        return patch_scores
    
    def compute_anomaly_scores(self, global_features):
        """Compute global anomaly scores"""
        
        if not self.centers_initialized[-1]:
            return torch.zeros(global_features.size(0), device=global_features.device)
        
        distances = torch.norm(global_features - self.global_center, p=2, dim=1)
        return distances


def create_model(model_type='simple', **kwargs):
    """
    Factory function to create push-pull models
    
    Args:
        model_type: 'advanced', 'simple', or 'region'
        **kwargs: Model parameters
        
    Returns:
        model: Push-pull model instance
    """
    
    if model_type == 'advanced':
        return AttentionPushPullCNN(**kwargs)
    elif model_type == 'simple':
        return SimplePushPullCNN(**kwargs)
    elif model_type == 'region':
        return RegionPushPullCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Options: advanced, simple, region")