"""
Advanced Data Handling for Push-Pull Wood Defect Detection
==========================================================

Smart data loading and preprocessing that addresses:
- Whole image vs patch trade-offs
- Better augmentations for wood textures  
- Balanced sampling strategies
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import random
import json
import logging
from pathlib import Path
import os


class WoodDefectDataset(Dataset):
    """
    Advanced dataset for wood defect detection
    
    Features:
    - Smart augmentations for wood textures
    - Balanced sampling support
    - Metadata tracking for analysis
    """
    
    def __init__(self, samples, phase='train', config=None):
        self.samples = samples
        self.phase = phase
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.img_size = self.config.get('img_size', 224)
        
        # Setup transforms
        self._setup_transforms()
        
        # Log dataset info
        self._log_dataset_info()
    
    def _setup_transforms(self):
        """Setup transforms optimized for wood textures"""
        
        base_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
        ]
        
        if self.phase == 'train':
            # Advanced augmentations for wood textures
            train_transforms = [
                transforms.RandomApply([
                    WoodTextureAugmentation(),  # Custom wood-specific augmentations
                ], p=0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.3),  # Wood can be oriented any way
                transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ColorJitter(
                    brightness=0.2,  # Wood lighting varies
                    contrast=0.15,   # Grain contrast important
                    saturation=0.1,  # Don't change wood color too much
                    hue=0.05        # Slight hue variation
                ),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
                ], p=0.1),
            ]
            
            self.transform = transforms.Compose(
                base_transforms + train_transforms + [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Simulate small occlusions
                ]
            )
        else:
            # Test transforms - minimal processing
            self.transform = transforms.Compose(
                base_transforms + [
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
    
    def _log_dataset_info(self):
        """Log dataset statistics"""
        
        normal_count = sum(1 for s in self.samples if s['label'] == 0)
        anomaly_count = sum(1 for s in self.samples if s['label'] == 1)
        
        self.logger.info(f"📊 {self.phase.upper()} Dataset: {len(self.samples)} samples")
        self.logger.info(f"   - Normal: {normal_count} ({normal_count/len(self.samples)*100:.1f}%)")
        self.logger.info(f"   - Anomaly: {anomaly_count} ({anomaly_count/len(self.samples)*100:.1f}%)")
        
        # Log anomaly type distribution if available
        if 'category' in self.samples[0]:
            category_counts = {}
            for sample in self.samples:
                if sample['label'] == 1:  # Only anomalies
                    cat = sample.get('category', 'unknown')
                    category_counts[cat] = category_counts.get(cat, 0) + 1
            
            if category_counts:
                self.logger.info("   📂 Anomaly categories:")
                for cat, count in sorted(category_counts.items()):
                    self.logger.info(f"      - {cat}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image with error handling
        try:
            image = Image.open(sample['path']).convert('RGB')
            
            # Quality check
            if image.size[0] < 32 or image.size[1] < 32:
                self.logger.warning(f"⚠️  Very small image: {sample['path']}")
            
            image = self.transform(image)
            
        except Exception as e:
            self.logger.warning(f"⚠️  Error loading {sample['path']}: {e}")
            # Return black image as fallback
            image = torch.zeros(3, self.img_size, self.img_size)
        
        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'path': sample['path'],
            'category': sample.get('category', 'unknown'),
            'is_expert': sample.get('is_expert', False)
        }
    
    def get_sample_weights(self):
        """Get weights for balanced sampling"""
        
        normal_count = sum(1 for s in self.samples if s['label'] == 0)
        anomaly_count = sum(1 for s in self.samples if s['label'] == 1)
        
        total = len(self.samples)
        normal_weight = total / (2 * normal_count) if normal_count > 0 else 1.0
        anomaly_weight = total / (2 * anomaly_count) if anomaly_count > 0 else 1.0
        
        weights = []
        for sample in self.samples:
            if sample['label'] == 0:
                weights.append(normal_weight)
            else:
                weights.append(anomaly_weight)
        
        return torch.tensor(weights, dtype=torch.float)


class WoodTextureAugmentation:
    """Custom augmentation for wood textures"""
    
    def __init__(self):
        self.grain_enhance_prob = 0.3
        self.lighting_change_prob = 0.4
    
    def __call__(self, image):
        """Apply wood-specific augmentations"""
        
        # Enhance wood grain (increase sharpness occasionally)
        if random.random() < self.grain_enhance_prob:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(1.0, 1.3))
        
        # Simulate lighting changes (brightness/contrast)
        if random.random() < self.lighting_change_prob:
            # Brightness change
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Contrast change  
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))
        
        return image


def load_expert_samples(dataset_path, n_normal=100, n_anomaly=200, seed=42):
    """
    Load expert samples with smart sampling
    
    Args:
        dataset_path: Path to dataset
        n_normal: Number of normal expert samples
        n_anomaly: Number of anomaly expert samples  
        seed: Random seed for reproducibility
        
    Returns:
        train_samples: List of training samples
        expert_paths: Set of expert sample paths (for exclusion)
    """
    
    logger = logging.getLogger(__name__)
    logger.info("📋 Loading expert samples...")
    
    dataset_path = Path(dataset_path)
    random.seed(seed)
    np.random.seed(seed)
    
    # Load normal expert samples
    normal_files = list((dataset_path / "normal").rglob("*.jpg"))
    if len(normal_files) < n_normal:
        logger.warning(f"⚠️  Only {len(normal_files)} normal files, requested {n_normal}")
        n_normal = len(normal_files)
    
    expert_normal = random.sample(normal_files, n_normal)
    
    # Load anomaly expert samples with balanced category sampling
    all_anomaly_files = []
    anomaly_path = dataset_path / "anomaly"
    category_files = {}
    
    # Group by category
    for item in os.listdir(str(anomaly_path)):
        item_path = anomaly_path / item
        if item_path.is_dir():
            files = list(item_path.rglob("*.jpg"))
            if files:
                category_files[item] = files
                all_anomaly_files.extend([(f, item) for f in files])
    
    # Balanced sampling across categories
    expert_anomaly = []
    remaining_quota = n_anomaly
    
    # Sort categories by size (largest first)
    sorted_categories = sorted(category_files.items(), key=lambda x: len(x[1]), reverse=True)
    
    for category, files in sorted_categories:
        if remaining_quota <= 0:
            break
        
        # Allocate samples proportionally but ensure minimum representation
        category_quota = max(
            min(25, len(files)),  # At most 25 per category
            min(remaining_quota // len(sorted_categories), len(files))  # Proportional
        )
        category_quota = min(category_quota, remaining_quota, len(files))
        
        if category_quota > 0:
            sampled = random.sample(files, category_quota)
            expert_anomaly.extend([(f, category) for f in sampled])
            remaining_quota -= category_quota
            
            logger.info(f"   📂 {category}: {category_quota}/{len(files)} samples")
    
    # Create training samples
    train_samples = []
    
    # Add expert normal samples
    for path in expert_normal:
        train_samples.append({
            'path': str(path),
            'label': 0,
            'category': 'normal',
            'is_expert': True
        })
    
    # Add expert anomaly samples
    for path, category in expert_anomaly:
        train_samples.append({
            'path': str(path),
            'label': 1,
            'category': category,
            'is_expert': True
        })
    
    expert_paths = set(str(p) for p in expert_normal + [p for p, _ in expert_anomaly])
    
    logger.info(f"✅ Expert samples: {len(expert_normal)} normal + {len(expert_anomaly)} anomaly = {len(train_samples)}")
    
    return train_samples, expert_paths


def load_all_test_samples(dataset_path, expert_paths):
    """
    Load all remaining samples for testing
    
    Args:
        dataset_path: Path to dataset
        expert_paths: Set of expert paths to exclude
        
    Returns:
        test_samples: List of test samples
    """
    
    logger = logging.getLogger(__name__)
    logger.info("📊 Loading test samples...")
    
    dataset_path = Path(dataset_path)
    test_samples = []
    
    # Load remaining normal samples
    normal_files = list((dataset_path / "normal").rglob("*.jpg"))
    for path in normal_files:
        if str(path) not in expert_paths:
            test_samples.append({
                'path': str(path),
                'label': 0,
                'category': 'normal',
                'is_expert': False
            })
    
    # Load remaining anomaly samples
    anomaly_path = dataset_path / "anomaly"
    for item in os.listdir(str(anomaly_path)):
        item_path = anomaly_path / item
        if item_path.is_dir():
            files = list(item_path.rglob("*.jpg"))
            for path in files:
                if str(path) not in expert_paths:
                    test_samples.append({
                        'path': str(path),
                        'label': 1,
                        'category': item,
                        'is_expert': False
                    })
    
    # Log test statistics
    normal_count = sum(1 for s in test_samples if s['label'] == 0)
    anomaly_count = sum(1 for s in test_samples if s['label'] == 1)
    
    logger.info(f"✅ Test samples: {len(test_samples)} total")
    logger.info(f"   - Normal: {normal_count}")
    logger.info(f"   - Anomaly: {anomaly_count}")
    
    return test_samples


def create_dataloaders(train_samples, test_samples, config):
    """
    Create optimized data loaders
    
    Args:
        train_samples: Training samples
        test_samples: Test samples
        config: Configuration dict
        
    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """
    
    logger = logging.getLogger(__name__)
    
    # Create datasets
    train_dataset = WoodDefectDataset(train_samples, 'train', config.get('data', {}))
    test_dataset = WoodDefectDataset(test_samples, 'test', config.get('data', {}))
    
    # Training configuration
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('hardware', {}).get('num_workers', 4)
    use_balanced_sampling = config.get('training', {}).get('balanced_sampling', True)
    
    # Create training loader with optional balanced sampling
    if use_balanced_sampling and len(train_samples) > 0:
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        logger.info("✅ Using balanced sampling for training")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
    
    # Test loader (no sampling needed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    logger.info(f"📦 DataLoaders created:")
    logger.info(f"   - Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"   - Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    logger.info(f"   - Batch size: {batch_size}, Workers: {num_workers}")
    
    return train_loader, test_loader


def save_dataset_info(train_samples, test_samples, output_path):
    """Save dataset information for analysis"""
    
    dataset_info = {
        'train_samples': len(train_samples),
        'test_samples': len(test_samples),
        'train_normal': sum(1 for s in train_samples if s['label'] == 0),
        'train_anomaly': sum(1 for s in train_samples if s['label'] == 1),
        'test_normal': sum(1 for s in test_samples if s['label'] == 0),
        'test_anomaly': sum(1 for s in test_samples if s['label'] == 1),
    }
    
    # Category breakdown
    train_categories = {}
    test_categories = {}
    
    for sample in train_samples:
        if sample['label'] == 1:  # Anomalies only
            cat = sample['category']
            train_categories[cat] = train_categories.get(cat, 0) + 1
    
    for sample in test_samples:
        if sample['label'] == 1:  # Anomalies only
            cat = sample['category']
            test_categories[cat] = test_categories.get(cat, 0) + 1
    
    dataset_info['train_anomaly_categories'] = train_categories
    dataset_info['test_anomaly_categories'] = test_categories
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)