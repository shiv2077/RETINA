#!/usr/bin/env python3
"""
RETINA Nightly Retrain Script
=============================

Automated MLOps pipeline that:
1. Loads base training dataset (normal images)
2. Fetches newly labeled annotations from AnnotationStore
3. Combines datasets with healthy sampling ratios
4. Fine-tunes BGAD model on new anomaly data
5. Deploys updated weights to production

This script is designed to run as a cron job (e.g., 2:00 AM daily).
All operations are logged for monitoring.

Usage:
    python scripts/nightly_retrain.py
    
    Or via cron:
    crontab -e
    0 2 * * * cd /home/shiv2077/dev/RETINA && python scripts/nightly_retrain.py >> logs/retrain.log 2>&1
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image, ImageDraw

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.models.bgad import BGADModel
from src.backend.services.labeling import LabelingService, AnnotationStore
from src.backend.config import ANNOTATIONS_DIR, MODELS_DIR

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Create logs directory
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = LOGS_DIR / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Sampling ratio: How many normal images per new anomaly
NORMAL_TO_ANOMALY_RATIO = 25

# Training hyperparameters
RETRAIN_EPOCHS = 10  # Fine-tune for fewer epochs
FINE_TUNE_LR = 0.0001  # Lower learning rate for fine-tuning
BATCH_SIZE = 32

# Thresholds
MIN_ANOMALIES_TO_RETRAIN = 5  # Only retrain if we have at least this many new anomalies
MIN_BBOX_CONFIDENCE = 0.5  # Only use annotations with this confidence

# Paths
PRODUCTION_MODEL_PATH = MODELS_DIR / "bgad_production.pt"
BACKUP_MODEL_PATH = MODELS_DIR / f"bgad_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"

# ============================================================================
# DATASET CLASSES
# ============================================================================

class NormalImageDataset(Dataset):
    """Dataset of normal training images."""
    
    def __init__(self, normal_dirs: Union[str, Path, List[Union[str, Path]]], transform=None, limit: Optional[int] = None):
        if not isinstance(normal_dirs, list):
            normal_dirs = [normal_dirs]
        self.normal_dirs = [Path(d) for d in normal_dirs]
        self.transform = transform
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        self.image_paths = []
        for normal_dir in self.normal_dirs:
            if normal_dir.exists():
                paths = [
                    p for p in normal_dir.rglob('*')
                    if p.suffix.lower() in image_extensions
                ]
                self.image_paths.extend(paths)
        
        # Limit if specified (useful for large datasets)
        if limit:
            self.image_paths = self.image_paths[:limit]
        
        logger.info(f"Loaded {len(self.image_paths)} normal training images from combining provided directories")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Label 0 = normal
        return image, 0


class AnnotationDataset(Dataset):
    """Dataset of newly labeled anomalies from AnnotationStore."""
    
    def __init__(self, annotations: List, transform=None):
        self.annotations = annotations
        self.transform = transform
        
        logger.info(f"Loaded {len(annotations)} annotation(s) for retraining")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Load image
        img_path = Path(annotation.image_path)
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}, skipping")
            # Return black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        else:
            image = Image.open(img_path).convert("RGB")
        
        # Generate mask from bounding boxes
        mask = self._generate_mask(image, annotation)
        
        if self.transform:
            image = self.transform(image)
        
        # Label 1 = anomaly
        return image, 1
    
    @staticmethod
    def _generate_mask(image: Image.Image, annotation) -> Image.Image:
        """
        Generate anomaly mask from bounding boxes.
        
        Returns a binary mask where 1 = anomalous region, 0 = normal.
        """
        mask = Image.new('L', image.size, 0)  # Black background (0 = normal)
        draw = ImageDraw.Draw(mask)
        
        # Draw each bounding box
        for bbox in annotation.bounding_boxes:
            # Bounding box is normalized (0-1), convert to pixel coordinates
            w, h = image.size
            x1 = int(bbox.x * w)
            y1 = int(bbox.y * h)
            x2 = int((bbox.x + bbox.width) * w)
            y2 = int((bbox.y + bbox.height) * h)
            
            # Draw rectangle (white = 255 = anomalous)
            draw.rectangle([x1, y1, x2, y2], fill=255)
        
        return mask


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_base_dataset_paths() -> List[Path]:
    """Find all relevant base datasets to combine MSVPD Unified Dataset correctly."""
    # First priority: check if unified dataset exists
    unified = PROJECT_ROOT / "MSVPD_Unified_Dataset" / "train" / "good"
    if unified.exists():
        logger.info(f"Found unified dataset at: {unified}")
        return [unified]
        
    # Second priority: load both individual datasets
    candidates = [
        PROJECT_ROOT / "decospan_small" / "train" / "good",
        PROJECT_ROOT / "mvtec_anomaly_detection" / "bottle" / "train" / "good",
    ]
    
    found = [p for p in candidates if p.exists()]
    if found:
        logger.info(f"Found multiple datasets to combine: {[str(p) for p in found]}")
        return found
        
    raise FileNotFoundError(
        f"No base datasets found. Unified dataset missing and missing source datasets."
    )


def load_new_annotations(store: AnnotationStore) -> List:
    """
    Load annotations from AnnotationStore.
    
    Filters for:
    - Label = "anomaly" (actual defects)
    - Has bounding boxes (not just image-level labels)
    - metadata["cascade_source"] = True (from cascade queue)
    - Not yet processed by retrain
    """
    annotations = store.store.list_all(label="anomaly")
    
    # Filter: Must have bounding boxes AND be from cascade
    filtered = []
    for ann in annotations:
        has_bboxes = len(ann.bounding_boxes) > 0
        is_cascade = ann.metadata.get("cascade_source", False)
        not_processed = ann.metadata.get("processed_by_retrain", False) is False
        
        if has_bboxes and is_cascade and not_processed:
            filtered.append(ann)
    
    logger.info(f"Found {len(filtered)} new anomaly annotations ready for retraining")
    return filtered


def create_combined_dataloader(
    normal_dataset: Dataset,
    anomaly_dataset: Dataset,
    batch_size: int = 32
) -> DataLoader:
    """
    Create a combined dataloader with healthy sampling ratios.
    
    Strategy:
    - For every N anomalies, sample 25*N normal images
    - This prevents overfitting to new defects
    - Maintains class balance in training batches
    """
    n_anomalies = len(anomaly_dataset)
    n_normal_to_use = min(len(normal_dataset), n_anomalies * NORMAL_TO_ANOMALY_RATIO)
    
    # Create subset of normal images (random sample)
    indices = np.random.choice(len(normal_dataset), size=n_normal_to_use, replace=False)
    normal_subset = torch.utils.data.Subset(normal_dataset, indices)
    
    logger.info(f"Sampling {n_normal_to_use} normal images (ratio {NORMAL_TO_ANOMALY_RATIO}:1)")
    logger.info(f"Total training samples: {n_normal_to_use + n_anomalies}")
    
    # Combine datasets
    combined = ConcatDataset([normal_subset, anomaly_dataset])
    
    # Create dataloader
    dataloader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def load_or_create_model(device: str) -> BGADModel:
    """
    Load existing production model for fine-tuning, or create new one.
    
    If production model exists and is valid, we fine-tune it.
    Otherwise, start fresh.
    """
    model = BGADModel(backbone="resnet18", feature_dim=256, margin=1.0)
    model = model.to(device)
    
    if PRODUCTION_MODEL_PATH.exists():
        try:
            logger.info(f"Loading existing model from {PRODUCTION_MODEL_PATH}")
            checkpoint = torch.load(PRODUCTION_MODEL_PATH, map_location=device)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            
            logger.info("✅ Successfully loaded production model")
        except Exception as e:
            logger.warning(f"Failed to load production model: {e}")
            logger.warning("Starting with fresh model")
    else:
        logger.info("No existing production model found, starting fresh")
    
    return model


def fine_tune_model(
    model: BGADModel,
    train_dataloader: DataLoader,
    device: str,
    epochs: int = 10,
    lr: float = 0.0001
) -> Dict:
    """
    Fine-tune BGAD model on combined dataset.
    
    Strategy:
    - Use lower learning rate (FINE_TUNE_LR) to preserve learned features
    - Freeze backbone initially, only train projection head
    - Monitor loss trends
    """
    logger.info(f"Starting fine-tuning for {epochs} epochs (lr={lr})")
    
    # Freeze backbone, only train projection head
    for param in model.encoder.backbone.parameters():
        param.requires_grad = False
    
    # Optimizer
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    
    # Training loop
    model.train()
    training_history = {
        "epochs": [],
        "losses": [],
        "pull_losses": [],
        "push_losses": []
    }
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_pull = 0.0
        epoch_push = 0.0
        n_batches = 0
        
        progress_bar = range(len(train_dataloader))
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            features = model.encoder(images)
            distances = torch.norm(features - model.center.unsqueeze(0), p=2, dim=1)
            
            # Compute loss
            loss, loss_dict = model.compute_loss(features, distances, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track losses
            epoch_loss += loss_dict["total_loss"]
            epoch_pull += loss_dict["pull_loss"]
            epoch_push += loss_dict["push_loss"]
            n_batches += 1
        
        # Average losses over epoch
        avg_loss = epoch_loss / n_batches
        avg_pull = epoch_pull / n_batches
        avg_push = epoch_push / n_batches
        
        training_history["epochs"].append(epoch + 1)
        training_history["losses"].append(avg_loss)
        training_history["pull_losses"].append(avg_pull)
        training_history["push_losses"].append(avg_push)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Pull: {avg_pull:.4f} | "
            f"Push: {avg_push:.4f}"
        )
    
    logger.info("✅ Fine-tuning completed")
    return training_history


def save_model(model: BGADModel, save_path: Path):
    """Save model weights to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "backbone": "resnet18",
            "feature_dim": 256,
            "margin": 1.0
        },
        "timestamp": datetime.now().isoformat(),
        "trained_for_retraining": True
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"✅ Model saved to {save_path}")


def mark_annotations_processed(
    labeling_service: LabelingService,
    annotations: List
):
    """
    Mark annotations as processed so they aren't trained on every night.
    
    (Optional: Can skip if you want to allow continuous fine-tuning on same data)
    """
    logger.info(f"Marking {len(annotations)} annotations as processed...")
    
    for annotation in annotations:
        annotation.metadata["processed_by_retrain"] = True
        annotation.metadata["last_retrain_used"] = datetime.now().isoformat()
        annotation.updated_at = datetime.now().isoformat()
        labeling_service.store.add(annotation)
    
    logger.info("✅ Annotations marked as processed")


def deploy_model(new_model_path: Path, production_path: Path):
    """
    Deploy new model to production.
    
    Strategy:
    1. Backup existing model (if exists)
    2. Copy new model to production
    3. Verify integrity
    """
    # Backup existing
    if production_path.exists():
        import shutil
        shutil.copy2(production_path, BACKUP_MODEL_PATH)
        logger.info(f"✅ Backed up existing model to {BACKUP_MODEL_PATH}")
    
    # Deploy new
    import shutil
    shutil.copy2(new_model_path, production_path)
    logger.info(f"✅ Deployed new model to {production_path}")
    
    # Verify
    if production_path.exists():
        size_mb = production_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Verified production model ({size_mb:.1f} MB)")
        return True
    else:
        logger.error("❌ Deployment verification failed!")
        return False


# ============================================================================
# MAIN RETRAIN PIPELINE
# ============================================================================

def run_nightly_retrain():
    """Execute the complete nightly retraining pipeline."""
    
    logger.info("=" * 80)
    logger.info("RETINA NIGHTLY RETRAIN PIPELINE STARTED")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    try:
        # ====================================================================
        # STAGE 1: LOAD BASE DATASET & ANNOTATIONS
        # ====================================================================
        logger.info("\n📦 STAGE 1: Loading datasets...")
        
        # Load base normal datasets (combining for full representation)
        base_dataset_paths = get_base_dataset_paths()
        transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        normal_dataset = NormalImageDataset(base_dataset_paths, transform=transform)
        
        # Load new annotations
        annotation_store = AnnotationStore(ANNOTATIONS_DIR)
        labeling_service = LabelingService(ANNOTATIONS_DIR)
        new_annotations = load_new_annotations(annotation_store)
        
        # Check if we have enough new data
        if len(new_annotations) < MIN_ANOMALIES_TO_RETRAIN:
            logger.warning(
                f"⚠️  Only {len(new_annotations)} new annotations. "
                f"Minimum required: {MIN_ANOMALIES_TO_RETRAIN}. Skipping retrain."
            )
            logger.info("=" * 80)
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "new_annotations": len(new_annotations)
            }
        
        # Create annotation dataset
        anomaly_dataset = AnnotationDataset(new_annotations, transform=transform)
        
        # ====================================================================
        # STAGE 2: CREATE COMBINED DATALOADER
        # ====================================================================
        logger.info("\n🔀 STAGE 2: Creating combined dataloader...")
        
        train_dataloader = create_combined_dataloader(
            normal_dataset,
            anomaly_dataset,
            batch_size=BATCH_SIZE
        )
        
        # ====================================================================
        # STAGE 3: FINE-TUNE MODEL
        # ====================================================================
        logger.info("\n⚙️  STAGE 3: Fine-tuning BGAD model...")
        
        model = load_or_create_model(device)
        training_history = fine_tune_model(
            model,
            train_dataloader,
            device=device,
            epochs=RETRAIN_EPOCHS,
            lr=FINE_TUNE_LR
        )
        
        # ====================================================================
        # STAGE 4: DEPLOY MODEL
        # ====================================================================
        logger.info("\n🚀 STAGE 4: Deploying new model...")
        
        # Save to temporary location
        temp_model_path = MODELS_DIR / "bgad_retrained_temp.pt"
        save_model(model, temp_model_path)
        
        # Deploy to production
        deploy_successful = deploy_model(temp_model_path, PRODUCTION_MODEL_PATH)
        
        # Clean up temp
        if temp_model_path.exists():
            temp_model_path.unlink()
        
        # ====================================================================
        # STAGE 5: MARK ANNOTATIONS PROCESSED
        # ====================================================================
        logger.info("\n✅ STAGE 5: Updating metadata...")
        
        mark_annotations_processed(labeling_service, new_annotations)
        
        # ====================================================================
        # COMPLETION
        # ====================================================================
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ NIGHTLY RETRAIN COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Duration: {elapsed:.1f} minutes")
        logger.info(f"New anomalies trained: {len(new_annotations)}")
        logger.info(f"Model deployed: {deploy_successful}")
        logger.info(f"Logfile: {log_file}")
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "duration_minutes": elapsed,
            "new_annotations": len(new_annotations),
            "epochs_trained": RETRAIN_EPOCHS,
            "deploy_successful": deploy_successful,
            "model_backup": str(BACKUP_MODEL_PATH) if BACKUP_MODEL_PATH.exists() else None,
            "training_history": training_history
        }
    
    except Exception as e:
        logger.error(f"\n❌ RETRAIN FAILED: {e}", exc_info=True)
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        logger.error(f"Duration before failure: {elapsed:.1f} minutes")
        logger.error(f"Logfile: {log_file}")
        
        # Try to restore backup if deployment failed
        if BACKUP_MODEL_PATH.exists():
            try:
                import shutil
                logger.warning("Attempting to restore backup model...")
                shutil.copy2(BACKUP_MODEL_PATH, PRODUCTION_MODEL_PATH)
                logger.info("✅ Backup restored successfully")
            except Exception as restore_error:
                logger.error(f"❌ Could not restore backup: {restore_error}")
        
        return {
            "status": "failed",
            "error": str(e),
            "duration_minutes": elapsed,
            "logfile": str(log_file)
        }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    result = run_nightly_retrain()
    
    # Exit with appropriate code
    if result["status"] == "success":
        sys.exit(0)  # Success
    elif result["status"] == "skipped":
        sys.exit(1)  # Skipped (not an error, but notify cron)
    else:  # failed
        sys.exit(2)  # Failure
