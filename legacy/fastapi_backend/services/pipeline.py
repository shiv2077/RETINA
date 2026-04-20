"""
Pipeline Service
Multi-stage anomaly detection pipeline orchestration.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

try:
    from ..models import PatchCoreModel, BGADModel
    from ..config import config, MODELS_DIR, ANNOTATIONS_DIR
    from .labeling import LabelingService, Annotation
    from .inference import InferenceService
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models import PatchCoreModel, BGADModel
    from config import config, MODELS_DIR, ANNOTATIONS_DIR
    from labeling import LabelingService, Annotation
    from inference import InferenceService


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    # Stage 1: Unsupervised
    patchcore_top_k: int = 100  # Top anomalies for labeling
    patchcore_fast_sampling: bool = True
    
    # Stage 2: Labeling
    min_labels_for_stage2: int = 50  # Minimum labels before supervised training
    active_learning_batch: int = 20  # Samples per active learning round
    
    # Stage 3: Supervised
    bgad_epochs: int = 30
    bgad_patience: int = 10


class MVTecDataset(Dataset):
    """MVTec AD Dataset wrapper."""
    
    def __init__(
        self,
        root: str,
        category: str,
        split: str = "train",
        img_size: int = 224,
        transform: Optional[transforms.Compose] = None
    ):
        self.root = Path(root)
        self.category = category
        self.split = split
        self.img_size = img_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[Path, int, Optional[Path]]]:
        samples = []
        split_dir = self.root / self.category / self.split
        
        if self.split == "train":
            good_dir = split_dir / "good"
            if good_dir.exists():
                for img_path in sorted(good_dir.glob("*.png")):
                    samples.append((img_path, 0, None))
        else:
            for defect_dir in sorted(split_dir.iterdir()):
                if defect_dir.is_dir():
                    is_anomaly = 0 if defect_dir.name == "good" else 1
                    
                    for img_path in sorted(defect_dir.glob("*.png")):
                        mask_path = None
                        if is_anomaly:
                            mask_dir = self.root / self.category / "ground_truth" / defect_dir.name
                            mask_name = img_path.stem + "_mask.png"
                            potential_mask = mask_dir / mask_name
                            if potential_mask.exists():
                                mask_path = potential_mask
                        
                        samples.append((img_path, is_anomaly, mask_path))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        img_path, label, mask_path = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.img_size, self.img_size)
        
        return {
            "image": image,
            "label": label,
            "mask": mask,
            "path": str(img_path)
        }


class LabeledDataset(Dataset):
    """Dataset from labeled annotations."""
    
    def __init__(
        self,
        annotations: List[Annotation],
        img_size: int = 224,
        augment: bool = False
    ):
        self.annotations = annotations
        
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ann = self.annotations[idx]
        image = Image.open(ann.image_path).convert("RGB")
        image = self.transform(image)
        label = 1 if ann.label == "anomaly" else 0
        return image, label


class PipelineService:
    """
    Multi-stage anomaly detection pipeline.
    
    Stage 1: Unsupervised (PatchCore)
    - Train on normal images only
    - Identify top anomalies for labeling
    
    Stage 2: Active Learning (Labeling)
    - Expert labels most uncertain samples
    - Incrementally build labeled dataset
    
    Stage 3: Supervised (BGAD)
    - Train with labeled data
    - Push-pull learning for boundary detection
    """
    
    def __init__(self, pipeline_config: Optional[PipelineConfig] = None):
        self.config = pipeline_config or PipelineConfig()
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Services
        self.inference = InferenceService()
        self.labeling = LabelingService(ANNOTATIONS_DIR)
        
        # Models
        self.patchcore: Optional[PatchCoreModel] = None
        self.bgad: Optional[BGADModel] = None
        
        # State
        self.category: Optional[str] = None
        self.stage: int = 0
        self.metrics: Dict = {}
    
    def run_stage1_unsupervised(
        self,
        category: str,
        data_root: str = None,
        fast_sampling: bool = True
    ) -> Dict:
        """
        Stage 1: Train PatchCore on normal samples.
        
        Returns:
            Training stats and top anomalies for labeling
        """
        self.category = category
        self.stage = 1
        
        data_root = data_root or config.mvtec_path
        
        print(f"\n{'='*60}")
        print(f"STAGE 1: Unsupervised Training (PatchCore) - {category}")
        print(f"{'='*60}")
        
        # Create datasets
        train_dataset = MVTecDataset(data_root, category, "train", config.patchcore.img_size)
        test_dataset = MVTecDataset(data_root, category, "test", config.patchcore.img_size)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )
        
        print(f"Train samples (normal): {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Create and train PatchCore
        self.patchcore = PatchCoreModel(
            backbone=config.patchcore.backbone,
            layers=config.patchcore.layers,
            img_size=config.patchcore.img_size,
            coreset_ratio=config.patchcore.coreset_ratio,
            num_neighbors=config.patchcore.num_neighbors,
            device=str(self.device)
        )
        
        train_stats = self.patchcore.fit(
            train_loader,
            category=category,
            fast_sampling=fast_sampling
        )
        
        # Evaluate
        print("\nEvaluating on test set...")
        eval_results = self._evaluate_patchcore(test_loader)
        
        # Get top anomalies for labeling
        print(f"\nIdentifying top-{self.config.patchcore_top_k} anomalies for labeling...")
        top_anomalies = self.patchcore.get_top_anomalies(
            test_loader,
            top_k=self.config.patchcore_top_k
        )
        
        # Add to labeling queue
        self.labeling.start_session(annotator="expert")
        self.labeling.add_to_queue([
            {
                "image_id": a["image_id"],
                "image_path": a["image_id"],  # Path stored in image_id
                "anomaly_score": a["anomaly_score"],
                "anomaly_map": a["anomaly_map"],
                "ground_truth": a["ground_truth"]
            }
            for a in top_anomalies
        ])
        
        # Save model
        model_path = MODELS_DIR / f"patchcore_{category}.pth"
        self.patchcore.save(model_path)
        
        self.metrics["stage1"] = {
            "category": category,
            "train_stats": train_stats,
            "eval_results": eval_results,
            "top_anomalies_count": len(top_anomalies),
            "model_path": str(model_path)
        }
        
        return self.metrics["stage1"]
    
    def _evaluate_patchcore(self, dataloader) -> Dict:
        """Evaluate PatchCore model."""
        all_scores = []
        all_labels = []
        all_maps = []
        all_masks = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                images = batch["image"].to(self.device)
                labels = batch["label"].numpy()
                masks = batch["mask"].numpy()
                
                scores, maps = self.patchcore.predict(images)
                
                all_scores.extend(scores.cpu().numpy())
                all_labels.extend(labels)
                all_maps.append(maps.numpy())
                all_masks.append(masks)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Image-level AUROC
        image_auroc = roc_auc_score(all_labels, all_scores)
        
        # Pixel-level AUROC (for anomalous images only)
        all_maps = np.concatenate(all_maps, axis=0)
        all_masks = np.concatenate(all_masks, axis=0)
        
        anomaly_mask = all_labels == 1
        if anomaly_mask.sum() > 0:
            pixel_labels = all_masks[anomaly_mask].flatten()
            pixel_scores = all_maps[anomaly_mask].flatten()
            pixel_auroc = roc_auc_score(pixel_labels, pixel_scores) if len(np.unique(pixel_labels)) > 1 else 0.0
        else:
            pixel_auroc = 0.0
        
        return {
            "image_auroc": float(image_auroc),
            "pixel_auroc": float(pixel_auroc),
            "total_samples": len(all_labels),
            "anomaly_samples": int(anomaly_mask.sum())
        }
    
    def run_stage2_labeling(self, batch_size: int = None) -> Dict:
        """
        Stage 2: Active learning labeling session.
        
        Returns next batch of samples to label.
        """
        self.stage = 2
        batch_size = batch_size or self.config.active_learning_batch
        
        print(f"\n{'='*60}")
        print(f"STAGE 2: Active Learning Labeling")
        print(f"{'='*60}")
        
        progress = self.labeling.get_progress()
        print(f"Current progress: {progress['completed']}/{progress['total']} labeled")
        print(f"Stats: {progress['stats']}")
        
        # Get next batch
        batch = []
        for _ in range(batch_size):
            sample = self.labeling.get_next_sample()
            if sample:
                batch.append(sample)
            else:
                break
        
        return {
            "batch": batch,
            "batch_size": len(batch),
            "progress": progress,
            "ready_for_stage3": progress["stats"]["total"] >= self.config.min_labels_for_stage2
        }
    
    def run_stage3_supervised(
        self,
        epochs: int = None,
        validation_split: float = 0.2
    ) -> Dict:
        """
        Stage 3: Train BGAD with labeled data.
        
        Returns:
            Training results
        """
        self.stage = 3
        epochs = epochs or self.config.bgad_epochs
        
        print(f"\n{'='*60}")
        print(f"STAGE 3: Supervised Training (BGAD)")
        print(f"{'='*60}")
        
        # Get labeled data
        annotations = self.labeling.store.list_all()
        
        if len(annotations) < self.config.min_labels_for_stage2:
            return {
                "success": False,
                "error": f"Not enough labels. Have {len(annotations)}, need {self.config.min_labels_for_stage2}"
            }
        
        print(f"Training with {len(annotations)} labeled samples")
        
        # Split into train/val
        from sklearn.model_selection import train_test_split
        labels = [1 if a.label == "anomaly" else 0 for a in annotations]
        
        train_ann, val_ann = train_test_split(
            annotations,
            test_size=validation_split,
            stratify=labels,
            random_state=42
        )
        
        # Create datasets
        train_dataset = LabeledDataset(train_ann, config.bgad.img_size, augment=True)
        val_dataset = LabeledDataset(val_ann, config.bgad.img_size, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Create and train BGAD
        self.bgad = BGADModel(
            backbone=config.bgad.backbone,
            feature_dim=config.bgad.feature_dim,
            margin=config.bgad.margin,
            pull_weight=config.bgad.pull_weight,
            push_weight=config.bgad.push_weight,
            device=str(self.device)
        )
        
        history = self.bgad.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            patience=self.config.bgad_patience
        )
        
        # Optimize threshold
        best_threshold = self.bgad.optimize_threshold(val_loader)
        print(f"Optimized threshold: {best_threshold:.4f}")
        
        # Save model
        model_path = MODELS_DIR / "bgad_model.pth"
        self.bgad.save(model_path)
        
        self.metrics["stage3"] = {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "history": history,
            "threshold": best_threshold,
            "model_path": str(model_path)
        }
        
        return self.metrics["stage3"]
    
    def evaluate_full_pipeline(self, data_root: str = None) -> Dict:
        """
        Evaluate the full pipeline on test data.
        
        Returns comprehensive metrics.
        """
        if self.category is None:
            raise RuntimeError("Pipeline not initialized. Run stage 1 first.")
        
        data_root = data_root or config.mvtec_path
        
        print(f"\n{'='*60}")
        print(f"Full Pipeline Evaluation - {self.category}")
        print(f"{'='*60}")
        
        test_dataset = MVTecDataset(data_root, self.category, "test", config.patchcore.img_size)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
        
        all_labels = []
        patchcore_scores = []
        bgad_scores = []
        ensemble_scores = []
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch["image"].to(self.device)
            labels = batch["label"].numpy()
            
            all_labels.extend(labels)
            
            # PatchCore
            if self.patchcore and self.patchcore.is_fitted:
                pc_scores, _ = self.patchcore.predict(images)
                patchcore_scores.extend(pc_scores.cpu().numpy())
            
            # BGAD
            if self.bgad and self.bgad.center_initialized:
                bgad_result = self.bgad.predict(images)
                bgad_scores.extend(bgad_result["scores"])
        
        results = {"labels": all_labels}
        
        # PatchCore metrics
        if patchcore_scores:
            pc_auroc = roc_auc_score(all_labels, patchcore_scores)
            results["patchcore"] = {
                "auroc": float(pc_auroc),
                "scores": patchcore_scores
            }
            print(f"PatchCore AUROC: {pc_auroc:.4f}")
        
        # BGAD metrics
        if bgad_scores:
            bgad_auroc = roc_auc_score(all_labels, bgad_scores)
            results["bgad"] = {
                "auroc": float(bgad_auroc),
                "scores": bgad_scores
            }
            print(f"BGAD AUROC: {bgad_auroc:.4f}")
        
        # Ensemble (if both available)
        if patchcore_scores and bgad_scores:
            # Normalize and combine
            pc_norm = np.array(patchcore_scores) / (np.max(patchcore_scores) + 1e-8)
            bgad_norm = np.array(bgad_scores) / (np.max(bgad_scores) + 1e-8)
            ensemble = 0.6 * pc_norm + 0.4 * bgad_norm
            
            ensemble_auroc = roc_auc_score(all_labels, ensemble)
            results["ensemble"] = {
                "auroc": float(ensemble_auroc),
                "scores": ensemble.tolist()
            }
            print(f"Ensemble AUROC: {ensemble_auroc:.4f}")
        
        self.metrics["evaluation"] = results
        return results
    
    def get_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            "category": self.category,
            "stage": self.stage,
            "patchcore_trained": self.patchcore is not None and self.patchcore.is_fitted,
            "bgad_trained": self.bgad is not None and self.bgad.center_initialized,
            "labeling_progress": self.labeling.get_progress(),
            "metrics": self.metrics
        }
