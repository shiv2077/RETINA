"""
Labeling Service
Roboflow-like annotation system for anomaly detection.
Includes cascade-triggered active learning queue.
"""
import json
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid


class LabelType(str, Enum):
    NORMAL = "normal"
    ANOMALY = "anomaly"
    UNCERTAIN = "uncertain"


class DefectType(str, Enum):
    SCRATCH = "scratch"
    DENT = "dent"
    CRACK = "crack"
    STAIN = "stain"
    HOLE = "hole"
    CONTAMINATION = "contamination"
    DISCOLORATION = "discoloration"
    MISSING_PART = "missing_part"
    DEFORMATION = "deformation"
    OTHER = "other"


@dataclass
class BoundingBox:
    """Bounding box annotation."""
    x: float  # Top-left x (0-1 normalized)
    y: float  # Top-left y (0-1 normalized)
    width: float  # Width (0-1 normalized)
    height: float  # Height (0-1 normalized)
    defect_type: str = "other"
    confidence: float = 1.0


@dataclass
class Annotation:
    """Single image annotation."""
    image_id: str
    image_path: str
    label: str  # normal, anomaly, uncertain
    defect_type: Optional[str] = None
    defect_types: List[str] = field(default_factory=list)
    bounding_boxes: List[BoundingBox] = field(default_factory=list)
    anomaly_score: Optional[float] = None
    confidence: str = "high"  # low, medium, high
    annotator: str = "unknown"
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data["bounding_boxes"] = [asdict(bb) for bb in self.bounding_boxes]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Annotation":
        bboxes = [BoundingBox(**bb) for bb in data.pop("bounding_boxes", [])]
        return cls(**data, bounding_boxes=bboxes)


class AnnotationStore:
    """
    Persistent annotation storage.
    Supports JSON export compatible with common ML frameworks.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.annotations_file = self.storage_path / "annotations.json"
        self.images_dir = self.storage_path / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.annotations: Dict[str, Annotation] = {}
        self._load()
    
    def _load(self):
        """Load annotations from disk."""
        if self.annotations_file.exists():
            with open(self.annotations_file, "r") as f:
                data = json.load(f)
                self.annotations = {
                    k: Annotation.from_dict(v) 
                    for k, v in data.get("annotations", {}).items()
                }
    
    def _save(self):
        """Save annotations to disk."""
        data = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_annotations": len(self.annotations),
            "annotations": {k: v.to_dict() for k, v in self.annotations.items()}
        }
        with open(self.annotations_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def add(self, annotation: Annotation, copy_image: bool = False) -> str:
        """Add or update an annotation."""
        if copy_image and Path(annotation.image_path).exists():
            # Copy image to storage
            src = Path(annotation.image_path)
            dst = self.images_dir / f"{annotation.image_id}{src.suffix}"
            shutil.copy2(src, dst)
            annotation.image_path = str(dst)
        
        annotation.updated_at = datetime.now().isoformat()
        self.annotations[annotation.image_id] = annotation
        self._save()
        return annotation.image_id
    
    def get(self, image_id: str) -> Optional[Annotation]:
        """Get annotation by ID."""
        return self.annotations.get(image_id)
    
    def delete(self, image_id: str) -> bool:
        """Delete an annotation."""
        if image_id in self.annotations:
            del self.annotations[image_id]
            self._save()
            return True
        return False
    
    def list_all(self, label: Optional[str] = None) -> List[Annotation]:
        """List all annotations, optionally filtered by label."""
        annotations = list(self.annotations.values())
        if label:
            annotations = [a for a in annotations if a.label == label]
        return annotations
    
    def get_stats(self) -> Dict:
        """Get annotation statistics."""
        labels = [a.label for a in self.annotations.values()]
        defects = [a.defect_type for a in self.annotations.values() if a.defect_type]
        
        return {
            "total": len(self.annotations),
            "by_label": {
                "normal": labels.count("normal"),
                "anomaly": labels.count("anomaly"),
                "uncertain": labels.count("uncertain")
            },
            "by_defect_type": {d: defects.count(d) for d in set(defects)},
            "with_bboxes": sum(1 for a in self.annotations.values() if a.bounding_boxes)
        }
    
    def export_coco(self, output_path: Path) -> Path:
        """Export annotations in COCO format."""
        coco = {
            "info": {
                "description": "RETINA Anomaly Detection Dataset",
                "version": "1.0",
                "date_created": datetime.now().isoformat()
            },
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "normal"},
                {"id": 1, "name": "anomaly"}
            ]
        }
        
        for idx, (image_id, ann) in enumerate(self.annotations.items()):
            coco["images"].append({
                "id": idx,
                "file_name": Path(ann.image_path).name,
                "width": ann.metadata.get("width", 224),
                "height": ann.metadata.get("height", 224)
            })
            
            coco["annotations"].append({
                "id": idx,
                "image_id": idx,
                "category_id": 1 if ann.label == "anomaly" else 0,
                "attributes": {
                    "defect_type": ann.defect_type,
                    "confidence": ann.confidence
                }
            })
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)
        
        return output_path
    
    def export_yolo(self, output_dir: Path) -> Path:
        """Export annotations in YOLO format."""
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        classes = list(DefectType)
        
        # Write classes.txt
        with open(output_dir / "classes.txt", "w") as f:
            for cls in classes:
                f.write(f"{cls.value}\n")
        
        for ann in self.annotations.values():
            # Copy image
            src = Path(ann.image_path)
            if src.exists():
                shutil.copy2(src, images_dir / src.name)
            
            # Write label file
            label_file = labels_dir / f"{src.stem}.txt"
            with open(label_file, "w") as f:
                for bbox in ann.bounding_boxes:
                    # YOLO format: class x_center y_center width height
                    class_idx = classes.index(DefectType(bbox.defect_type)) if bbox.defect_type in [d.value for d in classes] else len(classes) - 1
                    x_center = bbox.x + bbox.width / 2
                    y_center = bbox.y + bbox.height / 2
                    f.write(f"{class_idx} {x_center} {y_center} {bbox.width} {bbox.height}\n")
        
        return output_dir


class LabelingService:
    """
    Service for managing the labeling workflow.
    Integrates with PatchCore for active learning and Cascade inference.
    
    Features:
    - Thread-safe queue operations (prevents race conditions)
    - Cascade-triggered active learning (routes uncertain predictions)
    - Persistent queue (survives service restarts)
    """
    
    def __init__(self, storage_path: Path):
        self.store = AnnotationStore(storage_path)
        self.labeling_queue: List[Dict] = []
        self.queue_lock = threading.RLock()  # Thread-safe queue access
        self.session_id: Optional[str] = None
        
        # Cascade queue (separate from standard labeling queue)
        self.cascade_queue_path = Path(storage_path) / "cascade_queue.json"
        self.cascade_queue: List[Dict] = []
        self._load_cascade_queue()
    
    def _load_cascade_queue(self):
        """Load cascade queue from disk."""
        try:
            if self.cascade_queue_path.exists():
                with open(self.cascade_queue_path, "r") as f:
                    data = json.load(f)
                    self.cascade_queue = data.get("queue", [])
        except Exception as e:
            print(f"Warning: Could not load cascade queue: {e}")
            self.cascade_queue = []
    
    def _save_cascade_queue(self):
        """Save cascade queue to disk (thread-safe)."""
        try:
            with self.queue_lock:
                data = {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "queue_size": len(self.cascade_queue),
                    "queue": self.cascade_queue
                }
                with open(self.cascade_queue_path, "w") as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving cascade queue: {e}")
    
    def start_session(self, annotator: str = "expert") -> str:
        """Start a new labeling session."""
        self.session_id = f"session_{uuid.uuid4().hex[:8]}"
        self.annotator = annotator
        return self.session_id
    
    def add_to_queue(self, samples: List[Dict]):
        """
        Add samples to the labeling queue.
        Samples should come from PatchCore's get_top_anomalies().
        """
        for sample in samples:
            self.labeling_queue.append({
                "image_id": sample.get("image_id", str(uuid.uuid4().hex[:8])),
                "image_path": sample.get("image_path", ""),
                "anomaly_score": sample.get("anomaly_score", 0.0),
                "anomaly_map": sample.get("anomaly_map"),
                "ground_truth": sample.get("ground_truth"),
                "status": "pending"
            })
        
        # Sort by anomaly score (most anomalous first)
        self.labeling_queue.sort(key=lambda x: x["anomaly_score"], reverse=True)
    
    def get_next_sample(self) -> Optional[Dict]:
        """Get next sample to label."""
        for sample in self.labeling_queue:
            if sample["status"] == "pending":
                sample["status"] = "in_progress"
                return sample
        return None
    
    def submit_label(
        self,
        image_id: str,
        label: str,
        defect_type: Optional[str] = None,
        defect_types: Optional[List[str]] = None,
        bounding_boxes: Optional[List[Dict]] = None,
        confidence: str = "high",
        notes: str = ""
    ) -> Dict:
        """
        Submit a label for an image.
        
        Args:
            image_id: Image identifier
            label: "normal", "anomaly", or "uncertain"
            defect_type: Primary defect type
            defect_types: List of all defect types
            bounding_boxes: List of bounding box annotations
            confidence: Annotator confidence level
            notes: Optional notes
        
        Returns:
            Submission result
        """
        # Find in queue
        queue_item = None
        for item in self.labeling_queue:
            if item["image_id"] == image_id:
                queue_item = item
                break
        
        if not queue_item:
            return {"success": False, "error": "Image not found in queue"}
        
        # Create annotation
        bboxes = []
        if bounding_boxes:
            for bb in bounding_boxes:
                bboxes.append(BoundingBox(
                    x=bb["x"],
                    y=bb["y"],
                    width=bb["width"],
                    height=bb["height"],
                    defect_type=bb.get("defect_type", "other"),
                    confidence=bb.get("confidence", 1.0)
                ))
        
        annotation = Annotation(
            image_id=image_id,
            image_path=queue_item["image_path"],
            label=label,
            defect_type=defect_type,
            defect_types=defect_types or [],
            bounding_boxes=bboxes,
            anomaly_score=queue_item["anomaly_score"],
            confidence=confidence,
            annotator=getattr(self, "annotator", "unknown"),
            notes=notes,
            metadata={
                "session_id": self.session_id,
                "ground_truth": queue_item.get("ground_truth")
            }
        )
        
        # Save annotation
        self.store.add(annotation, copy_image=True)
        
        # Update queue
        queue_item["status"] = "completed"
        
        # Get stats
        stats = self.store.get_stats()
        
        return {
            "success": True,
            "image_id": image_id,
            "label": label,
            "stats": stats,
            "remaining": sum(1 for q in self.labeling_queue if q["status"] == "pending")
        }
    
    def skip_sample(self, image_id: str) -> Dict:
        """Skip a sample without labeling."""
        for item in self.labeling_queue:
            if item["image_id"] == image_id:
                item["status"] = "skipped"
                return {"success": True, "image_id": image_id}
        return {"success": False, "error": "Image not found"}
    
    def get_progress(self) -> Dict:
        """Get labeling progress."""
        total = len(self.labeling_queue)
        completed = sum(1 for q in self.labeling_queue if q["status"] == "completed")
        skipped = sum(1 for q in self.labeling_queue if q["status"] == "skipped")
        pending = sum(1 for q in self.labeling_queue if q["status"] == "pending")
        
        return {
            "total": total,
            "completed": completed,
            "skipped": skipped,
            "pending": pending,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "stats": self.store.get_stats()
        }
    
    # ========================================================================
    # CASCADE QUEUE METHODS (Active Learning from Cascade Router)
    # ========================================================================
    
    def add_to_cascade_queue(
        self,
        image_path: str,
        bgad_score: float,
        vlm_score: Optional[float] = None,
        routing_case: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Add image to cascade annotation queue.
        Called when predict_with_cascade() returns requires_expert_labeling=True.
        
        Args:
            image_path: Path to the image file
            bgad_score: BGAD anomaly score (0-2.0)
            vlm_score: VLM anomaly score if available (0-1.0)
            routing_case: Routing case (A_confident_normal, B_confident_anomaly, C_uncertain_vlm_routed)
            metadata: Additional metadata from cascade response
        
        Returns:
            Queue item info
        
        Thread-safe: Yes (uses queue_lock)
        """
        with self.queue_lock:
            # Check if already in queue
            image_id = Path(image_path).stem if Path(image_path).exists() else str(uuid.uuid4().hex[:8])
            for item in self.cascade_queue:
                if item["image_id"] == image_id and item["status"] == "pending":
                    # Already in queue, skip
                    return {"success": False, "error": "Image already in queue", "image_id": image_id}
            
            # Create queue item
            queue_item = {
                "image_id": image_id,
                "image_path": image_path,
                "bgad_score": bgad_score,
                "vlm_score": vlm_score,
                "routing_case": routing_case or "unknown",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "source": "cascade_inference",
                "metadata": metadata or {}
            }
            
            # Add to queue (insert at front so most recent are first)
            self.cascade_queue.insert(0, queue_item)
            self._save_cascade_queue()
            
            return {
                "success": True,
                "image_id": image_id,
                "queue_position": 0,
                "queue_size": len(self.cascade_queue)
            }
    
    def get_cascade_queue(self, limit: Optional[int] = None) -> Dict:
        """
        Fetch pending cascade annotations.
        Called by frontend to populate annotation studio.
        
        Args:
            limit: Max items to return (default: all pending)
        
        Returns:
            Dictionary with queue items and stats
        
        Thread-safe: Yes (uses queue_lock)
        """
        with self.queue_lock:
            pending = [item for item in self.cascade_queue if item["status"] == "pending"]
            
            if limit:
                pending = pending[:limit]
            
            # Get queue stats
            total = len(self.cascade_queue)
            pending_count = sum(1 for item in self.cascade_queue if item["status"] == "pending")
            labeled_count = sum(1 for item in self.cascade_queue if item["status"] == "labeled")
            skipped_count = sum(1 for item in self.cascade_queue if item["status"] == "skipped")
            
            return {
                "success": True,
                "queue": pending,
                "queue_size": len(pending),
                "stats": {
                    "total_in_queue": total,
                    "pending": pending_count,
                    "labeled": labeled_count,
                    "skipped": skipped_count
                }
            }
    
    def mark_cascade_labeled(
        self,
        image_id: str,
        label: str,
        bounding_boxes: Optional[List[Dict]] = None,
        defect_types: Optional[List[str]] = None,
        notes: str = ""
    ) -> Dict:
        """
        Mark cascade queue item as labeled and create annotation.
        Called when user submits annotation in studio.
        
        Args:
            image_id: Image ID
            label: "normal" or "anomaly"
            bounding_boxes: List of bbox dicts
            defect_types: Defect type labels
            notes: User notes
        
        Returns:
            Submission result
        
        Thread-safe: Yes (uses queue_lock)
        """
        with self.queue_lock:
            # Find in cascade queue
            queue_item = None
            for item in self.cascade_queue:
                if item["image_id"] == image_id:
                    queue_item = item
                    break
            
            if not queue_item:
                return {"success": False, "error": "Image not found in cascade queue"}
            
            # Create annotation
            bboxes = []
            if bounding_boxes:
                for bb in bounding_boxes:
                    bboxes.append(BoundingBox(
                        x=bb["x"],
                        y=bb["y"],
                        width=bb["width"],
                        height=bb["height"],
                        defect_type=bb.get("defect_type", "other"),
                        confidence=bb.get("confidence", 1.0)
                    ))
            
            annotation = Annotation(
                image_id=image_id,
                image_path=queue_item["image_path"],
                label=label,
                defect_type=defect_types[0] if defect_types else None,
                defect_types=defect_types or [],
                bounding_boxes=bboxes,
                anomaly_score=queue_item.get("bgad_score"),
                confidence="high",
                annotator=getattr(self, "annotator", "expert"),
                notes=notes,
                metadata={
                    "session_id": self.session_id,
                    "cascade_source": True,
                    "bgad_score": queue_item.get("bgad_score"),
                    "vlm_score": queue_item.get("vlm_score"),
                    "routing_case": queue_item.get("routing_case")
                }
            )
            
            # Save annotation
            self.store.add(annotation, copy_image=True)
            
            # Update queue status
            queue_item["status"] = "labeled"
            queue_item["labeled_at"] = datetime.now().isoformat()
            self._save_cascade_queue()
            
            return {
                "success": True,
                "image_id": image_id,
                "label": label,
                "remaining_in_queue": sum(1 for item in self.cascade_queue if item["status"] == "pending")
            }
    
    def skip_cascade_item(self, image_id: str) -> Dict:
        """
        Skip a cascade queue item without labeling.
        
        Thread-safe: Yes (uses queue_lock)
        """
        with self.queue_lock:
            for item in self.cascade_queue:
                if item["image_id"] == image_id:
                    item["status"] = "skipped"
                    item["skipped_at"] = datetime.now().isoformat()
                    self._save_cascade_queue()
                    return {
                        "success": True,
                        "image_id": image_id,
                        "remaining_in_queue": sum(1 for x in self.cascade_queue if x["status"] == "pending")
                    }
            
            return {"success": False, "error": "Image not found in cascade queue"}
    
    def get_cascade_stats(self) -> Dict:
        """
        Get cascade queue statistics.
        """
        with self.queue_lock:
            total = len(self.cascade_queue)
            pending = sum(1 for item in self.cascade_queue if item["status"] == "pending")
            labeled = sum(1 for item in self.cascade_queue if item["status"] == "labeled")
            skipped = sum(1 for item in self.cascade_queue if item["status"] == "skipped")
            
            # Get average scores
            pending_items = [item for item in self.cascade_queue if item["status"] == "pending"]
            avg_bgad_score = sum(item["bgad_score"] for item in pending_items) / len(pending_items) if pending_items else 0
            
            return {
                "total_queued": total,
                "pending": pending,
                "labeled": labeled,
                "skipped": skipped,
                "avg_bgad_score": round(avg_bgad_score, 4),
                "annotation_store_stats": self.store.get_stats()
            }
    
    def export(self, format: str = "json", output_path: Optional[Path] = None) -> Path:
        """Export annotations."""
        if output_path is None:
            output_path = self.store.storage_path / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if format == "coco":
            return self.store.export_coco(output_path / "annotations.json")
        elif format == "yolo":
            return self.store.export_yolo(output_path)
        else:
            # Default JSON
            return self.store.annotations_file
