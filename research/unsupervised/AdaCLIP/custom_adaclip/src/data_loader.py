# Data loader for defect dataset
import random
from pathlib import Path

def load_evaluation_dataset(dataset_path, max_samples=None):
    """Load dataset with labels for evaluation mode"""
    print("Loading defect dataset...")
    
    dataset_path = Path(dataset_path)
    normal_path = dataset_path / "normal"
    anomaly_path = dataset_path / "anomaly"
    
    # Check if paths exist
    if not normal_path.exists():
        raise FileNotFoundError(f"Normal path not found: {normal_path}")
    if not anomaly_path.exists():
        raise FileNotFoundError(f"Anomaly path not found: {anomaly_path}")
    
    # Supported image formats
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    
    # Get all normal images
    normal_images = []
    for ext in image_extensions:
        normal_images.extend(normal_path.glob(ext))
    
    # Get all anomaly images with defect types
    anomaly_images = []
    
    # Check if anomaly folder has subfolders or direct images
    has_subfolders = any(item.is_dir() for item in anomaly_path.iterdir())
    
    if has_subfolders:
        # Subfolders structure: anomaly/defect_type/images
        for defect_folder in anomaly_path.iterdir():
            if defect_folder.is_dir():
                for ext in image_extensions:
                    for img_path in defect_folder.glob(ext):
                        anomaly_images.append((img_path, defect_folder.name))
    else:
        # Direct images in anomaly folder
        for ext in image_extensions:
            for img_path in anomaly_path.glob(ext):
                anomaly_images.append((img_path, "anomaly"))
    
    # Create evaluation data: (image_path, label, defect_type)
    eval_data = []
    eval_data.extend([(img, 0, "normal") for img in normal_images])
    eval_data.extend([(img, 1, defect_type) for img, defect_type in anomaly_images])
    
    # Shuffle for random order
    random.shuffle(eval_data)
    
    # Limit samples if specified
    if max_samples is not None and max_samples > 0:
        eval_data = eval_data[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    # Count final amounts
    normal_count = sum(1 for _, label, _ in eval_data if label == 0)
    anomaly_count = sum(1 for _, label, _ in eval_data if label == 1)
    
    print(f"Normal images: {normal_count}")
    print(f"Anomaly images: {anomaly_count}")
    print(f"Total samples: {len(eval_data)}")
    
    return eval_data

def load_inference_dataset(dataset_path, max_samples=None):
    """Load unlabeled dataset for inference mode"""
    print("Loading unlabeled dataset for inference...")
    
    dataset_path = Path(dataset_path)
    
    # Supported image formats
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    
    # Get all images from dataset
    image_paths = []
    
    # Try flat structure first
    for ext in image_extensions:
        image_paths.extend(dataset_path.glob(ext))
    
    # Try normal folder structure
    normal_path = dataset_path / "normal"
    if normal_path.exists():
        for ext in image_extensions:
            image_paths.extend(normal_path.glob(ext))
    
    # Try anomaly folder structure (but treat as unlabeled)
    anomaly_path = dataset_path / "anomaly"
    if anomaly_path.exists():
        # Check if has subfolders or direct images
        has_subfolders = any(item.is_dir() for item in anomaly_path.iterdir())
        
        if has_subfolders:
            for defect_folder in anomaly_path.iterdir():
                if defect_folder.is_dir():
                    for ext in image_extensions:
                        image_paths.extend(defect_folder.glob(ext))
        else:
            # Direct images in anomaly folder
            for ext in image_extensions:
                image_paths.extend(anomaly_path.glob(ext))
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {dataset_path}")
    
    # Shuffle for random order
    random.shuffle(image_paths)
    
    # Limit samples if specified
    if max_samples is not None and max_samples > 0:
        image_paths = image_paths[:max_samples]
        print(f"Limited to {max_samples} samples for testing")
    
    # Create inference data: (image_path,)
    inference_data = [(img,) for img in image_paths]
    
    print(f"Total images for inference: {len(inference_data)}")
    
    return inference_data