import time
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import os

# Clean local import (since you are in your project root)
from src.backend.models.bgad import BGADModel

# Local Ubuntu Paths
dataset_path = "./MSVPD_Unified_Dataset"
final_model_save_path = "./output/bgad_production.pt"

class DirectoryImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = Path(directory)
        self.transform = transform
        self.samples = []
        
        # Crawl through subdirectories strictly looking for 'good' vs other names
        for class_dir in self.directory.iterdir():
            if not class_dir.is_dir(): continue
            
            # Label 'good' as 0 (Normal), anything else as 1 (Anomaly)
            label = 0 if class_dir.name == 'good' else 1
            
            for img_path in class_dir.glob('*.*'):
                if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                    self.samples.append((str(img_path), label))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Crawling local dataset folder...")
datasets = []

# Examine MVTec and Decospan subsets
for dataset_dir in Path(dataset_path).iterdir():
    if not dataset_dir.is_dir(): continue
    
    if dataset_dir.name == 'MVTec':
        # MVTec structure: category/train/good and category/test/defects
        for category_dir in dataset_dir.iterdir():
            if not category_dir.is_dir(): continue
            
            for split in ['train', 'test']:
                split_dir = category_dir / split
                if split_dir.exists():
                    ds = DirectoryImageDataset(split_dir, transform=transform)
                    if len(ds) > 0:
                        datasets.append(ds)
                        
    elif dataset_dir.name == 'Decospan':
        # Decospan structure: train/good and test/defects
        for split in ['train', 'test']:
            split_dir = dataset_dir / split
            if split_dir.exists():
                ds = DirectoryImageDataset(split_dir, transform=transform)
                if len(ds) > 0:
                    datasets.append(ds)

train_dataset_raw = ConcatDataset(datasets)
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataloader combined. Total samples: {len(train_dataset_raw)}")

# Train/Test Split (80% Train, 20% Test)
train_size = int(0.8 * len(train_dataset_raw))
test_size = len(train_dataset_raw) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset_raw, [train_size, test_size])
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Random Split: {train_size} training samples, {test_size} holdout test samples.")

# RTX HARDWARE FIX: Dropped batch_size to 64 to prevent CUDA Out of Memory
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4, 
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing BGAD Model on {device}...")

bgad_model = BGADModel(
    device=device,
    backbone="resnet18",
    margin=1.0,           
    pull_weight=1.0,      
    push_weight=2.0       
) 

epochs = 15
lr = 0.0001
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Fast Local Training Loop...")
train_start = time.time()

bgad_model.fit(train_loader, epochs=epochs, lr=lr)

print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training Completed in {time.time() - train_start:.2f} seconds.")

os.makedirs(os.path.dirname(final_model_save_path), exist_ok=True)
bgad_model.save(final_model_save_path)
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: Model saved securely to {final_model_save_path}")

# Evaluate Model Contextually
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Running AUROC Evaluation on Test Set...")
from sklearn.metrics import roc_auc_score
import numpy as np

bgad_model.eval()
y_true = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        scores = bgad_model.predict(images)
        y_true.extend(labels.numpy())
        y_scores.extend(scores.tolist() if isinstance(scores, np.ndarray) else [scores])

y_true = np.array(y_true)
y_scores = np.array(y_scores)
auroc = roc_auc_score(y_true, y_scores)

print(f"\n======================================")
print(f"🚀 FINAL AUROC SCORE: {auroc:.4f} ({auroc*100:.2f}%)")
print(f"======================================")
