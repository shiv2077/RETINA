import torch
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Import the model
from src.backend.models.bgad import BGADModel

MODEL_PATH = "output/bgad_production.pt"
DATASET_PATH = "MSVPD_Unified_Dataset"

class EvaluationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []
        
        # We need both normal and anomaly images to calculate AUROC
        # This assumes the dataset has 'good' and other defect folders inside each category
        
        for category_dir in self.root_dir.iterdir():
            if not category_dir.is_dir(): continue
                
            # Usually MVTec format has 'test' folder with 'good' and defect subfolders
            test_dir = category_dir / 'test'
            if not test_dir.exists():
                # Or maybe it's flat structure for training, let's look for standard patterns
                continue
                
            for class_dir in test_dir.iterdir():
                if not class_dir.is_dir(): continue
                
                is_anomaly = 0 if class_dir.name == 'good' else 1
                
                for img_path in class_dir.glob('*.*'):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        self.samples.append((str(img_path), is_anomaly))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, is_anomaly = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image), is_anomaly

def evaluate():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = EvaluationDataset(DATASET_PATH)
    
    if len(dataset) == 0:
        print("Error: Could not find structured evaluation data (test/good and test/defects).")
        print("Let's try a simpler approach assuming standard MVTec structure.")
        return
        
    print(f"Found {len(dataset)} evaluation images (Normal: {sum(1 for s in dataset.samples if s[1] == 0)}, Anomaly: {sum(1 for s in dataset.samples if s[1] == 1)})")
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model = BGADModel(device=device, backbone="resnet18")
    model.load(MODEL_PATH)
    model.model.eval()
    
    y_true = []
    y_scores = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            result = model.predict(images)
            
            y_true.extend(labels.numpy())
            y_scores.extend(result["scores"])
            
            if i % 10 == 0:
                print(f"Processed batch {i+1}/{len(loader)}")
                
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    auroc = roc_auc_score(y_true, y_scores)
    print(f"\n======================================")
    print(f"🚀 FINAL AUROC SCORE: {auroc:.4f} ({auroc*100:.2f}%)")
    print(f"======================================")
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('BGAD Model ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('output/evaluation/roc_curve.png')
    print("ROC curve saved to output/evaluation/roc_curve.png")

if __name__ == "__main__":
    evaluate()
