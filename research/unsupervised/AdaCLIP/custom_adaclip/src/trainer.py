#!/usr/bin/env python3
"""
AdaCLIP Fine-tuning Script
Fine-tune pre-trained AdaCLIP on custom dataset
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

# ============================================================================
# DATASET PATHS - FILL IN YOUR PATHS HERE
# ============================================================================
TRAIN_DATA_PATH = "/scratch/leuven/369/vsc36963/Vakantiejob/Decospan/D04_Model_AdaCLIP/AdaCLIP/dataset/Potatoes/train"  # Should contain: normal/
TEST_DATA_PATH = "/scratch/leuven/369/vsc36963/Vakantiejob/Decospan/D04_Model_AdaCLIP/AdaCLIP/dataset/Potatoes/test"    # Should contain: normal/ and anomaly/
ADACLIP_REPO = "/scratch/leuven/369/vsc36963/Vakantiejob/Decospan/D04_Model_AdaCLIP/AdaCLIP"        # AdaCLIP repository path
PRETRAINED_WEIGHTS = "/scratch/leuven/369/vsc36963/Vakantiejob/Decospan/D04_Model_AdaCLIP/AdaCLIP/weights/pretrained_all.pth"  # Pre-trained weights
OUTPUT_WEIGHTS = "/scratch/leuven/369/vsc36963/Vakantiejob/Decospan/D04_Model_AdaCLIP/AdaCLIP/weights/potato_finetuned.pth"    # Where to save fine-tuned model

# Training config
TRAINING_CONFIG = {
    "epochs": 20,
    "learning_rate": 1e-4,
    "batch_size": 4,     # Reduced from 16 to 4
    "save_every": 5,
    "k_clusters": 30,
    "image_size": 518,
    # Prompts for training
    "normal_prompt": "a photo of clean potatoes on industrial conveyor belt",
    "anomaly_prompt": "a photo of foreign objects mixed with potatoes on conveyor belt"
}

class PotatoDataset(Dataset):
    """Dataset for potato anomaly detection training"""
    
    def __init__(self, data_path, transform=None, is_training=True):
        self.data_path = Path(data_path)
        self.transform = transform
        self.is_training = is_training
        
        # Load image paths and labels
        self.samples = []
        
        if is_training:
            # Training: only normal images
            normal_path = self.data_path / "normal"
            if normal_path.exists():
                for img_path in normal_path.glob("*.jpg"):
                    self.samples.append((img_path, 0, "normal"))  # 0 = normal
        else:
            # Testing: normal and anomaly
            normal_path = self.data_path / "normal"
            anomaly_path = self.data_path / "anomaly"
            
            if normal_path.exists():
                for img_path in normal_path.glob("*.jpg"):
                    self.samples.append((img_path, 0, "normal"))
                    
            if anomaly_path.exists():
                for img_path in anomaly_path.glob("*.jpg"):
                    self.samples.append((img_path, 1, "anomaly"))
        
        print(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, defect_type = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'defect_type': defect_type,
            'path': str(img_path)
        }

def setup_model_and_optimizer(config, device):
    """Setup AdaCLIP model and optimizer"""
    
    # Setup AdaCLIP imports
    sys.path.insert(0, ADACLIP_REPO)
    os.chdir(ADACLIP_REPO)
    
    # Set cache directory
    os.makedirs(os.path.dirname(PRETRAINED_WEIGHTS).replace('weights', 'cache'), exist_ok=True)
    os.environ['TORCH_HOME'] = os.path.dirname(PRETRAINED_WEIGHTS).replace('weights', 'cache')
    
    from method.adaclip import AdaCLIP
    from method.trainer import AdaCLIP_Trainer
    
    # Model configuration
    n_layers = 24  # ViT-L/14
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]
    
    # Initialize model
    model = AdaCLIP_Trainer(
        backbone='ViT-L/14',
        feat_list=features_list,
        input_dim=1024,
        output_dim=768,
        learning_rate=config['learning_rate'],
        device=device,
        image_size=config['image_size'],
        prompting_depth=4,
        prompting_length=5,
        prompting_type='SD',
        prompting_branch='VL',
        use_hsf=True,
        k_clusters=config['k_clusters']
    )
    
    # Load pre-trained weights
    print(f"Loading pre-trained weights from: {PRETRAINED_WEIGHTS}")
    checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    
    # Set to training mode
    model.train()
    
    # Optimizer - fine-tune with lower learning rate  
    # Only optimize specific parameters to avoid overfitting
    params_to_optimize = []
    
    # Add prompting parameters (these are the main learnable parts)
    for name, param in model.named_parameters():
        if 'prompt' in name.lower() or 'adapter' in name.lower():
            params_to_optimize.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False  # Freeze most parameters
    
    # If no specific parameters found, use all parameters but freeze backbone
    if len(params_to_optimize) == 0:
        for name, param in model.named_parameters():
            if 'visual' not in name.lower() and 'transformer' not in name.lower():
                params_to_optimize.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    optimizer = optim.AdamW(params_to_optimize, lr=config['learning_rate'], weight_decay=0.01)
    
    print(f"Model loaded with k_clusters={config['k_clusters']}")
    print(f"Optimizing {len(params_to_optimize)} parameter groups")
    
    return model, optimizer

def train_one_epoch(model, dataloader, optimizer, normal_prompt, anomaly_prompt, device, epoch):
    """Train for one epoch - process images individually to avoid tensor size issues"""
    model.train()
    total_loss = 0
    valid_batches = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        torch.cuda.empty_cache()
        
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        batch_losses = []
        
        try:
            # Process each image individually to avoid tensor size mismatch
            for i in range(len(images)):
                if labels[i] == 0:  # Only process normal images
                    single_image = images[i:i+1]  # Keep batch dimension
                    
                    try:
                        # Forward pass with single image
                        normal_result = model.clip_model(single_image, [normal_prompt], aggregation=True)
                        anomaly_result = model.clip_model(single_image, [anomaly_prompt], aggregation=True)
                        
                        # Extract scores
                        _, normal_score = normal_result[:2]
                        _, anomaly_score = anomaly_result[:2]
                        
                        # Loss calculation
                        score_diff = normal_score - anomaly_score
                        margin = 1.0
                        img_loss = torch.clamp(margin - score_diff, min=0)
                        batch_losses.append(img_loss)
                        
                    except Exception as e:
                        continue  # Skip this image
            
            # Calculate batch loss
            if len(batch_losses) > 0:
                loss = torch.mean(torch.stack(batch_losses))
                
                if not torch.isnan(loss) and loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    valid_batches += 1
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'valid': valid_batches})
                else:
                    progress_bar.set_postfix({'loss': 'nan', 'valid': valid_batches})
            else:
                progress_bar.set_postfix({'loss': 'no_valid', 'valid': valid_batches})
                
        except Exception as e:
            progress_bar.set_postfix({'loss': 'batch_error', 'valid': valid_batches})
            continue
    
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f} ({valid_batches}/{len(dataloader)} valid batches)")
    
    return avg_loss

def evaluate_model(model, test_dataloader, normal_prompt, anomaly_prompt, device):
    """Evaluate model on test set"""
    model.eval()
    
    all_scores = []
    all_labels = []
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            images = batch['image'].to(device)
            labels = batch['label'].cpu().numpy()
            
            # Get anomaly scores
            for i, image in enumerate(images):
                try:
                    img_tensor = image.unsqueeze(0)
                    
                    # Get scores for both prompts
                    normal_result = model.clip_model(img_tensor, [normal_prompt], aggregation=True)
                    anomaly_result = model.clip_model(img_tensor, [anomaly_prompt], aggregation=True)
                    
                    # Extract scores
                    _, normal_score = normal_result[:2]
                    _, anomaly_score = anomaly_result[:2]
                    
                    # Combined score with sigmoid normalization
                    combined_score = normal_score.item() - anomaly_score.item()
                    normalized_score = torch.sigmoid(torch.tensor(combined_score)).item()
                    
                    all_scores.append(normalized_score)
                    all_labels.append(labels[i])
                    
                except Exception as e:
                    print(f"Evaluation failed for sample {i}: {e}")
                    continue
    
    # Calculate metrics
    if len(all_scores) > 0:
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        scores = np.array(all_scores)
        labels = np.array(all_labels)
        
        auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0
        ap = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0
        
        print(f"Test Results:")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  Average Precision: {ap:.4f}")
        print(f"  Score range: {scores.min():.4f} - {scores.max():.4f}")
        
        return auroc, ap, scores, labels
    else:
        print("No valid test results")
        return 0, 0, [], []

def main():
    """Main training function"""
    print("🚀 AdaCLIP Fine-tuning on Potato Dataset")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Validate paths
    train_path = Path(TRAIN_DATA_PATH)
    test_path = Path(TEST_DATA_PATH)
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    
    # Setup model
    model, optimizer = setup_model_and_optimizer(TRAINING_CONFIG, device)
    
    # Create datasets
    train_dataset = PotatoDataset(TRAIN_DATA_PATH, transform=model.preprocess, is_training=True)
    test_dataset = PotatoDataset(TEST_DATA_PATH, transform=model.preprocess, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Training loop
    best_auroc = 0
    training_history = []
    
    for epoch in range(1, TRAINING_CONFIG['epochs'] + 1):
        print(f"\n--- Epoch {epoch}/{TRAINING_CONFIG['epochs']} ---")
        
        # Train
        avg_loss = train_one_epoch(
            model, train_loader, optimizer,
            TRAINING_CONFIG['normal_prompt'],
            TRAINING_CONFIG['anomaly_prompt'],
            device, epoch
        )
        
        # Evaluate every few epochs
        if epoch % TRAINING_CONFIG['save_every'] == 0 or epoch == TRAINING_CONFIG['epochs']:
            auroc, ap, scores, labels = evaluate_model(
                model, test_loader,
                TRAINING_CONFIG['normal_prompt'],
                TRAINING_CONFIG['anomaly_prompt'],
                device
            )
            
            # Save best model
            if auroc > best_auroc:
                best_auroc = auroc
                print(f"New best AUROC: {best_auroc:.4f} - Saving model...")
                torch.save(model.state_dict(), OUTPUT_WEIGHTS)
            
            # Save training history
            training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'auroc': auroc,
                'ap': ap
            })
    
    # Save training log
    log_path = Path(OUTPUT_WEIGHTS).parent / "training_log.json"
    with open(log_path, 'w') as f:
        json.dump({
            'config': TRAINING_CONFIG,
            'best_auroc': best_auroc,
            'history': training_history
        }, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Model saved: {OUTPUT_WEIGHTS}")
    print(f"Training log: {log_path}")

if __name__ == "__main__":
    main()