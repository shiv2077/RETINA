# Model loader for AdaCLIP
import os
import sys
import torch
from pathlib import Path

def setup_adaclip_imports(adaclip_repo_path):
    """Setup AdaCLIP repository imports"""
    adaclip_path = Path(adaclip_repo_path)
    
    # Change to AdaCLIP directory and add to path
    os.chdir(adaclip_path)
    sys.path.insert(0, str(adaclip_path))
    
    print(f"AdaCLIP repo: {adaclip_path}")

def load_adaclip_model(config, device):
    """Load pre-trained AdaCLIP model"""
    print("Loading AdaCLIP model...")
    
    # Import AdaCLIP components with error handling
    try:
        from method.adaclip import AdaCLIP
        from method.trainer import AdaCLIP_Trainer
    except ImportError as e:
        print(f"Failed to import AdaCLIP: {e}")
        print("This might be a torch/torchvision version issue.")
        print("Try: pip install torch torchvision --force-reinstall")
        raise
    
    # Model configuration
    model_cfg = config['model']
    
    # Calculate feature hierarchy
    n_layers = model_cfg['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]
    
    # Initialize AdaCLIP trainer
    model = AdaCLIP_Trainer(
        backbone=model_cfg['backbone'],
        feat_list=features_list,
        input_dim=model_cfg['vision_cfg']['width'],
        output_dim=model_cfg['embed_dim'],
        learning_rate=float(model_cfg['learning_rate']),  # Convert to float
        device=device,
        image_size=model_cfg['image_size'],
        prompting_depth=model_cfg['prompting_depth'],
        prompting_length=model_cfg['prompting_length'],
        prompting_type=model_cfg['prompting_type'],
        prompting_branch=model_cfg['prompting_branch'],
        use_hsf=model_cfg['use_hsf'],
        k_clusters=model_cfg['k_clusters']
    )
    
    # Load pre-trained weights
    weights_path = config['paths']['weights']
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print(f"Model loaded: {model_cfg['backbone']} with {features_list}")
    print(f"Prompting length: {model_cfg['prompting_length']}")
    
    return model

def get_prompts(config):
    """Get normal and anomaly prompts from config"""
    prompts = config['prompts']
    normal_prompt = prompts['normal']
    anomaly_prompt = prompts['anomaly']
    
    print(f"Normal prompt: '{normal_prompt}'")
    print(f"Anomaly prompt: '{anomaly_prompt}'")
    
    return normal_prompt, anomaly_prompt