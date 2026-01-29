import os
import time
import yaml
import torch
import random
import numpy as np
from pathlib import Path
import shutil
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.models import Padim
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['dataset'].get('seed', 42))
    
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    dataset_path = Path(config['dataset']['path'])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    output_dir = Path(config['trainer']['default_root_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = Padim(
        backbone=config['model']['backbone'],
        layers=config['model']['layers']
    )
    
    datamodule = Folder(
        name=config['dataset']['name'],
        root=str(dataset_path / 'train'),
        normal_dir=config['dataset']['normal_dir'],
        abnormal_dir=config['dataset']['abnormal_dir']
    )
    datamodule.setup()
    
    engine = Engine(
        default_root_dir=str(output_dir),
        enable_checkpointing=config['trainer']['enable_checkpointing']
    )
    
    start_time = time.time()
    
    print("Starting PaDiM training...")
    engine.fit(model, datamodule)
    
    export_mode = config.get('optimization', {}).get('export_mode', 'torch')
    engine.export(
        model=model,
        export_type=export_mode,
        input_size=config['dataset']['image_size'],
        export_root=output_dir / "export",
        model_file_name="padim"
    )
    
    pt_file = output_dir / "export" / "weights" / "torch" / "padim.pt"
    new_pt_file = output_dir / "export" / "padim.pt"
    
    shutil.move(pt_file, new_pt_file)
        
    config_path = output_dir / "export" / "padim.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
       
    print(f"Model exported to {output_dir / 'export'}")
    print(f"Training time: {(time.time() - start_time)/60:.1f} minutes")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")
    
    return engine, model, datamodule


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    train_model("configs/config_padim.yaml")