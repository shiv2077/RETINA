
import yaml
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from anomalib.models import WinClip
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class WinClipInferencer:
    """Simplified WinCLIP inferencer using config-based loading"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.model = None
        self.config = None
        self._load_model()
    
    def _load_model(self):
        """Load WinCLIP model from config"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("Loading WinCLIP model from config...")
        
        # Create WinCLIP model
        self.model = WinClip(
            class_name=self.config['model']['class_name'],
            k_shot=self.config['model']['k_shot'],
            scales=tuple(self.config['model']['scales']),
            few_shot_source=self.config['model']['few_shot_source']
        )
        
        # Fix preprocessor
        self.model.pre_processor.test_transform = T.Compose([
            T.Resize((240, 240)),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                       std=(0.26862954, 0.26130258, 0.27577711))
        ])
        
        # Load few-shot reference images if needed
        ref_images = None
        if self.config['model']['k_shot'] > 0:
            print(f"Loading {self.config['model']['k_shot']} few-shot reference images...")
            few_shot_path = Path(self.config['model']['few_shot_source'])
            ref_paths = list(few_shot_path.glob('*.jpg'))[:self.config['model']['k_shot']]
            
            ref_tensors = []
            for path in ref_paths:
                img = Image.open(path).convert('RGB')
                img = self.model.pre_processor.test_transform(img)
                ref_tensors.append(img)
            
            if ref_tensors:
                ref_images = torch.stack(ref_tensors)
                print(f"Loaded {len(ref_tensors)} reference images")
        
        # Setup the model (WinCLIP uses class_name and reference images, not custom prompts)
        self.model.model.setup(self.config['model']['class_name'], ref_images)
        print("WinCLIP model setup complete")
        
        # Set to eval mode and move to GPU
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def predict(self, image: Image.Image):
        """Run prediction on PIL image"""
        # Preprocess image
        image_tensor = self.model.pre_processor.test_transform(image).unsqueeze(0)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(image_tensor)
            
            # Create result object similar to TorchInferencer output
            class PredictionResult:
                def __init__(self, pred_score):
                    self.pred_score = pred_score
            
            # Extract anomaly score
            if isinstance(prediction, dict):
                score = float(prediction.get("pred_score", prediction.get("anomaly_score", 0.0)))
            elif hasattr(prediction, 'pred_score'):
                score = float(prediction.pred_score)
            else:
                score = float(prediction.max())
            
            return PredictionResult(score)