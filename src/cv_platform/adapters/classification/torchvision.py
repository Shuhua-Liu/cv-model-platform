"""
Torchvision Classification Adapter - Supports pre-trained classification models from torchvision

Supported models:
- ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
- EfficientNet: efficientnet_b0-b7
- DenseNet: densenet121, densenet169, densenet201
- VGG: vgg11, vgg13, vgg16, vgg19
- MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- Vision Transformer: vit_b_16, vit_b_32, vit_l_16, vit_l_32
"""

import time
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from loguru import logger

from ..base import ClassificationAdapter

try:
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision is not installed; torchvision adapters will be unavailable.")


class TorchvisionAdapter(ClassificationAdapter):
    """Torchvision Classification Model Adapter"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 pretrained: bool = True,
                 num_classes: int = 1000,  # Number of ImageNet Categories
                 batch_size: int = 8,
                 **kwargs):
        """
        Initialize Torchvision Adapter
        
        Args:
            model_path: Model file path or `torchvision://model_name`
            device: Computing device
            pretrained: Whether to use pretrained weights
            num_classes: Number of classes (1000 for ImageNet)
            batch_size: Batch size
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Require install torchvision: pip install torchvision")
        
        super().__init__(model_path, device, **kwargs)
        
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Define the model architecture
        self.model_name = self._determine_model_name()
        
        # Data Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # ImageNet Class Names (Top 10 Examples)
        self.class_names = self._load_class_names()
    
    def _determine_model_name(self) -> str:
        """Determine the model name based on the path"""
        path_str = str(self.model_path).lower()
        
        # Process torchvision:// format
        if path_str.startswith('torchvision://'):
            return path_str.replace('torchvision://', '')
        
        # Inferring from the filename
        filename = self.model_path.name.lower()
        
        # ResNet Series
        if 'resnet18' in filename:
            return 'resnet18'
        elif 'resnet34' in filename:
            return 'resnet34'
        elif 'resnet50' in filename:
            return 'resnet50'
        elif 'resnet101' in filename:
            return 'resnet101'
        elif 'resnet152' in filename:
            return 'resnet152'
        
        # EfficientNet Series
        elif 'efficientnet_b0' in filename:
            return 'efficientnet_b0'
        elif 'efficientnet_b1' in filename:
            return 'efficientnet_b1'
        elif 'efficientnet_b2' in filename:
            return 'efficientnet_b2'
        elif 'efficientnet_b3' in filename:
            return 'efficientnet_b3'
        elif 'efficientnet_b4' in filename:
            return 'efficientnet_b4'
        elif 'efficientnet_b5' in filename:
            return 'efficientnet_b5'
        elif 'efficientnet_b6' in filename:
            return 'efficientnet_b6'
        elif 'efficientnet_b7' in filename:
            return 'efficientnet_b7'
        
        # MobileNet Series
        elif 'mobilenet_v2' in filename:
            return 'mobilenet_v2'
        elif 'mobilenet_v3_large' in filename:
            return 'mobilenet_v3_large'
        elif 'mobilenet_v3_small' in filename:
            return 'mobilenet_v3_small'
        
        # Vision Transformer
        elif 'vit_b_16' in filename:
            return 'vit_b_16'
        elif 'vit_b_32' in filename:
            return 'vit_b_32'
        elif 'vit_l_16' in filename:
            return 'vit_l_16'
        elif 'vit_l_32' in filename:
            return 'vit_l_32'
        
        # Default use of ResNet-50
        logger.warning(f"Unable to determine model architecture from filename: {filename}. Using ResNet50 by default.")
        return 'resnet50'
    
    def _load_class_names(self) -> List[str]:
        """Load Class Name"""
        # Only the first 20 ImageNet categories are provided here as examples
        # In practical applications, the full set of 1000 categories can be loaded from a file or via an API
        imagenet_classes = [
            'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
            'electric_ray', 'stingray', 'cock', 'hen', 'ostrich',
            'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting',
            'robin', 'bulbul', 'jay', 'magpie', 'chickadee'
        ]
        
        # If the number of categories is not 1000, generate a generic category name
        if self.num_classes != 1000:
            return [f'class_{i}' for i in range(self.num_classes)]
        
        # Expand to 1000 categories (the full list should be loaded in practical applications)
        while len(imagenet_classes) < 1000:
            imagenet_classes.append(f'class_{len(imagenet_classes)}')
        
        return imagenet_classes[:self.num_classes]
    
    def load_model(self) -> None:
        """Load classification model"""
        try:
            logger.info(f"Load Torchvision classification model: {self.model_name}")
            
            # First, attempt to load the user's model file directly
            if self.model_path.exists() and not str(self.model_path).startswith('torchvision://'):
                logger.info("Local model file detected; attempting to load directly...")
                
                try:
                    # Directly load the complete model
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    
                    if hasattr(checkpoint, 'state_dict') or not isinstance(checkpoint, dict):
                        self.model = checkpoint
                        if hasattr(self.model, 'to'):
                            self.model = self.model.to(self.device)
                        self.model.eval()
                        self.is_loaded = True
                        logger.info("Directly load the complete model successfully")
                        return
                except Exception as e:
                    logger.info(f"Direct loading failed. Attempting to use torchvision architecture: {e}")
            
            # Create a model using the torchvision architecture
            logger.info(f"Create a model using the torchvision architecture: {self.model_name}")
            
            # Obtain the model constructor
            model_func = self._get_model_function()
            
            if model_func is None:
                raise ValueError(f"Unsupported models: {self.model_name}")
            
            # Create model
            if self.pretrained and self.num_classes == 1000:
                # Use pre-trained weights
                weights = 'DEFAULT'  # The syntax for the new version of torchvision
                try:
                    self.model = model_func(weights=weights)
                except TypeError:
                    # Compatible with older versions of torchvision
                    self.model = model_func(pretrained=True)
                logger.info("Create a model using pre-trained weights")
            else:
                # No pre-training or custom category counts
                try:
                    self.model = model_func(weights=None, num_classes=self.num_classes)
                except TypeError:
                    self.model = model_func(pretrained=False, num_classes=self.num_classes)
                logger.info("Create a Random Initialization Model")
            
            # Load the user's weight file (if available)
            if self.model_path.exists() and not str(self.model_path).startswith('torchvision://'):
                logger.info("Load user weight file...")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
                
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    logger.info("User weight successfully loaded (strict mode)")
                except RuntimeError as e:
                    logger.warning(f"Strict mode loading failed. Attempting non-strict mode: {e}")
                    try:
                        self.model.load_state_dict(state_dict, strict=False)
                        logger.info("User weights successfully loaded (non-strict mode)")
                    except RuntimeError as e:
                        logger.error(f"Weight loading failed: {e}")
                        if not self.pretrained:
                            logger.info("Weights initialized randomly")
            
            # Move to specified device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"TorchvisionClassification model loaded successfully - Architecture: {self.model_name}")
            
        except Exception as e:
            logger.error(f"TorchvisionClassification model loaded failed: {e}")
            raise
    
    def _get_model_function(self):
        """Obtain the model constructor"""
        model_functions = {
            # ResNet Series
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
            
            # EfficientNet Series
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7,
            
            # MobileNet Series
            'mobilenet_v2': models.mobilenet_v2,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'mobilenet_v3_small': models.mobilenet_v3_small,
            
            # DenseNet Series
            'densenet121': models.densenet121,
            'densenet169': models.densenet169,
            'densenet201': models.densenet201,
            
            # VGG Series
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
        }
        
        # Vision Transformer requires special handling
        if 'vit_' in self.model_name:
            if hasattr(models, self.model_name):
                return getattr(models, self.model_name)
            else:
                logger.warning(f"Current version of torchvision does not support this: {self.model_name}")
                return None
        
        return model_functions.get(self.model_name)
    
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """Preprocess input data"""
        if isinstance(input_data, (str, Path)):
            # File path
            image = Image.open(input_data).convert('RGB')
        elif isinstance(input_data, Image.Image):
            # PIL image
            image = input_data.convert('RGB')
        elif isinstance(input_data, np.ndarray):
            # numpy array
            if input_data.ndim == 3 and input_data.shape[2] == 3:
                image = Image.fromarray(input_data)
            else:
                raise ValueError(f"Unsupported numpy array formats: {input_data.shape}")
        else:
            raise ValueError(f"Unsupported input formats: {type(input_data)}")
        
        # Apply preprocessing transformation
        tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return tensor.to(self.device)
    
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray], 
                top_k: int = 5,
                threshold: float = 0.0,
                **kwargs) -> Dict[str, Any]:
        """
        Perform image classification
        
        Args:
            image: Input image
            top_k: Return the top k prediction results
            threshold: Confidence threshold
            
        Returns:
            Classification results dictionary
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Preprocessed input
            input_tensor = self.preprocess(image)
            
            # Execute Inference
            start_time = time.time()
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                
            inference_time = time.time() - start_time
            
            # Post-processing results
            processed_results = self.postprocess(
                probabilities, 
                top_k=top_k,
                threshold=threshold,
                inference_time=inference_time
            )
            
            logger.debug(f"Torchvision Classification Completed - Duration: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"Torchvision classification prediction failed: {e}")
            raise
    
    def postprocess(self, 
                   probabilities: torch.Tensor,
                   top_k: int = 5,
                   threshold: float = 0.0,
                   **kwargs) -> Dict[str, Any]:
        """Post-processing classification output"""
        try:
            # Move to CPU and convert to numpy
            probs = probabilities.squeeze(0).cpu().numpy()  # [num_classes]
            
            # Retrieve the top-k results
            top_k = min(top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:top_k]
            
            # Construct prediction results
            predictions = []
            for idx in top_indices:
                confidence = float(probs[idx])
                if confidence >= threshold:
                    class_name = self.class_names[idx] if idx < len(self.class_names) else f'class_{idx}'
                    predictions.append({
                        'class': class_name,
                        'class_id': int(idx),
                        'confidence': confidence
                    })
            
            # Build result dictionary
            result = {
                'predictions': predictions,
                'top_class': predictions[0]['class'] if predictions else 'unknown',
                'top_confidence': predictions[0]['confidence'] if predictions else 0.0,
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'model_name': self.model_name,
                    'top_k': top_k,
                    'threshold': threshold,
                    'total_classes': len(probs)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Torchvision classification post-processing failed: {e}")
            raise
    
    def predict_batch(self, 
                     images: List[Union[str, Path, Image.Image, np.ndarray]],
                     top_k: int = 5,
                     threshold: float = 0.0,
                     **kwargs) -> List[Dict[str, Any]]:
        """Batch Prediction"""
        if not self.is_loaded:
            self.load_model()
        
        results = []
        
        # Batch processing
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            try:
                # Preprocess batch data
                batch_tensors = []
                for img in batch_images:
                    tensor = self.preprocess(img)
                    batch_tensors.append(tensor.squeeze(0))  # Remove batch dimension
                
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Batch Inference
                start_time = time.time()
                with torch.no_grad():
                    logits = self.model(batch_tensor)
                    probabilities = F.softmax(logits, dim=1)
                
                inference_time = time.time() - start_time
                
                # Process each result
                for j, probs in enumerate(probabilities):
                    result = self.postprocess(
                        probs.unsqueeze(0),
                        top_k=top_k,
                        threshold=threshold,
                        inference_time=inference_time / len(batch_images)
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Batch prediction failed (batch {i//self.batch_size + 1}): {e}")
                # Add incorrect results to maintain consistent list length
                for _ in batch_images:
                    results.append({
                        'predictions': [],
                        'top_class': 'error',
                        'top_confidence': 0.0,
                        'metadata': {'error': str(e)}
                    })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model details"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'classification',
            'framework': 'torchvision',
            'architecture': self.model_name,
            'pretrained': self.pretrained,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'input_size': (224, 224),
            'preprocessing': 'ImageNet normalization'
        })
        
        if self.is_loaded:
            try:
                # Obtain the number of model parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params
                })
                
            except Exception as e:
                logger.debug(f"Failed to retrieve model details: {e}")
        
        return info
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """Model Warmup"""
        if not self.is_loaded:
            self.load_model()
        
        # Create dummy entries for warmup
        dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                with torch.no_grad():
                    _ = self.model(dummy_image)
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
            except Exception as e:
                logger.warning(f"Warm-up operation {i+1} failed: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"Torchvision Classification Model Preheating Complete - Average Duration: {avg_time:.3f}s")
            
            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return super().warmup(num_runs)
