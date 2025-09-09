"""
DINOv3 Feature Extraction Adapter

DINOv3 is a self-supervised vision transformer for feature extraction
Supports various downstream tasks like classification, segmentation, detection
"""

import time
from typing import Dict, Any, Union, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger

from ..base import FeatureExtractionAdapter

try:
    import torchvision.transforms as T
    from torchvision.models import vision_transformer
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False
    logger.warning("torchvision not available for DINOv3")


class DINOv3Adapter(FeatureExtractionAdapter):
    """DINOv3 model adapter for feature extraction"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 patch_size: int = 14,
                 image_size: int = 518,
                 **kwargs):
        """
        Initialize DINOv3 adapter
        
        Args:
            model_path: Path to DINOv3 model weights
            device: Computing device
            patch_size: Patch size (14 for most models)
            image_size: Input image size (518 recommended)
        """
        if not DINOV3_AVAILABLE:
            raise ImportError("torchvision required for DINOv3: pip install torchvision")
        
        super().__init__(model_path, device, **kwargs)
        
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Model and preprocessing
        self.model = None
        self.transform = None
        
        # Determine model variant from path
        self.model_variant = self._determine_variant()
    
    def _determine_variant(self) -> str:
        """Determine DINOv3 model variant from path"""
        path_str = str(self.model_path).lower()
        
        if 'vits' in path_str:
            return 'vits14'
        elif 'vitb' in path_str:
            return 'vitb14'
        elif 'vitl' in path_str:
            return 'vitl14'
        elif 'vitg' in path_str:
            return 'vitg14'
        else:
            return 'vitl14'  # Default to large
    
    def load_model(self) -> None:
        """Load DINOv3 model"""
        try:
            logger.info(f"Loading DINOv3 model: {self.model_variant}")
            
            # Load model using torch hub or local weights
            if self.model_path.exists():
                # Load from local file
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                # Create model architecture based on variant
                if self.model_variant == 'vits14':
                    self.model = self._create_vit_small()
                elif self.model_variant == 'vitb14':
                    self.model = self._create_vit_base()
                elif self.model_variant == 'vitl14':
                    self.model = self._create_vit_large()
                else:
                    raise ValueError(f"Unknown variant: {self.model_variant}")
                
                # Load weights
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                # Load from torch hub
                self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{self.model_variant}')
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Create preprocessing transform
            self.transform = T.Compose([
                T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.is_loaded = True
            logger.info("DINOv3 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load DINOv3 model: {e}")
            raise
    
    def _create_vit_small(self):
        """Create ViT-Small architecture"""
        return vision_transformer.vit_b_16(weights=None)  # Placeholder - implement proper DINOv3 architecture
    
    def _create_vit_base(self):
        """Create ViT-Base architecture"""
        return vision_transformer.vit_b_16(weights=None)
    
    def _create_vit_large(self):
        """Create ViT-Large architecture"""
        return vision_transformer.vit_l_16(weights=None)
    
    def extract_features(self, 
                        images: Union[str, Path, Image.Image, List[Image.Image]],
                        return_cls_token: bool = True,
                        return_patch_tokens: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract features from images using DINOv3
        
        Args:
            images: Input image(s)
            return_cls_token: Return CLS token features
            return_patch_tokens: Return patch token features
            
        Returns:
            Dictionary containing extracted features
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Process input images
        if isinstance(images, (str, Path)):
            images = [Image.open(images).convert('RGB')]
        elif isinstance(images, Image.Image):
            images = [images]
        elif not isinstance(images, list):
            raise ValueError("Images must be path, PIL Image, or list of PIL Images")
        
        # Preprocess images
        batch = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(batch)
            
            if hasattr(self.model, 'forward_features'):
                # Get detailed features if available
                all_features = self.model.forward_features(batch)
                cls_tokens = all_features[:, 0]  # CLS token
                patch_tokens = all_features[:, 1:]  # Patch tokens
            else:
                # Fallback to standard output
                cls_tokens = features
                patch_tokens = None
        
        result = {}
        if return_cls_token:
            result['cls_tokens'] = cls_tokens
        if return_patch_tokens and patch_tokens is not None:
            result['patch_tokens'] = patch_tokens
        
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'feature_extraction',
            'framework': 'pytorch',
            'architecture': f'dinov3_{self.model_variant}',
            'patch_size': self.patch_size,
            'image_size': self.image_size,
            'output_dim': 768 if 'vits' in self.model_variant else (768 if 'vitb' in self.model_variant else 1024),
            'supports_batch': True
        })
        
        return info
    
    def unload_model(self) -> None:
        """Unload model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        logger.info("DINOv3 model unloaded")
