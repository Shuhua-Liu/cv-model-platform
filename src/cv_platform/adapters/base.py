"""
Base Model Adapter - Defines the interface for all model adapters

All model adapters should extend the BaseModelAdapter class and implement its abstract methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from loguru import logger


class BaseModelAdapter(ABC):
    """Model adapter base class"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 **kwargs):
        """
        Initialize the adapter

        Args:
            model_path: Model file path
            device: Compute device (cpu, cuda:0, auto, etc.)
            **kwargs: Other model-specific parameters
        """
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.model = None
        self.is_loaded = False
        self.config = kwargs
        
        logger.info(f"Initialize {self.__class__.__name__} - path: {self.model_path}, device: {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Parsing and verifying devices"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda:0" 
            else:
                return "cpu"
        return device
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        """Execute predictions"""
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """Preprocessing input data"""
        pass
    
    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> Any:
        """Post-processing model output"""
        pass
    
    def unload_model(self) -> None:
        """Unloading models to free up memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"{self.__class__.__name__} Model unloaded")
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """Model warm-up and return performance indicators"""
        if not self.is_loaded:
            self.load_model()
        
        # Subclasses should override this method to provide specific preheating logic
        logger.info(f"{self.__class__.__name__} Preheating completed")
        return {"warmup_runs": num_runs}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "adapter_class": self.__class__.__name__,
            "model_path": str(self.model_path),
            "device": self.device,
            "is_loaded": self.is_loaded,
            "config": self.config
        }
        
        # Add model file information
        if self.model_path.exists():
            stat = self.model_path.stat()
            info.update({
                "file_size_mb": stat.st_size / (1024 * 1024),
                "modified_time": stat.st_mtime
            })
        
        return info
    
    def __enter__(self):
        """Context manager entry"""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit"""
        self.unload_model()
    
    def __del__(self):
        """Destructors - ensure resource cleanup"""
        if hasattr(self, 'model') and self.model is not None:
            self.unload_model()


class DetectionAdapter(BaseModelAdapter):
    """Detection model adapter base class"""
    
    @abstractmethod
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray], 
                confidence: float = 0.25,
                nms_threshold: float = 0.45,
                **kwargs) -> List[Dict[str, Any]]:
        """
        Perform object detection

        Args:
            image: Input image
            confidence: Confidence threshold
            nms_threshold: NMS threshold

        Returns:
            A list of detection results, each containing:
            {
                'bbox': [x1, y1, x2, y2], # Bounding box coordinates
                'class': str, # Class name
                'class_id': int, # Class ID
                'confidence': float, # Confidence level
                'area': float # Area of the region
            }
        """
        pass
    
    def visualize_results(self, 
                         image: Union[str, Path, Image.Image, np.ndarray],
                         results: List[Dict[str, Any]],
                         save_path: Optional[str] = None) -> Image.Image:
        """Visualization of detection results"""
        # Default implementation - subclasses can override
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Here you should add the logic for drawing the bounding box
        # Simplify the implementation, you should actually use opencv or PIL to draw
        
        if save_path:
            image.save(save_path)
            
        return image


class SegmentationAdapter(BaseModelAdapter):
    """Segmentation model adapter base class"""
    
    @abstractmethod
    def predict(self,
                image: Union[str, Path, Image.Image, np.ndarray],
                **kwargs) -> Dict[str, Any]:
        """
        Perform image segmentation

        Args:
            image: Input image

        Returns:
            Segmentation result dictionary:
            {
                'masks': np.ndarray, # Segmentation masks [N, H, W]

                'scores': List[float], # Segmentation quality scores

                'areas': List[float], # Area of each mask

                'bbox': List[List[float]], # Bounding box of each mask

                'metadata': Dict # Other metadata
            }
        """
        pass


class ClassificationAdapter(BaseModelAdapter):
    """Classification model adapter base class"""
    
    @abstractmethod
    def predict(self,
                image: Union[str, Path, Image.Image, np.ndarray],
                top_k: int = 5,
                **kwargs) -> Dict[str, Any]:
        """
        Perform image classification
        
        Args:
            image: Input image
            top_k: Returns the top k predictions
            
        Returns:
            Classification result dictionary:
            {
                'predictions': [
                    {
                        'class': str,      # Class name
                        'class_id': int,   # Class ID
                        'confidence': float # Confidence level
                    },
                    ...
                ],
                'top_class': str,          # Class with the highest confidence level
                'top_confidence': float    # Highest confidence level
            }
        """
        pass


class GenerationAdapter(BaseModelAdapter):
    """Generate Model Adapter Base"""
    
    @abstractmethod
    def predict(self,
                prompt: str,
                negative_prompt: Optional[str] = None,
                num_steps: int = 20,
                guidance_scale: float = 7.5,
                width: int = 512,
                height: int = 512,
                **kwargs) -> Dict[str, Any]:
        """
        Execute image generation

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_steps: Number of inference steps
            guidance_scale: Guidance scale
            width: Image width
            height: Image height

        Returns:
            Generated result dictionary:
            {
                'images': List[Image.Image], # Generated images
                'metadata': Dict # Metadata such as generation parameters
            }
        """
        pass


class MultimodalAdapter(BaseModelAdapter):
    """Multimodal model adapter base class"""
    
    @abstractmethod
    def predict(self,
                image: Union[str, Path, Image.Image, np.ndarray],
                text: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Perform multimodal inference

        Args:
            image: Input image
            text: Input text (if required)

        Returns:
            Inference result dictionary (specific format depends on model type)
        """
        pass


# Adapter Type Mapping
ADAPTER_TYPE_MAP = {
    'detection': DetectionAdapter,
    'segmentation': SegmentationAdapter, 
    'classification': ClassificationAdapter,
    'generation': GenerationAdapter,
    'multimodal': MultimodalAdapter
}


def get_adapter_class(model_type: str) -> type:
    """Get the corresponding adapter base class according to the model type"""
    return ADAPTER_TYPE_MAP.get(model_type, BaseModelAdapter)
