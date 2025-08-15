"""
Adapter Registry - Manages the registration and instantiation of all model adapters

Provides dynamic registration, lookup, and creation of adapters.
"""

from typing import Dict, Type, List, Optional, Any
from pathlib import Path
import importlib
from loguru import logger

from .base import BaseModelAdapter


class AdapterRegistry:
    """Adapter Registry"""
    
    def __init__(self):
        """Initialize the registry"""
        self._adapters: Dict[str, Type[BaseModelAdapter]] = {}
        self._framework_mappings: Dict[str, str] = {}
        self._architecture_mappings: Dict[str, str] = {}
        
        # Automatically register built-in adapters
        self._register_builtin_adapters()
        
        logger.info("Adapter registration center initialization completed")
    
    def register(self, 
                 name: str, 
                 adapter_class: Type[BaseModelAdapter],
                 frameworks: Optional[List[str]] = None,
                 architectures: Optional[List[str]] = None) -> None:
        """
        Register an adapter

        Args:
            name: Adapter name
            adapter_class: Adapter class
            frameworks: List of supported frameworks
            architectures: List of supported architectures
        """
        if not issubclass(adapter_class, BaseModelAdapter):
            raise ValueError(f"The adapter class must inherit BaseModelAdapter: {adapter_class}")
        
        self._adapters[name] = adapter_class
        
        # Registering Framework Mappings
        if frameworks:
            for framework in frameworks:
                self._framework_mappings[framework] = name
        
        # Registering Schema Mappings
        if architectures:
            for arch in architectures:
                self._architecture_mappings[arch] = name
        
        logger.info(f"Registered adapters: {name} -> {adapter_class.__name__}")
    
    def get_adapter_class(self, name: str) -> Optional[Type[BaseModelAdapter]]:
        """Get the adapter class by name"""
        return self._adapters.get(name)
    
    def get_adapter_by_framework(self, framework: str) -> Optional[Type[BaseModelAdapter]]:
        """Get the adapter class according to the framework"""
        adapter_name = self._framework_mappings.get(framework)
        if adapter_name:
            return self._adapters.get(adapter_name)
        return None
    
    def get_adapter_by_architecture(self, architecture: str) -> Optional[Type[BaseModelAdapter]]:
        """Get the adapter class based on the architecture"""
        adapter_name = self._architecture_mappings.get(architecture)
        if adapter_name:
            return self._adapters.get(adapter_name)
        return None
    
    def create_adapter(self, 
                      model_path: str,
                      adapter_name: Optional[str] = None,
                      framework: Optional[str] = None,
                      architecture: Optional[str] = None,
                      **kwargs) -> BaseModelAdapter:
        """
        Create an adapter instance

        Args:
            model_path: Model file path
            adapter_name: Adapter name
            framework: Model framework
            architecture: Model architecture
            **kwargs: Parameters passed to the adapter

        Returns:
            Adapter instance
        """
        adapter_class = None
        
        # 1. Give priority to the specified adapter name
        if adapter_name:
            adapter_class = self.get_adapter_class(adapter_name)
            if adapter_class:
                logger.info(f"Use the specified adapter: {adapter_name}")
        
        # 2. Find adapters by architecture
        if not adapter_class and architecture:
            adapter_class = self.get_adapter_by_architecture(architecture)
            if adapter_class:
                logger.info(f"Find the adapter by architecture: {architecture}")
        
        # 3. Find adapters by framework
        if not adapter_class and framework:
            adapter_class = self.get_adapter_by_framework(framework)
            if adapter_class:
                logger.info(f"Find the adapter based on the framework: {framework}")
        
        # 4. If none are found, throw an exception
        if not adapter_class:
            available = list(self._adapters.keys())
            raise ValueError(
                f"No suitable adapter found - adapter_name: {adapter_name}, "
                f"framework: {framework}, architecture: {architecture}. "
                f"Available adapters: {available}"
            )
        
        # Creating an Adapter Instance
        try:
            adapter = adapter_class(model_path=model_path, **kwargs)
            logger.info(f"Adapter instance created successfully: {adapter_class.__name__}")
            return adapter
        except Exception as e:
            logger.error(f"Failed to create adapter: {e}")
            raise
    
    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """List all registered adapters"""
        adapters_info = {}
        
        for name, adapter_class in self._adapters.items():
            # Find supported frameworks and architectures
            frameworks = [k for k, v in self._framework_mappings.items() if v == name]
            architectures = [k for k, v in self._architecture_mappings.items() if v == name]
            
            adapters_info[name] = {
                'class': adapter_class.__name__,
                'module': adapter_class.__module__,
                'frameworks': frameworks,
                'architectures': architectures,
                'doc': adapter_class.__doc__ or "No description"
            }
        
        return adapters_info
    
    def _register_builtin_adapters(self):
        """Registering the built-in adapter"""
        # Register the built-in adapter we are about to implement here
        
        try:
            # YOLO Detection Adapter
            from .detection.ultralytics import UltralyticsAdapter
            self.register(
                'ultralytics',
                UltralyticsAdapter,
                frameworks=['ultralytics', 'yolo'],
                architectures=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                              'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov10x',
                              'yolo11n', 'yolo11s', 'yolo11m']
            )
        except ImportError:
            logger.debug("Ultralytics adapter not found, skipping registration")
        
        try:
            # SAM Segmentation Adapter
            from .segmentation.sam import SAMAdapter
            self.register(
                'sam',
                SAMAdapter,
                frameworks=['segment_anything'],
                architectures=['sam_vit_h', 'sam_vit_l', 'sam_vit_b', 'mobile_sam']
            )
        except ImportError:
            logger.debug("SAM adapter not found, skipping registration")
        
        try:
            # DeepLabV3 Segmentation Adapter
            from .segmentation.deeplabv3 import DeepLabV3Adapter
            self.register(
                'deeplabv3',
                DeepLabV3Adapter,
                frameworks=['torchvision'],
                architectures=['deeplabv3', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet']
            )
        except ImportError:
            logger.debug("DeepLabV3 adapter not found, skipping registration")
        
        try:
            #  Torchvision Classification Adapter
            from .classification.torchvision import TorchvisionAdapter
            self.register(
                'torchvision_classification',
                TorchvisionAdapter,
                frameworks=['torchvision'],
                architectures=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                              'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                              'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                              'efficientnet_b6', 'efficientnet_b7',
                              'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
                              'densenet121', 'densenet169', 'densenet201',
                              'vgg11', 'vgg13', 'vgg16', 'vgg19',
                              'vit_b_16', 'vit_b_32', 'vit_l_16', 'vit_l_32']
            )
        except ImportError:
            logger.debug("Torchvision adapter not found, skipping registration")
        
        try:
            # Stable Diffusion Generation Adapter
            from .generation.stable_diffusion import StableDiffusionAdapter
            self.register(
                'stable_diffusion',
                StableDiffusionAdapter,
                frameworks=['diffusers'],
                architectures=['stable_diffusion', 'stable_diffusion_xl', 'sdxl', 'sd1', 'sd2']
            )
        except ImportError:
            logger.debug("Stable Diffusion adapter not found, skipping registration")

        try:
            # FLUX Generation Adapter
            from .generation.flux import FluxAdapter
            self.register(
                'flux',
                FluxAdapter,
                frameworks=['diffusers'],
                architectures=['flux', 'flux-dev', 'flux-schnell', 'flux-pro']
            )
        except ImportError:
            logger.debug("FLUX adapter not found, skipping registration")
        
        try:
            # CLIP Multimodal Adapter（OpenAI CLIP）
            from .multimodal.clip import CLIPAdapter
            self.register(
                'clip',
                CLIPAdapter,
                frameworks=['clip', 'transformers'],
                architectures=['clip-vit-base', 'clip-vit-large', 'vit-b-32', 'vit-b-16', 
                              'vit-l-14', 'vit-l-14-336', 'rn50', 'rn101']
            )
        except ImportError:
            logger.debug("CLIP adapter not found, skipping registration")
        
        try:
            # OpenCLIP Multimodal Adapter
            from .multimodal.openclip import OpenCLIPAdapter
            self.register(
                'openclip',
                OpenCLIPAdapter,
                frameworks=['open_clip'],
                architectures=['convnext', 'coca', 'eva', 'openclip-vit']
            )
        except ImportError:
            logger.debug("OpenCLIP adapter not found, skipping registration")
    
    def auto_detect_adapter(self, 
                           model_path: str,
                           model_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Automatically detects the appropriate adapter.

        Args:
            model_path: Model path
            model_info: Model information (obtained from model_detector)

        Returns:
            Adapter name
        """
        if model_info:
            # 1. Matching by architecture
            architecture = model_info.get('architecture', '').lower()
            for arch, adapter_name in self._architecture_mappings.items():
                if arch.lower() in architecture:
                    logger.info(f"Automatically select an adapter based on architecture: {adapter_name}")
                    return adapter_name
            
            # 2. Matching by framework
            framework = model_info.get('framework', '').lower()
            for fw, adapter_name in self._framework_mappings.items():
                if fw.lower() in framework:
                    logger.info(f"Automatically select an adapter based on the framework: {adapter_name}")
                    return adapter_name
        
        # 3. Heuristic matching based on file path and name
        model_path_lower = str(model_path).lower()
        
        # Detecting the YOLO model
        if any(pattern in model_path_lower for pattern in ['yolo', 'yolov8', 'yolov9', 'yolov10', 'yolo11']):
            return 'ultralytics'
        
        # Detecting SAM Model
        if any(pattern in model_path_lower for pattern in ['sam_vit', 'mobile_sam']):
            return 'sam'
        
        # Detecting Stable Diffusion Model
        if any(pattern in model_path_lower for pattern in ['stable-diffusion', 'sd_', 'sdxl', 'flux']):
            return 'stable_diffusion'
        
        # Detecting Classification Models
        if any(pattern in model_path_lower for pattern in ['resnet', 'efficientnet', 'vit-']):
            return 'torchvision'
        
        # Detecting CLIP Model
        if any(pattern in model_path_lower for pattern in ['clip', 'vit-b-32', 'vit-l-14']):
            return 'clip'
        
        logger.warning(f"Unable to automatically detect adapter type: {model_path}")
        return None
    
    def get_compatible_adapters(self, model_type: str) -> List[str]:
        """Get a list of compatible adapters based on the model type"""
        compatible = []
        
        for adapter_name, adapter_class in self._adapters.items():
            # Check the base class type of the adapter
            if hasattr(adapter_class, '__bases__'):
                base_names = [base.__name__ for base in adapter_class.__bases__]
                
                if model_type == 'detection' and 'DetectionAdapter' in base_names:
                    compatible.append(adapter_name)
                elif model_type == 'segmentation' and 'SegmentationAdapter' in base_names:
                    compatible.append(adapter_name)
                elif model_type == 'classification' and 'ClassificationAdapter' in base_names:
                    compatible.append(adapter_name)
                elif model_type == 'generation' and 'GenerationAdapter' in base_names:
                    compatible.append(adapter_name)
                elif model_type == 'multimodal' and 'MultimodalAdapter' in base_names:
                    compatible.append(adapter_name)
        
        return compatible


# Global registry instance
_registry = None

def get_registry() -> AdapterRegistry:
    """Get the global adapter registry instance"""
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry


def register_adapter(name: str, 
                    adapter_class: Type[BaseModelAdapter],
                    frameworks: Optional[List[str]] = None,
                    architectures: Optional[List[str]] = None) -> None:
    """Convenience function: register the adapter to the global registry"""
    registry = get_registry()
    registry.register(name, adapter_class, frameworks, architectures)


def create_adapter(model_path: str, **kwargs) -> BaseModelAdapter:
    """Convenience function: creating an adapter instance"""
    registry = get_registry()
    return registry.create_adapter(model_path, **kwargs)


def list_available_adapters() -> Dict[str, Dict[str, Any]]:
    """Convenience function: List all available adapters"""
    registry = get_registry()
    return registry.list_adapters()