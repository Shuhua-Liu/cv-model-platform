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
        """Initialize registry center"""
        self._adapters: Dict[str, Type[BaseModelAdapter]] = {}
        self._framework_mappings: Dict[str, str] = {}
        self._architecture_mappings: Dict[str, str] = {}
        
        # Automatically register built-in adapters
        self._register_builtin_adapters()
        
        logger.info("Adapter registry initialization completed")
    
    def register(self, 
                 name: str, 
                 adapter_class: Type[BaseModelAdapter],
                 frameworks: Optional[List[str]] = None,
                 architectures: Optional[List[str]] = None) -> None:
        """
        Register Adapter
        
        Args:
            name: Adapter name
            adapter_class: Adapter class
            frameworks: List of supported frameworks
            architectures: List of supported architectures
        """
        if not issubclass(adapter_class, BaseModelAdapter):
            raise ValueError(f"Adapter classes must inherit from BaseModelAdapter.: {adapter_class}")
        
        self._adapters[name] = adapter_class
        
        # Registered Framework Mapping
        if frameworks:
            for framework in frameworks:
                self._framework_mappings[framework] = name
        
        # Registered Architecture Mapping
        if architectures:
            for arch in architectures:
                self._architecture_mappings[arch] = name
        
        logger.info(f"Registered Adapters: {name} -> {adapter_class.__name__}")
    
    def get_adapter_class(self, name: str) -> Optional[Type[BaseModelAdapter]]:
        """Retrieve adapter classes by name"""
        return self._adapters.get(name)
    
    def get_adapter_by_framework(self, framework: str) -> Optional[Type[BaseModelAdapter]]:
        """Retrieve the adapter class based on the framework"""
        adapter_name = self._framework_mappings.get(framework)
        if adapter_name:
            return self._adapters.get(adapter_name)
        return None
    
    def get_adapter_by_architecture(self, architecture: str) -> Optional[Type[BaseModelAdapter]]:
        """Retrieve the adapter class based on the architecture"""
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
        Create adapter instance
        
        Args:
            model_path: Path to model file
            adapter_name: Specified adapter name
            framework: Model framework
            architecture: Model architecture
            **kwargs: Parameters passed to the adapter
            
        Returns:
            Adapter instance
        """
        adapter_class = None
        
        # 1. Prioritize the use of the specified adapter name
        if adapter_name:
            adapter_class = self.get_adapter_class(adapter_name)
            if adapter_class:
                logger.info(f"Use the specified adapter: {adapter_name}")
        
        # 2. Find the adapter based on the architecture
        if not adapter_class and architecture:
            adapter_class = self.get_adapter_by_architecture(architecture)
            if adapter_class:
                logger.info(f"Find the adapter based on the architecture: {architecture}")
        
        # 3. Find the adapter based on the framework
        if not adapter_class and framework:
            adapter_class = self.get_adapter_by_framework(framework)
            if adapter_class:
                logger.info(f"Find the adapter based on the framework: {framework}")
        
        # 4. If none are found, throw an exception.
        if not adapter_class:
            available = list(self._adapters.keys())
            raise ValueError(
                f"No suitable adapter found - adapter_name: {adapter_name}, "
                f"framework: {framework}, architecture: {architecture}. "
                f"Available adapters: {available}"
            )
        
        # Create an adapter instance
        try:
            adapter = adapter_class(model_path=model_path, **kwargs)
            logger.info(f"Adapter instance successfully created: {adapter_class.__name__}")
            return adapter
        except Exception as e:
            logger.error(f"Adapter creation failed: {e}")
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
        """Register Built-in Adapter"""
        logger.info("Begin registering the built-in adapter...")
        
        registration_results = {}
        
        # YOLO Detection Adapter
        try:
            from .detection.ultralytics import UltralyticsAdapter
            self.register(
                'ultralytics',
                UltralyticsAdapter,
                frameworks=['ultralytics', 'yolo'],
                architectures=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                              'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov10x',
                              'yolo11n', 'yolo11s', 'yolo11m']
            )
            registration_results['ultralytics'] = True
            logger.info("✅ Ultralytics adapter registration successful")
        except ImportError as e:
            registration_results['ultralytics'] = False
            logger.warning(f"❌ Ultralytics adapter registration failed: {e}")
            logger.info("💡 Please install Ultralytics.: pip install ultralytics")
        except Exception as e:
            registration_results['ultralytics'] = False
            logger.error(f"❌ Ultralytics Adapter Registration Exception: {e}")
        
        # SAM Segmentation Adapter
        try:
            from .segmentation.sam import SAMAdapter
            self.register(
                'sam',
                SAMAdapter,
                frameworks=['segment_anything'],
                architectures=['sam_vit_h', 'sam_vit_l', 'sam_vit_b', 'mobile_sam']
            )
            registration_results['sam'] = True
            logger.info("✅ SAM adapter registration successful")
        except ImportError as e:
            registration_results['sam'] = False
            logger.warning(f"❌ SAM adapter registration failed: {e}")
            logger.info("💡 Please install segment-anything: pip install segment-anything")
        except Exception as e:
            registration_results['sam'] = False
            logger.error(f"❌ SAM Adapter Registration Exception: {e}")
        
        # DeepLabV3 Segmentation Adapter
        try:
            from .segmentation.deeplabv3 import DeepLabV3Adapter
            self.register(
                'deeplabv3',
                DeepLabV3Adapter,
                frameworks=['torchvision'],
                architectures=['deeplabv3', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet']
            )
            registration_results['deeplabv3'] = True
            logger.info("✅ DeepLabV3 adapter registration successful")
        except ImportError as e:
            registration_results['deeplabv3'] = False
            logger.debug(f"DeepLabV3 adapter registration failed: {e}")
        except Exception as e:
            registration_results['deeplabv3'] = False
            logger.error(f"❌ DeepLabV3 Adapter Registration Exception: {e}")
        
        # Torchvision Classification Adapter
        try:
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
            registration_results['torchvision_classification'] = True
            logger.info("✅ Torchvision Classification Adapter Registration Successful")
        except ImportError as e:
            registration_results['torchvision_classification'] = False
            logger.debug(f"Torchvision Classification adapter registration failed: {e}")
        except Exception as e:
            registration_results['torchvision_classification'] = False
            logger.error(f"❌ Torchvision Classification Adapter Registration Exception: {e}")

        # ControlNet Generation Adapter
        try:
            from.generation.controlnet import ControlNetAdapter
            self.register(
                'controlnet', 
                ControlNetAdapter,
                frameworks=['diffusers'],
                architectures=['controlnet', 'controlnet_canny', 'controlnet_depth', 
                               'controlnet_pose', 'controlnet_seg', 'controlnet_inpaint', 
                               'controlnet_hed', 'controlnet_mlsd', 'controlnt_normal']
                )
            registration_results['controlnet'] = True
            logger.info("✅ ControlNet Adapter Registration Successful")
        except ImportError as e:
            registration_results['controlnet'] = False
            logger.debug(f"ControlNet Adapter Registration Failed: {e}")
        except Exception as e:
            registration_results['controlnet'] = False
            logger.error(f"❌ ControlNet Adapter Registration Exception: {e}")
        
        # Stable Diffusion Generation Adapter
        try:
            from .generation.stable_diffusion import StableDiffusionAdapter
            self.register(
                'stable_diffusion',
                StableDiffusionAdapter,
                frameworks=['diffusers'],
                architectures=['stable_diffusion', 'stable_diffusion_xl', 'sdxl', 'sd1', 'sd2']
            )
            registration_results['stable_diffusion'] = True
            logger.info("✅ Stable Diffusion Adapter Registration Successful")
        except ImportError as e:
            registration_results['stable_diffusion'] = False
            logger.debug(f"Stable Diffusion Adapter Registration Failed: {e}")
        except Exception as e:
            registration_results['stable_diffusion'] = False
            logger.error(f"❌ Stable Diffusion Adapter Registration Exception: {e}")

        # FLUX Generation Adapter
        try:
            from .generation.flux import FluxAdapter
            self.register(
                'flux',
                FluxAdapter,
                frameworks=['diffusers'],
                architectures=['flux', 'flux-dev', 'flux-schnell', 'flux-pro']
            )
            registration_results['flux'] = True
            logger.info("✅ FLUX Adapter Registration Successful")
        except ImportError as e:
            registration_results['flux'] = False
            logger.debug(f"FLUX Adapter Registration Failed: {e}")
        except Exception as e:
            registration_results['flux'] = False
            logger.error(f"❌ FLUX Adapter Registration Exception: {e}")
        
        # CLIP Multimodal Adapter（OpenAI CLIP）
        try:
            from .multimodal.clip import CLIPAdapter
            self.register(
                'clip',
                CLIPAdapter,
                frameworks=['clip', 'transformers'],
                architectures=['clip-vit-base', 'clip-vit-large', 'vit-b-32', 'vit-b-16', 
                              'vit-l-14', 'vit-l-14-336', 'rn50', 'rn101']
            )
            registration_results['clip'] = True
            logger.info("✅ CLIP Adapter Registration Successful")
        except ImportError as e:
            registration_results['clip'] = False
            logger.debug(f"CLIP Adapter Registration Failed: {e}")
        except Exception as e:
            registration_results['clip'] = False
            logger.error(f"❌ CLIP Adapter Registration Exception: {e}")
        
        # OpenCLIP Multimodal Adapter
        try:
            from .multimodal.openclip import OpenCLIPAdapter
            self.register(
                'openclip',
                OpenCLIPAdapter,
                frameworks=['open_clip'],
                architectures=['convnext', 'coca', 'eva', 'openclip-vit']
            )
            registration_results['openclip'] = True
            logger.info("✅ OpenCLIP Adapter Registration Successful")
        except ImportError as e:
            registration_results['openclip'] = False
            logger.debug(f"OpenCLIP Adapter Registration Failed: {e}")
        except Exception as e:
            registration_results['openclip'] = False
            logger.error(f"❌ OpenCLIP Adapter Registration Exception: {e}")
        
        # Summary of Registration Results
        success_count = sum(registration_results.values())
        total_count = len(registration_results)
        
        logger.info(f"Adapter registration completed: {success_count}/{total_count} success")
        logger.info(f"Registered adapters: {list(self._adapters.keys())}")
        
        # If no adapters are successfully registered, issue a warning.
        if success_count == 0:
            logger.error("⚠️ No adapters have been successfully registered! Please check the installation of dependency packages.")
        elif 'ultralytics' not in self._adapters:
            logger.warning("⚠️ The critical adapter 'ultralytics' is not registered, which may affect the use of YOLO models.")
    
    def force_register_adapter(self, adapter_name: str) -> bool:
        """
        Force Registration of Specified Adapter
        
        Args:
            adapter_name: The name of the adapter to register
            
        Returns:
            True if successful, False otherwise
        """
        if adapter_name == 'ultralytics':
            try:
                from .detection.ultralytics import UltralyticsAdapter
                self.register(
                    'ultralytics',
                    UltralyticsAdapter,
                    frameworks=['ultralytics', 'yolo'],
                    architectures=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
                )
                logger.info(f"✅ Forced registration of {adapter_name} successful")
                return True
            except Exception as e:
                logger.error(f"❌ Forced registration of {adapter_name} failed: {e}")
                return False
        
        # Similar logic can be added for other adapters.
        logger.warning(f"Does not support forced registration of adapters: {adapter_name}")
        return False
    
    def auto_detect_adapter(self, 
                           model_path: str,
                           model_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Automatic detection of suitable adapters
        Args:
            model_path: Model path
            model_info: Model information (obtained from model_detector)
        Returns:
            Adapter name
        """
        logger.info(f"🔍 Auto-detecting adapter for: {model_path}")
        # Priority 1: Use model_info if available and reliable
        if model_info:
            # Check architecture mapping first
            architecture = model_info.get('architecture', '').lower()
            if architecture:
                for arch, adapter_name in self._architecture_mappings.items():
                    if arch.lower() == architecture or arch.lower() in architecture:
                        if adapter_name in self._adapters:
                            logger.info(f"✅ Matched by architecture '{architecture}' -> {adapter_name}")
                            return adapter_name
            # Check framework mapping
            framework = model_info.get('framework', '').lower()
            if framework:
                for fw, adapter_name in self._framework_mappings.items():
                    if fw.lower() == framework or fw.lower() in framework:
                        if adapter_name in self._adapters:
                            logger.info(f"✅ Matched by framework '{framework}' -> {adapter_name}")
                            return adapter_name
        # Priority 2: Path-based detection with strict ordering
        model_path_lower = str(model_path).lower()
        # MOST SPECIFIC PATTERNS FIRST
        # 1. YOLO models (highest priority - most specific)
        yolo_patterns = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                            'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov10x',
                            'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
        if any(pattern in model_path_lower for pattern in yolo_patterns):
            if 'ultralytics' in self._adapters:
                logger.info("✅ Detected YOLO model -> ultralytics")
                return 'ultralytics'
        # 2. SAM models
        sam_patterns = ['sam_vit_h', 'sam_vit_l', 'sam_vit_b', 'mobile_sam']
        if any(pattern in model_path_lower for pattern in sam_patterns):
            if 'sam' in self._adapters:
                logger.info("✅ Detected SAM model -> sam")
                return 'sam'
        # 3. ControlNet models (before general Stable Diffusion)
        controlnet_specific_patterns = ['controlnet-canny', 'controlnet-depth', 'controlnet-pose',
                                        'controlnet_seg', 'controlnet-inpaint', 'controlnet-normal']
        if any(pattern in model_path_lower for pattern in controlnet_specific_patterns):
            if 'controlnet' in self._adapters:
                logger.info("✅ Detected ControlNet model -> controlnet")
                return 'controlnet'
            elif 'stable_diffusion' in self._adapters:
                logger.info("✅ ControlNet fallback -> stable_diffusion")
                return 'stable_diffusion'
        if 'controlnet' in model_path_lower:
            if 'controlnet' in self._adapters:
                logger.info("✅ Detected ControlNet model -> controlnet")
                return 'controlnet'
            elif 'stable_diffusion' in self._adapters:
                logger.info("✅ ControlNet fallback -> stable_diffusion")
                return 'stable_diffusion'
        # 4. FLUX models (specific path patterns)
        flux_specific_patterns = ['flux.1-dev', 'flux.1-schnell', 'flux.1-pro', '/flux/']
        if any(pattern in model_path_lower for pattern in flux_specific_patterns):
            if 'flux' in self._adapters:
                logger.info("✅ Detected FLUX model -> flux")
                return 'flux'
            elif 'stable_diffusion' in self._adapters:
                logger.info("✅ FLUX fallback -> stable_diffusion")
                return 'stable_diffusion'
        # 5. Stable Diffusion models (broader patterns after specific ones)
        sd_patterns = ['stable-diffusion', '/sd_', 'sdxl', 'stable_diffusion']
        if any(pattern in model_path_lower for pattern in sd_patterns):
            if 'stable_diffusion' in self._adapters:
                logger.info("✅ Detected Stable Diffusion model -> stable_diffusion")
                return 'stable_diffusion'
        # 6. Classification models
        if any(pattern in model_path_lower for pattern in ['resnet', 'efficientnet', 'vit', 'classification']):
            if 'torchvision_classification' in self._adapters:
                logger.info("✅ Detected classification model -> torchvision_classification")
                return 'torchvision_classification'
        # 7. CLIP models
        if any(pattern in model_path_lower for pattern in ['clip', 'vit-b-32', 'vit-l-14']):
            if 'clip' in self._adapters:
                logger.info("✅ Detected CLIP model -> clip")
                return 'clip'
        # 8. Generic detection fallback (detection models)
        if any(keyword in model_path_lower for keyword in ['detect', 'object', 'bbox']):
            if 'ultralytics' in self._adapters:
                logger.info("🎯 Generic detection -> ultralytics")
                return 'ultralytics'
        # FINAL FALLBACK - NO DEFAULT TO FLUX!
        logger.warning(f"⚠️ Unable to auto-detect adapter for: {model_path}")
        logger.info(f"📊 Available adapters: {list(self._adapters.keys())}")
        return None  # Return None instead of defaulting to 'flux'
        
    
    def get_compatible_adapters(self, model_type: str) -> List[str]:
        """Retrieve a list of compatible adapters based on the model type."""
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


# Global Registry Center Instance
_registry = None

def get_registry() -> AdapterRegistry:
    """Obtain the instance of the global adapter registry"""
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry


def register_adapter(name: str, 
                    adapter_class: Type[BaseModelAdapter],
                    frameworks: Optional[List[str]] = None,
                    architectures: Optional[List[str]] = None) -> None:
    """Convenience Function: Registering Adapters to the Global Registry"""
    registry = get_registry()
    registry.register(name, adapter_class, frameworks, architectures)


def create_adapter(model_path: str, **kwargs) -> BaseModelAdapter:
    """Convenience Function: Create Adapter Instance"""
    registry = get_registry()
    return registry.create_adapter(model_path, **kwargs)


def list_available_adapters() -> Dict[str, Dict[str, Any]]:
    """Utility function: List all available adapters"""
    registry = get_registry()
    return registry.list_adapters()
