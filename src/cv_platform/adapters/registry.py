"""
Adapter Registry - Manages the registration and instantiation of all model adapters

Provides dynamic registration, lookup, and creation of adapters.
"""

from typing import Dict, Type, List, Optional, Any
from pathlib import Path
import importlib
from huggingface_hub import model_info
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
        
        # Detectron2 Detection/Segmentation Adapter
        try:
            from .detection.detectron2 import Detectron2Adapter
            self.register(
                'detectron2',
                Detectron2Adapter,
                frameworks=['detectron2'],
                architectures=[
                    'faster_rcnn', 'mask_rcnn', 'retinanet', 'fcos',
                    'mask2former', 'panoptic_fpn', 'keypoint_rcnn'
                ]
            )
            registration_results['detectron2'] = True
            logger.info("✅ Detectron2 Adapter Registration Successful")
        except ImportError as e:
            registration_results['detectron2'] = False
            logger.debug(f"Detectron2 Adapter Registration Failed: {e}")
        except Exception as e:
            registration_results['detectron2'] = False
            logger.error(f"❌ Detectron2 Adapter Registration Exception: {e}")
       
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
                frameworks=['diffusers', 'stable_diffusion'],
                architectures=['stable_diffusion', 'stable_diffusion_xl', 'sdxl', 'sd_1_5', 'sd_2_1', 'sd_2_1_unclip']
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
                frameworks=['diffusers_flux'],
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
                frameworks=['clip', 'transformers', 'openai_clip'],
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
                'open_clip',
                OpenCLIPAdapter,
                frameworks=['open_clip', 'openclip'],
                architectures=['open_clip', 'vit_b_32', 'vit_h_14']
            )
            registration_results['open_clip'] = True
            logger.info("✅ OpenCLIP Adapter Registration Successful")
        except ImportError as e:
            registration_results['open_clip'] = False
            logger.debug(f"OpenCLIP Adapter Registration Failed: {e}")
        except Exception as e:
            registration_results['open_clip'] = False
            logger.error(f"❌ OpenCLIP Adapter Registration Exception: {e}")

                # DINOv3 feature extraction adapter
        try:
            from .feature_extraction.dinov3 import DINOv3Adapter
            self.register(
                'dinov3',
                DINOv3Adapter,
                frameworks=['pytorch', 'transformers'],
                architectures=['dinov3', 'dinov3_vits14', 'dinov3_vitb14', 'dinov3_vitl14', 'dinov3_vitg14']
            )
            registration_results['dinov3'] = True
            logger.info("✅ DINOv3 Adapter Registration Successful")
        except ImportError as e:
            registration_results['dinov3'] = False
            logger.debug(f"DINOv3 Adapter Registration Failed: {e}")
        
        # LaMa inpainting adapter
        try:
            from .inpainting.lama import LaMaAdapter
            self.register(
                'lama',
                LaMaAdapter,
                frameworks=['pytorch'],
                architectures=['lama', 'large_mask_inpainting']
            )
            registration_results['lama'] = True
            logger.info("✅ LaMa Adapter Registration Successful")
        except ImportError as e:
            registration_results['lama'] = False
            logger.debug(f"LaMa Adapter Registration Failed: {e}")
        
        # SD Inpainting adapter
        try:
            from .inpainting.stable_diffusion_inpainting import StableDiffusionInpaintingAdapter
            self.register(
                'stable_diffusion_inpainting',
                StableDiffusionInpaintingAdapter,
                frameworks=['diffusers_inpainting', 'diffusers'],
                architectures=['stable_diffusion_inpainting', 'sd_inpainting', 'sd_2_inpainting']
            )
            registration_results['stable_diffusion_inpainting'] = True
            logger.info("✅ SD Inpainting Adapter Registration Successful")
        except ImportError as e:
            registration_results['stable_diffusion_inpainting'] = False
            logger.debug(f"SD Inpainting Adapter Registration Failed: {e}")
        
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
            framework = model_info.get('framework', '').lower()
            # Detectron2 framework detection
            if framework == 'detectron2':
                if 'detectron2' in self._adapters:
                    logger.info(f"✅ Matched by framework 'detectron2' -> detectron2")
                    return 'detectron2'
            # Check architecture mapping
            architecture = model_info.get('architecture', '').lower()
            if architecture:
                # Detectron2 architecture mappings
                detectron2_arch_mappings = {
                    'faster_rcnn': 'detectron2',
                    'mask_rcnn': 'detectron2',
                    'retinanet': 'detectron2',
                    'fcos': 'detectron2',
                    'mask2former': 'detectron2',
                    'panoptic_fpn': 'detectron2',
                    'keypoint_rcnn': 'detectron2'
                }
                for arch_pattern, adapter_name in detectron2_arch_mappings.items():
                    if arch_pattern in architecture:
                        if adapter_name in self._adapters:
                            logger.info(f"✅ Matched by architecture '{architecture}' -> {adapter_name}")
                            return adapter_name
                   
            # architecture = model_info.get('architecture', '').lower()
            # if architecture:
                # Specific SD architecture mapping
                sd_arch_mappings = {
                    'sd_2_1_unclip':
                    'stable_diffusion',
                    'sdxl':
                    'stable_diffusion',
                    'sd_2_1':
                    'stable_diffusion',
                    'sd_1_5':
                    'stable_diffusion',
                    'controlnet':
                    'controlnet',
                    'controlnet_canny':
                    'controlnet',
                    'controlnet_depth':
                    'controlnet',
                    'flux': 'flux',
                    'flux-dev': 'flux',
                    'flux-schnell': 'flux'
                }
                for arch_pattern, adapter_name in sd_arch_mappings.items():
                    if arch_pattern in architecture:
                        if adapter_name in self._adapters:
                            logger.info(f"✅ Matched by architecture '{architecture}' -> {adapter_name}")
                            return adapter_name
            # Check framework mapping
            framework = model_info.get('framework', '').lower()
            if framework == 'diffusers':
                pass
    
        # Priority 2: Path-based detection with strict ordering
        model_path_lower = str(model_path).lower()
        # DETECTRON2 PATTERNS FIRST (highest priority for detectron2)
        if any(pattern in model_path_lower for pattern in [
            'detectron2', 'faster_rcnn', 'mask_rcnn', 'retinanet',
            'fcos', 'mask2former', 'panoptic_fpn', 'keypoint_rcnn'
        ]):
            if 'detectron2' in self._adapters:
                logger.info("🎯 Detectron2 pattern detected -> detectron2")
                return 'detectron2'
        # Model zoo configuration names for detectron2
        detectron2_configs = [
            'faster_rcnn_r50', 'faster_rcnn_r101', 'retinanet_r50',
            'fcos_r50', 'mask_rcnn_r50', 'mask_rcnn_r101',
            'mask2former_r50', 'panoptic_fpn_r50', 'keypoint_rcnn_r50'
        ]
        if any(config in model_path_lower for config in detectron2_configs):
            if 'detectron2' in self._adapters:
                logger.info("🎯 Detectron2 model zoo config detected -> detectron2")
                return 'detectron2'
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
       
        # Priority: Inpainting models detection (before general SD)
        model_path_parts = str(model_path).lower().split('/')
        # Only apply inpainting detection to models actually in inpainting directories
        if 'inpainting' in model_path_parts:
            # SD Inpainting models in inpainting directory
            if any(pattern in model_path_lower for pattern in ['stable-diffusion', 'stable_diffusion', 'sd_']):
                if any(pattern in model_path_lower for pattern in ['2-inpainting', '2_inpainting', 'inpainting']):
                    if 'stable_diffusion_inpainting' in self._adapters:
                        logger.info("✅ SD Inpainting model detected → stable_diffusion_inpainting")
                        return 'stable_diffusion_inpainting'
            
            # LaMa models
            elif 'lama' in model_path_lower:
                if 'lama' in self._adapters:
                    logger.info("✅ LaMa model detected → lama")
                    return 'lama'
        # 4. FLUX models (specific path patterns)
        flux_specific_patterns = ['flux.1-dev', 'flux.1-schnell', 'flux.1-pro', '/flux/', 'flux-dev', 'flux-schnell']
        if any(pattern in model_path_lower for pattern in flux_specific_patterns):
            if 'flux' in self._adapters:
                logger.info("✅ Detected FLUX model -> flux")
                return 'flux'
            elif 'stable_diffusion' in self._adapters:
                logger.info("✅ FLUX fallback -> stable_diffusion")
                return 'stable_diffusion'
        # 5. Stable Diffusion models (broader patterns after specific ones)
        sd_patterns = ['stable-diffusion', '/sd_', '/sd_2_1_unclip/', '/sd_2_1/', '/sdxl_base_1.0/', 'stable_diffusion', 'sd_xl']
        if any(pattern in model_path_lower for pattern in sd_patterns):
            if 'stable_diffusion' in self._adapters:
                logger.info("✅ Detected Stable Diffusion model -> stable_diffusion")
                return 'stable_diffusion'        
        # 6. Generic patterns (fallback)
        if 'controlnet' in model_path_lower:
            if 'controlnet' in self._adapters:
                return 'controlnet'
        # 7. OpenCLIP models
        openclip_patterns = ['open_clip', 'openclip', 'open-clip']
        if any(pattern in model_path_lower for pattern in openclip_patterns):
            if 'open_clip' in self._adapters:
                logger.info("✅ Detected OpenCLIP model -> openclip")
                return 'openclip'
        # 8. CLIP models
        clip_patterns = ['/clip/', '\\\\clip\\\\', 'clip-vit', 'clip_vit']
        if any(pattern in model_path_lower for pattern in clip_patterns):
            if not any(pattern in model_path_lower for pattern in openclip_patterns):
                if 'clip' in self._adapters:
                    logger.info("✅ Detected CLIP model -> clip")
                    return 'clip'
        # Multimodal context detection (fallback for CLIP variants)
        if 'multimodal' in model_path_lower and any(pattern in model_path_lower for pattern in ['vit', 'vision']):
            if any(pattern in model_path_lower for pattern in openclip_patterns):
                if 'openclip' in self._adapters:
                    logger.info("✅ Detected multimodal OpenCLIP -> openclip")
                    return 'openclip'
            else:
                if 'clip' in self._adapters:
                    logger.info("✅ Detected multimodal CLIP -> clip")
                    return 'clip'
        # 9. Classification models
        if 'classification' in model_path_lower:
            if any(pattern in model_path_lower for pattern in ['resnet', 'efficientnet']):
                if 'torchvision_classification' in self._adapters:
                    logger.info("✅ Detected classification model -> torchvision_classification")
                    return 'torchvision_classification' 
            elif 'vit' in model_path_lower and 'multimodal' not in model_path_lower:
                if 'torchvision_classification' in self._adapters:
                    logger.info("✅ Detected ViT classification model -> torchvision_classification")
                    return 'torchvision_classification'        
        
        # Generic fallbacks by model type
        if model_info:
            model_type = model_info.get('type', '').lower()
            if model_type == 'detection':
                # Try detectron2 first, then ultralytics
                if 'detectron2' in self._adapters:
                    logger.info("🎯 Generic detection -> detectron2 (primary choice)")
                    return 'detectron2'
                elif 'ultralytics' in self._adapters:
                    logger.info("🎯 Generic detection -> ultralytics (fallback)")
                    return 'ultralytics'
            elif model_type == 'segmentation':
                # Try detectron2 for segmentation first, then SAM
                if any(pattern in model_path_lower for pattern in ['mask', 'rcnn', 'former']):
                    if 'detectron2' in self._adapters:
                        logger.info("🎯 Generic segmentation -> detectron2")
                        return 'detectron2'
                elif 'sam' in self._adapters:
                    logger.info("🎯 Generic segmentation -> sam")
                    return 'sam'
            elif model_type == 'generation':
                if 'stable_diffusion' in self._adapters:
                    logger.info("🎯 Generic generation -> stable_diffusion")
                    return 'stable_diffusion'
            elif model_type == 'classification':
                if 'torchvision_classification' in self._adapters:
                    logger.info("🎯 Generic classification -> torchvision_classification")
                    return 'torchvision_classification'
            elif model_type == 'multimodal':
                if 'clip' in self._adapters:
                    logger.info("🎯 Generic multimodal -> clip")
                    return 'clip'
                elif 'openclip' in self._adapters:
                    logger.info("🎯 Generic multimodal -> openclip")
                    return 'openclip'
            elif model_type == 'inpainting':
                if 'stable_diffusion_inpainting' in self._adapters:
                    logger.info("🎯 Generic inpainting -> stable_diffusion_inpainting")
                    return 'stable_diffusion_inpainting'
                elif 'lama' in self._adapters:
                    logger.info("🎯 Generic inpainting -> lama")
                    return 'lama'
        
        # 10. Generic detection fallback (detection models)
        if any(keyword in model_path_lower for keyword in ['detect', 'object', 'bbox']):
            if 'detectron2' in self._adapters:
                logger.info("🎯 Generic detection -> detectron2")
                return 'detectron2'
            elif 'ultralytics' in self._adapters:
                logger.info("🎯 Generic detection -> ultralytics")
                return 'ultralytics'
        
        # FINAL FALLBACK - NO DEFAULT TO FLUX!
        logger.warning(f"⚠️ No adapter detected for: {model_path}")
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
