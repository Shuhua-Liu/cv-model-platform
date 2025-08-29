"""
ControlNet Generation Adapter - Support for ControlNet models with diffusers
Supported ControlNet types:
- Canny edge detection
- Depth estimation
- Normal maps
- Pose estimation
- Segmentation masks
- OpenPose
- Inpainting
"""
import time
from typing import Dict, Any, Union, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2
from loguru import logger
from ..base import GenerationAdapter
try:
   from diffusers import (
       StableDiffusionControlNetPipeline,
       StableDiffusionXLControlNetPipeline,
       ControlNetModel,
       DDIMScheduler,
       DPMSolverMultistepScheduler,
       EulerAncestralDiscreteScheduler
   )
   from controlnet_aux import (
       CannyDetector,
       OpenposeDetector,
       MidasDetector,
       HEDdetector,
       MLSDdetector
   )
   CONTROLNET_AVAILABLE = True
except ImportError as e:
   CONTROLNET_AVAILABLE = False
   logger.warning(f"ControlNet dependencies not available: {e}")
   logger.info("Install with: pip install diffusers controlnet-aux transformers")

class ControlNetAdapter(GenerationAdapter):
    """ControlNet model adapter for controllable image generation"""
    def __init__(self,
                    model_path: Union[str, Path],
                    control_type: str = "canny",
                    base_model: str = "runwayml/stable-diffusion-v1-5",
                    device: str = "auto",
                    variant: str = "fp16",
                    enable_memory_efficient_attention: bool = True,
                    enable_xformers: bool = False,
                    cpu_offload: bool = False,
                    **kwargs):
        """
        Initialize ControlNet adapter
        Args:
            model_path: ControlNet model path or HuggingFace model ID
            control_type: Type of control (canny, depth, pose, seg, etc.)
            base_model: Base Stable Diffusion model to use
            device: Computing device
            variant: Model precision variant (fp16, fp32)
            enable_memory_efficient_attention: Enable memory efficient attention
            enable_xformers: Enable xformers optimization
            cpu_offload: Enable CPU offload to save GPU memory
        """
        if not CONTROLNET_AVAILABLE:
            raise ImportError("Need to install ControlNet dependencies: pip install diffusers controlnet-aux transformers")
        super().__init__(model_path, device, **kwargs)
        self.control_type = control_type.lower()
        self.base_model = base_model
        self.variant = variant
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.enable_xformers = enable_xformers
        self.cpu_offload = cpu_offload
        # Validate control type
        self.supported_control_types = {
            'canny', 'depth', 'pose', 'openpose', 'seg', 'normal',
            'hed', 'mlsd', 'scribble', 'inpaint'
        }
        if self.control_type not in self.supported_control_types:
            logger.warning(f"Unknown control type: {self.control_type}. Using 'canny' as fallback.")
            self.control_type = 'canny'
        # Inference pipeline components
        self.pipeline = None
        self.controlnet = None
        self.preprocessor = None
        # Determine model architecture
        self.model_architecture = self._determine_architecture()
    
    def _determine_architecture(self) -> str:
        """Determine ControlNet model architecture from path/config"""
        path_str = str(self.model_path).lower()
        if any(keyword in path_str for keyword in ['xl', 'sdxl']):
            return 'sdxl'
        elif any(keyword in self.base_model.lower() for keyword in ['xl', 'sdxl']):
            return 'sdxl'
        else:
            return 'sd15'  # Default to SD 1.5
        
    def _setup_preprocessor(self):
        """Setup the control preprocessing pipeline"""
        try:
            if self.control_type == 'canny':
                self.preprocessor = CannyDetector()
            elif self.control_type in ['pose', 'openpose']:
                self.preprocessor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            elif self.control_type == 'depth':
                self.preprocessor = MidasDetector.from_pretrained('lllyasviel/ControlNet')
            elif self.control_type == 'hed':
                self.preprocessor = HEDdetector.from_pretrained('lllyasviel/ControlNet')
            elif self.control_type == 'mlsd':
                self.preprocessor = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
            else:
                logger.warning(f"No preprocessor available for {self.control_type}")
                self.preprocessor = None
            logger.info(f"Preprocessor setup complete for {self.control_type}")
        except Exception as e:
            logger.error(f"Failed to setup preprocessor for {self.control_type}: {e}")
            self.preprocessor = None
    
    def load_model(self) -> None:
        """Load ControlNet model and pipeline"""
        try:
            logger.info(f"Loading ControlNet model: {self.model_path}")
            logger.info(f"Control type: {self.control_type}, Base model: {self.base_model}")
            # Load ControlNet model
            self.controlnet = ControlNetModel.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.variant == 'fp16' and 'cuda' in self.device else torch.float32,
                use_safetensors=True
            )
            # Choose appropriate pipeline based on architecture
            if self.model_architecture == 'sdxl':
                pipeline_class = StableDiffusionXLControlNetPipeline
            else:
                pipeline_class = StableDiffusionControlNetPipeline
            # Create pipeline
            self.pipeline = pipeline_class.from_pretrained(
                self.base_model,
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if self.variant == 'fp16' and 'cuda' in self.device else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            # Apply optimizations
            if self.enable_memory_efficient_attention:
                self.pipeline.enable_attention_slicing()
            if self.enable_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory optimization enabled")
                except Exception as e:
                    logger.warning(f"Failed to enable xformers: {e}")
            # Move to device
            if not self.cpu_offload:
                self.pipeline = self.pipeline.to(self.device)
            else:
                self.pipeline.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled")
            # Setup preprocessor
            self._setup_preprocessor()
            self.is_loaded = True
            logger.info(f"ControlNet model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load ControlNet model: {e}")
            raise
    
    def preprocess_control_image(self, control_image: Union[str, Path, Image.Image], **kwargs) -> Image.Image:
        """
        Preprocess control image based on control type
        Args:
            control_image: Input image for control
            **kwargs: Additional preprocessing parameters
        Returns:
            Processed control image
        """
        if isinstance(control_image, (str, Path)):
            control_image = Image.open(control_image).convert('RGB')
        elif not isinstance(control_image, Image.Image):
            raise ValueError("control_image must be a PIL Image, file path, or string")
        # Apply preprocessing based on control type
        if self.control_type == 'canny':
            if self.preprocessor:
                return self.preprocessor(control_image, **kwargs)
            else:
                # Fallback to OpenCV canny
                low_threshold = kwargs.get('low_threshold', 100)
                high_threshold = kwargs.get('high_threshold', 200)
                image_array = np.array(control_image)
                canny_image = cv2.Canny(image_array, low_threshold, high_threshold)
                canny_image = canny_image[:, :, None]
                canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
                return Image.fromarray(canny_image)
        elif self.preprocessor:
            return self.preprocessor(control_image, **kwargs)
        else:
            logger.warning(f"No preprocessing available for {self.control_type}, using original image")
            return control_image
        
    def generate(self,
                    prompt: str,
                    control_image: Union[str, Path, Image.Image],
                    negative_prompt: Optional[str] = None,
                    num_inference_steps: int = 20,
                    guidance_scale: float = 7.5,
                    controlnet_conditioning_scale: float = 1.0,
                    width: int = 512,
                    height: int = 512,
                    num_images_per_prompt: int = 1,
                    seed: Optional[int] = None,
                    preprocess_control: bool = True,
                    **kwargs) -> List[Image.Image]:
        """
        Generate images using ControlNet
        Args:
            prompt: Text prompt for image generation
            control_image: Control image (will be preprocessed if needed)
            negative_prompt: Negative prompt to avoid certain features
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            controlnet_conditioning_scale: How strongly to apply ControlNet guidance
            width: Output image width
            height: Output image height
            num_images_per_prompt: Number of images to generate per prompt
            seed: Random seed for reproducibility
            preprocess_control: Whether to preprocess the control image
            **kwargs: Additional generation parameters
        Returns:
            List of generated PIL Images
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        start_time = time.time()
        try:
            # Preprocess control image if requested
            if preprocess_control:
                processed_control = self.preprocess_control_image(control_image, **kwargs)
            else:
                if isinstance(control_image, (str, Path)):
                    processed_control = Image.open(control_image).convert('RGB')
                else:
                    processed_control = control_image
            # Resize control image to match output dimensions
            processed_control = processed_control.resize((width, height), Image.LANCZOS)
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
            # Prepare generation parameters
            generate_kwargs = {
                'prompt': prompt,
                'image': processed_control,
                'height': height,
                'width': width,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'controlnet_conditioning_scale': controlnet_conditioning_scale,
                'num_images_per_prompt': num_images_per_prompt,
            }
            if negative_prompt:
                generate_kwargs['negative_prompt'] = negative_prompt
            # Add any additional kwargs
            generate_kwargs.update({k: v for k, v in kwargs.items()
                                    if k not in ['low_threshold', 'high_threshold']})
            # Generate images
            logger.info(f"Generating {num_images_per_prompt} images with ControlNet ({self.control_type})")
            result = self.pipeline(**generate_kwargs)
            images = result.images if hasattr(result, 'images') else [result]
            generation_time = time.time() - start_time
            logger.info(f"Generated {len(images)} images in {generation_time:.2f}s")
            return images
        except Exception as e:
            logger.error(f"ControlNet generation failed: {e}")
            raise
       
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = super().get_model_info()
        info.update({
            'model_type': 'generation',
            'framework': 'diffusers',
            'architecture': f'controlnet_{self.control_type}',
            'control_type': self.control_type,
            'base_model': self.base_model,
            'model_architecture': self.model_architecture,
            'supported_control_types': list(self.supported_control_types),
            'default_resolution': '512x512' if self.model_architecture == 'sd15' else '1024x1024',
            'recommended_steps': 20,
            'recommended_guidance': 7.5,
            'recommended_controlnet_scale': 1.0,
            'preprocessor_available': self.preprocessor is not None,
            'optimizations': {
                'memory_efficient_attention': self.enable_memory_efficient_attention,
                'xformers': self.enable_xformers,
                'cpu_offload': self.cpu_offload
            }
        })
        if self.is_loaded and self.pipeline:
            try:
                # Get pipeline component info
                components = []
                if hasattr(self.pipeline, 'unet'):
                    components.append('unet')
                if hasattr(self.pipeline, 'vae'):
                    components.append('vae')
                if hasattr(self.pipeline, 'text_encoder'):
                    components.append('text_encoder')
                if hasattr(self.pipeline, 'controlnet'):
                    components.append('controlnet')
                info.update({
                    'pipeline_components': components,
                    'scheduler': type(self.pipeline.scheduler).__name__ if hasattr(self.pipeline, 'scheduler') else 'Unknown'
                })
            except Exception as e:
                logger.debug(f"Failed to get ControlNet model details: {e}")
        return info
       
    def set_scheduler(self, scheduler_name: str):
        """Set the diffusion scheduler"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        schedulers = {
            'ddim': 'DDIMScheduler',
            'pndm': 'PNDMScheduler',
            'lms': 'LMSDiscreteScheduler',
            'euler': 'EulerDiscreteScheduler',
            'euler_ancestral': 'EulerAncestralDiscreteScheduler',
            'dpm': 'DPMSolverMultistepScheduler',
            'ddpm': 'DDPMScheduler'
        }
        if scheduler_name not in schedulers:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        try:
            from diffusers import __dict__ as diffusers_dict
            scheduler_class = diffusers_dict[schedulers[scheduler_name]]
            self.pipeline.scheduler = scheduler_class.from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"Scheduler set to: {scheduler_name}")
        except Exception as e:
            logger.error(f"Failed to set scheduler: {e}")
            raise

    def unload_model(self) -> None:
        """Unload model and free memory"""
        if self.pipeline is not None:
            # Try to unload pipeline components
            if hasattr(self.pipeline, 'unet'):
                del self.pipeline.unet
            if hasattr(self.pipeline, 'vae'):
                del self.pipeline.vae
            if hasattr(self.pipeline, 'text_encoder'):
                del self.pipeline.text_encoder
            if hasattr(self.pipeline, 'controlnet'):
                del self.pipeline.controlnet
            del self.pipeline
            self.pipeline = None
        if self.controlnet is not None:
            del self.controlnet
            self.controlnet = None
        if self.preprocessor is not None:
            del self.preprocessor
            self.preprocessor = None
        self.is_loaded = False
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ControlNet model unloaded")    

