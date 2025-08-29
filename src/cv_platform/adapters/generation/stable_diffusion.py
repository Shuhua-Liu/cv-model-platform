"""
Stable Diffusion Generator Adapter - Supports Stable Diffusion Models from the Diffusers Library

Supported Models:
- Stable Diffusion 1.5
- Stable Diffusion 2.0/2.1
- Stable Diffusion XL (SDXL)
- FLUX (partial support)
"""

import time
from typing import Dict, Any, Union, List, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from loguru import logger

from ..base import GenerationAdapter

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DiffusionPipeline,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler
    )
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("diffusers are not installed; Stable Diffusion adapter will be unavailable.")


class StableDiffusionAdapter(GenerationAdapter):
    """Stable Diffusion Model Adapter"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 variant: str = "fp16",  # fp16, fp32
                 enable_memory_efficient_attention: bool = True,
                 enable_xformers: bool = False,
                 cpu_offload: bool = False,
                 **kwargs):
        """
        Initialize Stable Diffusion Adapter
        
        Args:
            model_path: Model file path or HuggingFace model ID
            device: Computing device
            variant: Model precision variant (fp16, fp32)
            enable_memory_efficient_attention: Enable memory-efficient attention
            enable_xformers: Enable xformers optimizations
            cpu_offload: Enable CPU offloading to conserve GPU memory
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("Require install diffusers: pip install diffusers transformers")
        
        super().__init__(model_path, device, **kwargs)
        
        self.variant = variant
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.enable_xformers = enable_xformers
        self.cpu_offload = cpu_offload
        
        # Determine model type
        self.model_type = self._determine_model_type()
        
        # Inference pipeline
        self.pipeline = None
    
    def _determine_model_type(self) -> str:
        """Determine Stable Diffusion model type"""
        path_str = str(self.model_path).lower()
        
        if any(keyword in path_str for keyword in ['sdxl', 'xl', 'stable-diffusion-xl']):
            return 'sdxl'
        elif any(keyword in path_str for keyword in ['flux']):
            return 'flux'
        elif any(keyword in path_str for keyword in ['v2', '2.0', '2.1']):
            return 'sd2'
        else:
            return 'sd1'  # Default to SD 1.x
    
    def load_model(self) -> None:
        """Load Stable Diffusion model"""
        try:
            logger.info(f"Load Stable Diffusion model: {self.model_path} (Type: {self.model_type})")
            
            # Prepare to load parameters
            load_kwargs = {
                'torch_dtype': torch.float16 if self.variant == 'fp16' and 'cuda' in self.device else torch.float32,
                'safety_checker': None,  # Disable the security checker to save memory.
                'requires_safety_checker': False
            }
            
            # Select pipeline based on model type
            if self.model_type == 'sdxl':
                pipeline_class = StableDiffusionXLPipeline
            else:
                pipeline_class = StableDiffusionPipeline
            
            # Try different loading methods
            model_path_str = str(self.model_path)
            
            # Method 1: If it is a HuggingFace model directory or ID
            if self.model_path.is_dir() or not self.model_path.exists():
                logger.info("Load models from Hugging Face or the directory...")
                self.pipeline = pipeline_class.from_pretrained(
                    model_path_str,
                    **load_kwargs
                )
            
            # Method 2: For a single file (.safetensors or .ckpt)
            elif self.model_path.is_file():
                logger.info("Load model from a single file...")
                
                if model_path_str.endswith('.safetensors'):
                    # SafeTensors format
                    self.pipeline = pipeline_class.from_single_file(
                        model_path_str,
                        **load_kwargs
                    )
                elif model_path_str.endswith(('.ckpt', '.pt', '.pth')):
                    # PyTorch checkpoint format
                    self.pipeline = pipeline_class.from_single_file(
                        model_path_str,
                        **load_kwargs
                    )
                else:
                    raise ValueError(f"Unsupported format: {self.model_path.suffix}")
            
            else:
                raise ValueError(f"Invalid model path: {self.model_path}")
            
            # Move to specific device
            self.pipeline = self.pipeline.to(self.device)
            
            # Apply optimization
            self._apply_optimizations()
            
            self.is_loaded = True
            logger.info(f"Stable Diffusion model loaded successfully - Type: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Stable Diffusion model failed to load: {e}")
            raise
    
    def _apply_optimizations(self) -> None:
        """Application Performance Optimization"""
        try:
            # Enable Memory-Efficient Attention
            if self.enable_memory_efficient_attention:
                self.pipeline.enable_attention_slicing()
                logger.info("Attention slicing has been enabled.")
            
            # Enable xformers optimization (if available)
            if self.enable_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Memory-efficient attention with xformers has been enabled.")
                except Exception as e:
                    logger.warning(f"xformers optimization activation failed: {e}")
            
            # Enable CPU Offload
            if self.cpu_offload:
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("CPU offloading has been enabled.")
                except Exception as e:
                    logger.warning(f"CPU unload activation failed: {e}")
            
            # Set a faster scheduler
            try:
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )
                logger.info("The DPMSolver scheduler has been configured.")
            except Exception as e:
                logger.warning(f"Scheduler configuration failed: {e}")
                
        except Exception as e:
            logger.warning(f"Application optimization failed: {e}")
    
    def preprocess(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Preprocessing Generate Parameters"""
        # Basic Parameter Verification and Setup
        params = {
            'prompt': prompt,
            'negative_prompt': kwargs.get('negative_prompt', None),
            'num_inference_steps': kwargs.get('num_steps', 20),
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'width': kwargs.get('width', 512),
            'height': kwargs.get('height', 512),
            'num_images_per_prompt': kwargs.get('num_images', 1),
            'seed': kwargs.get('seed', None)
        }
        
        # Verify Parameter Range
        params['num_inference_steps'] = max(1, min(100, params['num_inference_steps']))
        params['guidance_scale'] = max(1.0, min(20.0, params['guidance_scale']))
        params['width'] = max(64, min(2048, params['width']))
        params['height'] = max(64, min(2048, params['height']))
        params['num_images_per_prompt'] = max(1, min(4, params['num_images_per_prompt']))
        
        # Ensure size is a multiple of 8.（Stable Diffusion requirement）
        params['width'] = (params['width'] // 8) * 8
        params['height'] = (params['height'] // 8) * 8
        
        # Set random seed
        if params['seed'] is not None:
            generator = torch.Generator(device=self.device).manual_seed(params['seed'])
            params['generator'] = generator
        
        return params
    
    def predict(self, 
                prompt: str,
                negative_prompt: Optional[str] = None,
                num_steps: int = 20,
                guidance_scale: float = 7.5,
                width: int = 512,
                height: int = 512,
                num_images: int = 1,
                seed: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute Image Generation
        
        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            num_steps: Number of inference steps
            guidance_scale: Guidance scale
            width: Image width
            height: Image height
            num_images: Number of images to generate
            seed: Random seed
            
        Returns:
            Dictionary of generated results
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # Preprocessing Parameters
            params = self.preprocess(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_images=num_images,
                seed=seed,
                **kwargs
            )
            
            # Execute Generation
            start_time = time.time()
            
            logger.info(f"Begin generating image:  {params['width']}x{params['height']}, {params['num_inference_steps']} steps")
            
            with torch.no_grad():
                # Remove the seed parameter we added and pass the generator
                pipeline_params = params.copy()
                pipeline_params.pop('seed', None)
                
                result = self.pipeline(**pipeline_params)
                
            inference_time = time.time() - start_time
            
            # Post-processing results
            processed_results = self.postprocess(
                result,
                params=params,
                inference_time=inference_time
            )
            
            logger.info(f"Image generation complete - Duration: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"Stable Diffusion generation failed: {e}")
            raise
    
    def postprocess(self, 
                   pipeline_result: Any,
                   params: Dict[str, Any],
                   **kwargs) -> Dict[str, Any]:
        """Post-processing generates results"""
        try:
            # Retrieve the generated image
            images = pipeline_result.images
            
            # Prepare image information
            image_info = []
            for i, image in enumerate(images):
                # Generate filename
                timestamp = int(time.time() * 1000)
                filename = f"generated_{timestamp}_{i}.png"
                
                image_info.append({
                    'image': image,
                    'filename': filename,
                    'seed': params.get('seed'),
                    'width': image.width,
                    'height': image.height
                })
            
            # Build Results
            result = {
                'images': images,
                'image_info': image_info,
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'model_type': self.model_type,
                    'parameters': {
                        'prompt': params['prompt'],
                        'negative_prompt': params.get('negative_prompt'),
                        'num_inference_steps': params['num_inference_steps'],
                        'guidance_scale': params['guidance_scale'],
                        'width': params['width'],
                        'height': params['height'],
                        'num_images': len(images),
                        'seed': params.get('seed')
                    }
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Stable Diffusion post-processing failed: {e}")
            raise
    
    def generate_and_save(self,
                         prompt: str,
                         save_dir: Union[str, Path] = "outputs",
                         **kwargs) -> List[str]:
        """Generate an image and save it to a file"""
        # Execute generation
        results = self.predict(prompt, **kwargs)
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save images
        saved_paths = []
        for image_info in results['image_info']:
            image = image_info['image']
            filename = image_info['filename']
            file_path = save_path / filename
            
            image.save(file_path)
            saved_paths.append(str(file_path))
            logger.info(f"Image saved: {file_path}")
        
        return saved_paths
    
    def warmup(self, num_runs: int = 2) -> Dict[str, float]:
        """Model Warm-up"""
        if not self.is_loaded:
            self.load_model()
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                # Warm up using simple prompts
                _ = self.predict(
                    prompt="a simple test image",
                    num_steps=4,  # Use fewer steps to speed up warmup
                    width=256,    # Use a smaller size
                    height=256
                )
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                logger.info(f"Warm-up run {i+1}/{num_runs} completed: {warmup_time:.2f}s")
            except Exception as e:
                logger.warning(f"Warm-up run {i+1} failed: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"Stable Diffusion model warmup complete - Average Duration: {avg_time:.3f}s")
            
            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return super().warmup(num_runs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model details"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'generation',
            'framework': 'diffusers',
            'architecture': f'stable_diffusion_{self.model_type}',
            'variant': self.variant,
            'optimizations': {
                'memory_efficient_attention': self.enable_memory_efficient_attention,
                'xformers': self.enable_xformers,
                'cpu_offload': self.cpu_offload
            }
        })
        
        if self.is_loaded:
            try:
                # Retrieve pipeline component information
                components = []
                if hasattr(self.pipeline, 'unet'):
                    components.append('unet')
                if hasattr(self.pipeline, 'vae'):
                    components.append('vae')
                if hasattr(self.pipeline, 'text_encoder'):
                    components.append('text_encoder')
                if hasattr(self.pipeline, 'tokenizer'):
                    components.append('tokenizer')
                
                info.update({
                    'pipeline_components': components,
                    'scheduler': type(self.pipeline.scheduler).__name__
                })
                
            except Exception as e:
                logger.debug(f"Failed to retrieve model details: {e}")
        
        return info
    
    def set_scheduler(self, scheduler_name: str):
        """Set scheduler"""
        if not self.is_loaded:
            raise ValueError("Model has not been loaded.")
        
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
            raise ValueError(f"Unsupported schedulers: {scheduler_name}")
        
        try:
            from diffusers import __dict__ as diffusers_dict
            scheduler_class = diffusers_dict[schedulers[scheduler_name]]
            self.pipeline.scheduler = scheduler_class.from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"The scheduler has been set to: {scheduler_name}")
        except Exception as e:
            logger.error(f"Setting up the scheduler failed: {e}")
            raise
    
    def unload_model(self) -> None:
        """Unload the model to free up memory"""
        if self.pipeline is not None:
            # Attempt to uninstall pipeline components
            if hasattr(self.pipeline, 'unet'):
                del self.pipeline.unet
            if hasattr(self.pipeline, 'vae'):
                del self.pipeline.vae
            if hasattr(self.pipeline, 'text_encoder'):
                del self.pipeline.text_encoder
                
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            
            # Clear GPU cache 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Stable Diffusion model has been unloaded")
