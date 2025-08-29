"""
FLUX Generation Adapter - Supports Black Forest Labs' FLUX Models

Supported Models:
- FLUX.1-dev
- FLUX.1-schnell
- FLUX.1-pro (via API)
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
    from diffusers import FluxPipeline, FluxTransformer2DModel
    FLUX_AVAILABLE = True
except ImportError:
    FLUX_AVAILABLE = False
    logger.warning("diffusers are either not installed or their version does not support FLUX; the FLUX adapter will be unavailable")


class FluxAdapter(GenerationAdapter):
    """FLUX Model Adapter"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 variant: str = "fp16",  # fp16, fp32
                 enable_memory_efficient_attention: bool = True,
                 cpu_offload: bool = False,
                 **kwargs):
        """
        Initialize FLUX adapter
        
        Args:
            model_path: Model file path or HuggingFace model ID
            device: Computing device
            variant: Model precision variant (fp16, fp32)
            enable_memory_efficient_attention: Enable memory-efficient attention
            cpu_offload: Enable CPU offloading to conserve GPU memory
        """
        if not FLUX_AVAILABLE:
            raise ImportError("Version of diffusers that supports FLUX needs to be installed.: pip install diffusers>=0.30.0")
        
        super().__init__(model_path, device, **kwargs)
        
        self.variant = variant
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.cpu_offload = cpu_offload
        
        # Determine model type
        self.model_type = self._determine_model_type()
        
        # Inference Pipeline
        self.pipeline = None
    
    def _determine_model_type(self) -> str:
        """Determine FLUX model type"""
        path_str = str(self.model_path).lower()
        
        if 'schnell' in path_str:
            return 'flux-schnell'
        elif 'dev' in path_str:
            return 'flux-dev'
        elif 'pro' in path_str:
            return 'flux-pro'
        else:
            return 'flux-dev'  # Default to dev version
    
    def load_model(self) -> None:
        """Load FLUX model"""
        try:
            logger.info(f"Load FLUX model: {self.model_path} (Type: {self.model_type})")
            
            # Ready to load parameters
            load_kwargs = {
                'torch_dtype': torch.float16 if self.variant == 'fp16' and 'cuda' in self.device else torch.float32,
            }
            
            # Select loading method based on model path type
            model_path_str = str(self.model_path)
            
            # Method 1: If it is a Hugging Face model ID or directory
            if not self.model_path.exists() or self.model_path.is_dir():
                logger.info("Load the FLUX model from HuggingFace or the directory...")
                
                # Try common FLUX model IDs
                possible_model_ids = [
                    model_path_str,
                    "black-forest-labs/FLUX.1-dev",
                    "black-forest-labs/FLUX.1-schnell"
                ]
                
                for model_id in possible_model_ids:
                    try:
                        self.pipeline = FluxPipeline.from_pretrained(
                            model_id,
                            **load_kwargs
                        )
                        logger.info(f"FLUX model loaded successfully: {model_id}")
                        break
                    except Exception as e:
                        logger.debug(f"Failed to load {model_id}: {e}")
                        continue
                else:
                    raise ValueError("Unable to load any FLUX models")
            
            # Method 2: If local file
            elif self.model_path.is_file():
                logger.info("Load FLUX models from local files...")
                
                if model_path_str.endswith('.safetensors'):
                    # SafeTensors Format - FLUX typically requires the complete pipeline directory
                    raise ValueError("FLUX model requires a complete pipeline directory and does not support loading individual files")
                else:
                    raise ValueError(f"Unsupported FLUX model file formats: {self.model_path.suffix}")
            
            else:
                raise ValueError(f"Invalid model path: {self.model_path}")
            
            # Move to specific device
            self.pipeline = self.pipeline.to(self.device)
            
            # Appy optimizations
            self._apply_optimizations()
            
            self.is_loaded = True
            logger.info(f"FLUX model loaded successfully - Type: {self.model_type}")
            
        except Exception as e:
            logger.error(f"FLUX model loading failed: {e}")
            raise
    
    def _apply_optimizations(self) -> None:
        """Application Performance Optimization"""
        try:
            # Enable Memory-Efficient Attention
            if self.enable_memory_efficient_attention:
                self.pipeline.enable_attention_slicing()
                logger.info("Attention slicing has been enabled.")
            
            # Enable CPU Offload
            if self.cpu_offload:
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("CPU offload enabled")
                except Exception as e:
                    logger.warning(f"CPU offload failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Application optimization failed: {e}")
    
    def preprocess(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Preprocessing Generate Parameters"""
        # FLUX-specific parameter settings
        params = {
            'prompt': prompt,
            'num_inference_steps': kwargs.get('num_steps', 50 if self.model_type == 'flux-dev' else 4),
            'guidance_scale': kwargs.get('guidance_scale', 3.5),  # FLUX uses a lower guidance_scale
            'width': kwargs.get('width', 1024),  # FLUX defaults to 1024x1024
            'height': kwargs.get('height', 1024),
            'num_images_per_prompt': kwargs.get('num_images', 1),
            'seed': kwargs.get('seed', None)
        }
        
        # Verify parameter range
        params['num_inference_steps'] = max(1, min(100, params['num_inference_steps']))
        params['guidance_scale'] = max(0.0, min(10.0, params['guidance_scale']))
        params['width'] = max(256, min(2048, params['width']))
        params['height'] = max(256, min(2048, params['height']))
        params['num_images_per_prompt'] = max(1, min(4, params['num_images_per_prompt']))
        
        # Ensure the size is a multiple of 16 (FLUX requirement)
        params['width'] = (params['width'] // 16) * 16
        params['height'] = (params['height'] // 16) * 16
        
        # Set random seed
        if params['seed'] is not None:
            generator = torch.Generator(device=self.device).manual_seed(params['seed'])
            params['generator'] = generator
        
        return params
    
    def predict(self, 
                prompt: str,
                num_steps: int = None,
                guidance_scale: float = 3.5,
                width: int = 1024,
                height: int = 1024,
                num_images: int = 1,
                seed: Optional[int] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Execute Image Generation
        
        Args:
            prompt: Forward prompt
            num_steps: Inference steps (FLUX-dev defaults to 50, FLUX-schnell defaults to 4)
            guidance_scale: Guidance scale (FLUX recommends 3.5)
            width: Image width
            height: Image height
            num_images: Number of images to generate
            seed: Random seed
            
        Returns:
            Generated results dictionary
        """
        if not self.is_loaded:
            self.load_model()
        
        # Set default step count based on model type
        if num_steps is None:
            num_steps = 50 if self.model_type == 'flux-dev' else 4
        
        try:
            # Preprocessing Parameters
            params = self.preprocess(
                prompt=prompt,
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
            
            logger.info(f"开始FLUX生成: {params['width']}x{params['height']}, {params['num_inference_steps']}步")
            
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
            
            logger.info(f"FLUX image generation complete - Duration: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"FLUX generation failed: {e}")
            raise
    
    def postprocess(self, 
                   pipeline_result: Any,
                   params: Dict[str, Any],
                   **kwargs) -> Dict[str, Any]:
        """Post-processing generates results"""
        try:
            # Retrieve generated image
            images = pipeline_result.images
            
            # Prepare image information
            image_info = []
            for i, image in enumerate(images):
                # Generate filename
                timestamp = int(time.time() * 1000)
                filename = f"flux_generated_{timestamp}_{i}.png"
                
                image_info.append({
                    'image': image,
                    'filename': filename,
                    'seed': params.get('seed'),
                    'width': image.width,
                    'height': image.height
                })
            
            # Construction Results
            result = {
                'images': images,
                'image_info': image_info,
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'model_type': self.model_type,
                    'parameters': {
                        'prompt': params['prompt'],
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
            logger.error(f"FLUX post-processing failed: {e}")
            raise
    
    def generate_and_save(self,
                         prompt: str,
                         save_dir: Union[str, Path] = "flux_outputs",
                         **kwargs) -> List[str]:
        """Generate an image and save it to a file"""
        # Execute Generation
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
            logger.info(f"FLUX image saved: {file_path}")
        
        return saved_paths
    
    def warmup(self, num_runs: int = 1) -> Dict[str, float]:
        """Model Preheating (FLUX has a longer preheating time; fewer iterations are recommended)"""
        if not self.is_loaded:
            self.load_model()
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                # Warm up using simple prompts and fewer steps.
                _ = self.predict(
                    prompt="a simple test",
                    num_steps=1,  # Minimum number of steps
                    width=512,    # Smaller size
                    height=512
                )
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                logger.info(f"FLUX warmup {i+1}/{num_runs} completed: {warmup_time:.2f}s")
            except Exception as e:
                logger.warning(f"FLUX warmup operation {i+1} failed: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"FLUX model warmup complete - Average Duration: {avg_time:.3f}s")
            
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
            'architecture': 'flux',
            'flux_variant': self.model_type,
            'variant': self.variant,
            'default_resolution': '1024x1024',
            'recommended_steps': 50 if self.model_type == 'flux-dev' else 4,
            'recommended_guidance': 3.5,
            'optimizations': {
                'memory_efficient_attention': self.enable_memory_efficient_attention,
                'cpu_offload': self.cpu_offload
            }
        })
        
        if self.is_loaded:
            try:
                # Retrieve pipeline component information
                components = []
                if hasattr(self.pipeline, 'transformer'):
                    components.append('transformer')
                if hasattr(self.pipeline, 'vae'):
                    components.append('vae')
                if hasattr(self.pipeline, 'text_encoder'):
                    components.append('text_encoder')
                if hasattr(self.pipeline, 'text_encoder_2'):
                    components.append('text_encoder_2')
                if hasattr(self.pipeline, 'tokenizer'):
                    components.append('tokenizer')
                
                info.update({
                    'pipeline_components': components,
                    'scheduler': type(self.pipeline.scheduler).__name__ if hasattr(self.pipeline, 'scheduler') else 'Unknown'
                })
                
            except Exception as e:
                logger.debug(f"Failed to retrieve detailed information about FLUX model: {e}")
        
        return info
    
    def unload_model(self) -> None:
        """Unload model to free up memory."""
        if self.pipeline is not None:
            # Attempt to uninstall pipeline components
            if hasattr(self.pipeline, 'transformer'):
                del self.pipeline.transformer
            if hasattr(self.pipeline, 'vae'):
                del self.pipeline.vae
            if hasattr(self.pipeline, 'text_encoder'):
                del self.pipeline.text_encoder
            if hasattr(self.pipeline, 'text_encoder_2'):
                del self.pipeline.text_encoder_2
                
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("FLUX model has been unloaded")
