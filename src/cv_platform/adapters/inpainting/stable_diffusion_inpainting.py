"""
Stable Diffusion Inpainting Adapter

Supports SD 1.5, SD 2.0/2.1 inpainting models
High-quality diffusion-based inpainting with text guidance
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
    from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logger.warning("diffusers not available for SD inpainting")


class StableDiffusionInpaintingAdapter(GenerationAdapter):
    """Stable Diffusion inpainting adapter"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 variant: str = "fp16",
                 enable_memory_efficient_attention: bool = True,
                 cpu_offload: bool = False,
                 **kwargs):
        """
        Initialize SD inpainting adapter
        
        Args:
            model_path: Path to SD inpainting model
            device: Computing device
            variant: Model precision (fp16/fp32)
            enable_memory_efficient_attention: Enable memory optimization
            cpu_offload: Enable CPU offload
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers required: pip install diffusers")
        
        super().__init__(model_path, device, **kwargs)
        
        self.variant = variant
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.cpu_offload = cpu_offload
        
        self.pipeline = None
        self.model_version = self._determine_version()
    
    def _determine_version(self) -> str:
        """Determine SD inpainting model version"""
        path_str = str(self.model_path).lower()
        
        if any(v in path_str for v in ['2.1', '2-1', 'v2-1']):
            return 'sd21_inpainting'
        elif any(v in path_str for v in ['2.0', '2-0', 'v2-0', '_2_']):
            return 'sd20_inpainting'
        elif any(v in path_str for v in ['1.5', '1-5', 'v1-5']):
            return 'sd15_inpainting'
        else:
            return 'sd20_inpainting'  # Default
    
    def load_model(self) -> None:
        """Load SD inpainting model"""
        try:
            logger.info(f"Loading SD inpainting model: {self.model_version}")
            
            # Load pipeline
            if self.model_path.exists():
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    str(self.model_path),
                    torch_dtype=torch.float16 if self.variant == 'fp16' else torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:
                # Load from hub
                model_id = "stabilityai/stable-diffusion-2-inpainting"
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.variant == 'fp16' else torch.float32
                )
            
            # Apply optimizations
            if self.enable_memory_efficient_attention:
                self.pipeline.enable_attention_slicing()
            
            # Move to device
            if not self.cpu_offload:
                self.pipeline = self.pipeline.to(self.device)
            else:
                self.pipeline.enable_sequential_cpu_offload()
            
            self.is_loaded = True
            logger.info("SD inpainting model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load SD inpainting model: {e}")
            raise
    
    def inpaint(self, 
                prompt: str,
                image: Union[str, Path, Image.Image],
                mask: Union[str, Path, Image.Image],
                negative_prompt: Optional[str] = None,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                strength: float = 1.0,
                width: int = 512,
                height: int = 512,
                seed: Optional[int] = None) -> Image.Image:
        """
        Perform text-guided inpainting
        
        Args:
            prompt: Text prompt describing desired inpainting
            image: Input image
            mask: Mask indicating areas to inpaint
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            strength: Inpainting strength
            width: Output width
            height: Output height
            seed: Random seed
            
        Returns:
            Inpainted image
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Process inputs
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        if isinstance(mask, (str, Path)):
            mask = Image.open(mask).convert('L')
        
        # Resize inputs
        image = image.resize((width, height), Image.LANCZOS)
        mask = mask.resize((width, height), Image.LANCZOS)
        
        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
        
        # Generate
        with torch.no_grad():
            result = self.pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                width=width,
                height=height
            )
        
        return result.images[0]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'inpainting',
            'framework': 'diffusers',
            'architecture': 'stable_diffusion_inpainting',
            'version': self.model_version,
            'text_guided': True,
            'max_resolution': '768x768',
            'recommended_steps': 50
        })
        
        return info
    
    def unload_model(self) -> None:
        """Unload model and free memory"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        logger.info("SD inpainting model unloaded")