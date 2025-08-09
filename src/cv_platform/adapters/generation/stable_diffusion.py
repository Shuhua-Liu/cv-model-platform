"""
Stable Diffusion 生成适配器 - 支持diffusers库的Stable Diffusion模型

支持的模型：
- Stable Diffusion 1.5
- Stable Diffusion 2.0/2.1
- Stable Diffusion XL (SDXL)
- FLUX（部分支持）
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
    logger.warning("diffusers未安装，Stable Diffusion适配器将不可用")


class StableDiffusionAdapter(GenerationAdapter):
    """Stable Diffusion模型适配器"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 variant: str = "fp16",  # fp16, fp32
                 enable_memory_efficient_attention: bool = True,
                 enable_xformers: bool = False,
                 cpu_offload: bool = False,
                 **kwargs):
        """
        初始化Stable Diffusion适配器
        
        Args:
            model_path: 模型文件路径或HuggingFace模型ID
            device: 计算设备
            variant: 模型精度变体 (fp16, fp32)
            enable_memory_efficient_attention: 启用内存高效注意力
            enable_xformers: 启用xformers优化
            cpu_offload: 启用CPU卸载以节省GPU内存
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("需要安装diffusers: pip install diffusers transformers")
        
        super().__init__(model_path, device, **kwargs)
        
        self.variant = variant
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.enable_xformers = enable_xformers
        self.cpu_offload = cpu_offload
        
        # 确定模型类型
        self.model_type = self._determine_model_type()
        
        # 推理管道
        self.pipeline = None
    
    def _determine_model_type(self) -> str:
        """确定Stable Diffusion模型类型"""
        path_str = str(self.model_path).lower()
        
        if any(keyword in path_str for keyword in ['sdxl', 'xl', 'stable-diffusion-xl']):
            return 'sdxl'
        elif any(keyword in path_str for keyword in ['flux']):
            return 'flux'
        elif any(keyword in path_str for keyword in ['v2', '2.0', '2.1']):
            return 'sd2'
        else:
            return 'sd1'  # 默认为SD 1.x
    
    def load_model(self) -> None:
        """加载Stable Diffusion模型"""
        try:
            logger.info(f"加载Stable Diffusion模型: {self.model_path} (类型: {self.model_type})")
            
            # 准备加载参数
            load_kwargs = {
                'torch_dtype': torch.float16 if self.variant == 'fp16' and 'cuda' in self.device else torch.float32,
                'safety_checker': None,  # 禁用安全检查器以节省内存
                'requires_safety_checker': False
            }
            
            # 根据模型类型选择管道
            if self.model_type == 'sdxl':
                pipeline_class = StableDiffusionXLPipeline
            else:
                pipeline_class = StableDiffusionPipeline
            
            # 尝试不同的加载方法
            model_path_str = str(self.model_path)
            
            # 方法1: 如果是HuggingFace模型目录或ID
            if self.model_path.is_dir() or not self.model_path.exists():
                logger.info("从HuggingFace或目录加载模型...")
                self.pipeline = pipeline_class.from_pretrained(
                    model_path_str,
                    **load_kwargs
                )
            
            # 方法2: 如果是单个文件（.safetensors或.ckpt）
            elif self.model_path.is_file():
                logger.info("从单个文件加载模型...")
                
                if model_path_str.endswith('.safetensors'):
                    # SafeTensors格式
                    self.pipeline = pipeline_class.from_single_file(
                        model_path_str,
                        **load_kwargs
                    )
                elif model_path_str.endswith(('.ckpt', '.pt', '.pth')):
                    # PyTorch检查点格式
                    self.pipeline = pipeline_class.from_single_file(
                        model_path_str,
                        **load_kwargs
                    )
                else:
                    raise ValueError(f"不支持的文件格式: {self.model_path.suffix}")
            
            else:
                raise ValueError(f"无效的模型路径: {self.model_path}")
            
            # 移动到指定设备
            self.pipeline = self.pipeline.to(self.device)
            
            # 应用优化
            self._apply_optimizations()
            
            self.is_loaded = True
            logger.info(f"Stable Diffusion模型加载成功 - 类型: {self.model_type}")
            
        except Exception as e:
            logger.error(f"Stable Diffusion模型加载失败: {e}")
            raise
    
    def _apply_optimizations(self) -> None:
        """应用性能优化"""
        try:
            # 启用内存高效注意力
            if self.enable_memory_efficient_attention:
                self.pipeline.enable_attention_slicing()
                logger.info("已启用注意力切片")
            
            # 启用xformers优化（如果可用）
            if self.enable_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("已启用xformers内存高效注意力")
                except Exception as e:
                    logger.warning(f"xformers优化启用失败: {e}")
            
            # 启用CPU卸载
            if self.cpu_offload:
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("已启用CPU卸载")
                except Exception as e:
                    logger.warning(f"CPU卸载启用失败: {e}")
            
            # 设置更快的调度器
            try:
                self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )
                logger.info("已设置DPMSolver调度器")
            except Exception as e:
                logger.warning(f"调度器设置失败: {e}")
                
        except Exception as e:
            logger.warning(f"应用优化失败: {e}")
    
    def preprocess(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """预处理生成参数"""
        # 基本参数验证和设置
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
        
        # 验证参数范围
        params['num_inference_steps'] = max(1, min(100, params['num_inference_steps']))
        params['guidance_scale'] = max(1.0, min(20.0, params['guidance_scale']))
        params['width'] = max(64, min(2048, params['width']))
        params['height'] = max(64, min(2048, params['height']))
        params['num_images_per_prompt'] = max(1, min(4, params['num_images_per_prompt']))
        
        # 确保尺寸是8的倍数（Stable Diffusion要求）
        params['width'] = (params['width'] // 8) * 8
        params['height'] = (params['height'] // 8) * 8
        
        # 设置随机种子
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
        执行图像生成
        
        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            num_steps: 推理步数
            guidance_scale: 引导尺度
            width: 图像宽度
            height: 图像高度
            num_images: 生成图像数量
            seed: 随机种子
            
        Returns:
            生成结果字典
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # 预处理参数
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
            
            # 执行生成
            start_time = time.time()
            
            logger.info(f"开始生成图像: {params['width']}x{params['height']}, {params['num_inference_steps']}步")
            
            with torch.no_grad():
                # 移除我们添加的seed参数，传递generator
                pipeline_params = params.copy()
                pipeline_params.pop('seed', None)
                
                result = self.pipeline(**pipeline_params)
                
            inference_time = time.time() - start_time
            
            # 后处理结果
            processed_results = self.postprocess(
                result,
                params=params,
                inference_time=inference_time
            )
            
            logger.info(f"图像生成完成 - 耗时: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"Stable Diffusion生成失败: {e}")
            raise
    
    def postprocess(self, 
                   pipeline_result: Any,
                   params: Dict[str, Any],
                   **kwargs) -> Dict[str, Any]:
        """后处理生成结果"""
        try:
            # 获取生成的图像
            images = pipeline_result.images
            
            # 准备图像信息
            image_info = []
            for i, image in enumerate(images):
                # 生成文件名
                timestamp = int(time.time() * 1000)
                filename = f"generated_{timestamp}_{i}.png"
                
                image_info.append({
                    'image': image,
                    'filename': filename,
                    'seed': params.get('seed'),
                    'width': image.width,
                    'height': image.height
                })
            
            # 构建结果
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
            logger.error(f"Stable Diffusion后处理失败: {e}")
            raise
    
    def generate_and_save(self,
                         prompt: str,
                         save_dir: Union[str, Path] = "outputs",
                         **kwargs) -> List[str]:
        """生成图像并保存到文件"""
        # 执行生成
        results = self.predict(prompt, **kwargs)
        
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        saved_paths = []
        for image_info in results['image_info']:
            image = image_info['image']
            filename = image_info['filename']
            file_path = save_path / filename
            
            image.save(file_path)
            saved_paths.append(str(file_path))
            logger.info(f"图像已保存: {file_path}")
        
        return saved_paths
    
    def warmup(self, num_runs: int = 2) -> Dict[str, float]:
        """模型预热"""
        if not self.is_loaded:
            self.load_model()
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                # 使用简单提示词进行预热
                _ = self.predict(
                    prompt="a simple test image",
                    num_steps=4,  # 使用较少步数以加快预热
                    width=256,    # 使用较小尺寸
                    height=256
                )
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                logger.info(f"预热运行 {i+1}/{num_runs} 完成: {warmup_time:.2f}s")
            except Exception as e:
                logger.warning(f"预热运行 {i+1} 失败: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"Stable Diffusion模型预热完成 - 平均耗时: {avg_time:.3f}s")
            
            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return super().warmup(num_runs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
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
                # 获取管道组件信息
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
                logger.debug(f"获取模型详细信息失败: {e}")
        
        return info
    
    def set_scheduler(self, scheduler_name: str):
        """设置调度器"""
        if not self.is_loaded:
            raise ValueError("模型未加载")
        
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
            raise ValueError(f"不支持的调度器: {scheduler_name}")
        
        try:
            from diffusers import __dict__ as diffusers_dict
            scheduler_class = diffusers_dict[schedulers[scheduler_name]]
            self.pipeline.scheduler = scheduler_class.from_config(
                self.pipeline.scheduler.config
            )
            logger.info(f"调度器已设置为: {scheduler_name}")
        except Exception as e:
            logger.error(f"设置调度器失败: {e}")
            raise
    
    def unload_model(self) -> None:
        """卸载模型释放内存"""
        if self.pipeline is not None:
            # 尝试卸载管道组件
            if hasattr(self.pipeline, 'unet'):
                del self.pipeline.unet
            if hasattr(self.pipeline, 'vae'):
                del self.pipeline.vae
            if hasattr(self.pipeline, 'text_encoder'):
                del self.pipeline.text_encoder
                
            del self.pipeline
            self.pipeline = None
            self.is_loaded = False
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Stable Diffusion模型已卸载")
