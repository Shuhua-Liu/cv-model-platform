"""
FLUX 生成适配器 - 支持Black Forest Labs的FLUX模型

支持的模型：
- FLUX.1-dev
- FLUX.1-schnell
- FLUX.1-pro (通过API)
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
    logger.warning("diffusers未安装或版本不支持FLUX，FLUX适配器将不可用")


class FluxAdapter(GenerationAdapter):
    """FLUX模型适配器"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 variant: str = "fp16",  # fp16, fp32
                 enable_memory_efficient_attention: bool = True,
                 cpu_offload: bool = False,
                 **kwargs):
        """
        初始化FLUX适配器
        
        Args:
            model_path: 模型文件路径或HuggingFace模型ID
            device: 计算设备
            variant: 模型精度变体 (fp16, fp32)
            enable_memory_efficient_attention: 启用内存高效注意力
            cpu_offload: 启用CPU卸载以节省GPU内存
        """
        if not FLUX_AVAILABLE:
            raise ImportError("需要安装支持FLUX的diffusers版本: pip install diffusers>=0.30.0")
        
        super().__init__(model_path, device, **kwargs)
        
        self.variant = variant
        self.enable_memory_efficient_attention = enable_memory_efficient_attention
        self.cpu_offload = cpu_offload
        
        # 确定模型类型
        self.model_type = self._determine_model_type()
        
        # 推理管道
        self.pipeline = None
    
    def _determine_model_type(self) -> str:
        """确定FLUX模型类型"""
        path_str = str(self.model_path).lower()
        
        if 'schnell' in path_str:
            return 'flux-schnell'
        elif 'dev' in path_str:
            return 'flux-dev'
        elif 'pro' in path_str:
            return 'flux-pro'
        else:
            return 'flux-dev'  # 默认为dev版本
    
    def load_model(self) -> None:
        """加载FLUX模型"""
        try:
            logger.info(f"加载FLUX模型: {self.model_path} (类型: {self.model_type})")
            
            # 准备加载参数
            load_kwargs = {
                'torch_dtype': torch.float16 if self.variant == 'fp16' and 'cuda' in self.device else torch.float32,
            }
            
            # 根据模型路径类型选择加载方式
            model_path_str = str(self.model_path)
            
            # 方法1: 如果是HuggingFace模型ID或目录
            if not self.model_path.exists() or self.model_path.is_dir():
                logger.info("从HuggingFace或目录加载FLUX模型...")
                
                # 尝试常见的FLUX模型ID
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
                        logger.info(f"成功加载FLUX模型: {model_id}")
                        break
                    except Exception as e:
                        logger.debug(f"尝试加载 {model_id} 失败: {e}")
                        continue
                else:
                    raise ValueError("无法加载任何FLUX模型")
            
            # 方法2: 如果是本地文件
            elif self.model_path.is_file():
                logger.info("从本地文件加载FLUX模型...")
                
                if model_path_str.endswith('.safetensors'):
                    # SafeTensors格式 - FLUX通常需要完整的pipeline目录
                    raise ValueError("FLUX模型需要完整的pipeline目录，不支持单文件加载")
                else:
                    raise ValueError(f"不支持的FLUX模型文件格式: {self.model_path.suffix}")
            
            else:
                raise ValueError(f"无效的模型路径: {self.model_path}")
            
            # 移动到指定设备
            self.pipeline = self.pipeline.to(self.device)
            
            # 应用优化
            self._apply_optimizations()
            
            self.is_loaded = True
            logger.info(f"FLUX模型加载成功 - 类型: {self.model_type}")
            
        except Exception as e:
            logger.error(f"FLUX模型加载失败: {e}")
            raise
    
    def _apply_optimizations(self) -> None:
        """应用性能优化"""
        try:
            # 启用内存高效注意力
            if self.enable_memory_efficient_attention:
                self.pipeline.enable_attention_slicing()
                logger.info("已启用注意力切片")
            
            # 启用CPU卸载
            if self.cpu_offload:
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("已启用CPU卸载")
                except Exception as e:
                    logger.warning(f"CPU卸载启用失败: {e}")
                    
        except Exception as e:
            logger.warning(f"应用优化失败: {e}")
    
    def preprocess(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """预处理生成参数"""
        # FLUX特定的参数设置
        params = {
            'prompt': prompt,
            'num_inference_steps': kwargs.get('num_steps', 50 if self.model_type == 'flux-dev' else 4),
            'guidance_scale': kwargs.get('guidance_scale', 3.5),  # FLUX使用较低的guidance_scale
            'width': kwargs.get('width', 1024),  # FLUX默认1024x1024
            'height': kwargs.get('height', 1024),
            'num_images_per_prompt': kwargs.get('num_images', 1),
            'seed': kwargs.get('seed', None)
        }
        
        # 验证参数范围
        params['num_inference_steps'] = max(1, min(100, params['num_inference_steps']))
        params['guidance_scale'] = max(0.0, min(10.0, params['guidance_scale']))
        params['width'] = max(256, min(2048, params['width']))
        params['height'] = max(256, min(2048, params['height']))
        params['num_images_per_prompt'] = max(1, min(4, params['num_images_per_prompt']))
        
        # 确保尺寸是16的倍数（FLUX要求）
        params['width'] = (params['width'] // 16) * 16
        params['height'] = (params['height'] // 16) * 16
        
        # 设置随机种子
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
        执行图像生成
        
        Args:
            prompt: 正向提示词
            num_steps: 推理步数（FLUX-dev默认50，FLUX-schnell默认4）
            guidance_scale: 引导尺度（FLUX推荐3.5）
            width: 图像宽度
            height: 图像高度
            num_images: 生成图像数量
            seed: 随机种子
            
        Returns:
            生成结果字典
        """
        if not self.is_loaded:
            self.load_model()
        
        # 根据模型类型设置默认步数
        if num_steps is None:
            num_steps = 50 if self.model_type == 'flux-dev' else 4
        
        try:
            # 预处理参数
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
            
            # 执行生成
            start_time = time.time()
            
            logger.info(f"开始FLUX生成: {params['width']}x{params['height']}, {params['num_inference_steps']}步")
            
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
            
            logger.info(f"FLUX图像生成完成 - 耗时: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"FLUX生成失败: {e}")
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
                filename = f"flux_generated_{timestamp}_{i}.png"
                
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
            logger.error(f"FLUX后处理失败: {e}")
            raise
    
    def generate_and_save(self,
                         prompt: str,
                         save_dir: Union[str, Path] = "flux_outputs",
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
            logger.info(f"FLUX图像已保存: {file_path}")
        
        return saved_paths
    
    def warmup(self, num_runs: int = 1) -> Dict[str, float]:
        """模型预热（FLUX预热时间较长，建议少次数）"""
        if not self.is_loaded:
            self.load_model()
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                # 使用简单提示词和较少步数进行预热
                _ = self.predict(
                    prompt="a simple test",
                    num_steps=1,  # 最少步数
                    width=512,    # 较小尺寸
                    height=512
                )
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                logger.info(f"FLUX预热运行 {i+1}/{num_runs} 完成: {warmup_time:.2f}s")
            except Exception as e:
                logger.warning(f"FLUX预热运行 {i+1} 失败: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"FLUX模型预热完成 - 平均耗时: {avg_time:.3f}s")
            
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
                # 获取管道组件信息
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
                logger.debug(f"获取FLUX模型详细信息失败: {e}")
        
        return info
    
    def unload_model(self) -> None:
        """卸载模型释放内存"""
        if self.pipeline is not None:
            # 尝试卸载管道组件
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
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("FLUX模型已卸载")
