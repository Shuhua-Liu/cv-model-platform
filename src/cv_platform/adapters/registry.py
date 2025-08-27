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
    """适配器注册中心"""
    
    def __init__(self):
        """初始化注册中心"""
        self._adapters: Dict[str, Type[BaseModelAdapter]] = {}
        self._framework_mappings: Dict[str, str] = {}
        self._architecture_mappings: Dict[str, str] = {}
        
        # 自动注册内置适配器
        self._register_builtin_adapters()
        
        logger.info("适配器注册中心初始化完成")
    
    def register(self, 
                 name: str, 
                 adapter_class: Type[BaseModelAdapter],
                 frameworks: Optional[List[str]] = None,
                 architectures: Optional[List[str]] = None) -> None:
        """
        注册适配器
        
        Args:
            name: 适配器名称
            adapter_class: 适配器类
            frameworks: 支持的框架列表
            architectures: 支持的架构列表
        """
        if not issubclass(adapter_class, BaseModelAdapter):
            raise ValueError(f"适配器类必须继承BaseModelAdapter: {adapter_class}")
        
        self._adapters[name] = adapter_class
        
        # 注册框架映射
        if frameworks:
            for framework in frameworks:
                self._framework_mappings[framework] = name
        
        # 注册架构映射
        if architectures:
            for arch in architectures:
                self._architecture_mappings[arch] = name
        
        logger.info(f"已注册适配器: {name} -> {adapter_class.__name__}")
    
    def get_adapter_class(self, name: str) -> Optional[Type[BaseModelAdapter]]:
        """根据名称获取适配器类"""
        return self._adapters.get(name)
    
    def get_adapter_by_framework(self, framework: str) -> Optional[Type[BaseModelAdapter]]:
        """根据框架获取适配器类"""
        adapter_name = self._framework_mappings.get(framework)
        if adapter_name:
            return self._adapters.get(adapter_name)
        return None
    
    def get_adapter_by_architecture(self, architecture: str) -> Optional[Type[BaseModelAdapter]]:
        """根据架构获取适配器类"""
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
        创建适配器实例
        
        Args:
            model_path: 模型文件路径
            adapter_name: 指定适配器名称
            framework: 模型框架
            architecture: 模型架构
            **kwargs: 传递给适配器的参数
            
        Returns:
            适配器实例
        """
        adapter_class = None
        
        # 1. 优先使用指定的适配器名称
        if adapter_name:
            adapter_class = self.get_adapter_class(adapter_name)
            if adapter_class:
                logger.info(f"使用指定适配器: {adapter_name}")
        
        # 2. 根据架构查找适配器
        if not adapter_class and architecture:
            adapter_class = self.get_adapter_by_architecture(architecture)
            if adapter_class:
                logger.info(f"根据架构找到适配器: {architecture}")
        
        # 3. 根据框架查找适配器
        if not adapter_class and framework:
            adapter_class = self.get_adapter_by_framework(framework)
            if adapter_class:
                logger.info(f"根据框架找到适配器: {framework}")
        
        # 4. 如果都没找到，抛出异常
        if not adapter_class:
            available = list(self._adapters.keys())
            raise ValueError(
                f"未找到合适的适配器 - adapter_name: {adapter_name}, "
                f"framework: {framework}, architecture: {architecture}. "
                f"可用适配器: {available}"
            )
        
        # 创建适配器实例
        try:
            adapter = adapter_class(model_path=model_path, **kwargs)
            logger.info(f"成功创建适配器实例: {adapter_class.__name__}")
            return adapter
        except Exception as e:
            logger.error(f"创建适配器失败: {e}")
            raise
    
    def list_adapters(self) -> Dict[str, Dict[str, Any]]:
        """列出所有注册的适配器"""
        adapters_info = {}
        
        for name, adapter_class in self._adapters.items():
            # 查找支持的框架和架构
            frameworks = [k for k, v in self._framework_mappings.items() if v == name]
            architectures = [k for k, v in self._architecture_mappings.items() if v == name]
            
            adapters_info[name] = {
                'class': adapter_class.__name__,
                'module': adapter_class.__module__,
                'frameworks': frameworks,
                'architectures': architectures,
                'doc': adapter_class.__doc__ or "无描述"
            }
        
        return adapters_info
    
    def _register_builtin_adapters(self):
        """注册内置适配器 - 增强版本"""
        logger.info("开始注册内置适配器...")
        
        registration_results = {}
        
        # YOLO 检测适配器
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
            logger.info("✅ Ultralytics适配器注册成功")
        except ImportError as e:
            registration_results['ultralytics'] = False
            logger.warning(f"❌ Ultralytics适配器注册失败: {e}")
            logger.info("💡 请安装ultralytics: pip install ultralytics")
        except Exception as e:
            registration_results['ultralytics'] = False
            logger.error(f"❌ Ultralytics适配器注册异常: {e}")
        
        # SAM 分割适配器
        try:
            from .segmentation.sam import SAMAdapter
            self.register(
                'sam',
                SAMAdapter,
                frameworks=['segment_anything'],
                architectures=['sam_vit_h', 'sam_vit_l', 'sam_vit_b', 'mobile_sam']
            )
            registration_results['sam'] = True
            logger.info("✅ SAM适配器注册成功")
        except ImportError as e:
            registration_results['sam'] = False
            logger.warning(f"❌ SAM适配器注册失败: {e}")
            logger.info("💡 请安装segment-anything: pip install segment-anything")
        except Exception as e:
            registration_results['sam'] = False
            logger.error(f"❌ SAM适配器注册异常: {e}")
        
        # DeepLabV3 分割适配器
        try:
            from .segmentation.deeplabv3 import DeepLabV3Adapter
            self.register(
                'deeplabv3',
                DeepLabV3Adapter,
                frameworks=['torchvision'],
                architectures=['deeplabv3', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet']
            )
            registration_results['deeplabv3'] = True
            logger.info("✅ DeepLabV3适配器注册成功")
        except ImportError as e:
            registration_results['deeplabv3'] = False
            logger.debug(f"DeepLabV3适配器注册失败: {e}")
        except Exception as e:
            registration_results['deeplabv3'] = False
            logger.error(f"❌ DeepLabV3适配器注册异常: {e}")
        
        # Torchvision 分类适配器
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
            logger.info("✅ Torchvision分类适配器注册成功")
        except ImportError as e:
            registration_results['torchvision_classification'] = False
            logger.debug(f"Torchvision分类适配器注册失败: {e}")
        except Exception as e:
            registration_results['torchvision_classification'] = False
            logger.error(f"❌ Torchvision分类适配器注册异常: {e}")
        
        # Stable Diffusion 生成适配器
        try:
            from .generation.stable_diffusion import StableDiffusionAdapter
            self.register(
                'stable_diffusion',
                StableDiffusionAdapter,
                frameworks=['diffusers'],
                architectures=['stable_diffusion', 'stable_diffusion_xl', 'sdxl', 'sd1', 'sd2']
            )
            registration_results['stable_diffusion'] = True
            logger.info("✅ Stable Diffusion适配器注册成功")
        except ImportError as e:
            registration_results['stable_diffusion'] = False
            logger.debug(f"Stable Diffusion适配器注册失败: {e}")
        except Exception as e:
            registration_results['stable_diffusion'] = False
            logger.error(f"❌ Stable Diffusion适配器注册异常: {e}")

        # FLUX 生成适配器
        try:
            from .generation.flux import FluxAdapter
            self.register(
                'flux',
                FluxAdapter,
                frameworks=['diffusers'],
                architectures=['flux', 'flux-dev', 'flux-schnell', 'flux-pro']
            )
            registration_results['flux'] = True
            logger.info("✅ FLUX适配器注册成功")
        except ImportError as e:
            registration_results['flux'] = False
            logger.debug(f"FLUX适配器注册失败: {e}")
        except Exception as e:
            registration_results['flux'] = False
            logger.error(f"❌ FLUX适配器注册异常: {e}")
        
        # CLIP 多模态适配器（OpenAI CLIP）
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
            logger.info("✅ CLIP适配器注册成功")
        except ImportError as e:
            registration_results['clip'] = False
            logger.debug(f"CLIP适配器注册失败: {e}")
        except Exception as e:
            registration_results['clip'] = False
            logger.error(f"❌ CLIP适配器注册异常: {e}")
        
        # OpenCLIP 多模态适配器
        try:
            from .multimodal.openclip import OpenCLIPAdapter
            self.register(
                'openclip',
                OpenCLIPAdapter,
                frameworks=['open_clip'],
                architectures=['convnext', 'coca', 'eva', 'openclip-vit']
            )
            registration_results['openclip'] = True
            logger.info("✅ OpenCLIP适配器注册成功")
        except ImportError as e:
            registration_results['openclip'] = False
            logger.debug(f"OpenCLIP适配器注册失败: {e}")
        except Exception as e:
            registration_results['openclip'] = False
            logger.error(f"❌ OpenCLIP适配器注册异常: {e}")
        
        # 汇总注册结果
        success_count = sum(registration_results.values())
        total_count = len(registration_results)
        
        logger.info(f"适配器注册完成: {success_count}/{total_count} 成功")
        logger.info(f"已注册的适配器: {list(self._adapters.keys())}")
        
        # 如果没有任何适配器注册成功，发出警告
        if success_count == 0:
            logger.error("⚠️ 没有任何适配器注册成功！请检查依赖包安装")
        elif 'ultralytics' not in self._adapters:
            logger.warning("⚠️ 关键适配器 'ultralytics' 未注册，这可能影响YOLO模型的使用")
    
    def force_register_adapter(self, adapter_name: str) -> bool:
        """
        强制注册指定适配器
        
        Args:
            adapter_name: 要注册的适配器名称
            
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
                logger.info(f"✅ 强制注册 {adapter_name} 成功")
                return True
            except Exception as e:
                logger.error(f"❌ 强制注册 {adapter_name} 失败: {e}")
                return False
        
        # 可以为其他适配器添加类似的逻辑
        logger.warning(f"不支持强制注册适配器: {adapter_name}")
        return False
    
    def auto_detect_adapter(self, 
                           model_path: str,
                           model_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        增强的自动检测适合的适配器
        Args:
            model_path: 模型路径
            model_info: 模型信息（从model_detector获得）
        Returns:
            适配器名称
        """
        logger.info(f"🔍 开始检测模型适配器: {model_path}")
        # 优先级1: 基于model_info的精确匹配
        if model_info:
            # 根据架构匹配
            architecture = model_info.get('architecture', '').lower()
            for arch, adapter_name in self._architecture_mappings.items():
                if arch.lower() in architecture:
                    logger.info(f"✅ 根据架构 '{architecture}' 选择适配器: {adapter_name}")
                    return adapter_name
            # 根据框架匹配
            framework = model_info.get('framework', '').lower()
            for fw, adapter_name in self._framework_mappings.items():
                if fw.lower() in framework:
                    logger.info(f"✅ 根据框架 '{framework}' 选择适配器: {adapter_name}")
                    return adapter_name
        # 优先级2: 基于文件路径的智能检测
        model_path_lower = str(model_path).lower()
        # 检测YOLO模型 (最高优先级)
        yolo_patterns = ['yolo', 'yolov8', 'yolov9', 'yolov10', 'yolo11']
        if any(pattern in model_path_lower for pattern in yolo_patterns):
            detected_name = 'ultralytics'
            if detected_name in self._adapters:
                logger.info(f"✅ 检测到YOLO模型，选择适配器: {detected_name}")
                return detected_name
            else:
                logger.warning(f"⚠️ 检测到YOLO模型但适配器 '{detected_name}' 未注册")
        # 检测DETR模型 (检测模型但不是YOLO)
        detr_patterns = ['detr', 'detection']
        if any(pattern in model_path_lower for pattern in detr_patterns):
            # DETR模型通常也可以用ultralytics处理，或者需要专门的适配器
            detected_name = 'ultralytics'  # 默认使用ultralytics
            if detected_name in self._adapters:
                logger.info(f"✅ 检测到DETR模型，使用适配器: {detected_name}")
                return detected_name
        # 检测SAM模型
        sam_patterns = ['sam_vit', 'mobile_sam', 'sam']
        if any(pattern in model_path_lower for pattern in sam_patterns):
            detected_name = 'sam'
            if detected_name in self._adapters:
                logger.info(f"✅ 检测到SAM模型，选择适配器: {detected_name}")
                return detected_name
        # 检测Stable Diffusion模型
        sd_patterns = ['stable-diffusion', 'sd_', 'sdxl', 'flux']
        if any(pattern in model_path_lower for pattern in sd_patterns):
            detected_name = 'stable_diffusion'
            if detected_name in self._adapters:
                logger.info(f"✅ 检测到Stable Diffusion模型，选择适配器: {detected_name}")
                return detected_name
        # 检测CLIP模型
        clip_patterns = ['clip', 'vit-b-32', 'vit-l-14']
        if any(pattern in model_path_lower for pattern in clip_patterns):
            detected_name = 'clip'
            if detected_name in self._adapters:
                logger.info(f"✅ 检测到CLIP模型，选择适配器: {detected_name}")
                return detected_name
        # 检测分类模型 (较低优先级，避免误判)
        classification_patterns = ['resnet', 'efficientnet', 'densenet', 'vgg', 'mobilenet']
        if any(pattern in model_path_lower for pattern in classification_patterns):
            # 进一步检查是否真的是分类模型
            if not any(exclusion in model_path_lower for exclusion in ['yolo', 'detr', 'detection']):
                detected_name = 'torchvision_classification'
                if detected_name in self._adapters:
                    logger.info(f"✅ 检测到分类模型，选择适配器: {detected_name}")
                    return detected_name
        # 优先级3: 基于文件内容的深度分析（如果文件存在）
        try:
            from pathlib import Path
            import torch
            model_file = Path(model_path)
            if model_file.exists() and model_file.suffix in ['.pt', '.pth', '.ckpt']:
                logger.info("📁 文件存在，尝试内容分析...")
                try:
                    # 只加载文件头部信息，不加载完整模型
                    checkpoint = torch.load(model_file, map_location='cpu')
                    if isinstance(checkpoint, dict):
                        # 检查YOLO特征
                        yolo_keys = ['model', 'epoch', 'best_fitness', 'optimizer']
                        if any(key in checkpoint for key in yolo_keys):
                            if 'ultralytics' in self._adapters:
                                logger.info("🔍 内容分析: 检测到YOLO模型特征")
                                return 'ultralytics'
                        # 检查分类模型特征
                        if 'state_dict' in checkpoint or 'model_state_dict' in checkpoint:
                            if 'torchvision_classification' in self._adapters:
                                logger.info("🔍 内容分析: 检测到分类模型特征")
                                return 'torchvision_classification'
                except Exception as e:
                    logger.debug(f"文件内容分析失败: {e}")
        except ImportError:
            logger.debug("torch未安装，跳过文件内容分析")
        # 优先级4: 默认策略
        # 如果路径包含detection相关词汇，默认使用ultralytics
        if any(keyword in model_path_lower for keyword in ['detect', 'object', 'bbox']):
            if 'ultralytics' in self._adapters:
                logger.info("🎯 默认策略: 检测相关路径，使用ultralytics适配器")
                return 'ultralytics'
        # 最后的fallback
        logger.warning(f"⚠️ 无法自动检测适配器类型: {model_path}")
        logger.info(f"📊 可用适配器: {list(self._adapters.keys())}")
        return None
    
    def get_compatible_adapters(self, model_type: str) -> List[str]:
        """根据模型类型获取兼容的适配器列表"""
        compatible = []
        
        for adapter_name, adapter_class in self._adapters.items():
            # 检查适配器的基类类型
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


# 全局注册中心实例
_registry = None

def get_registry() -> AdapterRegistry:
    """获取全局适配器注册中心实例"""
    global _registry
    if _registry is None:
        _registry = AdapterRegistry()
    return _registry


def register_adapter(name: str, 
                    adapter_class: Type[BaseModelAdapter],
                    frameworks: Optional[List[str]] = None,
                    architectures: Optional[List[str]] = None) -> None:
    """便利函数：注册适配器到全局注册中心"""
    registry = get_registry()
    registry.register(name, adapter_class, frameworks, architectures)


def create_adapter(model_path: str, **kwargs) -> BaseModelAdapter:
    """便利函数：创建适配器实例"""
    registry = get_registry()
    return registry.create_adapter(model_path, **kwargs)


def list_available_adapters() -> Dict[str, Dict[str, Any]]:
    """便利函数：列出所有可用适配器"""
    registry = get_registry()
    return registry.list_adapters()
