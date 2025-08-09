"""
适配器注册中心 - 管理所有模型适配器的注册和实例化

提供适配器的动态注册、查找和创建功能。
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
        """注册内置适配器"""
        # 这里注册我们即将实现的内置适配器
        
        try:
            # YOLOv8 检测适配器
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
            logger.debug("Ultralytics适配器未找到，跳过注册")
        
        try:
            # SAM 分割适配器
            from .segmentation.sam import SAMAdapter
            self.register(
                'sam',
                SAMAdapter,
                frameworks=['segment_anything'],
                architectures=['sam_vit_h', 'sam_vit_l', 'sam_vit_b', 'mobile_sam']
            )
        except ImportError:
            logger.debug("SAM适配器未找到，跳过注册")
        
        try:
            # DeepLabV3 分割适配器
            from .segmentation.deeplabv3 import DeepLabV3Adapter
            self.register(
                'deeplabv3',
                DeepLabV3Adapter,
                frameworks=['torchvision'],
                architectures=['deeplabv3', 'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet']
            )
        except ImportError:
            logger.debug("DeepLabV3适配器未找到，跳过注册")
        
        try:
            # ResNet 分类适配器
            from .classification.torchvision import TorchvisionAdapter
            self.register(
                'torchvision',
                TorchvisionAdapter,
                frameworks=['torchvision'],
                architectures=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                              'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']
            )
        except ImportError:
            logger.debug("Torchvision适配器未找到，跳过注册")
        
        try:
            # Stable Diffusion 生成适配器
            from .generation.stable_diffusion import StableDiffusionAdapter
            self.register(
                'stable_diffusion',
                StableDiffusionAdapter,
                frameworks=['diffusers'],
                architectures=['stable_diffusion', 'stable_diffusion_xl', 'flux']
            )
        except ImportError:
            logger.debug("Stable Diffusion适配器未找到，跳过注册")
        
        try:
            # CLIP 多模态适配器
            from .multimodal.clip import CLIPAdapter
            self.register(
                'clip',
                CLIPAdapter,
                frameworks=['transformers', 'open_clip'],
                architectures=['clip-vit-base', 'clip-vit-large']
            )
        except ImportError:
            logger.debug("CLIP适配器未找到，跳过注册")
    
    def auto_detect_adapter(self, 
                           model_path: str,
                           model_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        自动检测适合的适配器
        
        Args:
            model_path: 模型路径
            model_info: 模型信息（从model_detector获得）
            
        Returns:
            适配器名称
        """
        if model_info:
            # 1. 根据架构匹配
            architecture = model_info.get('architecture', '').lower()
            for arch, adapter_name in self._architecture_mappings.items():
                if arch.lower() in architecture:
                    logger.info(f"根据架构 {architecture} 自动选择适配器: {adapter_name}")
                    return adapter_name
            
            # 2. 根据框架匹配
            framework = model_info.get('framework', '').lower()
            for fw, adapter_name in self._framework_mappings.items():
                if fw.lower() in framework:
                    logger.info(f"根据框架 {framework} 自动选择适配器: {adapter_name}")
                    return adapter_name
        
        # 3. 根据文件路径和名称进行启发式匹配
        model_path_lower = str(model_path).lower()
        
        # 检测YOLO模型
        if any(pattern in model_path_lower for pattern in ['yolo', 'yolov8', 'yolov9', 'yolov10', 'yolo11']):
            return 'ultralytics'
        
        # 检测SAM模型
        if any(pattern in model_path_lower for pattern in ['sam_vit', 'mobile_sam']):
            return 'sam'
        
        # 检测Stable Diffusion模型
        if any(pattern in model_path_lower for pattern in ['stable-diffusion', 'sd_', 'sdxl', 'flux']):
            return 'stable_diffusion'
        
        # 检测分类模型
        if any(pattern in model_path_lower for pattern in ['resnet', 'efficientnet', 'vit-']):
            return 'torchvision'
        
        # 检测CLIP模型
        if any(pattern in model_path_lower for pattern in ['clip', 'vit-b-32', 'vit-l-14']):
            return 'clip'
        
        logger.warning(f"无法自动检测适配器类型: {model_path}")
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