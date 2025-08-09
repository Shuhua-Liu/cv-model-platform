"""
基础模型适配器 - 定义所有模型适配器的接口

所有的模型适配器都应该继承BaseModelAdapter类，并实现其抽象方法。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from loguru import logger


class BaseModelAdapter(ABC):
    """模型适配器基类"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 **kwargs):
        """
        初始化适配器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备 (cpu, cuda:0, auto等)
            **kwargs: 其他模型特定参数
        """
        self.model_path = Path(model_path)
        self.device = self._resolve_device(device)
        self.model = None
        self.is_loaded = False
        self.config = kwargs
        
        logger.info(f"初始化 {self.__class__.__name__} - 路径: {self.model_path}, 设备: {self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """解析和验证设备"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda:0" 
            else:
                return "cpu"
        return device
    
    @abstractmethod
    def load_model(self) -> None:
        """加载模型到内存"""
        pass
    
    @abstractmethod
    def predict(self, input_data: Any, **kwargs) -> Any:
        """执行预测"""
        pass
    
    @abstractmethod
    def preprocess(self, input_data: Any) -> Any:
        """预处理输入数据"""
        pass
    
    @abstractmethod
    def postprocess(self, raw_output: Any, **kwargs) -> Any:
        """后处理模型输出"""
        pass
    
    def unload_model(self) -> None:
        """卸载模型释放内存"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"{self.__class__.__name__} 模型已卸载")
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """模型预热，返回性能指标"""
        if not self.is_loaded:
            self.load_model()
        
        # 子类应该重写此方法提供具体的预热逻辑
        logger.info(f"{self.__class__.__name__} 预热完成")
        return {"warmup_runs": num_runs}
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "adapter_class": self.__class__.__name__,
            "model_path": str(self.model_path),
            "device": self.device,
            "is_loaded": self.is_loaded,
            "config": self.config
        }
        
        # 添加模型文件信息
        if self.model_path.exists():
            stat = self.model_path.stat()
            info.update({
                "file_size_mb": stat.st_size / (1024 * 1024),
                "modified_time": stat.st_mtime
            })
        
        return info
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.is_loaded:
            self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.unload_model()
    
    def __del__(self):
        """析构函数 - 确保资源清理"""
        if hasattr(self, 'model') and self.model is not None:
            self.unload_model()


class DetectionAdapter(BaseModelAdapter):
    """检测模型适配器基类"""
    
    @abstractmethod
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray], 
                confidence: float = 0.25,
                nms_threshold: float = 0.45,
                **kwargs) -> List[Dict[str, Any]]:
        """
        执行目标检测
        
        Args:
            image: 输入图像
            confidence: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测结果列表，每个结果包含：
            {
                'bbox': [x1, y1, x2, y2],  # 边界框坐标
                'class': str,               # 类别名称
                'class_id': int,           # 类别ID
                'confidence': float,       # 置信度
                'area': float             # 区域面积
            }
        """
        pass
    
    def visualize_results(self, 
                         image: Union[str, Path, Image.Image, np.ndarray],
                         results: List[Dict[str, Any]],
                         save_path: Optional[str] = None) -> Image.Image:
        """可视化检测结果"""
        # 默认实现 - 子类可以重写
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # 这里应该添加绘制边界框的逻辑
        # 简化实现，实际应该使用opencv或PIL绘制
        
        if save_path:
            image.save(save_path)
            
        return image


class SegmentationAdapter(BaseModelAdapter):
    """分割模型适配器基类"""
    
    @abstractmethod
    def predict(self,
                image: Union[str, Path, Image.Image, np.ndarray],
                **kwargs) -> Dict[str, Any]:
        """
        执行图像分割
        
        Args:
            image: 输入图像
            
        Returns:
            分割结果字典：
            {
                'masks': np.ndarray,       # 分割掩码 [N, H, W]
                'scores': List[float],     # 分割质量分数
                'areas': List[float],      # 每个掩码的面积
                'bbox': List[List[float]], # 每个掩码的边界框
                'metadata': Dict           # 其他元数据
            }
        """
        pass


class ClassificationAdapter(BaseModelAdapter):
    """分类模型适配器基类"""
    
    @abstractmethod
    def predict(self,
                image: Union[str, Path, Image.Image, np.ndarray],
                top_k: int = 5,
                **kwargs) -> Dict[str, Any]:
        """
        执行图像分类
        
        Args:
            image: 输入图像
            top_k: 返回前k个预测结果
            
        Returns:
            分类结果字典：
            {
                'predictions': [
                    {
                        'class': str,      # 类别名称
                        'class_id': int,   # 类别ID
                        'confidence': float # 置信度
                    },
                    ...
                ],
                'top_class': str,          # 最高置信度类别
                'top_confidence': float    # 最高置信度
            }
        """
        pass


class GenerationAdapter(BaseModelAdapter):
    """生成模型适配器基类"""
    
    @abstractmethod
    def predict(self,
                prompt: str,
                negative_prompt: Optional[str] = None,
                num_steps: int = 20,
                guidance_scale: float = 7.5,
                width: int = 512,
                height: int = 512,
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
            
        Returns:
            生成结果字典：
            {
                'images': List[Image.Image],  # 生成的图像
                'metadata': Dict              # 生成参数等元数据
            }
        """
        pass


class MultimodalAdapter(BaseModelAdapter):
    """多模态模型适配器基类"""
    
    @abstractmethod
    def predict(self,
                image: Union[str, Path, Image.Image, np.ndarray],
                text: Optional[str] = None,
                **kwargs) -> Dict[str, Any]:
        """
        执行多模态推理
        
        Args:
            image: 输入图像
            text: 输入文本（如果需要）
            
        Returns:
            推理结果字典（具体格式依据模型类型）
        """
        pass


# 适配器类型映射
ADAPTER_TYPE_MAP = {
    'detection': DetectionAdapter,
    'segmentation': SegmentationAdapter, 
    'classification': ClassificationAdapter,
    'generation': GenerationAdapter,
    'multimodal': MultimodalAdapter
}


def get_adapter_class(model_type: str) -> type:
    """根据模型类型获取对应的适配器基类"""
    return ADAPTER_TYPE_MAP.get(model_type, BaseModelAdapter)
