"""
Torchvision 分类适配器 - 支持torchvision的预训练分类模型

支持的模型：
- ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
- EfficientNet: efficientnet_b0-b7
- DenseNet: densenet121, densenet169, densenet201
- VGG: vgg11, vgg13, vgg16, vgg19
- MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- Vision Transformer: vit_b_16, vit_b_32, vit_l_16, vit_l_32
"""

import time
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from loguru import logger

from ..base import ClassificationAdapter

try:
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision未安装，Torchvision适配器将不可用")


class TorchvisionAdapter(ClassificationAdapter):
    """Torchvision分类模型适配器"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 pretrained: bool = True,
                 num_classes: int = 1000,  # ImageNet类别数
                 batch_size: int = 8,
                 **kwargs):
        """
        初始化Torchvision适配器
        
        Args:
            model_path: 模型文件路径或torchvision://model_name
            device: 计算设备
            pretrained: 是否使用预训练权重
            num_classes: 类别数（1000为ImageNet）
            batch_size: 批处理大小
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("需要安装torchvision: pip install torchvision")
        
        super().__init__(model_path, device, **kwargs)
        
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # 确定模型架构
        self.model_name = self._determine_model_name()
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # ImageNet类别名称（前10个作为示例）
        self.class_names = self._load_class_names()
    
    def _determine_model_name(self) -> str:
        """根据路径确定模型名称"""
        path_str = str(self.model_path).lower()
        
        # 处理torchvision://格式
        if path_str.startswith('torchvision://'):
            return path_str.replace('torchvision://', '')
        
        # 从文件名推断
        filename = self.model_path.name.lower()
        
        # ResNet系列
        if 'resnet18' in filename:
            return 'resnet18'
        elif 'resnet34' in filename:
            return 'resnet34'
        elif 'resnet50' in filename:
            return 'resnet50'
        elif 'resnet101' in filename:
            return 'resnet101'
        elif 'resnet152' in filename:
            return 'resnet152'
        
        # EfficientNet系列
        elif 'efficientnet_b0' in filename:
            return 'efficientnet_b0'
        elif 'efficientnet_b1' in filename:
            return 'efficientnet_b1'
        elif 'efficientnet_b2' in filename:
            return 'efficientnet_b2'
        elif 'efficientnet_b3' in filename:
            return 'efficientnet_b3'
        elif 'efficientnet_b4' in filename:
            return 'efficientnet_b4'
        elif 'efficientnet_b5' in filename:
            return 'efficientnet_b5'
        elif 'efficientnet_b6' in filename:
            return 'efficientnet_b6'
        elif 'efficientnet_b7' in filename:
            return 'efficientnet_b7'
        
        # MobileNet系列
        elif 'mobilenet_v2' in filename:
            return 'mobilenet_v2'
        elif 'mobilenet_v3_large' in filename:
            return 'mobilenet_v3_large'
        elif 'mobilenet_v3_small' in filename:
            return 'mobilenet_v3_small'
        
        # Vision Transformer
        elif 'vit_b_16' in filename:
            return 'vit_b_16'
        elif 'vit_b_32' in filename:
            return 'vit_b_32'
        elif 'vit_l_16' in filename:
            return 'vit_l_16'
        elif 'vit_l_32' in filename:
            return 'vit_l_32'
        
        # 默认使用ResNet-50
        logger.warning(f"无法从文件名确定模型架构: {filename}，默认使用resnet50")
        return 'resnet50'
    
    def _load_class_names(self) -> List[str]:
        """加载类别名称"""
        # 这里只提供前20个ImageNet类别作为示例
        # 实际应用中可以从文件或API加载完整的1000个类别
        imagenet_classes = [
            'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead',
            'electric_ray', 'stingray', 'cock', 'hen', 'ostrich',
            'brambling', 'goldfinch', 'house_finch', 'junco', 'indigo_bunting',
            'robin', 'bulbul', 'jay', 'magpie', 'chickadee'
        ]
        
        # 如果类别数不是1000，生成通用类别名
        if self.num_classes != 1000:
            return [f'class_{i}' for i in range(self.num_classes)]
        
        # 扩展到1000个类别（实际应用中应该加载完整列表）
        while len(imagenet_classes) < 1000:
            imagenet_classes.append(f'class_{len(imagenet_classes)}')
        
        return imagenet_classes[:self.num_classes]
    
    def load_model(self) -> None:
        """加载分类模型"""
        try:
            logger.info(f"加载Torchvision分类模型: {self.model_name}")
            
            # 首先尝试直接加载用户的模型文件
            if self.model_path.exists() and not str(self.model_path).startswith('torchvision://'):
                logger.info("检测到本地模型文件，尝试直接加载...")
                
                try:
                    # 直接加载完整模型
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    
                    if hasattr(checkpoint, 'state_dict') or not isinstance(checkpoint, dict):
                        self.model = checkpoint
                        if hasattr(self.model, 'to'):
                            self.model = self.model.to(self.device)
                        self.model.eval()
                        self.is_loaded = True
                        logger.info("成功直接加载完整模型")
                        return
                except Exception as e:
                    logger.info(f"直接加载失败，尝试使用torchvision架构: {e}")
            
            # 使用torchvision架构创建模型
            logger.info(f"使用torchvision架构创建模型: {self.model_name}")
            
            # 获取模型构造函数
            model_func = self._get_model_function()
            
            if model_func is None:
                raise ValueError(f"不支持的模型: {self.model_name}")
            
            # 创建模型
            if self.pretrained and self.num_classes == 1000:
                # 使用预训练权重
                weights = 'DEFAULT'  # torchvision新版本的写法
                try:
                    self.model = model_func(weights=weights)
                except TypeError:
                    # 兼容旧版本torchvision
                    self.model = model_func(pretrained=True)
                logger.info("使用预训练权重创建模型")
            else:
                # 不使用预训练或自定义类别数
                try:
                    self.model = model_func(weights=None, num_classes=self.num_classes)
                except TypeError:
                    self.model = model_func(pretrained=False, num_classes=self.num_classes)
                logger.info("创建随机初始化模型")
            
            # 加载用户的权重文件（如果存在）
            if self.model_path.exists() and not str(self.model_path).startswith('torchvision://'):
                logger.info("加载用户权重文件...")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # 处理不同的检查点格式
                if isinstance(checkpoint, dict):
                    if 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
                
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    logger.info("成功加载用户权重（严格模式）")
                except RuntimeError as e:
                    logger.warning(f"严格模式加载失败，尝试非严格模式: {e}")
                    try:
                        self.model.load_state_dict(state_dict, strict=False)
                        logger.info("成功加载用户权重（非严格模式）")
                    except RuntimeError as e:
                        logger.error(f"权重加载失败: {e}")
                        if not self.pretrained:
                            logger.info("使用随机初始化的权重")
            
            # 移动到指定设备
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Torchvision分类模型加载成功 - 架构: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Torchvision分类模型加载失败: {e}")
            raise
    
    def _get_model_function(self):
        """获取模型构造函数"""
        model_functions = {
            # ResNet系列
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
            
            # EfficientNet系列
            'efficientnet_b0': models.efficientnet_b0,
            'efficientnet_b1': models.efficientnet_b1,
            'efficientnet_b2': models.efficientnet_b2,
            'efficientnet_b3': models.efficientnet_b3,
            'efficientnet_b4': models.efficientnet_b4,
            'efficientnet_b5': models.efficientnet_b5,
            'efficientnet_b6': models.efficientnet_b6,
            'efficientnet_b7': models.efficientnet_b7,
            
            # MobileNet系列
            'mobilenet_v2': models.mobilenet_v2,
            'mobilenet_v3_large': models.mobilenet_v3_large,
            'mobilenet_v3_small': models.mobilenet_v3_small,
            
            # DenseNet系列
            'densenet121': models.densenet121,
            'densenet169': models.densenet169,
            'densenet201': models.densenet201,
            
            # VGG系列
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
        }
        
        # Vision Transformer需要特殊处理
        if 'vit_' in self.model_name:
            if hasattr(models, self.model_name):
                return getattr(models, self.model_name)
            else:
                logger.warning(f"当前torchvision版本不支持 {self.model_name}")
                return None
        
        return model_functions.get(self.model_name)
    
    def preprocess(self, input_data: Any) -> torch.Tensor:
        """预处理输入数据"""
        if isinstance(input_data, (str, Path)):
            # 文件路径
            image = Image.open(input_data).convert('RGB')
        elif isinstance(input_data, Image.Image):
            # PIL图像
            image = input_data.convert('RGB')
        elif isinstance(input_data, np.ndarray):
            # numpy数组
            if input_data.ndim == 3 and input_data.shape[2] == 3:
                image = Image.fromarray(input_data)
            else:
                raise ValueError(f"不支持的numpy数组格式: {input_data.shape}")
        else:
            raise ValueError(f"不支持的输入格式: {type(input_data)}")
        
        # 应用预处理变换
        tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
        return tensor.to(self.device)
    
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray], 
                top_k: int = 5,
                threshold: float = 0.0,
                **kwargs) -> Dict[str, Any]:
        """
        执行图像分类
        
        Args:
            image: 输入图像
            top_k: 返回前k个预测结果
            threshold: 置信度阈值
            
        Returns:
            分类结果字典
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # 预处理输入
            input_tensor = self.preprocess(image)
            
            # 执行推理
            start_time = time.time()
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                
            inference_time = time.time() - start_time
            
            # 后处理结果
            processed_results = self.postprocess(
                probabilities, 
                top_k=top_k,
                threshold=threshold,
                inference_time=inference_time
            )
            
            logger.debug(f"Torchvision分类完成 - 耗时: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"Torchvision分类预测失败: {e}")
            raise
    
    def postprocess(self, 
                   probabilities: torch.Tensor,
                   top_k: int = 5,
                   threshold: float = 0.0,
                   **kwargs) -> Dict[str, Any]:
        """后处理分类输出"""
        try:
            # 移到CPU并转换为numpy
            probs = probabilities.squeeze(0).cpu().numpy()  # [num_classes]
            
            # 获取top-k结果
            top_k = min(top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:top_k]
            
            # 构建预测结果
            predictions = []
            for idx in top_indices:
                confidence = float(probs[idx])
                if confidence >= threshold:
                    class_name = self.class_names[idx] if idx < len(self.class_names) else f'class_{idx}'
                    predictions.append({
                        'class': class_name,
                        'class_id': int(idx),
                        'confidence': confidence
                    })
            
            # 构建结果字典
            result = {
                'predictions': predictions,
                'top_class': predictions[0]['class'] if predictions else 'unknown',
                'top_confidence': predictions[0]['confidence'] if predictions else 0.0,
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'model_name': self.model_name,
                    'top_k': top_k,
                    'threshold': threshold,
                    'total_classes': len(probs)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Torchvision分类后处理失败: {e}")
            raise
    
    def predict_batch(self, 
                     images: List[Union[str, Path, Image.Image, np.ndarray]],
                     top_k: int = 5,
                     threshold: float = 0.0,
                     **kwargs) -> List[Dict[str, Any]]:
        """批量预测"""
        if not self.is_loaded:
            self.load_model()
        
        results = []
        
        # 批量处理
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i + self.batch_size]
            
            try:
                # 预处理批量数据
                batch_tensors = []
                for img in batch_images:
                    tensor = self.preprocess(img)
                    batch_tensors.append(tensor.squeeze(0))  # 移除batch维度
                
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # 批量推理
                start_time = time.time()
                with torch.no_grad():
                    logits = self.model(batch_tensor)
                    probabilities = F.softmax(logits, dim=1)
                
                inference_time = time.time() - start_time
                
                # 处理每个结果
                for j, probs in enumerate(probabilities):
                    result = self.postprocess(
                        probs.unsqueeze(0),
                        top_k=top_k,
                        threshold=threshold,
                        inference_time=inference_time / len(batch_images)
                    )
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"批量预测失败 (batch {i//self.batch_size + 1}): {e}")
                # 添加错误结果以保持列表长度一致
                for _ in batch_images:
                    results.append({
                        'predictions': [],
                        'top_class': 'error',
                        'top_confidence': 0.0,
                        'metadata': {'error': str(e)}
                    })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'classification',
            'framework': 'torchvision',
            'architecture': self.model_name,
            'pretrained': self.pretrained,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'input_size': (224, 224),
            'preprocessing': 'ImageNet normalization'
        })
        
        if self.is_loaded:
            try:
                # 获取模型参数数量
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                info.update({
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params
                })
                
            except Exception as e:
                logger.debug(f"获取模型详细信息失败: {e}")
        
        return info
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """模型预热"""
        if not self.is_loaded:
            self.load_model()
        
        # 创建dummy输入进行预热
        dummy_image = torch.randn(1, 3, 224, 224).to(self.device)
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                with torch.no_grad():
                    _ = self.model(dummy_image)
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
            except Exception as e:
                logger.warning(f"预热运行 {i+1} 失败: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"Torchvision分类模型预热完成 - 平均耗时: {avg_time:.3f}s")
            
            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return super().warmup(num_runs)
