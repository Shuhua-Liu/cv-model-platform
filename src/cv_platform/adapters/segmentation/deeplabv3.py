"""
DeepLabV3 分割适配器 - 支持torchvision的DeepLabV3模型

支持的模型：
- deeplabv3_resnet50
- deeplabv3_resnet101  
- deeplabv3_mobilenet_v3_large
"""

import time
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from loguru import logger

from ..base import SegmentationAdapter

try:
    import torchvision.transforms as transforms
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    logger.warning("torchvision未安装，DeepLabV3适配器将不可用")


class DeepLabV3Adapter(SegmentationAdapter):
    """DeepLabV3模型适配器"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 num_classes: int = 21,  # PASCAL VOC类别数
                 batch_size: int = 1,
                 **kwargs):
        """
        初始化DeepLabV3适配器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
            num_classes: 类别数（21为PASCAL VOC，81为COCO）
            batch_size: 批处理大小
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("需要安装torchvision: pip install torchvision")
        
        super().__init__(model_path, device, **kwargs)
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # DeepLabV3通常使用512x512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # PASCAL VOC类别名称
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # 如果是COCO数据集，类别数会更多
        if num_classes > 21:
            self.class_names = [f'class_{i}' for i in range(num_classes)]
    
    def load_model(self) -> None:
        """加载DeepLabV3模型"""
        try:
            logger.info(f"加载DeepLabV3模型: {self.model_path}")
            
            # 根据文件名判断具体的模型架构
            model_name = self.model_path.name.lower()
            
            if 'resnet101' in model_name:
                self.model = models.segmentation.deeplabv3_resnet101(
                    weights=None, 
                    num_classes=self.num_classes
                )
                logger.info("使用ResNet-101作为backbone")
            elif 'resnet50' in model_name:
                self.model = models.segmentation.deeplabv3_resnet50(
                    weights=None, 
                    num_classes=self.num_classes
                )
                logger.info("使用ResNet-50作为backbone")
            elif 'mobilenet' in model_name:
                self.model = models.segmentation.deeplabv3_mobilenet_v3_large(
                    weights=None, 
                    num_classes=self.num_classes
                )
                logger.info("使用MobileNetV3-Large作为backbone")
            else:
                # 默认使用ResNet-101
                self.model = models.segmentation.deeplabv3_resnet101(
                    weights=None, 
                    num_classes=self.num_classes
                )
                logger.info("使用默认ResNet-101作为backbone")
            
            # 加载预训练权重
            if self.model_path.exists():
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
                    state_dict = checkpoint
                
                # 加载状态字典
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                except RuntimeError as e:
                    logger.warning(f"严格模式加载失败，尝试非严格模式: {e}")
                    self.model.load_state_dict(state_dict, strict=False)
                
                logger.info("成功加载预训练权重")
            else:
                logger.warning(f"模型文件不存在: {self.model_path}")
                logger.info("使用随机初始化的权重")
            
            # 移动到指定设备
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"DeepLabV3模型加载成功 - 类别数: {self.num_classes}")
            
        except Exception as e:
            logger.error(f"DeepLabV3模型加载失败: {e}")
            raise
    
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
        
        # 保存原始尺寸用于后处理
        self.original_size = image.size  # (width, height)
        
        # 应用预处理变换
        tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
        return tensor.to(self.device)
    
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray], 
                threshold: float = 0.5,
                return_probabilities: bool = False,
                **kwargs) -> Dict[str, Any]:
        """
        执行图像分割
        
        Args:
            image: 输入图像
            threshold: 分割阈值
            return_probabilities: 是否返回概率图
            
        Returns:
            分割结果字典
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # 预处理输入
            input_tensor = self.preprocess(image)
            
            # 执行推理
            start_time = time.time()
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # DeepLabV3返回字典，包含'out'键
                if isinstance(output, dict):
                    logits = output['out']
                else:
                    logits = output
                
                # 应用softmax获取概率
                probabilities = F.softmax(logits, dim=1)
                
                # 获取预测类别
                predictions = torch.argmax(probabilities, dim=1)
                
            inference_time = time.time() - start_time
            
            # 后处理结果
            processed_results = self.postprocess(
                predictions, 
                probabilities if return_probabilities else None,
                threshold=threshold,
                inference_time=inference_time
            )
            
            logger.debug(f"DeepLabV3分割完成 - 耗时: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"DeepLabV3预测失败: {e}")
            raise
    
    def postprocess(self, 
                   predictions: torch.Tensor,
                   probabilities: torch.Tensor = None,
                   threshold: float = 0.5,
                   **kwargs) -> Dict[str, Any]:
        """后处理DeepLabV3输出"""
        try:
            # 移到CPU并转换为numpy
            pred_mask = predictions.squeeze(0).cpu().numpy()  # [H, W]
            
            if probabilities is not None:
                prob_maps = probabilities.squeeze(0).cpu().numpy()  # [C, H, W]
            else:
                prob_maps = None
            
            # 调整到原始图像尺寸
            from PIL import Image as PILImage
            pred_mask_pil = PILImage.fromarray(pred_mask.astype(np.uint8))
            pred_mask_resized = pred_mask_pil.resize(self.original_size, PILImage.NEAREST)
            pred_mask_final = np.array(pred_mask_resized)
            
            # 生成分割掩码（每个类别一个掩码）
            masks = []
            scores = []
            areas = []
            bboxes = []
            
            unique_classes = np.unique(pred_mask_final)
            
            for class_id in unique_classes:
                if class_id == 0:  # 跳过背景类
                    continue
                
                # 创建该类别的二值掩码
                class_mask = (pred_mask_final == class_id).astype(np.uint8)
                
                if np.sum(class_mask) == 0:
                    continue
                
                masks.append(class_mask)
                
                # 计算该类别的平均置信度
                if prob_maps is not None:
                    # 调整概率图尺寸
                    prob_class = prob_maps[class_id]
                    prob_pil = PILImage.fromarray((prob_class * 255).astype(np.uint8))
                    prob_resized = prob_pil.resize(self.original_size, PILImage.BILINEAR)
                    prob_final = np.array(prob_resized) / 255.0
                    
                    avg_score = np.mean(prob_final[class_mask > 0])
                    scores.append(float(avg_score))
                else:
                    scores.append(1.0)  # 如果没有概率图，设为1.0
                
                # 计算面积
                area = np.sum(class_mask)
                areas.append(float(area))
                
                # 计算边界框
                y_indices, x_indices = np.where(class_mask > 0)
                if len(y_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                else:
                    bboxes.append([0.0, 0.0, 0.0, 0.0])
            
            # 构建结果
            result = {
                'masks': np.array(masks) if masks else np.empty((0, *pred_mask_final.shape)),
                'scores': scores,
                'areas': areas,
                'bboxes': bboxes,
                'class_ids': [int(cls) for cls in unique_classes if cls != 0],
                'class_names': [self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}' 
                              for cls in unique_classes if cls != 0],
                'prediction_mask': pred_mask_final,  # 完整的预测掩码
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'original_size': self.original_size,
                    'model_input_size': (512, 512),
                    'num_classes': self.num_classes,
                    'threshold': threshold
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"DeepLabV3后处理失败: {e}")
            raise
    
    def visualize_results(self, 
                         image: Union[str, Path, Image.Image, np.ndarray],
                         results: Dict[str, Any],
                         save_path: str = None,
                         alpha: float = 0.6) -> Image.Image:
        """可视化分割结果"""
        try:
            # 加载原始图像
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert('RGB')
            else:
                raise ValueError(f"不支持的图像格式: {type(image)}")
            
            # 确保图像尺寸匹配
            img = img.resize(self.original_size)
            img_array = np.array(img)
            
            # 获取预测掩码
            pred_mask = results['prediction_mask']
            
            # 创建彩色分割图
            colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
            
            # 为每个类别分配颜色
            colors = [
                [0, 0, 0],      # 背景
                [128, 0, 0],    # 飞机
                [0, 128, 0],    # 自行车
                [128, 128, 0],  # 鸟
                [0, 0, 128],    # 船
                [128, 0, 128],  # 瓶子
                [0, 128, 128],  # 巴士
                [128, 128, 128], # 车
                [64, 0, 0],     # 猫
                [192, 0, 0],    # 椅子
                [64, 128, 0],   # 牛
                [192, 128, 0],  # 餐桌
                [64, 0, 128],   # 狗
                [192, 0, 128],  # 马
                [64, 128, 128], # 摩托车
                [192, 128, 128], # 人
                [0, 64, 0],     # 盆栽
                [128, 64, 0],   # 羊
                [0, 192, 0],    # 沙发
                [128, 192, 0],  # 火车
                [0, 64, 128],   # 电视
            ]
            
            # 应用颜色
            for class_id in np.unique(pred_mask):
                if class_id < len(colors):
                    colored_mask[pred_mask == class_id] = colors[class_id]
            
            # 混合原图和分割图
            blended = (1 - alpha) * img_array + alpha * colored_mask
            blended = blended.astype(np.uint8)
            
            result_img = Image.fromarray(blended)
            
            # 保存图像
            if save_path:
                result_img.save(save_path)
                logger.info(f"分割可视化结果已保存: {save_path}")
            
            return result_img
            
        except Exception as e:
            logger.error(f"可视化失败: {e}")
            # 返回原图像
            if isinstance(image, Image.Image):
                return image
            else:
                return Image.open(image) if isinstance(image, (str, Path)) else Image.fromarray(image)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'segmentation',
            'framework': 'torchvision',
            'architecture': 'deeplabv3',
            'num_classes': self.num_classes,
            'batch_size': self.batch_size,
            'input_size': (512, 512),
            'class_names': self.class_names[:10] if len(self.class_names) > 10 else self.class_names
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
