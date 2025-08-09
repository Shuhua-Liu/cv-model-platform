"""
Ultralytics YOLO适配器 - 支持YOLOv8, YOLOv9, YOLOv10, YOLOv11等

支持的模型：
- YOLOv8: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
- YOLOv9: yolov9c, yolov9e
- YOLOv10: yolov10n, yolov10s, yolov10m, yolov10l, yolov10x
- YOLOv11: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
"""

import time
from typing import List, Dict, Any, Union
from pathlib import Path
import numpy as np
from PIL import Image
from loguru import logger

from ..base import DetectionAdapter

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics未安装，YOLO适配器将不可用")


class UltralyticsAdapter(DetectionAdapter):
    """Ultralytics YOLO模型适配器"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 confidence: float = 0.25,
                 nms_threshold: float = 0.45,
                 batch_size: int = 4,
                 **kwargs):
        """
        初始化YOLO适配器
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
            confidence: 检测置信度阈值
            nms_threshold: NMS阈值
            batch_size: 批处理大小
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("需要安装ultralytics: pip install ultralytics")
        
        super().__init__(model_path, device, **kwargs)
        
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.batch_size = batch_size
        
        # COCO类别名称
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def load_model(self) -> None:
        """加载YOLO模型"""
        try:
            logger.info(f"加载YOLO模型: {self.model_path}")
            
            # 创建YOLO实例
            self.model = YOLO(str(self.model_path))
            
            # 移动到指定设备
            self.model.to(self.device)
            
            # 获取模型信息
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            
            self.is_loaded = True
            logger.info(f"YOLO模型加载成功 - 类别数: {len(self.class_names)}")
            
        except Exception as e:
            logger.error(f"YOLO模型加载失败: {e}")
            raise
    
    def preprocess(self, input_data: Any) -> Any:
        """预处理输入数据"""
        # Ultralytics YOLO会自动处理预处理，这里主要做格式转换
        if isinstance(input_data, (str, Path)):
            # 文件路径
            return str(input_data)
        elif isinstance(input_data, Image.Image):
            # PIL图像
            return input_data
        elif isinstance(input_data, np.ndarray):
            # numpy数组
            return input_data
        else:
            raise ValueError(f"不支持的输入格式: {type(input_data)}")
    
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray], 
                confidence: float = None,
                nms_threshold: float = None,
                **kwargs) -> List[Dict[str, Any]]:
        """
        执行目标检测
        
        Args:
            image: 输入图像
            confidence: 置信度阈值（覆盖默认值）
            nms_threshold: NMS阈值（覆盖默认值）
            
        Returns:
            检测结果列表
        """
        if not self.is_loaded:
            self.load_model()
        
        # 使用传入参数或默认参数
        conf = confidence if confidence is not None else self.confidence
        iou = nms_threshold if nms_threshold is not None else self.nms_threshold
        
        try:
            # 预处理输入
            processed_input = self.preprocess(image)
            
            # 执行推理
            start_time = time.time()
            results = self.model(
                processed_input,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            # 后处理结果
            processed_results = self.postprocess(results, inference_time=inference_time)
            
            logger.debug(f"YOLO检测完成 - 耗时: {inference_time:.3f}s, 检测到: {len(processed_results)}个对象")
            return processed_results
            
        except Exception as e:
            logger.error(f"YOLO预测失败: {e}")
            raise
    
    def postprocess(self, raw_output: Any, **kwargs) -> List[Dict[str, Any]]:
        """后处理YOLO输出"""
        results = []
        
        try:
            # Ultralytics返回的是Results对象列表
            for result in raw_output:
                boxes = result.boxes
                if boxes is not None:
                    # 获取检测框信息
                    xyxy = boxes.xyxy.cpu().numpy()  # 边界框坐标
                    conf = boxes.conf.cpu().numpy()  # 置信度
                    cls = boxes.cls.cpu().numpy()    # 类别ID
                    
                    # 处理每个检测结果
                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        confidence = float(conf[i])
                        class_id = int(cls[i])
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        # 计算面积
                        area = (x2 - x1) * (y2 - y1)
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class': class_name,
                            'class_id': class_id,
                            'confidence': confidence,
                            'area': float(area)
                        }
                        results.append(detection)
            
            # 按置信度排序
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
        except Exception as e:
            logger.error(f"YOLO后处理失败: {e}")
            raise
        
        return results
    
    def predict_batch(self, 
                     images: List[Union[str, Path, Image.Image, np.ndarray]],
                     confidence: float = None,
                     nms_threshold: float = None,
                     **kwargs) -> List[List[Dict[str, Any]]]:
        """批量预测"""
        if not self.is_loaded:
            self.load_model()
        
        conf = confidence if confidence is not None else self.confidence
        iou = nms_threshold if nms_threshold is not None else self.nms_threshold
        
        try:
            # 预处理所有输入
            processed_inputs = [self.preprocess(img) for img in images]
            
            # 批量推理
            start_time = time.time()
            results = self.model(
                processed_inputs,
                conf=conf,
                iou=iou,
                device=self.device,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            # 处理每个结果
            batch_results = []
            for result in results:
                processed = self.postprocess([result], inference_time=inference_time/len(images))
                batch_results.append(processed)
            
            logger.debug(f"YOLO批量检测完成 - 批量大小: {len(images)}, 总耗时: {inference_time:.3f}s")
            return batch_results
            
        except Exception as e:
            logger.error(f"YOLO批量预测失败: {e}")
            raise
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """模型预热"""
        if not self.is_loaded:
            self.load_model()
        
        # 创建dummy输入进行预热
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                results = self.model(dummy_image, verbose=False)
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
            except Exception as e:
                logger.warning(f"预热运行 {i+1} 失败: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"YOLO模型预热完成 - 平均耗时: {avg_time:.3f}s")
            
            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return super().warmup(num_runs)
    
    def visualize_results(self, 
                         image: Union[str, Path, Image.Image, np.ndarray],
                         results: List[Dict[str, Any]],
                         save_path: str = None,
                         show_labels: bool = True,
                         show_confidence: bool = True) -> Image.Image:
        """可视化检测结果"""
        try:
            # 加载图像
            if isinstance(image, (str, Path)):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image).convert('RGB')
            else:
                raise ValueError(f"不支持的图像格式: {type(image)}")
            
            # 使用PIL进行简单绘制（实际项目中可能需要更复杂的可视化）
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(img)
            
            # 尝试加载字体
            try:
                # 可以根据系统调整字体路径
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 颜色列表
            colors = [
                '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
                '#800000', '#008000', '#000080', '#808000', '#800080', '#008080'
            ]
            
            # 绘制检测框
            for i, detection in enumerate(results):
                bbox = detection['bbox']
                class_name = detection['class']
                confidence = detection['confidence']
                
                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]
                
                # 绘制边界框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # 绘制标签
                if show_labels:
                    label = class_name
                    if show_confidence:
                        label += f" {confidence:.2f}"
                    
                    # 计算文本大小
                    bbox_font = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox_font[2] - bbox_font[0]
                    text_height = bbox_font[3] - bbox_font[1]
                    
                    # 绘制标签背景
                    draw.rectangle([x1, y1-text_height-4, x1+text_width+4, y1], fill=color)
                    
                    # 绘制文本
                    draw.text((x1+2, y1-text_height-2), label, fill='white', font=font)
            
            # 保存图像
            if save_path:
                img.save(save_path)
                logger.info(f"可视化结果已保存: {save_path}")
            
            return img
            
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
            'model_type': 'detection',
            'framework': 'ultralytics',
            'confidence_threshold': self.confidence,
            'nms_threshold': self.nms_threshold,
            'batch_size': self.batch_size,
            'num_classes': len(self.class_names),
            'class_names': self.class_names[:10] if len(self.class_names) > 10 else self.class_names  # 只显示前10个类别
        })
        
        if self.is_loaded:
            try:
                # 获取模型架构信息
                if hasattr(self.model, 'model'):
                    model_info = self.model.info(verbose=False)
                    info['model_summary'] = str(model_info)
                    
                # 获取模型参数数量
                if hasattr(self.model.model, 'parameters'):
                    total_params = sum(p.numel() for p in self.model.model.parameters())
                    info['total_parameters'] = total_params
                    
            except Exception as e:
                logger.debug(f"获取模型详细信息失败: {e}")
        
        return info
