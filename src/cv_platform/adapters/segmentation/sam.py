"""
SAM (Segment Anything Model) Adapter - 修复抽象方法实现

修复内容：
1. 实现基类的抽象方法 postprocess
2. 保持原有功能完整性
3. 添加更好的错误处理

支持的模型：
- sam_vit_h (ViT-Huge)
- sam_vit_l (ViT-Large) 
- sam_vit_b (ViT-Base)
- mobile_sam
"""

import time
from typing import Dict, Any, Union, List, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from loguru import logger

from ..base import SegmentationAdapter

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("segment-anything未安装，SAMAdapter将不可用")


class SAMAdapter(SegmentationAdapter):
    """Segment Anything Model Adapter"""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 device: str = "auto",
                 points_per_side: int = 32,
                 pred_iou_thresh: float = 0.88,
                 stability_score_thresh: float = 0.95,
                 crop_n_layers: int = 0,
                 crop_n_points_downscale_factor: int = 1,
                 min_mask_region_area: int = 0,
                 **kwargs):
        """
        初始化SAMAdapter
        
        Args:
            model_path: 模型文件路径
            device: 计算设备
            points_per_side: 每边的点数（用于自动掩码生成）
            pred_iou_thresh: 预测IoU阈值
            stability_score_thresh: 稳定性分数阈值
            crop_n_layers: 裁剪层数
            crop_n_points_downscale_factor: 裁剪点下采样因子
            min_mask_region_area: 最小掩码区域面积
        """
        if not SAM_AVAILABLE:
            raise ImportError("需要安装segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")
        
        super().__init__(model_path, device, **kwargs)
        
        # SAM参数
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        
        # 确定模型类型
        self.model_type = self._determine_model_type()
        
        # SAM预测器和掩码生成器
        self.predictor = None
        self.mask_generator = None
    
    def _determine_model_type(self) -> str:
        """根据文件名确定SAM模型类型"""
        filename = self.model_path.name.lower()
        
        if 'vit_h' in filename or 'huge' in filename:
            return 'vit_h'
        elif 'vit_l' in filename or 'large' in filename:
            return 'vit_l'
        elif 'vit_b' in filename or 'base' in filename:
            return 'vit_b'
        elif 'mobile' in filename:
            return 'vit_b'  # Mobile SAM基于ViT-B
        else:
            # 默认假设是ViT-B
            logger.warning(f"无法从文件名确定SAM模型类型: {filename}，默认使用vit_b")
            return 'vit_b'
    
    def load_model(self) -> None:
        """加载SAM模型"""
        try:
            logger.info(f"加载SAM模型: {self.model_path} (类型: {self.model_type})")
            
            # 加载SAM模型
            self.model = sam_model_registry[self.model_type](checkpoint=str(self.model_path))
            self.model.to(device=self.device)
            
            # 创建预测器（用于交互式分割）
            self.predictor = SamPredictor(self.model)
            
            # 创建自动掩码生成器（用于全图分割）
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                min_mask_region_area=self.min_mask_region_area,
            )
            
            self.is_loaded = True
            logger.info(f"SAM模型加载成功 - 类型: {self.model_type}")
            
        except Exception as e:
            logger.error(f"SAM模型加载失败: {e}")
            raise
    
    def preprocess(self, input_data: Any) -> np.ndarray:
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
                image = input_data
            else:
                raise ValueError(f"不支持的numpy数组格式: {input_data.shape}")
        else:
            raise ValueError(f"不支持的输入格式: {type(input_data)}")
        
        # 转换为numpy数组（RGB格式）
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # 保存原始尺寸
        self.original_size = image_array.shape[:2]  # (height, width)
        
        return image_array
    
    def predict(self, 
                image: Union[str, Path, Image.Image, np.ndarray], 
                mode: str = "automatic",
                points: Optional[List[List[float]]] = None,
                point_labels: Optional[List[int]] = None,
                boxes: Optional[List[List[float]]] = None,
                mask_input: Optional[np.ndarray] = None,
                multimask_output: bool = True,
                **kwargs) -> Dict[str, Any]:
        """
        执行图像分割
        
        Args:
            image: 输入图像
            mode: 分割模式 ("automatic" 或 "interactive")
            points: 提示点坐标 [[x1, y1], [x2, y2], ...]
            point_labels: 点标签 [1, 0, 1, ...] (1为前景，0为背景)
            boxes: 边界框 [[x1, y1, x2, y2], ...]
            mask_input: 输入掩码
            multimask_output: 是否输出多个掩码
            
        Returns:
            分割结果字典
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # 预处理输入
            image_array = self.preprocess(image)
            
            start_time = time.time()
            
            if mode == "automatic":
                # 自动分割模式
                masks = self.mask_generator.generate(image_array)
                inference_time = time.time() - start_time
                
                # 后处理自动分割结果
                processed_results = self._postprocess_automatic(
                    masks, 
                    inference_time=inference_time
                )
                
            elif mode == "interactive":
                # 交互式分割模式
                self.predictor.set_image(image_array)
                
                # 转换输入格式
                input_points = np.array(points) if points else None
                input_labels = np.array(point_labels) if point_labels else None
                input_boxes = np.array(boxes) if boxes else None
                
                # 执行预测
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    box=input_boxes[0] if input_boxes is not None and len(input_boxes) > 0 else None,
                    mask_input=mask_input,
                    multimask_output=multimask_output,
                )
                
                inference_time = time.time() - start_time
                
                # 后处理交互式分割结果
                processed_results = self._postprocess_interactive(
                    masks, 
                    scores, 
                    logits,
                    inference_time=inference_time
                )
                
            else:
                raise ValueError(f"不支持的分割模式: {mode}")
            
            logger.debug(f"SAM分割完成 - 模式: {mode}, 耗时: {inference_time:.3f}s")
            return processed_results
            
        except Exception as e:
            logger.error(f"SAM预测失败: {e}")
            raise
    
    def postprocess(self, raw_output: Any, **kwargs) -> Dict[str, Any]:
        """
        后处理SAM输出 - 实现基类抽象方法
        
        这个方法是为了兼容BaseModelAdapter的抽象方法要求。
        实际的后处理逻辑在 _postprocess_automatic 和 _postprocess_interactive 中实现。
        """
        if isinstance(raw_output, dict):
            # 如果已经是处理后的字典格式，直接返回
            return raw_output
        elif isinstance(raw_output, list):
            # 如果是SAM自动模式的原始输出（掩码列表）
            return self._postprocess_automatic(raw_output, **kwargs)
        elif isinstance(raw_output, tuple) and len(raw_output) == 3:
            # 如果是SAM交互模式的原始输出（masks, scores, logits）
            masks, scores, logits = raw_output
            return self._postprocess_interactive(masks, scores, logits, **kwargs)
        else:
            # 未知格式，返回空结果
            logger.warning(f"未知的SAM输出格式: {type(raw_output)}")
            return {
                'masks': np.empty((0, 0, 0)),
                'scores': [],
                'areas': [],
                'bboxes': [],
                'metadata': {
                    'error': 'Unknown output format',
                    **kwargs
                }
            }
    
    def _postprocess_interactive(self, 
                               masks: np.ndarray, 
                               scores: np.ndarray, 
                               logits: np.ndarray,
                               **kwargs) -> Dict[str, Any]:
        """后处理交互式分割结果"""
        try:
            # masks: [N, H, W], scores: [N], logits: [N, H, W]
            
            mask_arrays = []
            score_list = []
            areas = []
            bboxes = []
            
            for i in range(len(masks)):
                mask = masks[i].astype(np.uint8)
                mask_arrays.append(mask)
                
                score_list.append(float(scores[i]))
                
                # 计算面积
                area = np.sum(mask)
                areas.append(float(area))
                
                # 计算边界框
                if area > 0:
                    y_indices, x_indices = np.where(mask > 0)
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                else:
                    bboxes.append([0.0, 0.0, 0.0, 0.0])
            
            result = {
                'masks': np.array(mask_arrays) if mask_arrays else np.empty((0, *self.original_size)),
                'scores': score_list,
                'areas': areas,
                'bboxes': bboxes,
                'logits': logits,  # 保留logits用于进一步处理
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'mode': 'interactive',
                    'num_masks': len(masks),
                    'original_size': self.original_size
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"SAM交互式分割后处理失败: {e}")
            raise
    
    def _postprocess_automatic(self, masks: List[Dict], **kwargs) -> Dict[str, Any]:
        """后处理自动分割结果"""
        try:
            if not masks:
                return {
                    'masks': np.empty((0, *self.original_size)),
                    'scores': [],
                    'areas': [],
                    'bboxes': [],
                    'metadata': {
                        'inference_time': kwargs.get('inference_time', 0),
                        'mode': 'automatic',
                        'num_masks': 0
                    }
                }
            
            # 按稳定性分数排序
            masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
            
            # 提取数据
            mask_arrays = []
            scores = []
            areas = []
            bboxes = []
            
            for mask_data in masks:
                mask = mask_data['segmentation']
                mask_arrays.append(mask.astype(np.uint8))
                
                scores.append(float(mask_data['stability_score']))
                areas.append(float(mask_data['area']))
                
                # 转换边界框格式
                bbox = mask_data['bbox']  # [x, y, w, h]
                bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                bboxes.append([float(x) for x in bbox_xyxy])
            
            result = {
                'masks': np.array(mask_arrays),
                'scores': scores,
                'areas': areas,
                'bboxes': bboxes,
                'metadata': {
                    'inference_time': kwargs.get('inference_time', 0),
                    'mode': 'automatic',
                    'num_masks': len(masks),
                    'original_size': self.original_size,
                    'points_per_side': self.points_per_side
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"SAM自动分割后处理失败: {e}")
            raise

    def predict_point(self, 
                     image: Union[str, Path, Image.Image, np.ndarray],
                     point: Tuple[float, float],
                     label: int = 1,
                     **kwargs) -> Dict[str, Any]:
        """使用单点提示进行分割"""
        return self.predict(
            image=image,
            mode="interactive",
            points=[list(point)],
            point_labels=[label],
            **kwargs
        )
    
    def predict_box(self, 
                   image: Union[str, Path, Image.Image, np.ndarray],
                   box: Tuple[float, float, float, float],
                   **kwargs) -> Dict[str, Any]:
        """使用边界框提示进行分割"""
        return self.predict(
            image=image,
            mode="interactive",
            boxes=[list(box)],
            **kwargs
        )
    
    def predict_points(self, 
                      image: Union[str, Path, Image.Image, np.ndarray],
                      points: List[Tuple[float, float]],
                      labels: List[int],
                      **kwargs) -> Dict[str, Any]:
        """使用多点提示进行分割"""
        points_list = [list(p) for p in points]
        return self.predict(
            image=image,
            mode="interactive",
            points=points_list,
            point_labels=labels,
            **kwargs
        )
    
    def visualize_results(self, 
                         image: Union[str, Path, Image.Image, np.ndarray],
                         results: Dict[str, Any],
                         save_path: str = None,
                         show_points: bool = False,
                         show_boxes: bool = False,
                         alpha: float = 0.6) -> Image.Image:
        """可视化SAM分割结果"""
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
            
            img_array = np.array(img)
            
            # 获取掩码
            masks = results['masks']
            
            if len(masks) == 0:
                logger.warning("没有找到分割掩码")
                return img
            
            # 创建彩色掩码
            overlay = img_array.copy()
            
            # 为每个掩码分配不同颜色
            colors = [
                [255, 0, 0],    # 红色
                [0, 255, 0],    # 绿色
                [0, 0, 255],    # 蓝色
                [255, 255, 0],  # 黄色
                [255, 0, 255],  # 洋红
                [0, 255, 255],  # 青色
                [255, 128, 0],  # 橙色
                [128, 0, 255],  # 紫色
                [255, 128, 128], # 浅红
                [128, 255, 128], # 浅绿
            ]
            
            for i, mask in enumerate(masks):
                if np.sum(mask) == 0:
                    continue
                
                color = colors[i % len(colors)]
                
                # 应用颜色到掩码区域
                overlay[mask > 0] = color
            
            # 混合原图和掩码
            blended = (1 - alpha) * img_array + alpha * overlay
            result_img = Image.fromarray(blended.astype(np.uint8))
            
            # 如果需要显示点或框，可以在这里添加绘制逻辑
            if show_points or show_boxes:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(result_img)
                
                # 这里可以添加点和框的绘制逻辑
                # 需要从metadata或其他地方获取点和框的信息
                pass
            
            # 保存图像
            if save_path:
                result_img.save(save_path)
                logger.info(f"SAM可视化结果已保存: {save_path}")
            
            return result_img
            
        except Exception as e:
            logger.error(f"SAM可视化失败: {e}")
            # 返回原图像
            if isinstance(image, Image.Image):
                return image
            else:
                return Image.open(image) if isinstance(image, (str, Path)) else Image.fromarray(image)
    
    def warmup(self, num_runs: int = 3) -> Dict[str, float]:
        """模型预热"""
        if not self.is_loaded:
            self.load_model()
        
        # 创建dummy输入进行预热
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        warmup_times = []
        for i in range(num_runs):
            start_time = time.time()
            try:
                # 使用自动模式进行预热（点数较少以加快速度）
                original_points = self.points_per_side
                self.points_per_side = 8  # 临时减少点数
                
                # 重新创建掩码生成器
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.model,
                    points_per_side=self.points_per_side,
                    pred_iou_thresh=self.pred_iou_thresh,
                    stability_score_thresh=self.stability_score_thresh,
                    crop_n_layers=0,  # 简化处理
                    min_mask_region_area=100,
                )
                
                results = self.predict(dummy_image, mode="automatic")
                warmup_time = time.time() - start_time
                warmup_times.append(warmup_time)
                
                # 恢复原始设置
                self.points_per_side = original_points
                
            except Exception as e:
                logger.warning(f"SAM预热运行 {i+1} 失败: {e}")
        
        if warmup_times:
            avg_time = np.mean(warmup_times)
            min_time = np.min(warmup_times)
            max_time = np.max(warmup_times)
            
            logger.info(f"SAM模型预热完成 - 平均耗时: {avg_time:.3f}s")
            
            return {
                "warmup_runs": len(warmup_times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time
            }
        
        return super().warmup(num_runs)
    
    def predict_batch(self, 
                     images: List[Union[str, Path, Image.Image, np.ndarray]],
                     mode: str = "automatic",
                     **kwargs) -> List[Dict[str, Any]]:
        """批量预测"""
        if not self.is_loaded:
            self.load_model()
        
        results = []
        for image in images:
            try:
                result = self.predict(image, mode=mode, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"批量预测中的图像失败: {e}")
                # 添加空结果以保持列表长度一致
                results.append({
                    'masks': np.empty((0, 0, 0)),
                    'scores': [],
                    'areas': [],
                    'bboxes': [],
                    'metadata': {'error': str(e)}
                })
        
        return results
    
    def set_automatic_config(self, preset: str = "default"):
        """设置自动分割的预设配置"""
        presets = {
            "fast": {
                "points_per_side": 16,
                "pred_iou_thresh": 0.8,
                "stability_score_thresh": 0.9,
                "crop_n_layers": 0,
                "min_mask_region_area": 500
            },
            "default": {
                "points_per_side": 32,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
                "crop_n_layers": 0,
                "min_mask_region_area": 0
            },
            "quality": {
                "points_per_side": 64,
                "pred_iou_thresh": 0.92,
                "stability_score_thresh": 0.97,
                "crop_n_layers": 1,
                "min_mask_region_area": 100
            }
        }
        
        if preset not in presets:
            raise ValueError(f"未知预设: {preset}. 可用预设: {list(presets.keys())}")
        
        config = presets[preset]
        
        # 更新参数
        self.points_per_side = config["points_per_side"]
        self.pred_iou_thresh = config["pred_iou_thresh"] 
        self.stability_score_thresh = config["stability_score_thresh"]
        self.crop_n_layers = config["crop_n_layers"]
        self.min_mask_region_area = config["min_mask_region_area"]
        
        # 重新创建掩码生成器
        if self.is_loaded:
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=self.points_per_side,
                pred_iou_thresh=self.pred_iou_thresh,
                stability_score_thresh=self.stability_score_thresh,
                crop_n_layers=self.crop_n_layers,
                crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
                min_mask_region_area=self.min_mask_region_area,
            )
        
        logger.info(f"SAM配置已设置为: {preset}")
    
    def get_mask_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """获取掩码统计信息"""
        masks = results.get('masks', [])
        scores = results.get('scores', [])
        areas = results.get('areas', [])
        
        if len(masks) == 0:
            return {
                'num_masks': 0,
                'total_area': 0,
                'avg_area': 0,
                'avg_score': 0,
                'coverage_ratio': 0
            }
        
        total_area = sum(areas)
        avg_area = total_area / len(areas)
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 计算覆盖率（假设图像大小）
        image_area = self.original_size[0] * self.original_size[1] if hasattr(self, 'original_size') else 1
        coverage_ratio = total_area / image_area
        
        return {
            'num_masks': len(masks),
            'total_area': total_area,
            'avg_area': avg_area,
            'avg_score': avg_score,
            'coverage_ratio': coverage_ratio,
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0
        }
    
    def filter_masks(self, 
                    results: Dict[str, Any],
                    min_area: int = 100,
                    min_score: float = 0.8,
                    max_masks: int = None) -> Dict[str, Any]:
        """过滤掩码结果"""
        masks = results.get('masks', [])
        scores = results.get('scores', [])
        areas = results.get('areas', [])
        bboxes = results.get('bboxes', [])
        
        if len(masks) == 0:
            return results
        
        # 创建过滤索引
        valid_indices = []
        for i, (area, score) in enumerate(zip(areas, scores)):
            if area >= min_area and score >= min_score:
                valid_indices.append(i)
        
        # 如果指定了最大掩码数，按分数排序并取前N个
        if max_masks and len(valid_indices) > max_masks:
            scored_indices = [(i, scores[i]) for i in valid_indices]
            scored_indices.sort(key=lambda x: x[1], reverse=True)
            valid_indices = [i for i, _ in scored_indices[:max_masks]]
        
        # 应用过滤
        filtered_results = results.copy()
        filtered_results['masks'] = masks[valid_indices] if len(valid_indices) > 0 else np.empty((0, *masks.shape[1:]))
        filtered_results['scores'] = [scores[i] for i in valid_indices]
        filtered_results['areas'] = [areas[i] for i in valid_indices]
        filtered_results['bboxes'] = [bboxes[i] for i in valid_indices]
        
        # 更新元数据
        if 'metadata' not in filtered_results:
            filtered_results['metadata'] = {}
        filtered_results['metadata']['filtered'] = True
        filtered_results['metadata']['original_num_masks'] = len(masks)
        filtered_results['metadata']['filtered_num_masks'] = len(valid_indices)
        filtered_results['metadata']['filter_params'] = {
            'min_area': min_area,
            'min_score': min_score,
            'max_masks': max_masks
        }
        
        logger.info(f"掩码过滤: {len(masks)} -> {len(valid_indices)}")
        return filtered_results
    
    def merge_masks(self, results: Dict[str, Any], overlap_threshold: float = 0.8) -> Dict[str, Any]:
        """合并重叠的掩码"""
        masks = results.get('masks', [])
        
        if len(masks) <= 1:
            return results
        
        try:
            # 计算掩码间的IoU
            def calculate_iou(mask1, mask2):
                intersection = np.logical_and(mask1, mask2).sum()
                union = np.logical_or(mask1, mask2).sum()
                return intersection / union if union > 0 else 0
            
            # 找到需要合并的掩码对
            to_merge = []
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    iou = calculate_iou(masks[i], masks[j])
                    if iou > overlap_threshold:
                        to_merge.append((i, j, iou))
            
            if not to_merge:
                return results
            
            # 执行合并（简化版本：只合并IoU最高的一对）
            to_merge.sort(key=lambda x: x[2], reverse=True)
            i, j, _ = to_merge[0]
            
            # 创建合并后的结果
            merged_results = results.copy()
            merged_mask = np.logical_or(masks[i], masks[j]).astype(np.uint8)
            
            # 更新数组
            new_masks = []
            new_scores = []
            new_areas = []
            new_bboxes = []
            
            scores = results.get('scores', [])
            areas = results.get('areas', [])
            bboxes = results.get('bboxes', [])
            
            for k in range(len(masks)):
                if k == i:
                    # 使用合并后的掩码
                    new_masks.append(merged_mask)
                    new_scores.append(max(scores[i], scores[j]) if k < len(scores) else 1.0)
                    new_areas.append(float(np.sum(merged_mask)))
                    
                    # 计算合并后的边界框
                    y_indices, x_indices = np.where(merged_mask > 0)
                    if len(y_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        new_bboxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
                    else:
                        new_bboxes.append([0.0, 0.0, 0.0, 0.0])
                        
                elif k != j:  # 跳过被合并的掩码
                    new_masks.append(masks[k])
                    new_scores.append(scores[k] if k < len(scores) else 1.0)
                    new_areas.append(areas[k] if k < len(areas) else float(np.sum(masks[k])))
                    new_bboxes.append(bboxes[k] if k < len(bboxes) else [0.0, 0.0, 0.0, 0.0])
            
            merged_results['masks'] = np.array(new_masks)
            merged_results['scores'] = new_scores
            merged_results['areas'] = new_areas  
            merged_results['bboxes'] = new_bboxes
            
            if 'metadata' not in merged_results:
                merged_results['metadata'] = {}
            merged_results['metadata']['merged'] = True
            merged_results['metadata']['original_num_masks'] = len(masks)
            merged_results['metadata']['merged_num_masks'] = len(new_masks)
            
            logger.info(f"掩码合并: {len(masks)} -> {len(new_masks)}")
            return merged_results
            
        except Exception as e:
            logger.warning(f"掩码合并失败: {e}")
            return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型详细信息"""
        info = super().get_model_info()
        
        info.update({
            'model_type': 'segmentation',
            'framework': 'segment_anything',
            'architecture': f'sam_{self.model_type}',
            'points_per_side': self.points_per_side,
            'pred_iou_thresh': self.pred_iou_thresh,
            'stability_score_thresh': self.stability_score_thresh,
            'supported_modes': ['automatic', 'interactive'],
            'input_prompts': ['points', 'boxes', 'masks']
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
                logger.debug(f"获取SAM模型详细信息失败: {e}")
        
        return info