"""
模型自动发现器 - 修复HuggingFace模型重复检测问题和DeepLabV3变体识别

修复内容：
1. 避免HuggingFace格式目录中的子组件被检测为独立模型
2. 改进HuggingFace模型的检测逻辑
3. 添加更严格的路径过滤
4. 正确识别DeepLabV3的不同变体（resnet50, resnet101, mobilenet）
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger


@dataclass 
class ModelInfo:
    """模型信息数据类"""
    name: str
    type: str  # detection, segmentation, classification, generation, multimodal
    path: str
    format: str  # pytorch, safetensors, onnx, tensorrt, huggingface
    size_mb: float
    framework: str  # ultralytics, sam, detectron2, diffusers, etc.
    architecture: str  # yolov8, sam_vit_h, resnet50, stable_diffusion, etc.
    confidence: float  # 检测置信度 (0-1)
    metadata: Dict[str, Any]  # 额外的元数据


class ModelDetector:
    """模型自动发现器"""
    
    # 支持的模型文件扩展名
    SUPPORTED_EXTENSIONS = {
        '.pt': 'pytorch',
        '.pth': 'pytorch', 
        '.ckpt': 'pytorch',
        '.safetensors': 'safetensors',
        '.onnx': 'onnx',
        '.trt': 'tensorrt',
        '.engine': 'tensorrt',
        '.bin': 'huggingface',  # HuggingFace binary format
    }
    
    # HuggingFace模型的标识文件
    HUGGINGFACE_INDICATORS = {
        'model_index.json',     # Diffusers模型
        'config.json',          # Transformers模型
        'pytorch_model.bin',    # PyTorch权重
        'model.safetensors',    # SafeTensors权重
    }
    
    # HuggingFace子组件目录（应该被跳过）
    HUGGINGFACE_COMPONENT_DIRS = {
        'text_encoder', 'text_encoder_2', 'unet', 'vae', 'safety_checker',
        'feature_extractor', 'scheduler', 'tokenizer', 'image_processor',
        'controlnet', 'adapter'
    }
    
    # 模型名称模式匹配
    MODEL_PATTERNS = {
        # YOLO 检测模型
        'yolo': {
            'patterns': ['yolov8', 'yolov9', 'yolov10', 'yolo11', 'yolov5'],
            'type': 'detection',
            'framework': 'ultralytics'
        },
        # SAM 分割模型
        'sam': {
            'patterns': ['sam_vit_h', 'sam_vit_l', 'sam_vit_b', 'mobile_sam'],
            'type': 'segmentation',
            'framework': 'segment_anything'
        },
        # DeepLab 分割模型 - 支持更多变体
        'deeplabv3': {
            'patterns': ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet', 'deeplabv3', 'deeplab'],
            'type': 'segmentation',
            'framework': 'torchvision'
        },
        # Mask2Former 分割模型
        'mask2former': {
            'patterns': ['mask2former'],
            'type': 'segmentation',
            'framework': 'detectron2'
        },
        # U-Net 分割模型
        'unet': {
            'patterns': ['unet', 'u_net'],
            'type': 'segmentation',
            'framework': 'pytorch'
        },
        # ResNet 分类模型
        'resnet': {
            'patterns': ['resnet18-', 'resnet34-', 'resnet50-', 'resnet101-', 'resnet152-'],
            'type': 'classification',
            'framework': 'torchvision'
        },
        # EfficientNet 分类模型
        'efficientnet': {
            'patterns': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                        'efficientnet_b6', 'efficientnet_b7'],
            'type': 'classification', 
            'framework': 'timm'
        },
        # Stable Diffusion 生成模型
        'stable_diffusion': {
            'patterns': ['v1-5-pruned', 'v2-1', 'sdxl', 'sd_xl', 'stable-diffusion'],
            'type': 'generation',
            'framework': 'diffusers'
        },
        # CLIP 多模态模型
        'clip': {
            'patterns': ['clip-vit', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14'],
            'type': 'multimodal',
            'framework': 'transformers'
        }
    }
    
    def __init__(self, models_root: Optional[Path] = None):
        """
        初始化模型检测器
        
        Args:
            models_root: 模型根目录，默认使用配置管理器中的路径
        """
        if models_root:
            self.models_root = Path(models_root)
        else:
            from .config_manager import get_config_manager
            self.models_root = get_config_manager().get_models_root()
        
        # 用于跟踪已检测的HuggingFace模型目录
        self._detected_hf_dirs: Set[Path] = set()
        
        logger.info(f"初始化模型检测器 - 根目录: {self.models_root}")
    
    def detect_models(self, 
                     include_patterns: Optional[List[str]] = None,
                     exclude_patterns: Optional[List[str]] = None,
                     min_size_mb: float = 0.1,
                     max_size_mb: float = 50000) -> List[ModelInfo]:
        """
        检测所有可用的模型
        """
        logger.info("开始扫描模型文件...")
        
        if not self.models_root.exists():
            logger.warning(f"模型根目录不存在: {self.models_root}")
            return []
        
        detected_models = []
        self._detected_hf_dirs.clear()  # 重置HuggingFace目录追踪
        
        # 第一步：检测HuggingFace格式的模型
        hf_models = self._detect_all_huggingface_models()
        detected_models.extend(hf_models)
        
        # 第二步：扫描各个类别目录的其他格式模型
        for category_dir in self.models_root.iterdir():
            if not category_dir.is_dir():
                continue
                
            logger.debug(f"扫描目录: {category_dir}")
            category_models = self._scan_directory(
                category_dir, 
                include_patterns, 
                exclude_patterns,
                min_size_mb,
                max_size_mb
            )
            detected_models.extend(category_models)
        
        # 按置信度和文件大小排序
        detected_models.sort(key=lambda x: (x.confidence, x.size_mb), reverse=True)
        
        logger.info(f"模型扫描完成 - 发现 {len(detected_models)} 个模型")
        return detected_models
    
    def _detect_all_huggingface_models(self) -> List[ModelInfo]:
        """检测所有HuggingFace格式的模型"""
        hf_models = []
        
        for root, dirs, files in os.walk(self.models_root):
            root_path = Path(root)
            
            # 跳过已经检测过的HuggingFace目录的子目录
            if any(root_path.is_relative_to(hf_dir) and root_path != hf_dir 
                   for hf_dir in self._detected_hf_dirs):
                continue
            
            # 检查是否是HuggingFace模型目录
            if self._is_huggingface_model_dir(root_path):
                hf_model = self._detect_huggingface_model(root_path)
                if hf_model:
                    hf_models.append(hf_model)
                    self._detected_hf_dirs.add(root_path)
                    logger.info(f"检测到HuggingFace模型: {hf_model.name} at {root_path}")
        
        return hf_models
    
    def _is_huggingface_model_dir(self, dir_path: Path) -> bool:
        """判断目录是否是HuggingFace模型目录"""
        # 检查是否包含HuggingFace模型的标识文件
        for indicator in self.HUGGINGFACE_INDICATORS:
            if (dir_path / indicator).exists():
                return True
        return False
    
    def _scan_directory(self, 
                       directory: Path,
                       include_patterns: Optional[List[str]],
                       exclude_patterns: Optional[List[str]], 
                       min_size_mb: float,
                       max_size_mb: float) -> List[ModelInfo]:
        """扫描指定目录（跳过HuggingFace模型内部文件）"""
        models = []
        
        # 递归扫描所有文件
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            
            # 跳过HuggingFace模型目录内的文件
            if self._is_inside_huggingface_model(file_path):
                continue
            
            # 跳过HuggingFace组件子目录中的文件
            if self._is_huggingface_component_file(file_path):
                continue
            
            # 检查文件扩展名
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            
            # 检查文件大小
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb < min_size_mb or size_mb > max_size_mb:
                    continue
            except OSError:
                logger.warning(f"无法读取文件大小: {file_path}")
                continue
            
            # 应用包含/排除模式
            if include_patterns and not any(pattern in file_path.name for pattern in include_patterns):
                continue
            if exclude_patterns and any(pattern in file_path.name for pattern in exclude_patterns):
                continue
            
            # 检测模型信息
            model_info = self._analyze_model_file(file_path)
            if model_info:
                models.append(model_info)
        
        return models
    
    def _is_inside_huggingface_model(self, file_path: Path) -> bool:
        """检查文件是否在已检测的HuggingFace模型目录内"""
        for hf_dir in self._detected_hf_dirs:
            try:
                file_path.relative_to(hf_dir)
                return True
            except ValueError:
                continue
        return False
    
    def _is_huggingface_component_file(self, file_path: Path) -> bool:
        """检查文件是否在HuggingFace组件子目录中"""
        for parent in file_path.parents:
            if parent.name in self.HUGGINGFACE_COMPONENT_DIRS:
                # 进一步检查上级目录是否可能是HuggingFace模型
                for ancestor in parent.parents:
                    if self._is_huggingface_model_dir(ancestor):
                        return True
        return False
    
    def _detect_huggingface_model(self, model_dir: Path) -> Optional[ModelInfo]:
        """检测HuggingFace格式的模型"""
        try:
            # 优先检查diffusers模型（有model_index.json）
            model_index_file = model_dir / 'model_index.json'
            if model_index_file.exists():
                return self._detect_diffusers_model(model_dir, model_index_file)
            
            # 检查transformers模型（有config.json）
            config_file = model_dir / 'config.json'
            if config_file.exists():
                return self._detect_transformers_model(model_dir, config_file)
            
            # 检查是否只是包含权重文件的目录
            if any((model_dir / f).exists() for f in ['pytorch_model.bin', 'model.safetensors']):
                return self._detect_generic_hf_model(model_dir)
            
            return None
            
        except Exception as e:
            logger.warning(f"检测HuggingFace模型失败 {model_dir}: {e}")
            return None
    
    def _detect_diffusers_model(self, model_dir: Path, model_index_file: Path) -> Optional[ModelInfo]:
        """检测Diffusers模型（如Stable Diffusion）"""
        try:
            with open(model_index_file, 'r', encoding='utf-8') as f:
                model_index = json.load(f)
            
            # 计算目录总大小
            total_size = self._calculate_directory_size(model_dir)
            size_mb = total_size / (1024 * 1024)
            
            # 推断模型类型和架构
            model_type = 'generation'
            framework = 'diffusers'
            
            dir_name = model_dir.name.lower()
            if 'sdxl' in dir_name or 'xl' in dir_name:
                architecture = 'stable_diffusion_xl'
            elif 'flux' in dir_name:
                architecture = 'flux'
            elif 'controlnet' in dir_name:
                architecture = 'controlnet'
            else:
                architecture = 'stable_diffusion'
            
            model_name = model_dir.name
            
            metadata = {
                'model_index': model_index,
                'huggingface_format': True,
                'diffusers_model': True,
                'components': list(model_index.keys()) if isinstance(model_index, dict) else []
            }
            
            return ModelInfo(
                name=model_name,
                type=model_type,
                path=str(model_dir),
                format='huggingface',
                size_mb=size_mb,
                framework=framework,
                architecture=architecture,
                confidence=0.95,  # 高置信度，因为有明确的model_index.json
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"检测Diffusers模型失败 {model_dir}: {e}")
            return None
    
    def _detect_transformers_model(self, model_dir: Path, config_file: Path) -> Optional[ModelInfo]:
        """检测Transformers模型（如CLIP、BERT等）"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            total_size = self._calculate_directory_size(model_dir)
            size_mb = total_size / (1024 * 1024)
            
            # 根据config推断模型类型
            model_type = 'multimodal'  # 默认
            framework = 'transformers'
            architecture = 'unknown'
            
            # 根据架构信息推断具体类型
            arch = config.get('architectures', [''])[0].lower() if config.get('architectures') else ''
            model_type_from_config = config.get('model_type', '').lower()
            
            if 'clip' in arch or 'clip' in model_type_from_config:
                model_type = 'multimodal'
                architecture = 'clip'
            elif 'bert' in arch or 'bert' in model_type_from_config:
                model_type = 'text'
                architecture = 'bert'
            elif 'gpt' in arch or 'gpt' in model_type_from_config:
                model_type = 'text'
                architecture = 'gpt'
            
            model_name = model_dir.name
            
            metadata = {
                'config': config,
                'huggingface_format': True,
                'transformers_model': True,
                'architecture_from_config': arch
            }
            
            return ModelInfo(
                name=model_name,
                type=model_type,
                path=str(model_dir),
                format='huggingface',
                size_mb=size_mb,
                framework=framework,
                architecture=architecture,
                confidence=0.9,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"检测Transformers模型失败 {model_dir}: {e}")
            return None
    
    def _detect_generic_hf_model(self, model_dir: Path) -> Optional[ModelInfo]:
        """检测通用HuggingFace模型（只有权重文件）"""
        try:
            total_size = self._calculate_directory_size(model_dir)
            size_mb = total_size / (1024 * 1024)
            
            model_name = model_dir.name
            
            # 基于目录名称推断
            dir_name = model_name.lower()
            if 'diffusion' in dir_name or 'sd' in dir_name:
                model_type = 'generation'
                architecture = 'stable_diffusion'
            elif 'clip' in dir_name:
                model_type = 'multimodal'
                architecture = 'clip'
            else:
                model_type = 'unknown'
                architecture = 'unknown'
            
            metadata = {
                'huggingface_format': True,
                'generic_hf_model': True
            }
            
            return ModelInfo(
                name=model_name,
                type=model_type,
                path=str(model_dir),
                format='huggingface',
                size_mb=size_mb,
                framework='huggingface',
                architecture=architecture,
                confidence=0.7,  # 中等置信度
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"检测通用HuggingFace模型失败 {model_dir}: {e}")
            return None
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """计算目录总大小"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.debug(f"计算目录大小失败 {directory}: {e}")
        return total_size
    
    def _analyze_model_file(self, file_path: Path) -> Optional[ModelInfo]:
        """分析单个模型文件"""
        try:
            # 获取基本信息
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_format = self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), 'unknown')
            
            # 推断模型类型和框架
            model_type, framework, architecture, confidence = self._infer_model_properties(file_path)
            
            # 生成模型名称
            model_name = self._generate_model_name(file_path, architecture)
            
            # 收集元数据
            metadata = self._collect_metadata(file_path)
            
            return ModelInfo(
                name=model_name,
                type=model_type,
                path=str(file_path),
                format=file_format,
                size_mb=size_mb,
                framework=framework,
                architecture=architecture,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"分析模型文件失败 {file_path}: {e}")
            return None
    
    def _infer_model_properties(self, file_path: Path) -> Tuple[str, str, str, float]:
        """推断模型属性"""
        filename = file_path.name.lower()
        parent_dirs = [p.name.lower() for p in file_path.parents]
        
        # 特殊处理：优先检查明确的模型模式
        # YOLO模型的精确匹配
        yolo_variants = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                        'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov10m', 'yolov10l', 'yolov10x',
                        'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
        
        for variant in yolo_variants:
            if variant in filename:
                return 'detection', 'ultralytics', variant, 0.95
        
        # DeepLabV3 检测 - 精确匹配不同变体
        if 'deeplabv3' in filename:
            if 'resnet101' in filename:
                arch = 'deeplabv3_resnet101'
            elif 'resnet50' in filename:
                arch = 'deeplabv3_resnet50'
            elif 'mobilenet' in filename:
                arch = 'deeplabv3_mobilenet_v3_large'
            else:
                arch = 'deeplabv3'
            return 'segmentation', 'torchvision', arch, 0.9
        
        # Mask2Former 检测
        if 'mask2former' in filename:
            return 'segmentation', 'detectron2', 'mask2former', 0.9
        
        # U-Net 检测
        if 'unet' in filename or 'u_net' in filename:
            return 'segmentation', 'pytorch', 'unet', 0.9
        
        # 基于路径和文件名推断
        for pattern_group, info in self.MODEL_PATTERNS.items():
            for pattern in info['patterns']:
                confidence = 0.0
                
                # 文件名匹配
                if pattern in filename:
                    confidence += 0.6
                
                # 父目录匹配
                if any(pattern in parent_dir for parent_dir in parent_dirs):
                    confidence += 0.3
                    
                # 路径结构匹配 - 这是最重要的判断依据
                if info['type'] in parent_dirs:
                    confidence += 0.4  # 提高路径结构的权重
                
                if confidence > 0.5:  # 置信度阈值
                    return info['type'], info['framework'], pattern, confidence
        
        # 基于目录结构的后备推断（这是最可靠的方法）
        if 'segmentation' in parent_dirs:
            # 在分割目录中，进一步判断具体框架
            if 'deeplabv3' in filename:
                if 'resnet101' in filename:
                    return 'segmentation', 'torchvision', 'deeplabv3_resnet101', 0.8
                elif 'resnet50' in filename:
                    return 'segmentation', 'torchvision', 'deeplabv3_resnet50', 0.8
                elif 'mobilenet' in filename:
                    return 'segmentation', 'torchvision', 'deeplabv3_mobilenet_v3_large', 0.8
                else:
                    return 'segmentation', 'torchvision', 'deeplabv3', 0.8
            elif 'deeplab' in filename:
                return 'segmentation', 'torchvision', 'deeplabv3', 0.8
            elif 'sam' in filename:
                return 'segmentation', 'segment_anything', 'sam', 0.8
            elif 'mask2former' in filename:
                return 'segmentation', 'detectron2', 'mask2former', 0.8
            elif 'unet' in filename:
                return 'segmentation', 'pytorch', 'unet', 0.8
            else:
                # 通用分割模型
                return 'segmentation', 'unknown', 'unknown', 0.7
                
        elif 'detection' in parent_dirs:
            if 'yolo' in filename:
                return 'detection', 'ultralytics', 'yolo', 0.8
            else:
                return 'detection', 'unknown', 'unknown', 0.7
                
        elif 'classification' in parent_dirs:
            if 'resnet' in filename:
                return 'classification', 'torchvision', 'resnet', 0.8
            elif 'efficientnet' in filename:
                return 'classification', 'timm', 'efficientnet', 0.8
            else:
                return 'classification', 'unknown', 'unknown', 0.7
                
        elif 'generation' in parent_dirs:
            if 'stable' in filename or 'sd' in filename or 'diffusion' in filename:
                return 'generation', 'diffusers', 'stable_diffusion', 0.8
            else:
                return 'generation', 'unknown', 'unknown', 0.7
                
        elif 'multimodal' in parent_dirs:
            if 'clip' in filename:
                return 'multimodal', 'transformers', 'clip', 0.8
            else:
                return 'multimodal', 'unknown', 'unknown', 0.7
        
        # 如果没有明确的目录结构，使用更保守的推断
        return 'unknown', 'unknown', 'unknown', 0.1
    
    def _generate_model_name(self, file_path: Path, architecture: str) -> str:
        """生成模型名称"""
        if architecture != 'unknown':
            # 对于YOLO模型，保持完整的文件名（去掉扩展名）
            if 'yolo' in architecture.lower():
                filename = file_path.stem
                # 保留yolov8n, yolov8s等完整名称
                if any(variant in filename.lower() for variant in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                                                                   'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov10m']):
                    return filename.lower()
            
            # 对于DeepLabV3，精确识别不同变体
            if 'deeplabv3' in architecture.lower():
                if 'resnet101' in file_path.name.lower():
                    return 'deeplabv3_resnet101'
                elif 'resnet50' in file_path.name.lower():
                    return 'deeplabv3_resnet50'
                elif 'mobilenet' in file_path.name.lower():
                    return 'deeplabv3_mobilenet_v3_large'
                else:
                    return 'deeplabv3'
            
            return architecture
        
        # 使用文件名（去掉扩展名）
        name = file_path.stem.lower()
        
        # 清理常见后缀，但保留重要的标识符
        suffixes_to_remove = ['-pruned', '-emaonly', '_pruned', '_ema', '_best', '_final', '_coco', '-586e9e4e']
        for suffix in suffixes_to_remove:
            name = name.replace(suffix, '')
        
        return name
    
    def _collect_metadata(self, file_path: Path) -> Dict[str, Any]:
        """收集模型元数据"""
        metadata = {
            'file_path': str(file_path),
            'parent_dir': str(file_path.parent),
            'modified_time': file_path.stat().st_mtime,
            'file_hash': self._calculate_file_hash(file_path),
        }
        
        # 尝试读取模型特定的元数据
        if file_path.suffix.lower() == '.safetensors':
            metadata.update(self._read_safetensors_metadata(file_path))
        elif file_path.suffix.lower() in ['.pt', '.pth']:
            metadata.update(self._read_pytorch_metadata(file_path))
        
        return metadata
    
    def _calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """计算文件哈希值（仅读取文件头部以提高速度）"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                # 只读取前1MB来计算哈希，避免大文件读取过慢
                chunk = f.read(1024 * 1024)
                hasher.update(chunk)
            return hasher.hexdigest()[:16]  # 返回前16位
        except Exception as e:
            logger.warning(f"计算文件哈希失败 {file_path}: {e}")
            return "unknown"
    
    def _read_safetensors_metadata(self, file_path: Path) -> Dict[str, Any]:
        """读取SafeTensors文件的元数据"""
        metadata = {}
        try:
            # SafeTensors文件头包含JSON元数据
            with open(file_path, 'rb') as f:
                # 读取前8字节获取头部长度
                header_size = int.from_bytes(f.read(8), byteorder='little')
                if header_size < 1024 * 1024:  # 合理的头部大小
                    header_bytes = f.read(header_size)
                    header_str = header_bytes.decode('utf-8')
                    header_data = json.loads(header_str)
                    
                    # 提取有用信息
                    if '__metadata__' in header_data:
                        metadata.update(header_data['__metadata__'])
                    
                    # 统计参数数量
                    total_params = 0
                    for tensor_info in header_data.values():
                        if isinstance(tensor_info, dict) and 'shape' in tensor_info:
                            shape = tensor_info['shape']
                            params = 1
                            for dim in shape:
                                params *= dim
                            total_params += params
                    
                    metadata['total_parameters'] = total_params
                    
        except Exception as e:
            logger.debug(f"读取SafeTensors元数据失败 {file_path}: {e}")
        
        return metadata
    
    def _read_pytorch_metadata(self, file_path: Path) -> Dict[str, Any]:
        """读取PyTorch模型元数据"""
        metadata = {}
        try:
            # 这里只获取基本信息，避免完全加载模型
            import torch
            
            # 尝试加载模型检查点信息
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            
            if isinstance(checkpoint, dict):
                # 常见的检查点键
                if 'model' in checkpoint:
                    model_state = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    model_state = checkpoint['state_dict']
                else:
                    model_state = checkpoint
                
                # 统计参数数量
                if isinstance(model_state, dict):
                    total_params = sum(p.numel() for p in model_state.values() if hasattr(p, 'numel'))
                    metadata['total_parameters'] = total_params
                
                # 其他元数据
                for key in ['epoch', 'version', 'arch', 'best_acc1']:
                    if key in checkpoint:
                        metadata[key] = checkpoint[key]
                        
        except Exception as e:
            logger.debug(f"读取PyTorch元数据失败 {file_path}: {e}")
        
        return metadata
    
    def generate_config(self, detected_models: List[ModelInfo], output_file: Optional[Path] = None) -> Dict[str, Any]:
        """根据检测到的模型生成配置文件"""
        config = {
            "models_root": str(self.models_root),
            "models": {}
        }
        
        for model in detected_models:
            # 跳过置信度太低的模型
            if model.confidence < 0.3:
                continue
            
            model_config = {
                "type": model.type,
                "path": model.path,
                "format": model.format,
                "framework": model.framework,
                "architecture": model.architecture,
                "device": "auto",
                "metadata": {
                    "size_mb": model.size_mb,
                    "confidence": model.confidence,
                    "auto_detected": True
                }
            }
            
            # 添加模型类型特定的默认配置
            if model.type == "detection":
                model_config.update({
                    "batch_size": 4,
                    "confidence": 0.25,
                    "nms_threshold": 0.45
                })
            elif model.type == "segmentation":
                model_config.update({
                    "batch_size": 1,
                    "points_per_side": 32 if "sam" in model.framework else 16
                })
            elif model.type == "classification":
                model_config.update({
                    "batch_size": 8,
                    "top_k": 5
                })
            elif model.type == "generation":
                model_config.update({
                    "batch_size": 1,
                    "enable_memory_efficient_attention": True
                })
            
            config["models"][model.name] = model_config
        
        # 保存配置文件
        if output_file:
            import yaml
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"模型配置已保存到: {output_file}")
        
        return config
    
    def get_summary(self, models: List[ModelInfo]) -> Dict[str, Any]:
        """获取检测结果摘要"""
        summary = {
            "total_models": len(models),
            "by_type": {},
            "by_framework": {},
            "total_size_mb": sum(m.size_mb for m in models),
            "high_confidence": len([m for m in models if m.confidence > 0.7]),
            "medium_confidence": len([m for m in models if 0.3 <= m.confidence <= 0.7]),
            "low_confidence": len([m for m in models if m.confidence < 0.3])
        }
        
        # 按类型统计
        for model in models:
            model_type = model.type
            if model_type not in summary["by_type"]:
                summary["by_type"][model_type] = {"count": 0, "size_mb": 0}
            summary["by_type"][model_type]["count"] += 1
            summary["by_type"][model_type]["size_mb"] += model.size_mb
        
        # 按框架统计
        for model in models:
            framework = model.framework
            if framework not in summary["by_framework"]:
                summary["by_framework"][framework] = {"count": 0, "size_mb": 0}
            summary["by_framework"][framework]["count"] += 1
            summary["by_framework"][framework]["size_mb"] += model.size_mb
        
        return summary


# 便利函数
def detect_models_in_directory(directory: str, **kwargs) -> List[ModelInfo]:
    """便利函数：检测指定目录中的模型"""
    detector = ModelDetector(Path(directory))
    return detector.detect_models(**kwargs)


def generate_models_config(directory: str, output_file: str = None) -> Dict[str, Any]:
    """便利函数：生成模型配置文件"""
    detector = ModelDetector(Path(directory))
    models = detector.detect_models()
    output_path = Path(output_file) if output_file else None
    return detector.generate_config(models, output_path)