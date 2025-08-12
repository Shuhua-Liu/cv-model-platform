import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger


@dataclass 
class ModelInfo:
    """Model Information Data Class"""
    name: str
    type: str  # detection, segmentation, classification, generation, multimodal
    path: str
    format: str  # pytorch, safetensors, onnx, tensorrt, huggingface
    size_mb: float
    framework: str  # ultralytics, sam, detectron2, diffusers, etc.
    architecture: str  # yolov8, sam_vit_h, resnet50, stable_diffusion, etc.
    confidence: float  # Detection confidence (0-1)
    metadata: Dict[str, Any]  # Additional metadata


class ModelDetector:
    """Automatic Model Discoverer"""
    
    # Supported model file extensions
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
    
    # HuggingFace model identification file
    HUGGINGFACE_INDICATORS = {
        'model_index.json',     # Diffusers model
        'config.json',          # Transformers model 
        'pytorch_model.bin',    # PyTorch weights
        'model.safetensors',    # SafeTensors weights
    }
    
    # HuggingFace subcomponent directory (should be skipped)
    HUGGINGFACE_COMPONENT_DIRS = {
        'text_encoder', 'text_encoder_2', 'unet', 'vae', 'safety_checker',
        'feature_extractor', 'scheduler', 'tokenizer', 'image_processor',
        'controlnet', 'adapter'
    }
    
    # Model name pattern matching
    MODEL_PATTERNS = {
        # YOLO detection model
        'yolo': {
            'patterns': ['yolov8', 'yolov9', 'yolov10', 'yolo11', 'yolov5'],
            'type': 'detection',
            'framework': 'ultralytics'
        },
        # SAM segmentation model
        'sam': {
            'patterns': ['sam_vit_h', 'sam_vit_l', 'sam_vit_b', 'mobile_sam'],
            'type': 'segmentation',
            'framework': 'segment_anything'
        },
        # DeepLab segmentation model - Support more variants
        'deeplabv3': {
            'patterns': ['deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3_mobilenet', 'deeplabv3', 'deeplab'],
            'type': 'segmentation',
            'framework': 'torchvision'
        },
        # Mask2Former segmentation model
        'mask2former': {
            'patterns': ['mask2former'],
            'type': 'segmentation',
            'framework': 'detectron2'
        },
        # U-Net segmentation model
        'unet': {
            'patterns': ['unet', 'u_net'],
            'type': 'segmentation',
            'framework': 'pytorch'
        },
        # ResNet classification model
        'resnet': {
            'patterns': ['resnet18-', 'resnet34-', 'resnet50-', 'resnet101-', 'resnet152-'],
            'type': 'classification',
            'framework': 'torchvision'
        },
        # EfficientNet classification model
        'efficientnet': {
            'patterns': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 
                        'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
                        'efficientnet_b6', 'efficientnet_b7'],
            'type': 'classification', 
            'framework': 'timm'
        },
        # Stable Diffusion generation model
        'stable_diffusion': {
            'patterns': ['v1-5-pruned', 'v2-1', 'sdxl', 'sd_xl', 'stable-diffusion'],
            'type': 'generation',
            'framework': 'diffusers'
        },
        # CLIP multimodal model
        'clip': {
            'patterns': ['clip-vit', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14'],
            'type': 'multimodal',
            'framework': 'transformers'
        }
    }
    
    def __init__(self, models_root: Optional[Path] = None):
        """
        Initialize the model detector
        
        Args:
            models_root: The model root directory. By default, the path specified in the configuration manager is used.
        """
        if models_root:
            self.models_root = Path(models_root)
        else:
            from .config_manager import get_config_manager
            self.models_root = get_config_manager().get_models_root()
        
        # HuggingFace model directory for tracking detected faces
        self._detected_hf_dirs: Set[Path] = set()
        
        logger.info(f"Initialize the model detector - root directory: {self.models_root}")
    
    def detect_models(self, 
                     include_patterns: Optional[List[str]] = None,
                     exclude_patterns: Optional[List[str]] = None,
                     min_size_mb: float = 0.1,
                     max_size_mb: float = 50000) -> List[ModelInfo]:
        """
        Check all available models
        """
        logger.info("Start scanning the model file...")
        
        if not self.models_root.exists():
            logger.warning(f"The model root directory does not exist: {self.models_root}")
            return []
        
        detected_models = []
        self._detected_hf_dirs.clear()  # Reset HuggingFace directory tracking
        
        # Step 1: Detect the HuggingFace format model
        hf_models = self._detect_all_huggingface_models()
        detected_models.extend(hf_models)
        
        # Step 2: Scan for other format models in each category directory
        for category_dir in self.models_root.iterdir():
            if not category_dir.is_dir():
                continue
                
            logger.debug(f"Scan directory: {category_dir}")
            category_models = self._scan_directory(
                category_dir, 
                include_patterns, 
                exclude_patterns,
                min_size_mb,
                max_size_mb
            )
            detected_models.extend(category_models)
        
        # Sort by confidence and file size
        detected_models.sort(key=lambda x: (x.confidence, x.size_mb), reverse=True)
        
        logger.info(f"Model scan completed - {len(detected_models)} models found")
        return detected_models
    
    def _detect_all_huggingface_models(self) -> List[ModelInfo]:
        """Detect all HuggingFace format models"""
        hf_models = []
        
        for root, dirs, files in os.walk(self.models_root):
            root_path = Path(root)
            
            # Skip subdirectories of HuggingFace directories that have already been detected
            if any(root_path.is_relative_to(hf_dir) and root_path != hf_dir 
                   for hf_dir in self._detected_hf_dirs):
                continue
            
            # Check if it is the HuggingFace model directory
            if self._is_huggingface_model_dir(root_path):
                hf_model = self._detect_huggingface_model(root_path)
                if hf_model:
                    hf_models.append(hf_model)
                    self._detected_hf_dirs.add(root_path)
                    logger.info(f"HuggingFace model detected: {hf_model.name} at {root_path}")
        
        return hf_models
    
    def _is_huggingface_model_dir(self, dir_path: Path) -> bool:
        """Determine whether the directory is a HuggingFace model directory"""
        # Check whether the identification file of the HuggingFace model is included
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
        """Scan the specified directory (skip the internal files of the HuggingFace model)"""
        models = []
        
        # Recursively scan all files
        for file_path in directory.rglob('*'):
            if not file_path.is_file():
                continue
            
            # Skip files in the HuggingFace model directory
            if self._is_inside_huggingface_model(file_path):
                continue
            
            # Skip files in the HuggingFace component subdirectory
            if self._is_huggingface_component_file(file_path):
                continue
            
            # Check the file extension
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            
            # Check file size
            try:
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb < min_size_mb or size_mb > max_size_mb:
                    continue
            except OSError:
                logger.warning(f"Unable to read file size: {file_path}")
                continue
            
            # Apply include/exclude patterns
            if include_patterns and not any(pattern in file_path.name for pattern in include_patterns):
                continue
            if exclude_patterns and any(pattern in file_path.name for pattern in exclude_patterns):
                continue
            
            # Detect model information
            model_info = self._analyze_model_file(file_path)
            if model_info:
                models.append(model_info)
        
        return models
    
    def _is_inside_huggingface_model(self, file_path: Path) -> bool:
        """Check if the file is in the detected HuggingFace model directory"""
        for hf_dir in self._detected_hf_dirs:
            try:
                file_path.relative_to(hf_dir)
                return True
            except ValueError:
                continue
        return False
    
    def _is_huggingface_component_file(self, file_path: Path) -> bool:
        """Check if the file is in the HuggingFace component subdirectory"""
        for parent in file_path.parents:
            if parent.name in self.HUGGINGFACE_COMPONENT_DIRS:
                # Further check whether the parent directory may be a HuggingFace model
                for ancestor in parent.parents:
                    if self._is_huggingface_model_dir(ancestor):
                        return True
        return False
    
    def _detect_huggingface_model(self, model_dir: Path) -> Optional[ModelInfo]:
        """Detecting HuggingFace format models"""
        try:
            # Prioritize checking the diffusers model (with model_index.json)
            model_index_file = model_dir / 'model_index.json'
            if model_index_file.exists():
                return self._detect_diffusers_model(model_dir, model_index_file)
            
            # Check transformers model (with config.json)
            config_file = model_dir / 'config.json'
            if config_file.exists():
                return self._detect_transformers_model(model_dir, config_file)
            
            # Check if it is just the directory containing the weights file
            if any((model_dir / f).exists() for f in ['pytorch_model.bin', 'model.safetensors']):
                return self._detect_generic_hf_model(model_dir)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to detect HuggingFace model {model_dir}: {e}")
            return None
    
    def _detect_diffusers_model(self, model_dir: Path, model_index_file: Path) -> Optional[ModelInfo]:
        """Detect Diffusers models (such as Stable Diffusion)"""
        try:
            with open(model_index_file, 'r', encoding='utf-8') as f:
                model_index = json.load(f)
            
            # Calculate the total size of a directory
            total_size = self._calculate_directory_size(model_dir)
            size_mb = total_size / (1024 * 1024)
            
            # Inference model type and architecture
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
                confidence=0.95,  # High confidence, because there is a clear model_index.json
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Failed to detect Diffusers model {model_dir}: {e}")
            return None
    
    def _detect_transformers_model(self, model_dir: Path, config_file: Path) -> Optional[ModelInfo]:
        """Detecting Transformers models (such as CLIP, BERT, etc.)"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            total_size = self._calculate_directory_size(model_dir)
            size_mb = total_size / (1024 * 1024)
            
            # Infer model type based on config
            model_type = 'multimodal'  # default
            framework = 'transformers'
            architecture = 'unknown'
            
            # Inferring concrete types based on schema information
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
            logger.warning(f"Detection of Transformers model failed {model_dir}: {e}")
            return None
    
    def _detect_generic_hf_model(self, model_dir: Path) -> Optional[ModelInfo]:
        """Detecting the generic HuggingFace model (weights file only)"""
        try:
            total_size = self._calculate_directory_size(model_dir)
            size_mb = total_size / (1024 * 1024)
            
            model_name = model_dir.name
            
            # Inferred based on directory name
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
                confidence=0.7,  # Medium confidence
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Detection of generic HuggingFace model fails {model_dir}: {e}")
            return None
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate the total size of a directory"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.debug(f"Failed to calculate directory size {directory}: {e}")
        return total_size
    
    def _analyze_model_file(self, file_path: Path) -> Optional[ModelInfo]:
        """Analyzing a single model file"""
        try:
            # Get basic information
            size_mb = file_path.stat().st_size / (1024 * 1024)
            file_format = self.SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), 'unknown')
            
            # Inference model types and frameworks
            model_type, framework, architecture, confidence = self._infer_model_properties(file_path)
            
            # Generate model name
            model_name = self._generate_model_name(file_path, architecture)
            
            # Collect metadata
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
            logger.warning(f"Failed to analyze the model file {file_path}: {e}")
            return None
    
    def _infer_model_properties(self, file_path: Path) -> Tuple[str, str, str, float]:
        """Inferring model properties"""
        filename = file_path.name.lower()
        parent_dirs = [p.name.lower() for p in file_path.parents]
        
        # Special handling: Prioritize checking for explicit model patterns
        # Exact Matching of the YOLO Model
        yolo_variants = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                        'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov10m', 'yolov10l', 'yolov10x',
                        'yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
        
        for variant in yolo_variants:
            if variant in filename:
                return 'detection', 'ultralytics', variant, 0.95
        
        # DeepLabV3 Detection - Exact Matching of Different Variants
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
        
        # Mask2Former Detection
        if 'mask2former' in filename:
            return 'segmentation', 'detectron2', 'mask2former', 0.9
        
        # U-Net Detection
        if 'unet' in filename or 'u_net' in filename:
            return 'segmentation', 'pytorch', 'unet', 0.9
        
        # Inferred based on path and file name
        for pattern_group, info in self.MODEL_PATTERNS.items():
            for pattern in info['patterns']:
                confidence = 0.0
                
                # File name matching
                if pattern in filename:
                    confidence += 0.6
                
                # Parent directory matching
                if any(pattern in parent_dir for parent_dir in parent_dirs):
                    confidence += 0.3
                    
                # Path structure matching - this is the most important basis for judgment
                if info['type'] in parent_dirs:
                    confidence += 0.4  # Increase the weight of the path structure
                
                if confidence > 0.5:  # Confidence threshold
                    return info['type'], info['framework'], pattern, confidence
        
        # Fallback inference based on directory structure (this is the most reliable approach)
        if 'segmentation' in parent_dirs:
            # In the segmentation directory, further determine the specific framework
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
                # General segmentation model
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
        
        # If there is no clear directory structure, use a more conservative inference
        return 'unknown', 'unknown', 'unknown', 0.1
    
    def _generate_model_name(self, file_path: Path, architecture: str) -> str:
        """Generate model name"""
        if architecture != 'unknown':
            # For the YOLO model, keep the full file name (minus the extension)
            if 'yolo' in architecture.lower():
                filename = file_path.stem
                # Keep the full name of yolov8n, yolov8s, etc.
                if any(variant in filename.lower() for variant in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                                                                   'yolov9c', 'yolov9e', 'yolov10n', 'yolov10s', 'yolov10m']):
                    return filename.lower()
            
            # For DeepLabV3, accurate identification of different variants
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
        
        # Use the file name (minus the extension)
        name = file_path.stem.lower()
        
        # Clean up common suffixes, but keep important identifiers
        suffixes_to_remove = ['-pruned', '-emaonly', '_pruned', '_ema', '_best', '_final', '_coco', '-586e9e4e']
        for suffix in suffixes_to_remove:
            name = name.replace(suffix, '')
        
        return name
    
    def _collect_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Collecting model metadata"""
        metadata = {
            'file_path': str(file_path),
            'parent_dir': str(file_path.parent),
            'modified_time': file_path.stat().st_mtime,
            'file_hash': self._calculate_file_hash(file_path),
        }
        
        # Attempt to read model-specific metadata
        if file_path.suffix.lower() == '.safetensors':
            metadata.update(self._read_safetensors_metadata(file_path))
        elif file_path.suffix.lower() in ['.pt', '.pth']:
            metadata.update(self._read_pytorch_metadata(file_path))
        
        return metadata
    
    def _calculate_file_hash(self, file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate the hash value of the file (only read the file header to increase speed)"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                # Only read the first 1MB to calculate the hash to avoid slow reading of large files
                chunk = f.read(1024 * 1024)
                hasher.update(chunk)
            return hasher.hexdigest()[:16]  # Return the first 16 bits
        except Exception as e:
            logger.warning(f"Failed to calculate file hash {file_path}: {e}")
            return "unknown"
    
    def _read_safetensors_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Read metadata of SafeTensors files"""
        metadata = {}
        try:
            # The SafeTensors file header contains JSON metadata
            with open(file_path, 'rb') as f:
                # Read the first 8 bytes to get the header length
                header_size = int.from_bytes(f.read(8), byteorder='little')
                if header_size < 1024 * 1024:  # Reasonable head size
                    header_bytes = f.read(header_size)
                    header_str = header_bytes.decode('utf-8')
                    header_data = json.loads(header_str)
                    
                    # Extract useful information
                    if '__metadata__' in header_data:
                        metadata.update(header_data['__metadata__'])
                    
                    # Number of statistical parameters
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
            logger.debug(f"Failed to read SafeTensors metadata {file_path}: {e}")
        
        return metadata
    
    def _read_pytorch_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Reading PyTorch model metadata"""
        metadata = {}
        try:
            # Only basic information is obtained here to avoid fully loading the model
            import torch
            
            # Try loading model checkpoint information
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            
            if isinstance(checkpoint, dict):
                # Common checkpoint keys
                if 'model' in checkpoint:
                    model_state = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    model_state = checkpoint['state_dict']
                else:
                    model_state = checkpoint
                
                # Number of statistical parameters
                if isinstance(model_state, dict):
                    total_params = sum(p.numel() for p in model_state.values() if hasattr(p, 'numel'))
                    metadata['total_parameters'] = total_params
                
                # Other metadata
                for key in ['epoch', 'version', 'arch', 'best_acc1']:
                    if key in checkpoint:
                        metadata[key] = checkpoint[key]
                        
        except Exception as e:
            logger.debug(f"Failed to read PyTorch metadata {file_path}: {e}")
        
        return metadata
    
    def generate_config(self, detected_models: List[ModelInfo], output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate a configuration file based on the detected model"""
        config = {
            "models_root": str(self.models_root),
            "models": {}
        }
        
        for model in detected_models:
            # Skip models with too low confidence
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
            
            # Add model type specific default configuration
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
        
        # Save the configuration file
        if output_file:
            import yaml
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Model configuration saved to: {output_file}")
        
        return config
    
    def get_summary(self, models: List[ModelInfo]) -> Dict[str, Any]:
        """Get a summary of test results"""
        summary = {
            "total_models": len(models),
            "by_type": {},
            "by_framework": {},
            "total_size_mb": sum(m.size_mb for m in models),
            "high_confidence": len([m for m in models if m.confidence > 0.7]),
            "medium_confidence": len([m for m in models if 0.3 <= m.confidence <= 0.7]),
            "low_confidence": len([m for m in models if m.confidence < 0.3])
        }
        
        # Statistics by type
        for model in models:
            model_type = model.type
            if model_type not in summary["by_type"]:
                summary["by_type"][model_type] = {"count": 0, "size_mb": 0}
            summary["by_type"][model_type]["count"] += 1
            summary["by_type"][model_type]["size_mb"] += model.size_mb
        
        # Statistics by framework
        for model in models:
            framework = model.framework
            if framework not in summary["by_framework"]:
                summary["by_framework"][framework] = {"count": 0, "size_mb": 0}
            summary["by_framework"][framework]["count"] += 1
            summary["by_framework"][framework]["size_mb"] += model.size_mb
        
        return summary


# Convenience functions
def detect_models_in_directory(directory: str, **kwargs) -> List[ModelInfo]:
    """Convenience functions: Detect models in the specified directory"""
    detector = ModelDetector(Path(directory))
    return detector.detect_models(**kwargs)


def generate_models_config(directory: str, output_file: str = None) -> Dict[str, Any]:
    """Convenience functions: Generate model configuration file"""
    detector = ModelDetector(Path(directory))
    models = detector.detect_models()
    output_path = Path(output_file) if output_file else None
    return detector.generate_config(models, output_path)