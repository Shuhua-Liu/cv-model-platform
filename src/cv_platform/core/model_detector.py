"""
Enhanced Model Detector - Inheriting from BaseManager

Model Detector with full BaseManager integration for state management,
health monitoring, metrics tracking, and automatic model discovery.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger

from .base_manager import BaseManager, ManagerState, HealthStatus, HealthCheckResult


@dataclass
class ModelInfo:
    """Model information data structure"""
    name: str
    path: Path
    type: str  # detection, segmentation, classification, generation, multimodal
    framework: str  # pytorch, tensorflow, onnx, etc.
    architecture: str  # yolov8n, resnet50, etc.
    confidence: float  # detection confidence (0.0-1.0)
    file_size_mb: float
    last_modified: float
    metadata: Dict[str, Any]


class ModelDetector(BaseManager):
    """Enhanced Model Detector inheriting from BaseManager"""
    
    def __init__(self, models_root: Optional[Path] = None):
        """Initialize the model detector with BaseManager capabilities"""
        super().__init__("ModelDetector")
        
        # Core configuration
        self.models_root = Path(models_root) if models_root else Path("./cv_models")
        
        # Detection patterns and rules
        self.MODEL_PATTERNS = {
            'yolo': {
                'patterns': ['yolo', 'yolov8', 'yolov9', 'yolov10', 'yolo11'],
                'type': 'detection',
                'framework': 'ultralytics'
            },
            'sam': {
                'patterns': ['sam_vit', 'mobile_sam', 'segment_anything'],
                'type': 'segmentation',
                'framework': 'segment_anything'
            },
            'stable_diffusion': {
                'patterns': ['stable-diffusion', 'sd_', 'sdxl', 'flux'],
                'type': 'generation',
                'framework': 'diffusers'
            },
            'clip': {
                'patterns': ['clip', 'vit-b-32', 'vit-l-14'],
                'type': 'multimodal',
                'framework': 'transformers'
            },
            'resnet': {
                'patterns': ['resnet'],
                'type': 'classification',
                'framework': 'torchvision'
            },
            'efficientnet': {
                'patterns': ['efficientnet'],
                'type': 'classification',
                'framework': 'torchvision'
            }
        }
        
        # Supported file extensions
        self.SUPPORTED_EXTENSIONS = {
            '.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.pkl',
            '.h5', '.pb', '.onnx', '.trt', '.engine', '.mlmodel'
        }
        
        # Detection state
        self._detected_models: List[ModelInfo] = []
        self._last_scan_time: Optional[float] = None
        self._scan_in_progress = False
        
        logger.info(f"ModelDetector initialized for path: {self.models_root}")
    
    def initialize(self) -> bool:
        """
        Initialize detector - implements BaseManager abstract method
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Validate models root directory
            if not self.models_root.exists():
                logger.warning(f"Models root directory does not exist: {self.models_root}")
                try:
                    self.models_root.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created models root directory: {self.models_root}")
                except Exception as e:
                    logger.error(f"Failed to create models root directory: {e}")
                    return False
            
            # Set initial metrics
            self.update_metric('models_root', str(self.models_root))
            self.update_metric('supported_extensions', len(self.SUPPORTED_EXTENSIONS))
            self.update_metric('pattern_groups', len(self.MODEL_PATTERNS))
            self.update_metric('initialization_time', time.time())
            
            # Perform initial scan
            detected_count = self._perform_model_scan()
            self.update_metric('initial_scan_models', detected_count)
            
            logger.info(f"ModelDetector initialization completed - found {detected_count} models")
            return True
            
        except Exception as e:
            logger.error(f"ModelDetector initialization failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """
        Cleanup detector resources - implements BaseManager abstract method
        """
        try:
            # Clear detected models
            self._detected_models.clear()
            
            # Update final metrics
            self.update_metric('cleanup_time', time.time())
            
            logger.info("ModelDetector cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during ModelDetector cleanup: {e}")
    
    def perform_health_check(self) -> HealthCheckResult:
        """
        Perform comprehensive health check
        
        Returns:
            Health check result with detailed status
        """
        start_time = time.time()
        
        try:
            # Check basic state
            if self.state not in [ManagerState.RUNNING, ManagerState.READY]:
                return HealthCheckResult(
                    status=HealthStatus.CRITICAL,
                    message=f"ModelDetector not running (state: {self.state.value})",
                    details={'state': self.state.value},
                    timestamp=time.time(),
                    check_duration=time.time() - start_time
                )
            
            # Check models root accessibility
            models_root_accessible = self.models_root.exists() and os.access(self.models_root, os.R_OK)
            
            # Check detected models
            models_count = len(self._detected_models)
            last_scan_age = time.time() - self._last_scan_time if self._last_scan_time else float('inf')
            
            # Determine health status
            if not models_root_accessible:
                status = HealthStatus.CRITICAL
                message = f"Models root directory not accessible: {self.models_root}"
            elif self._scan_in_progress:
                status = HealthStatus.WARNING
                message = "Model scan in progress"
            elif last_scan_age > 3600:  # 1 hour
                status = HealthStatus.WARNING
                message = f"Last scan was {last_scan_age/3600:.1f} hours ago"
            elif models_count == 0:
                status = HealthStatus.WARNING
                message = "No models detected"
            else:
                status = HealthStatus.HEALTHY
                message = f"{models_count} models detected, last scan {last_scan_age/60:.1f} minutes ago"
            
            details = {
                'models_root_accessible': models_root_accessible,
                'models_detected': models_count,
                'last_scan_time': self._last_scan_time,
                'last_scan_age_minutes': last_scan_age / 60,
                'scan_in_progress': self._scan_in_progress,
                'models_root_path': str(self.models_root)
            }
            
            return HealthCheckResult(
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                check_duration=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {e}",
                details={'error': str(e)},
                timestamp=time.time(),
                check_duration=time.time() - start_time
            )
    
    def detect_models(self, force_rescan: bool = False) -> List[ModelInfo]:
        """
        Detect available models with caching and metrics
        
        Args:
            force_rescan: Force a new scan even if recent data exists
            
        Returns:
            List of detected model information
        """
        scan_start_time = time.time()
        
        # Check if we need to rescan
        if not force_rescan and self._detected_models and self._last_scan_time:
            scan_age = time.time() - self._last_scan_time
            if scan_age < 300:  # 5 minutes cache
                logger.debug(f"Using cached model detection results (age: {scan_age:.1f}s)")
                self.increment_metric('cache_hits')
                return self._detected_models.copy()
        
        # Perform new scan
        self.increment_metric('scans_performed')
        detected_count = self._perform_model_scan()
        
        scan_duration = time.time() - scan_start_time
        self.update_metric('last_scan_duration', scan_duration)
        self.update_metric('last_scan_models_found', detected_count)
        
        logger.info(f"Model detection completed in {scan_duration:.2f}s - found {detected_count} models")
        
        return self._detected_models.copy()
    
    def _perform_model_scan(self) -> int:
        """
        Perform the actual model scanning
        
        Returns:
            Number of models detected
        """
        if self._scan_in_progress:
            logger.warning("Model scan already in progress")
            return len(self._detected_models)
        
        self._scan_in_progress = True
        scan_start_time = time.time()
        
        try:
            logger.info(f"Starting model scan in: {self.models_root}")
            
            detected_models = []
            scanned_files = 0
            
            # Walk through directory tree
            for root_path in self._walk_model_directories():
                for file_path in root_path.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        scanned_files += 1
                        
                        try:
                            model_info = self._analyze_model_file(file_path)
                            if model_info and model_info.confidence > 0.5:  # Confidence threshold
                                detected_models.append(model_info)
                                logger.debug(f"Detected model: {model_info.name} ({model_info.type}/{model_info.framework})")
                        
                        except Exception as e:
                            logger.warning(f"Failed to analyze model file {file_path}: {e}")
            
            # Update state
            self._detected_models = detected_models
            self._last_scan_time = time.time()
            
            # Update metrics
            self.update_metric('files_scanned', scanned_files)
            self.update_metric('models_detected', len(detected_models))
            self.update_metric('scan_success_rate', len(detected_models) / max(scanned_files, 1))
            
            logger.info(f"Scan completed: {scanned_files} files scanned, {len(detected_models)} models detected")
            
            return len(detected_models)
            
        finally:
            self._scan_in_progress = False
    
    def _walk_model_directories(self) -> List[Path]:
        """
        Get list of directories to scan for models
        
        Returns:
            List of directory paths to scan
        """
        directories_to_scan = [self.models_root]
        
        # Add common subdirectories if they exist
        common_subdirs = [
            'detection', 'segmentation', 'classification', 'generation', 'multimodal',
            'yolo', 'sam', 'stable_diffusion', 'clip', 'resnet', 'efficientnet'
        ]
        
        for subdir in common_subdirs:
            subdir_path = self.models_root / subdir
            if subdir_path.exists() and subdir_path.is_dir():
                directories_to_scan.append(subdir_path)
        
        return directories_to_scan
    
    def _analyze_model_file(self, file_path: Path) -> Optional[ModelInfo]:
        """
        Analyze a model file and extract information
        
        Args:
            file_path: Path to the model file
            
        Returns:
            ModelInfo if analysis successful, None otherwise
        """
        try:
            # Get basic file information
            stat = file_path.stat()
            file_size_mb = stat.st_size / (1024 * 1024)
            
            # Generate model name from file path
            model_name = self._generate_model_name(file_path)
            
            # Detect model type, framework, and architecture
            model_type, framework, architecture, confidence = self._detect_model_characteristics(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(file_path)
            
            return ModelInfo(
                name=model_name,
                path=file_path,
                type=model_type,
                framework=framework,
                architecture=architecture,
                confidence=confidence,
                file_size_mb=file_size_mb,
                last_modified=stat.st_mtime,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Failed to analyze model file {file_path}: {e}")
            return None
    
    def _generate_model_name(self, file_path: Path) -> str:
        """
        Generate a unique model name from file path
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Generated model name
        """
        # Use relative path from models root
        try:
            relative_path = file_path.relative_to(self.models_root)
            # Remove file extension and use path structure
            name_parts = list(relative_path.parent.parts) + [relative_path.stem]
            # Filter out common directory names
            filtered_parts = [part for part in name_parts if part not in ['.', 'models', 'weights', 'checkpoints']]
            
            if filtered_parts:
                return '_'.join(filtered_parts)
            else:
                return file_path.stem
                
        except ValueError:
            # File is not under models_root
            return file_path.stem
    
    def _detect_model_characteristics(self, file_path: Path) -> Tuple[str, str, str, float]:
        """
        Detect model type, framework, and architecture from file path and name
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Tuple of (type, framework, architecture, confidence)
        """
        filename = file_path.name.lower()
        parent_dirs = [p.lower() for p in file_path.parent.parts]
        
        # Specific pattern matching with high confidence
        
        # YOLO Detection
        if any(pattern in filename for pattern in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']):
            arch = next((pattern for pattern in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'] 
                        if pattern in filename), 'yolov8')
            return 'detection', 'ultralytics', arch, 0.95
        
        if any(pattern in filename for pattern in ['yolov9', 'yolov10', 'yolo11']):
            return 'detection', 'ultralytics', 'yolo', 0.9
        
        # SAM Segmentation
        if 'sam_vit_b' in filename:
            return 'segmentation', 'segment_anything', 'sam_vit_b', 0.95
        elif 'sam_vit_l' in filename:
            return 'segmentation', 'segment_anything', 'sam_vit_l', 0.95
        elif 'sam_vit_h' in filename:
            return 'segmentation', 'segment_anything', 'sam_vit_h', 0.95
        elif 'mobile_sam' in filename:
            return 'segmentation', 'segment_anything', 'mobile_sam', 0.9
        
        # Stable Diffusion
        if any(pattern in filename for pattern in ['stable-diffusion', 'sd_1_5', 'sdxl']):
            if 'sdxl' in filename:
                return 'generation', 'diffusers', 'sdxl', 0.9
            else:
                return 'generation', 'diffusers', 'stable_diffusion', 0.9
        
        # CLIP
        if any(pattern in filename for pattern in ['clip', 'vit-b-32', 'vit-l-14']):
            arch = 'vit-b-32' if 'vit-b-32' in filename else ('vit-l-14' if 'vit-l-14' in filename else 'clip')
            return 'multimodal', 'transformers', arch, 0.9
        
        # ResNet Classification
        if 'resnet' in filename:
            if any(variant in filename for variant in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']):
                arch = next(variant for variant in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] 
                           if variant in filename)
                return 'classification', 'torchvision', arch, 0.9
            return 'classification', 'torchvision', 'resnet', 0.8
        
        # EfficientNet Classification
        if 'efficientnet' in filename:
            return 'classification', 'torchvision', 'efficientnet', 0.9
        
        # DeepLab Segmentation
        if 'deeplabv3' in filename:
            if 'resnet101' in filename:
                return 'segmentation', 'torchvision', 'deeplabv3_resnet101', 0.9
            elif 'resnet50' in filename:
                return 'segmentation', 'torchvision', 'deeplabv3_resnet50', 0.9
            elif 'mobilenet' in filename:
                return 'segmentation', 'torchvision', 'deeplabv3_mobilenet_v3_large', 0.9
            else:
                return 'segmentation', 'torchvision', 'deeplabv3', 0.8
        
        # Mask2Former Detection
        if 'mask2former' in filename:
            return 'segmentation', 'detectron2', 'mask2former', 0.9
        
        # U-Net Detection
        if 'unet' in filename or 'u_net' in filename:
            return 'segmentation', 'pytorch', 'unet', 0.8
        
        # Fallback inference based on directory structure
        if 'segmentation' in parent_dirs:
            return 'segmentation', 'unknown', 'unknown', 0.6
        elif 'detection' in parent_dirs:
            return 'detection', 'unknown', 'unknown', 0.6
        elif 'classification' in parent_dirs:
            return 'classification', 'unknown', 'unknown', 0.6
        elif 'generation' in parent_dirs:
            return 'generation', 'unknown', 'unknown', 0.6
        elif 'multimodal' in parent_dirs:
            return 'multimodal', 'unknown', 'unknown', 0.6
        
        # General pattern matching with lower confidence
        for pattern_group, info in self.MODEL_PATTERNS.items():
            for pattern in info['patterns']:
                confidence = 0.0
                
                # File name matching
                if pattern in filename:
                    confidence += 0.6
                
                # Parent directory matching
                if any(pattern in parent_dir for parent_dir in parent_dirs):
                    confidence += 0.3
                
                # Path structure matching
                if info['type'] in parent_dirs:
                    confidence += 0.4
                
                if confidence > 0.5:
                    return info['type'], info['framework'], pattern, confidence
        
        # Default fallback
        return 'unknown', 'unknown', 'unknown', 0.1
    
    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract additional metadata from model file
        
        Args:
            file_path: Path to the model file
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            'file_extension': file_path.suffix,
            'file_name': file_path.name,
            'relative_path': str(file_path.relative_to(self.models_root)) if self.models_root in file_path.parents else str(file_path)
        }
        
        # Try to extract more detailed metadata based on file type
        try:
            if file_path.suffix.lower() in ['.pt', '.pth']:
                # PyTorch model - could inspect with torch.load if needed
                metadata['framework_hint'] = 'pytorch'
            elif file_path.suffix.lower() == '.onnx':
                metadata['framework_hint'] = 'onnx'
            elif file_path.suffix.lower() == '.h5':
                metadata['framework_hint'] = 'tensorflow/keras'
            elif file_path.suffix.lower() == '.safetensors':
                metadata['framework_hint'] = 'safetensors'
                
        except Exception as e:
            logger.debug(f"Failed to extract detailed metadata from {file_path}: {e}")
        
        return metadata
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get information about a specific detected model
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelInfo if found, None otherwise
        """
        for model_info in self._detected_models:
            if model_info.name == model_name:
                return model_info
        return None
    
    def get_models_by_type(self, model_type: str) -> List[ModelInfo]:
        """
        Get all detected models of a specific type
        
        Args:
            model_type: Type of models to retrieve
            
        Returns:
            List of ModelInfo objects
        """
        return [model for model in self._detected_models if model.type == model_type]
    
    def get_models_by_framework(self, framework: str) -> List[ModelInfo]:
        """
        Get all detected models using a specific framework
        
        Args:
            framework: Framework name
            
        Returns:
            List of ModelInfo objects
        """
        return [model for model in self._detected_models if model.framework == framework]
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive detection summary
        
        Returns:
            Dictionary with detection statistics
        """
        if not self._detected_models:
            return {
                'total_models': 0,
                'last_scan_time': self._last_scan_time,
                'scan_in_progress': self._scan_in_progress
            }
        
        # Calculate statistics
        types_count = {}
        frameworks_count = {}
        total_size_mb = 0
        
        for model in self._detected_models:
            types_count[model.type] = types_count.get(model.type, 0) + 1
            frameworks_count[model.framework] = frameworks_count.get(model.framework, 0) + 1
            total_size_mb += model.file_size_mb
        
        return {
            'total_models': len(self._detected_models),
            'models_by_type': types_count,
            'models_by_framework': frameworks_count,
            'total_size_mb': total_size_mb,
            'average_size_mb': total_size_mb / len(self._detected_models),
            'last_scan_time': self._last_scan_time,
            'scan_in_progress': self._scan_in_progress,
            'models_root': str(self.models_root)
        }
    
    def export_detection_results(self, format: str = 'dict') -> Any:
        """
        Export detection results in various formats
        
        Args:
            format: Export format ('dict', 'json', 'csv')
            
        Returns:
            Detection results in requested format
        """
        if format == 'dict':
            return {
                'summary': self.get_detection_summary(),
                'models': [
                    {
                        'name': model.name,
                        'path': str(model.path),
                        'type': model.type,
                        'framework': model.framework,
                        'architecture': model.architecture,
                        'confidence': model.confidence,
                        'file_size_mb': model.file_size_mb,
                        'last_modified': model.last_modified,
                        'metadata': model.metadata
                    }
                    for model in self._detected_models
                ]
            }
        
        elif format == 'json':
            import json
            data = self.export_detection_results('dict')
            return json.dumps(data, indent=2, default=str)
        
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                'name', 'path', 'type', 'framework', 'architecture', 
                'confidence', 'file_size_mb', 'last_modified'
            ])
            
            # Write model data
            for model in self._detected_models:
                writer.writerow([
                    model.name, str(model.path), model.type, model.framework,
                    model.architecture, model.confidence, model.file_size_mb,
                    model.last_modified
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def watch_directory(self, interval: int = 300) -> None:
        """
        Start watching directory for changes (placeholder for future implementation)
        
        Args:
            interval: Check interval in seconds
        """
        # This could be implemented with file system watchers
        # For now, just log the intention
        logger.info(f"Directory watching not yet implemented (would check every {interval}s)")
        self.update_metric('watch_interval', interval)


# Enhanced global function
def get_model_detector(models_root: Optional[Path] = None) -> ModelDetector:
    """
    Get a model detector instance
    
    Args:
        models_root: Root directory for model scanning
        
    Returns:
        ModelDetector instance
    """
    detector = ModelDetector(models_root)
    
    # Auto-start the detector
    if not detector.start():
        logger.error("Failed to start ModelDetector")
    
    return detector