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

        self._cached_models = None
        self._last_scan_time = 0
        self.cache_deruation = 300
        self.MIN_FILE_SIZE = 1024 * 1024
        
        # Core configuration
        self.models_root = Path(models_root) if models_root else Path("./cv_models")
        
        # Detection patterns and rules
        self.MODEL_PATTERNS = {
            'yolo': {
                'patterns': ['yolo', 'yolov8', 'yolov9', 'yolov10', 'yolo11'],
                'type': 'detection',
                'framework': 'ultralytics'
            },
            'detr': {
                'patterns': ['detr', 'detr-resnet-101'],
                'type': 'detection',
                'framework': 'huggingface'
            },
            'sam': {
                'patterns': ['sam_vit', 'mobile_sam', 'segment_anything'],
                'type': 'segmentation',
                'framework': 'segment_anything'
            },
            'controlnet':{
                'patterns': ['controlnet-canny', 'controlnet-seg', 'controlnet-depth', 'controlnet-pose'],
                'type': 'generation',
                'framwork': 'diffusers'
            },
            'stable_diffusion': {
                'patterns': ['stable_diffusion', 'sd_2_1', 'sd_2_1_unclip', 'sdxl'],
                'type': 'generation',
                'framework': 'diffusers'
            },
            'flux': {
                'patterns': ['FLUX.1-schnell', 'flux'],
                'type': 'generation',
                'framework': 'diffusers'
            },
            'clip': {
                'patterns': ['clip', 'vit-b-32', 'vit-h-14'],
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
            },
            'vit': {
                'patterns': ['vit', 'vit-base-patch16-224'],
                'type': 'classification',
                'framework': 'transformers'
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
        if self._cached_models and not force_rescan:
            logger.debug(f"Using cached model detection results ")
            return self._cached_models
        
        logger.info("Starting enhanced model detection...")
        start_time = time.time()

        all_models =[]
        processed_directories = set()

        # Phase 1: Detect HuggingFace model directories
        logger.info("Phase 1: Scanning for HuggingFace model directories...")
        hf_models = self._detect_huggingface_directories()

        for hf_dir, hf_info in hf_models.items():
            processed_directories.add(hf_dir)

            # Create directory-level model entry
            model_info = self._create_directory_model(hf_dir, hf_info)
            if model_info:
                all_models.append(model_info)
                logger.debug(f"HF Directory model: {model_info.name}")
        
        logger.info (f"Found {len(hf_models)} HuggingFace model directories")

        # Phase 2: Scan remaining individual files
        logger.info('Phase 2: Scanning individual model files...')
        directories_to_scan = self._walk_model_directories()
        individual_count = 0

        for directory in directories_to_scan:
            for file_path in directory.rglob('*'):
                # Skip if file is inside a processed HF directory
                if self._is_inside_processed_directory(file_path, processed_directories):
                    continue
                # Check if it's a model file
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS and 
                    file_path.stat().st_size > self.MIN_FILE_SIZE):
                    model_info = self._analyze_model_file(file_path)
                    if model_info and model_info.confidence >= 0.5:
                        all_models.append(model_info)
                        individual_count += 1
                        logger.debug(f"Individual model: {model_info.name}")
        
        logger.info(f"Found {individual_count} individual model files")

        # Phase 3: Final cleanup and dedulication
        final_models = self._deduplicate_model(all_models)

        detection_time = time.time() - start_time
        logger.info(f"Model detection completed in {detection_time:.2f}s")
        logger.info(f"Total models found: {len(final_models)} (from {len(all_models)} cadidates)")

        # Update cache
        self._cached_models = final_models
        self._last_scan_time = time.time()

        return final_models

        # scan_start_time = time.time()

        # # Check if we need to rescan
        # if not force_rescan and self._detected_models and self._last_scan_time:
        #     scan_age = time.time() - self._last_scan_time
        #     if scan_age < 300:  # 5 minutes cache
        #         logger.debug(f"Using cached model detection results (age: {scan_age:.1f}s)")
        #         self.increment_metric('cache_hits')
        #         return self._detected_models.copy()
        
        # # Perform new scan
        # self.increment_metric('scans_performed')
        # detected_count = self._perform_model_scan()
        
        # scan_duration = time.time() - scan_start_time
        # self.update_metric('last_scan_duration', scan_duration)
        # self.update_metric('last_scan_models_found', detected_count)
        
        # logger.info(f"Model detection completed in {scan_duration:.2f}s - found {detected_count} models")
        
        # return self._detected_models.copy()

    def _detect_huggingface_directories(self) -> Dict[Path, Dict[str, Any]]:
        """
        Detect HuggingFace model directories

        Returns:
        Dict mapping directory paths to their metadata
        """
        hf_directories = {}

        # HuggingFace indicators
        hf_indicators = {
            'config_files': ['config.json', 'model_index.json', 'pytorch_model.bin.index.json'],
            'component_dirs': ['unet', 'vae', 'text_encoder', 'text_encoder_2', 'transformer', 'tokenizer', 'scheduler', 'safety_checker'],
            'model_files': ['pytorch_model.bin', 'model.safetensors', 'diffusion_pytorch_model.safetensors']
        }

        directories_to_scan = self._walk_model_directories()

        for scan_dir in directories_to_scan:
            for potential_hf_dir in scan_dir.rglob('*'):
                if not potential_hf_dir.is_dir():
                    continue

                # Skip if already processed as parent
                if any(potential_hf_dir.is_relative_to(existing) for existing in hf_directories.keys()):
                    continue

                # Check HuggingFace indicators
                hf_score = self._calculate_hf_score(potential_hf_dir, hf_indicators)
                if hf_score >= 0.6: # Threshold for HF directory detection
                    hf_info = self._analyze_hf_directory(potential_hf_dir, hf_indicators)
                    hf_directories[potential_hf_dir] = hf_info
                    logger.debug(f"HF Directory detected: {potential_hf_dir} (score: {hf_score:.2f})")

        return hf_directories
    
    def _calculate_hf_score(self, directory: Path, indicators: Dict[str, List[str]]) -> float:
        """Calculate HuggingFace detection score for a directory"""
        score = 0.0

        # Check or config files (high weight)
        config_files = [f for f in directory.iterdir() if f.is_file()]
        config_count = sum(1 for f in config_files if f.name in indicators['config_files'])
        if config_count > 0:
            score += 0.4

        # Check for component directories (medium weight)
        subdirs = [d for d in directory.iterdir() if d.is_dir()]
        component_count = sum(1 for d in subdirs if d.name in indicators['component_dirs'])
        if component_count >= 2:
            score += 0.3
        elif component_count == 1:
            score += 0.15

        # Check for model files (medium weight)
        model_files = [f for f in directory.rglob('*.safetensors') if f.is_file()]
        model_files.extend([f for f in directory.rglob('*.bin') if f.is_file()])

        if len(model_files) > 1:
            score += 0.3
        elif len(model_files) == 1:
            score += 0.1

        # Check naming patterns (low weight)
        dir_name = directory.name.lower()
        model_patterns = ['sd_', 'stable', 'flux', 'controlnet', 'clip', 'vit']
        if any(pattern in dir_name for pattern in model_patterns):
            score += 0.1
        
        return min(score, 1.0)
    
    def _create_directory_model(self, hf_dir: Path, hf_info: Dict[str, Any]) -> Optional[ModelInfo]:
        """Create a ModelInfo for a HuggingFace directory"""
        try:
            # Calculate total directory size
            total_size = sum(f.stat().st_size for f in hf_dir.rglob('*') if f.is_file())
            total_size_mb = total_size / (1024 * 1024)

            # Get latest modification time
            latest_mtime = max(
                (f.stat().st_mtime for f in hf_dir.rglob('*') if f.is_file()),
                default=hf_dir.stat().st_mtime
            )

            # Determine model characteristics
            model_type, framework, architecture = self._analyze_hf_directory_type(hf_dir)

            # Generate model name from directory
            model_name = self._generate_model_name(hf_dir)

            # Create metadata
            metadata = {
                'is_huggingface_directory': True,
                'component_count': len(list(hf_dir.iterdir())),
                'config_files': hf_info.get('config_files', []),
                'model_files_count': len(list(hf_dir.rglob('*.safetensor')) + list(hf_dir.rglob('*.bin'))),
                'directory_path': str(hf_dir)                                       
            }

            return ModelInfo(
                name=model_name,
                path=hf_dir, # Point to directory, not individual file
                type=model_type,
                framework=framework,
                architecture=architecture,
                confidence=0.95, # High confidence for HF directories
                file_size_mb=total_size_mb,
                last_modified=latest_mtime,
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Failed to create directory model for {hf_dir}: {e}")
        return None

    def _analyze_hf_directory(self, directory: Path, indicators: Dict[str, List[str]]) -> Dict[str, Any]:
        """Minimal HF directory analysis"""
        return {
            'config_files': [f.name for f in directory.iterdir() if f.is_file() and f.name in indicators.get('config_files', [])],
            'component_dirs': [d.name for d in directory.iterdir() if d.is_dir()],
            'model_files': [f.name for f in directory.rglob('*.safetensors')] + [f.name for f in directory.rglob('*.bin')],
            'total_size_mb': sum(f.stat().st_size for f in directory.rglob('*') if f.is_file()) / (1024 * 1024),
            'file_count': len(list(directory.rglob('*')))
        }
    
    def _analyze_hf_directory_type(self, hf_dir: Path) -> Tuple[str, str, str]:
        """Analyze HuggingFace directory to determine model type"""
        dir_name = hf_dir.name.lower()
        dir_path = str(hf_dir).lower()

        # Generation models
        if any(pattern in dir_name for pattern in ['sd_2_1', 'stable-diffusion-2-1']):
            if 'unclip' in dir_name:
                return 'generation', 'diffusers', 'stable_diffusion_2_1_unclip'
            else:
                return 'generation', 'diffusers', 'stable_diffusion_2_1'
        
        elif any(pattern in dir_name for pattern in ['sd_1_5', 'stable-diffusion-v1-5']):
            return 'generation', 'diffusers', 'stable_diffusion_1_5'
        
        elif any(pattern in dir_name for pattern in ['sdxl', 'stable-diffusion-xl']):
            return 'generation', 'diffusers', 'stable_diffusion_xl'
        
        elif 'flux' in dir_name:
            if 'schnell' in dir_name:
                return 'generation', 'diffusers', 'flux_schnell'
            elif 'dev' in dir_name:
                return 'generation', 'diffusers', 'flux_dev'
            else:
                return 'generation', 'diffusers', 'flux'
            
        elif 'controlnet' in dir_name:
            if 'canny' in dir_name:
                return 'generation', 'diffusers', 'controlnet_canny'
            elif 'depth' in dir_name:
                return 'generation', 'diffusers', 'controlnet_depth'
            else:
                return 'generation', 'diffusers', 'controlnet'
        
        # Inpainting models
        elif any(pattern in dir_name for pattern in ['inpainting', 'inpaint']):
            if any(pattern in dir_name for pattern in ['stable-diffusion', 'stable_diffusion', 'sd_']):
                if '2' in dir_name:
                    return 'inpainting', 'diffusers', 'stable_diffusion_2_inpainting'
                else:
                    return 'inpainting', 'diffusers', 'stable_diffusion_inpainting'
            else:
                return 'inpainting', 'diffusers', 'inpainting'
        
        # Multimodal models
        elif any(pattern in dir_name for pattern in ['clip', 'open_clip', 'vit']):
            return 'multimodal', 'transformers', 'clip'
        
        # Default fallback
        if 'generation' in dir_path:
            return 'generation', 'diffusers', 'unknown'
        elif 'multimodal' in dir_path:
            return 'multimodal', 'transformers', 'unknown'
        else:
            return 'unknown', 'unknown', 'unknown' 
        
    def _is_inside_processed_directory(self, file_path: Path, processed_dirs: set) -> bool:
        """Check if a file is inside any processed HuggingFace directory"""
        for processed_dir in processed_dirs:
            try:
                file_path.relative_to(processed_dir)
                return True # File is inside this processed directory 
            except ValueError:
                continue # File is not inside this directory
        return False

    def _deduplicate_model(slef, models: List[ModelInfo]) -> List[ModelInfo]:
        """Remove duplicaet models and prefer directory-level models"""
        seen_names = {}
        unique_models = []

        # Sort by preference: HF directories first, then individual files
        sorted_models = sorted(models, key=lambda m: (
            not m.metadata.get('is_huggingface_directory', False), # HF dirs first
            m.name.lower() # Then alphabetically
        ))

        for model in sorted_models:
            key = model.name.lower()

            if key not in seen_names:
                seen_names[key] = model
                unique_models.append(model)
            else:
                # Prefer HF directory over individual files
                existing = seen_names[key]
                if (model.metadata.get('is_huggingface_directory', False) and not existing.metadata.get('is_huggingface_directory', False)):
                    # Replace individual file with HF directory
                    idx = unique_models.index(existing)
                    unique_models[idx] = model
                    seen_names[key] = model
        return unique_models

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
        full_path_lower = str(file_path).lower()
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
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
        

        # OpenCLIP
        openclip_patterns = ['open-clip', 'openclip', 'open_clip']
        if any(pattern in full_path_lower for pattern in openclip_patterns):
            if 'vit-b-32' in full_path_lower:
                return 'multimodal', 'open_clip', 'openclip_vit_b_32', 0.95
            elif 'vit-h-14' in full_path_lower:
                return 'multimodal', 'open_clip', 'openclip_vit_h_14', 0.95
            elif 'vit-l-14' in full_path_lower:
                return 'multimodal', 'open_clip', 'openclip_vit_l_14', 0.95
            else:
                return 'multimodal', 'open_clip', 'openclip', 0.90
        
        # CLIP
        clip_patterns = ['/clip/', '\\\\clip\\\\', 'clip-vit', 'clip_vit']
        if any(pattern in full_path_lower for pattern in clip_patterns):
            if not any(pattern in full_path_lower for pattern in openclip_patterns):
                if 'vit-b-32' in full_path_lower:
                    return 'multimodal', 'transformers', 'clip_vit_base_patch32', 0.95
                elif 'vit-l-14' in full_path_lower:
                    return 'multimodal', 'transformers', 'clip_vit_large_patch14', 0.95
                else:
                    return 'multimodal', 'transformers', 'clip', 0.90      
        
        # Multimodal directory context (fallback)
        if 'multimodal' in full_path_lower and 'vit' in full_path_lower:
            if any(pattern in full_path_lower for pattern in openclip_patterns):
                return 'multimodal', 'open_clip', 'openclip', 0.80
            else:
                return 'multimodal', 'transformers', 'clip', 0.80
        
        # ControlNet
        if 'generation' in parent_dirs and 'controlnet' in parent_dirs:
            controlnet_model_dir = None
            for parent_dir in parent_dirs:
                if 'controlnet' in parent_dir or ('controlnet' in parent_dir and parent_dir != 'controlnet'):
                    controlnet_model_dir = parent_dir
                    break
            
            if controlnet_model_dir:
                if 'canny' in controlnet_model_dir:
                    return 'generation', 'diffusers', 'controlnet_canny', 0.95
                elif 'depth' in controlnet_model_dir:
                    return 'generation', 'diffusers', 'controlnet_depth', 0.95
                elif 'seg' in controlnet_model_dir:
                    return 'generation', 'diffusers', 'controlnet_seg', 0.95
                elif 'pose' in controlnet_model_dir:
                    return 'generation', 'diffusers', 'controlnet_pose', 0.95
                elif 'inpaint' in controlnet_model_dir:
                    return 'generation', 'diffusers', 'controlnet_inpaint', 0.95
                else:
                    return 'generation' , 'diffusers', 'controlnet', 0.9
            else:
                return 'generation' , 'diffusers', 'controlnet', 0.85

        # INPAINTING MODELS (HIGH PRIORITY - before general SD)
        if 'inpainting' in full_path_lower:
            # SD Inpainting models
            if any(pattern in full_path_lower for pattern in ['stable-diffusion', 'stable_diffusion', 'sd_']):
                if any(pattern in full_path_lower for pattern in ['2-inpainting', '2_inpainting', 'inpainting']):
                    return 'inpainting', 'diffusers', 'stable_diffusion_2_inpainting', 0.95
                else:
                    return 'inpainting', 'diffusers', 'stable_diffusion_inpainting', 0.90
            
            # LaMa models
            elif 'lama' in full_path_lower:
                return 'inpainting', 'pytorch', 'lama', 0.95
            
            # Generic inpainting
            else:
                return 'inpainting', 'pytorch', 'inpainting', 0.70

        # Stable Diffusion
        if 'generation' in parent_dirs and 'stable_diffusion' in parent_dirs:
            sd_version_dir = None
            for parent_dir in parent_dirs:
                if any(version in parent_dir for version in ['sd_1_5', 'sd_2_1', 'sd_2_0', 'sdxl']):
                    sd_version_dir = parent_dir
                    break
            
            if sd_version_dir:
                if 'sd_2_1_unclip' in sd_version_dir:
                    return 'generation', 'diffusers', 'stable-diffusion_2_1_unclip', 0.95 
                elif 'sd_2_1' in sd_version_dir:
                    return 'generation', 'diffusers', 'stable-diffusion_2_1', 0.95 
                elif 'sd_1_5' in sd_version_dir:
                    return 'generation', 'diffusers', 'stable-diffusion_1_5', 0.95 
                elif 'sd_2_0' in sd_version_dir:
                    return 'generation', 'diffusers', 'stable-diffusion_2_0', 0.95 
                elif 'sdxl' in sd_version_dir:
                    return 'generation', 'diffusers', 'stable-diffusion_xl', 0.95 
            else:
                return 'generation', 'diffusers', 'stable_diffusion', 0.8
        
        # Other models in generation dir  
        if 'generation' in parent_dirs:
            if any(pattern in full_path_lower for pattern in ['flux', 'dalle']):
                return 'generation', 'diffusers', 'text_to_image', 0.8
            elif any(pattern in full_path_lower for pattern in ['stable-diffusion', 'stable_diffusion']):
                return 'generation', 'diffusers', 'stable_diffusion', 0.7
            elif any(pattern in filename for pattern in ['2.1', '2_1', 'sd-2-1', 'sd_2_1', 'v2.1']):
                if 'unclip' not in filename:
                    return 'generation', 'diffusers', 'stable_diffusion_2_1', 0.95
                else:
                    return 'generation', 'diffusers', 'stable_diffusion_2_1_unclip', 0.95
            elif any(pattern in filename for pattern in ['v1.5', '1.5', '1_5']): 
                return 'generation', 'diffusers', 'stable_diffusion_1_5', 0.9
            else:
                if any(keyword in filename for keyword in ['768-nonema-pruned', '768-ema-pruned']):
                    if 'unclip' in filename:
                        return 'generation', 'diffusers', 'stable_diffusion_2_1_unclip', 0.9
                    else:
                        return 'generation', 'diffusers', 'stable_diffusion_2_1', 0.9
                return 'generation', 'diffusers', 'stable_diffusion', 0.7
        
        elif any(pattern in filename for pattern in ['sd_2_1', 'sd-2-1']):
            if 'unclip' in filename:
                return 'generation', 'diffusers', 'stable_diffusion_2_1_unclip', 0.95
            else:
                return 'generation', 'diffusers', 'stable_diffusion_2_1', 0.95
        elif any(pattern in filename for pattern in ['sd_1_5', 'sd-1-5']):
            return ' generation', 'diffusers', 'stable_diffusion_1_5', 0.9
        elif 'sdxl' in filename:
            return 'generation', 'diffusers', 'stable_diffusion_xl', 0.9
        
        # ResNet Classification
        if 'resnet' in filename and 'classification' in full_path_lower  and 'multimodal' not in full_path_lower:
            if any(variant in filename for variant in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']):
                arch = next(variant for variant in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'] 
                           if variant in filename)
                return 'classification', 'torchvision', arch, 0.9
            return 'classification', 'torchvision', 'resnet', 0.8
        
        # EfficientNet Classification
        if 'efficientnet' in filename:
            return 'classification', 'torchvision', 'efficientnet', 0.9

        if 'vit' in filename and 'classification' in full_path_lower  and 'multimodal' not in full_path_lower:
            return 'classification', 'transformers', 'vit', 0.75 
        
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

        # DINOv3 models
        dinov3_patterns = ['dinov3', 'dinov2', 'dino_v3']
        if any(pattern in full_path_lower for pattern in dinov3_patterns):
            if 'vits' in filename:
                return 'feature_extraction', 'pytorch', 'dinov3_vits14', 0.95
            elif 'vitb' in filename:
                return 'feature_extraction', 'pytorch', 'dinov3_vitb14', 0.95
            elif 'vitl' in filename:
                return 'feature_extraction', 'pytorch', 'dinov3_vitl14', 0.95
            elif 'vitg' in filename:
                return 'feature_extraction', 'pytorch', 'dinov3_vitg14', 0.95
            else:
                return 'feature_extraction', 'pytorch', 'dinov3', 0.85
        
        # Fallback inference based on directory structure
        if 'segmentation' in parent_dirs:
            return 'segmentation', 'unknown', 'unknown', 0.6
        elif 'detection' in parent_dirs:
            return 'detection', 'unknown', 'unknown', 0.6
        elif 'classification' in parent_dirs:
            return 'classification', 'unknown', 'unknown', 0.6
        elif 'generation' in parent_dirs:
            if 'unet' in filename and file_size_mb > 500:
                return 'generation', 'diffusers', 'unet', 0.9
            elif 'vae' in filename and file_size_mb > 80:
                return 'generation', 'diffusers', 'vae', 0.8
            elif 'controlnet' in filename and file_size_mb > 100:
                return 'generation', 'diffusers', 'controlnet', 0.9
            elif 'flux' in filename and file_size_mb > 1000:
                return 'generation', 'diffusers', 'flux', 0.9
            elif any(pattern in filename for pattern in ['sd_2_1', 'sdxl']) and file_size_mb > 2000:
                return 'generation', 'diffusers', 'stable_diffusion', 0.95
            else:
                return 'generation', 'unknown', 'unknown', 0.3
        elif 'multimodal' in parent_dirs:
            return 'multimodal', 'transformers', 'unknown', 0.3
        
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
