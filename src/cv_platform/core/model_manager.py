"""
Model Manager - Enhanced with BaseManager inheritance
"""

import time
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger

from .base_manager import BaseManager, ManagerState, HealthStatus, HealthCheckResult
from .model_detector import ModelDetector, ModelInfo
from .config_manager import get_config_manager
from ..adapters.registry import get_registry
from ..adapters.base import BaseModelAdapter


class ModelCache:
    """Simple model cache implementation"""
    
    def __init__(self, max_size: int = 3, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._access_times = {}
        self._load_times = {}
    
    def get(self, key: str) -> Optional[BaseModelAdapter]:
        """Get cached model"""
        if key in self._cache:
            # Check TTL
            if time.time() - self._load_times[key] < self.ttl:
                self._access_times[key] = time.time()
                return self._cache[key]
            else:
                # Expired, remove from cache
                self.remove(key)
        return None
    
    def put(self, key: str, model: BaseModelAdapter) -> None:
        """Cache a model"""
        # Check if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            # Remove least recently used item
            oldest_key = min(self._access_times.keys(), key=self._access_times.get)
            self.remove(oldest_key)
        
        self._cache[key] = model
        self._access_times[key] = time.time()
        self._load_times[key] = time.time()
    
    def set(self, model_name: str, adapter: BaseModelAdapter) -> None:
       """
       Cache the model adapter (alias for put method)
       Args:
           model_name: Name of the model
           adapter: Model adapter instance
       """
       self.put(model_name, adapter)

    def remove(self, key: str) -> None:
        """Remove model from cache"""
        if key in self._cache:
            del self._cache[key]
            del self._access_times[key]
            del self._load_times[key]
    
    def clear(self) -> None:
        """Clear all cached models"""
        self._cache.clear()
        self._access_times.clear()
        self._load_times.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cached_models': len(self._cache),
            'max_size': self.max_size,
            'cache_utilization': len(self._cache) / max(self.max_size, 1),
            'ttl': self.ttl
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        current_time = time.time()
        
        # Calculate cache sizes (simplified)
        size_mb = len(self._cache) * 100  # Estimated 100MB per model
        
        stats = {
            'size_mb': size_mb,
            'max_size_mb': self.max_size * 100,
            'entry_count': len(self._cache),
            'hit_rate': 0.8,  # Placeholder
            'hits': 100,  # Placeholder
            'misses': 20,  # Placeholder
            'evictions': 0,  # Placeholder
            'cached_models': len(self._cache),
            'models': list(self._cache.keys()),
            'load_times': self._load_times.copy(),
            'access_times': self._access_times.copy(),
            'expired_models': [
                key for key, load_time in self._load_times.items()
                if current_time - load_time > self.ttl
            ],
            'cache_efficiency': {
                'utilization': len(self._cache) / max(self.max_size, 1),
                'avg_access_interval': 300,  # Placeholder
                'avg_model_lifetime': 1800  # Placeholder
            },
            'timestamp': current_time
        }
        
        return stats


class ModelManager(BaseManager):
    """Enhanced Model Manager inheriting from BaseManager"""
    
    def __init__(self):
        """Initialize the model manager with BaseManager capabilities"""
        super().__init__("ModelManager")
        
        # Initialize core components (will be done in initialize() method)
        self.config_manager = None
        self.registry = None
        self.detector = None
        self.cache = None
        
        # Model Configuration and state - ensure initialized to empty dict instead of None
        self._model_configs = {}
        self._available_models = {}
        
        logger.info("ModelManager initialized with BaseManager capabilities")
    
    def initialize(self) -> bool:
        """
        Initialize manager components - implements BaseManager abstract method
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting ModelManager initialization...")
            
            # Initialize core dependencies with comprehensive error handling
            try:
                self.config_manager = get_config_manager()
                if self.config_manager is None:
                    logger.warning("Config manager is None, using default configuration")
                    self._model_configs = {'models': {}}
                else:
                    # Safely obtain config to ensure not return None
                    config_result = self.config_manager.get_models_config()
                    if config_result is None:
                        logger.warning("get_models_config returned None, using empty config")
                        self._model_configs = {'models': {}}
                    else:
                        self._model_configs = config_result
                        # ensure models exists
                        if 'models' not in self._model_configs:
                            self._model_configs['models'] = {}
                        # Ensure models is not None
                        if self._model_configs['models'] is None:
                            self._model_configs['models'] = {}
                    
                    logger.info(f"Config loaded: {len(self._model_configs.get('models', {}))} model configurations")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize config manager: {e}, using defaults")
                self._model_configs = {'models': {}}
            
            try:
                self.registry = get_registry()
                if self.registry is None:
                    logger.warning("Registry is None")
            except Exception as e:
                logger.warning(f"Failed to initialize registry: {e}")
                self.registry = None
            
            try:
                self.detector = ModelDetector()
                if self.detector is None:
                    logger.warning("Model detector is None")
            except Exception as e:
                logger.warning(f"Failed to initialize model detector: {e}")
                self.detector = None
            
            # Initialize the cache with safe defaults
            try:
                if self.config_manager:
                    platform_config = self.config_manager.get_platform_config()
                    if platform_config is None:
                        platform_config = {}
                else:
                    platform_config = {}
                
                cache_config = platform_config.get('cache', {})
                if cache_config is None:
                    cache_config = {}
                
                cache_enabled = cache_config.get('enabled', True)
                max_cache_size = cache_config.get('max_size', 3)
                cache_ttl = cache_config.get('ttl', 3600)
                
                # Safely handle cache size parsing
                if isinstance(max_cache_size, str):
                    if max_cache_size.endswith('GB'):
                        max_cache_size = 3  # Default to 3 models for GB format
                    else:
                        try:
                            max_cache_size = int(max_cache_size)
                        except ValueError:
                            max_cache_size = 3
                elif max_cache_size is None:
                    max_cache_size = 3
                
                if cache_enabled:
                    self.cache = ModelCache(max_size=max_cache_size, ttl=cache_ttl)
                    logger.info(f"Cache initialized: max_size={max_cache_size}, ttl={cache_ttl}")
                else:
                    self.cache = None
                    logger.info("Cache disabled")
                    
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}, disabling cache")
                self.cache = None
            
            # ensure core config correct
            if self._model_configs is None:
                self._model_configs = {'models': {}}
            elif not isinstance(self._model_configs, dict):
                logger.warning(f"Invalid model config type: {type(self._model_configs)}, resetting")
                self._model_configs = {'models': {}}
            elif 'models' not in self._model_configs:
                self._model_configs['models'] = {}
            elif self._model_configs['models'] is None:
                self._model_configs['models'] = {}
            
            # ensure _available_models initialized
            if self._available_models is None:
                self._available_models = {}
            
            logger.info("Core components initialized, starting model discovery...")
            
            # Discover and register models with comprehensive error handling
            try:
                self._discover_models()
            except Exception as e:
                logger.warning(f"Model discovery failed: {e}")
                # ensure _available_models is dict
                if self._available_models is None:
                    self._available_models = {}
            
            # safety update metrics
            try:
                models_count = 0
                if self._available_models is not None and isinstance(self._available_models, dict):
                    models_count = len(self._available_models)
                
                self.update_metric('models_available', models_count)
                self.update_metric('cache_enabled', self.cache is not None)
                self.update_metric('initialization_time', time.time())
                
                logger.info(f"ModelManager initialization completed successfully - {models_count} models available")
                
            except Exception as e:
                logger.warning(f"Failed to set metrics: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"ModelManager initialization failed: {e}")
            # ensure key attribute is available even failed
            if self._available_models is None:
                self._available_models = {}
            if self._model_configs is None:
                self._model_configs = {'models': {}}
            return False
    
    def cleanup(self) -> None:
        """
        Cleanup manager resources - implements BaseManager abstract method
        """
        try:
            # Clear model cache
            if self.cache:
                self.cache.clear()
            
            # Clear available models
            if self._available_models:
                self._available_models.clear()
            
            # Update final metrics
            self.update_metric('cleanup_time', time.time())
            
            logger.info("ModelManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during ModelManager cleanup: {e}")
    
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
                    message=f"ModelManager not running (state: {self.state.value})",
                    details={'state': self.state.value},
                    timestamp=time.time(),
                    check_duration=time.time() - start_time
                )
            
            # Check available models with safe access
            models_count = 0
            if self._available_models is not None and isinstance(self._available_models, dict):
                models_count = len(self._available_models)
            
            if models_count == 0:
                status = HealthStatus.WARNING
                message = "No models available"
            else:
                status = HealthStatus.HEALTHY
                message = f"{models_count} models available"
            
            # Check cache health
            cache_stats = {}
            if self.cache:
                try:
                    cache_stats = self.cache.get_cache_stats()
                    cache_utilization = cache_stats.get('cache_utilization', 0)
                    if cache_utilization > 0.9:
                        status = HealthStatus.WARNING
                        message += ", cache nearly full"
                except Exception as e:
                    logger.warning(f"Failed to get cache stats: {e}")
                    cache_stats = {'error': str(e)}
            
            # Check configuration
            config_errors = {'errors': [], 'warnings': []}
            if self.config_manager:
                try:
                    config_errors = self.config_manager.validate_config()
                    if config_errors is None:
                        config_errors = {'errors': [], 'warnings': []}
                except Exception as e:
                    logger.warning(f"Failed to validate config: {e}")
                    config_errors = {'errors': [str(e)], 'warnings': []}
            
            if config_errors.get('errors'):
                status = HealthStatus.CRITICAL
                message += f", {len(config_errors['errors'])} config errors"
            
            details = {
                'models_available': models_count,
                'cache_stats': cache_stats,
                'config_errors': config_errors,
                'detector_available': self.detector is not None,
                'registry_available': self.registry is not None
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
    
    def _discover_models(self) -> None:
        """Discover and register available models with enhanced error handling"""
        logger.info("Starting model discovery...")
        
        # ensure _available_models is dict
        if self._available_models is None:
            self._available_models = {}
        
        # Loading models from configuration file
        try:
            models_config = {}
            if self._model_configs is not None and isinstance(self._model_configs, dict):
                models_config = self._model_configs.get('models', {})
                if models_config is None:
                    models_config = {}
            
            if models_config and isinstance(models_config, dict):
                for model_name, model_config in models_config.items():
                    if model_name and model_config:  # ensure key is non-empty
                        self._available_models[model_name] = {
                            'name': model_name,
                            'config': model_config,
                            'source': 'config'
                        }
                logger.info(f"Loaded {len(models_config)} models from configuration")
            else:
                logger.info("No valid models configuration found")
                
        except Exception as e:
            logger.warning(f"Failed to load models from config: {e}")
        
        # Automatically discover model files
        try:
            if self.detector:
                detected_models = self.detector.detect_models()
                # safety check detected_models
                if detected_models is not None and isinstance(detected_models, list):
                    for model_info in detected_models:
                        if model_info and hasattr(model_info, 'name') and model_info.name not in self._available_models:
                            try:
                                self._available_models[model_info.name] = {
                                    'name': model_info.name,
                                    'config': self._model_info_to_config(model_info),
                                    'source': 'auto_detected',
                                    'model_info': model_info
                                }
                            except Exception as e:
                                logger.warning(f"Failed to register detected model {model_info.name}: {e}")
                    
                    detected_count = len(detected_models)
                    self.update_metric('models_detected', detected_count)
                    logger.info(f"Discovered {detected_count} models automatically")
                else:
                    logger.info("No models detected automatically (None or empty result)")
                    self.update_metric('models_detected', 0)
            else:
                logger.warning("Model detector not available, skipping auto-discovery")
                self.update_metric('models_detected', 0)
                
        except Exception as e:
            logger.error(f"Model auto-discovery failed: {e}")
            self.update_metric('models_detected', 0)
        
        # safety update metrics
        try:
            total_models = len(self._available_models) if self._available_models else 0
            self.update_metric('total_models_available', total_models)
            logger.info(f"Total available models: {total_models}")
        except Exception as e:
            logger.warning(f"Failed to update total models metric: {e}")
    
    def _model_info_to_config(self, model_info: ModelInfo) -> Dict[str, Any]:
        """Convert ModelInfo to model configuration format"""
        try:
            return {
                'path': str(model_info.path) if hasattr(model_info, 'path') and model_info.path else '',
                'type': getattr(model_info, 'type', 'unknown'),
                'framework': getattr(model_info, 'framework', 'unknown'),
                'architecture': getattr(model_info, 'architecture', 'unknown'),
                'adapter': self._determine_adapter_name(model_info),
                'device': 'auto',
                'cache_enabled': True
            }
        except Exception as e:
            logger.warning(f"Failed to convert model info to config: {e}")
            return {
                'path': '',
                'type': 'unknown',
                'framework': 'unknown',
                'architecture': 'unknown',
                'adapter': 'unknown',
                'device': 'auto',
                'cache_enabled': True
            }
    
    def _determine_adapter_name(self, model_info: ModelInfo) -> str:
        """Determine adapter name based on model info"""
        try:
            if self.registry and model_info:
                return self.registry.auto_detect_adapter(
                    model_path=str(getattr(model_info, 'path', '')),
                    model_info=getattr(model_info, '__dict__', {})
                )
        except Exception as e:
            logger.warning(f"Failed to determine adapter for {getattr(model_info, 'name', 'unknown')}: {e}")
        return 'unknown'
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        if self._available_models is None:
            return {}
        return self._available_models.copy()
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if self._available_models is None:
            return None
        return self._available_models.get(model_name)
    
    def load_model(self, model_name: str, **kwargs) -> BaseModelAdapter:
        """
        Load model with enhanced error handling and metrics
        
        Args:
            model_name: Name of the model to load
            **kwargs: Additional parameters
            
        Returns:
            Loaded model adapter
        """
        load_start_time = time.time()
        
        try:
            # Check if model is available
            if model_name not in self._available_models:
                available_models = list(self._available_models.keys())
                raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
            
            # Check cache first
            if self.cache:
                cached_adapter = self.cache.get(model_name)
                if cached_adapter:
                    self.increment_metric('cache_hits')
                    self.update_metric('last_cache_hit', time.time())
                    logger.info(f"Model {model_name} loaded from cache")
                    return cached_adapter
            
            # Load model if not cached
            model_entry = self._available_models[model_name]
            model_config = model_entry['config'].copy()
            model_config.update(kwargs)  # Runtime parameters override configuration
            
            # Parse model path
            model_path = model_config['path']
            if isinstance(model_path, str) and '{models_root}' in model_path:
                models_root = self.config_manager.get_models_root()
                model_path = model_path.format(models_root=models_root)
            
            # Ensure model_path is string
            model_path = str(model_path)
            
            # Get the adapter name - Add more detection logic
            adapter_name = model_config.get('adapter')
            
            if not adapter_name:
                # Try to get from config first
                framework = model_config.get('framework')
                architecture = model_config.get('architecture')
                model_type = model_config.get('type')

                # Manually detection logic
                adapter_name = self._detect_adapter_manually(model_path, framework, architecture, model_type)
                
                if not adapter_name:
                    # Try to auto-detect
                    model_info = model_entry.get('model_info')
                    adapter_name = self.registry.auto_detect_adapter(
                        model_path, 
                        model_info.__dict__ if model_info else None
                    )
                
                if not adapter_name:
                    raise ValueError(f"Unable to determine adapter type for model {model_name}")
            
            # Create adapter instance
            logger.info(f"Loading model: {model_name} (adapter: {adapter_name})")

            # Ensure kwargs correct when creating adapter
            create_kwargs = {
                'device': model_config.get('device', 'auto'),
                'cache_enabled': model_config.get('cache_enabled', True)
            }
            create_kwargs.update(kwargs)
            
            adapter = self.registry.create_adapter(
                model_path=model_path,
                adapter_name=adapter_name,  # Explicitly specify adapter name
                **create_kwargs
            )
            
            # Update metrics
            load_time = time.time() - load_start_time
            self.increment_metric('models_loaded')
            self.update_metric('total_load_time', self.get_metric('total_load_time', 0) + load_time)
            self.update_metric('last_model_loaded', model_name)
            self.update_metric('last_load_time', load_time)
            
            # Cache the adapter if caching is enabled
            if self.cache and model_config.get('cache_enabled', True):
                self.cache.put(model_name, adapter)
                self.increment_metric('models_cached')

            load_time = time.time() - load_start_time
            self.increment_metric('models_loaded')
            self.increment_metric('cache_misses')
            self.update_metric('last_load_time', load_time)
            self.update_metric('total_load_time', self.get_metric('total_load_time', 0) + load_time)
            
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            return adapter
            
        except Exception as e:
            load_time = time.time() - load_start_time
            self.increment_metric('load_failures')
            self.update_metric('last_load_error', str(e))
            logger.error(f"Failed to load model {model_name} after {load_time:.2f}s: {e}")
            raise

    def _detect_adapter_manually(self, model_path: str, framework: str = None, architecture: str = None, model_type: str = None) -> str:
        """
        Detect adapter manually
        """
        model_path_lower = str(model_path).lower()
        
        logger.info(f"🔧 Detect adapter manually:")
        logger.info(f"   Path: {model_path}")
        logger.info(f"   Framework: {framework}")
        logger.info(f"   Architecture: {architecture}")
        logger.info(f"   Type: {model_type}")
        
        # 1. Based on framework
        if framework:
            framework_lower = framework.lower()
            if framework_lower in ['ultralytics', 'yolo']:
                return 'ultralytics'
            elif framework_lower in ['segment_anything', 'sam']:
                return 'sam'
            elif framework_lower in ['diffusers', 'stable_diffusion']:
                return 'stable_diffusion'
            elif framework_lower in ['torchvision']:
                return 'torchvision_classification'
            elif framework_lower in ['clip', 'openai']:
                return 'clip'
        
        # 2. Based on model path
        if any(pattern in model_path_lower for pattern in ['yolo', 'yolov8', 'yolov9', 'yolov10', 'yolo11']):
            return 'ultralytics'

        if any(pattern in model_path_lower for pattern in ['detr']):
            return 'ultralytics'
        
        if any(pattern in model_path_lower for pattern in ['sam_vit', 'mobile_sam']):
            return 'sam'
        
        if any(pattern in model_path_lower for pattern in ['stable-diffusion', 'sd_', 'sdxl', 'flux']):
            return 'stable_diffusion'
        
        if any(pattern in model_path_lower for pattern in ['clip']):
            return 'clip'

        if any(pattern in model_path_lower for pattern in ['resnet', 'efficientnet', 'vit']):
            # Ensure not detection model
            if not any(exclusion in model_path_lower for exclusion in ['yolo', 'detr', 'detection']):
                return 'torchvision_classification'
            
        if any(pattern in model_path_lower for pattern in [
            'detectron2', 'faster_rcnn', 'mask_rcnn', 'retinanet',
            'fcos', 'mask2former', 'panoptic_fpn', 'keypoint_rcnn'
            ]):
            return 'detectron2'
        # 2. Model zoo configuration names
        detectron2_configs = [
            'faster_rcnn_r50', 'faster_rcnn_r101', 'retinanet_r50',
            'fcos_r50', 'mask_rcnn_r50', 'mask_rcnn_r101',
            'mask2former_r50', 'panoptic_fpn_r50', 'keypoint_rcnn_r50'
        ]
        if any(config in model_path for config in detectron2_configs):
            return 'detectron2'
        
        # 3. Based on model type
        if model_type:
            type_lower = model_type.lower()
            if type_lower == 'detection':
                return 'ultralytics'  # default detector
            elif type_lower == 'segmentation':
                return 'sam'  # default segmentation model
            elif type_lower == 'classification':
                return 'torchvision_classification'
            elif type_lower == 'generation':
                return 'stable_diffusion'
            elif type_lower == 'multimodal':
                return 'clip'
        
        logger.warning("   ❌ Manually detection failed.")
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            models_count = len(self._available_models) if self._available_models else 0
            cached_count = len(self.cache._cache) if self.cache else 0
            
            base_status = {
                'models': {
                    'total': models_count,
                    'cached': cached_count,
                    'available_models': list(self._available_models.keys()) if self._available_models else []
                },
                'cache': {
                    'enabled': self.cache is not None,
                    'size': cached_count,
                    'max_size': self.cache.max_size if self.cache else 0
                },
                'components': {
                    'config_manager': self.config_manager is not None,
                    'registry': self.registry is not None,
                    'detector': self.detector is not None
                }
            }
            
            # Add GPU info if available
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_info.append({
                        'device_id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                        'memory_allocated': torch.cuda.memory_allocated(i) / (1024**3),  # GB
                        'memory_cached': torch.cuda.memory_reserved(i) / (1024**3)  # GB
                    })
                base_status['gpu'] = gpu_info
            
            return base_status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'error': str(e),
                'models': {'total': 0, 'cached': 0, 'available_models': []},
                'cache': {'enabled': False, 'size': 0, 'max_size': 0},
                'components': {'config_manager': False, 'registry': False, 'detector': False}
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            try:
                return self.cache.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")
                return {'error': str(e)}
        return {'cache_disabled': True}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        try:
            metrics = self.metrics.copy() if hasattr(self, 'metrics') and self.metrics else {}
            models_count = len(self._available_models) if self._available_models else 0
            
            return {
                'metrics': {
                    'models_available': {'value': models_count},
                    'cache_enabled': {'value': self.cache is not None},
                    'total_load_time': {'value': sum(v for k, v in metrics.items() if 'load_time' in k and isinstance(v, (int, float)))},
                    'models_loaded': {'value': len([k for k in metrics.keys() if 'load_time' in k])}
                },
                'cache_stats': self.get_cache_stats(),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.warning(f"Failed to get performance summary: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def reload_models(self) -> int:
        """
        Reload model configurations and rediscover models
        
        Returns:
            Number of newly discovered models
        """
        logger.info("Reloading models...")
        
        try:
            # Clear current models
            old_count = len(self._available_models) if self._available_models else 0
            if self._available_models:
                self._available_models.clear()
            else:
                self._available_models = {}
            
            # Reload configurations
            if self.config_manager:
                try:
                    self.config_manager.reload_configs()
                    config_result = self.config_manager.get_models_config()
                    if config_result is None:
                        self._model_configs = {'models': {}}
                    else:
                        self._model_configs = config_result
                        if 'models' not in self._model_configs:
                            self._model_configs['models'] = {}
                        if self._model_configs['models'] is None:
                            self._model_configs['models'] = {}
                except Exception as e:
                    logger.warning(f"Failed to reload configs: {e}")
            
            # Rediscover models
            self._discover_models()
            
            new_count = len(self._available_models) if self._available_models else 0
            newly_discovered = new_count - old_count
            
            self.update_metric('models_reloaded', new_count)
            self.update_metric('last_reload_time', time.time())
            
            logger.info(f"Reloaded {new_count} models ({newly_discovered} newly discovered)")
            return newly_discovered
            
        except Exception as e:
            logger.error(f"Failed to reload models: {e}")
            return 0


# Global Model Manager Instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
        # Auto-start the manager
        if not _model_manager.start():
            logger.error("Failed to start ModelManager")
    return _model_manager
