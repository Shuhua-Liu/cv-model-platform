"""
Enhanced Model Manager - Inheriting from BaseManager

Model Manager with full BaseManager integration for state management,
health monitoring, metrics tracking, and lifecycle management.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import threading
from collections import OrderedDict
from loguru import logger

from .base_manager import BaseManager, ManagerState, HealthStatus, HealthCheckResult
from .config_manager import get_config_manager
from .model_detector import ModelDetector, ModelInfo
from ..adapters.registry import get_registry
from ..adapters.base import BaseModelAdapter


class ModelCache:
    """Model cache management (keeping existing implementation)"""
    
    def __init__(self, max_size: int = 3, ttl: int = 3600):
        """
        Initialize the model cache

        Args:
            max_size: Maximum number of cached models
            ttl: Cache lifetime (seconds)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        
        logger.info(f"Model cache initialization - maximum number: {max_size}, TTL: {ttl}s")
    
    def get(self, model_name: str) -> Optional[BaseModelAdapter]:
        """Get the cached model"""
        with self._lock:
            if model_name not in self._cache:
                return None
            
            cache_entry = self._cache[model_name]
            
            # Check TTL
            if time.time() - cache_entry['timestamp'] > self.ttl:
                logger.debug(f"Model cache expires: {model_name}")
                self._remove(model_name)
                return None
            
            # Move to recently used location
            self._cache.move_to_end(model_name)
            cache_entry['last_accessed'] = time.time()
            cache_entry['access_count'] += 1
            
            return cache_entry['adapter']
    
    def put(self, model_name: str, adapter: BaseModelAdapter) -> None:
        """Cache the model adapter"""
        with self._lock:
            # Remove oldest if cache is full
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)
                logger.debug(f"Evicted model from cache: {oldest_key}")
            
            self._cache[model_name] = {
                'adapter': adapter,
                'timestamp': time.time(),
                'last_accessed': time.time(),
                'access_count': 1
            }
            
            logger.debug(f"Model cached: {model_name}")
    
    def _remove(self, model_name: str) -> None:
        """Remove model from cache"""
        if model_name in self._cache:
            del self._cache[model_name]
    
    def clear(self) -> None:
        """Clear all cached models"""
        with self._lock:
            for model_name in list(self._cache.keys()):
                self._remove(model_name)
            logger.info("Model cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            stats = {
                'cached_models': len(self._cache),
                'max_size': self.max_size,
                'ttl': self.ttl,
                'models': {}
            }
            
            for model_name, cache_entry in self._cache.items():
                stats['models'][model_name] = {
                    'timestamp': cache_entry['timestamp'],
                    'last_accessed': cache_entry['last_accessed'],
                    'access_count': cache_entry['access_count'],
                    'age': time.time() - cache_entry['timestamp']
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
        
        # Model Configuration and state
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
            # Initialize core dependencies
            self.config_manager = get_config_manager()
            self.registry = get_registry()
            self.detector = ModelDetector()
            
            # Initialize the cache
            platform_config = self.config_manager.get_platform_config()
            cache_config = platform_config.get('cache', {})
            
            cache_enabled = cache_config.get('enabled', True)
            max_cache_size = cache_config.get('max_size', 3)
            cache_ttl = cache_config.get('ttl', 3600)
            
            if cache_enabled:
                # Parsing cache size (if in string format such as "4GB")
                if isinstance(max_cache_size, str) and max_cache_size.endswith('GB'):
                    max_cache_size = 3  # By default, 3 models are cached.
                
                self.cache = ModelCache(max_size=max_cache_size, ttl=cache_ttl)
            else:
                self.cache = None
            
            # Load model configurations
            self._model_configs = self.config_manager.get_models_config()
            
            # Discover and register models
            self._discover_models()
            
            # Set up initial metrics
            self.update_metric('models_available', len(self._available_models))
            self.update_metric('cache_enabled', cache_enabled)
            self.update_metric('initialization_time', time.time())
            
            logger.info("ModelManager initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ModelManager initialization failed: {e}")
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
            
            # Check available models
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
                cache_stats = self.cache.get_cache_stats()
                cache_utilization = cache_stats['cached_models'] / cache_stats['max_size']
                if cache_utilization > 0.9:
                    status = HealthStatus.WARNING
                    message += ", cache nearly full"
            
            # Check configuration
            config_errors = self.config_manager.validate_config() if self.config_manager else {'errors': [], 'warnings': []}
            if config_errors['errors']:
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
        """Discover and register available models"""
        logger.info("Starting model discovery...")
        
        # Loading models from configuration file
        models_config = self._model_configs.get('models', {})
        for model_name, model_config in models_config.items():
            self._available_models[model_name] = {
                'name': model_name,
                'config': model_config,
                'source': 'config'
            }
        
        # Automatically discover model files
        try:
            detected_models = self.detector.detect_models()
            for model_info in detected_models:
                if model_info.name not in self._available_models:
                    # Automatically register detected models
                    self._available_models[model_info.name] = {
                        'name': model_info.name,
                        'config': self._model_info_to_config(model_info),
                        'source': 'auto_detected',
                        'model_info': model_info
                    }
            
            self.update_metric('models_detected', len(detected_models))
            logger.info(f"Discovered {len(detected_models)} models automatically")
            
        except Exception as e:
            logger.error(f"Model auto-discovery failed: {e}")
        
        self.update_metric('total_models_available', len(self._available_models))
        logger.info(f"Total available models: {len(self._available_models)}")
    
    def _model_info_to_config(self, model_info: ModelInfo) -> Dict[str, Any]:
        """Convert ModelInfo to model configuration format"""
        return {
            'path': str(model_info.path),
            'type': model_info.type,
            'framework': model_info.framework,
            'architecture': model_info.architecture,
            'adapter': self._determine_adapter_name(model_info),
            'device': 'auto',
            'cache_enabled': True
        }
    
    def _determine_adapter_name(self, model_info: ModelInfo) -> str:
        """Determine adapter name based on model info"""
        try:
            return self.registry.auto_detect_adapter(
                model_path=str(model_info.path),
                model_info=model_info.__dict__
            )
        except Exception as e:
            logger.warning(f"Failed to determine adapter for {model_info.name}: {e}")
            return 'unknown'
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        return self._available_models.copy()
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
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
            
            # Get the adapter name
            adapter_name = model_config.get('adapter')
            if not adapter_name:
                # Try automatic detection
                model_info = model_entry.get('model_info')
                adapter_name = self.registry.auto_detect_adapter(
                    model_path, 
                    model_info.__dict__ if model_info else None
                )
                
                if not adapter_name:
                    raise ValueError(f"Unable to determine adapter type for model {model_name}")
            
            # Create adapter instance
            logger.info(f"Loading model: {model_name} (adapter: {adapter_name})")
            adapter = self.registry.create_adapter(
                model_path=model_path,
                adapter_name=adapter_name,
                **{k: v for k, v in model_config.items() if k not in ['path', 'adapter']}
            )
            
            # Load the model into memory
            adapter.load_model()
            
            # Cache the loaded model
            if self.cache and model_config.get('cache_enabled', True):
                self.cache.put(model_name, adapter)
                self.increment_metric('models_cached')
            
            # Update metrics
            load_time = time.time() - load_start_time
            self.increment_metric('models_loaded')
            self.increment_metric('cache_misses')
            self.update_metric('last_load_time', load_time)
            self.update_metric('total_load_time', 
                             self.get_metric('total_load_time', 0) + load_time)
            
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            return adapter
            
        except Exception as e:
            # Update error metrics
            self.increment_metric('load_failures')
            self.update_metric('last_error', str(e))
            self.update_metric('last_error_time', time.time())
            
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload model from cache
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if model was unloaded, False if not found
        """
        if self.cache:
            # Remove from cache
            self.cache._remove(model_name)
            self.increment_metric('models_unloaded')
            logger.info(f"Model {model_name} unloaded from cache")
            return True
        return False
    
    def clear_cache(self) -> None:
        """Clear all cached models"""
        if self.cache:
            cache_stats = self.cache.get_cache_stats()
            self.cache.clear()
            self.update_metric('cache_clears', 1)
            logger.info(f"Cleared {cache_stats['cached_models']} models from cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_cache_stats()
        return {'cache_enabled': False}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        import torch
        import psutil
        
        base_status = {
            'manager_status': super().get_status(),  # Get BaseManager status
            'models': {
                'total': len(self._available_models),
                'cached': len(self.cache._cache) if self.cache else 0,
                'available_models': list(self._available_models.keys())
            },
            'cache': self.get_cache_stats(),
            'system': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else None
            },
            'torch': {
                'version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        # GPU information
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
    
    def reload_models(self) -> int:
        """
        Reload model configurations and rediscover models
        
        Returns:
            Number of newly discovered models
        """
        logger.info("Reloading models...")
        
        # Clear current models
        old_count = len(self._available_models)
        self._available_models.clear()
        
        # Reload configurations
        self.config_manager.reload_configs()
        self._model_configs = self.config_manager.get_models_config()
        
        # Rediscover models
        self._discover_models()
        
        new_count = len(self._available_models)
        newly_discovered = new_count - old_count
        
        self.update_metric('models_reloaded', new_count)
        self.update_metric('last_reload_time', time.time())
        
        logger.info(f"Reloaded {new_count} models ({newly_discovered} newly discovered)")
        return newly_discovered


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