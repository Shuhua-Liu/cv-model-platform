"""
Model Manager - Centrally manages the loading, caching, and calling of all models.

Provides a unified model interface, handling model loading, caching, device management, and more.
"""

from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time
import threading
from collections import OrderedDict
from loguru import logger

from .config_manager import get_config_manager
from .model_detector import ModelDetector, ModelInfo
from ..adapters.registry import get_registry
from ..adapters.base import BaseModelAdapter


class ModelCache:
    """Model cache management"""
    
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
            
            logger.debug(f"Hitting the model cache: {model_name}")
            return cache_entry['adapter']
    
    def put(self, model_name: str, adapter: BaseModelAdapter) -> None:
        """Cache Model"""
        with self._lock:
            # If the cache is full, remove the least recently used model
            while len(self._cache) >= self.max_size:
                oldest_model = next(iter(self._cache))
                self._remove(oldest_model)
            
            # Add to Cache
            self._cache[model_name] = {
                'adapter': adapter,
                'timestamp': time.time(),
                'last_accessed': time.time(),
                'access_count': 1
            }
            
            logger.info(f"Model cached: {model_name}")
    
    def remove(self, model_name: str) -> None:
        """Remove the model from the cache"""
        with self._lock:
            self._remove(model_name)
    
    def _remove(self, model_name: str) -> None:
        """Internal removal method"""
        if model_name in self._cache:
            cache_entry = self._cache.pop(model_name)
            adapter = cache_entry['adapter']
            
            try:
                adapter.unload_model()
            except Exception as e:
                logger.warning(f"Unloading model failed {model_name}: {e}")
            
            logger.info(f"Model removed from cache: {model_name}")
    
    def clear(self) -> None:
        """Clear the cache"""
        with self._lock:
            model_names = list(self._cache.keys())
            for model_name in model_names:
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


class ModelManager:
    """Model Manager"""
    
    def __init__(self):
        """Initialize the model manager"""
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
        
        # Model Configuration
        self._model_configs = self.config_manager.get_models_config()
        self._available_models = {}
        
        # Discovering and registering models
        self._discover_models()
        
        logger.info("Model manager initialization completed")
    
    def _discover_models(self) -> None:
        """Discover and register available models"""
        logger.info("Start discovering the model...")
        
        # Loading a model from a configuration file
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
                    # Automatic adapter detection
                    adapter_name = self.registry.auto_detect_adapter(
                        model_info.path, 
                        model_info.__dict__
                    )
                    
                    if adapter_name:
                        self._available_models[model_info.name] = {
                            'name': model_info.name,
                            'config': {
                                'type': model_info.type,
                                'path': model_info.path,
                                'format': model_info.format,
                                'framework': model_info.framework,
                                'architecture': model_info.architecture,
                                'device': 'auto',
                                'adapter': adapter_name
                            },
                            'source': 'auto_detected',
                            'model_info': model_info
                        }
        except Exception as e:
            logger.warning(f"Automatic model discovery failed: {e}")
        
        logger.info(f"Found {len(self._available_models)} available models")
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models"""
        return self._available_models.copy()
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        return self._available_models.get(model_name)
    
    def load_model(self, 
                   model_name: str, 
                   force_reload: bool = False,
                   **kwargs) -> BaseModelAdapter:
        """
        Loads a model

        Args:
            model_name: Model name
            force_reload: Whether to force a reload
            **kwargs: Parameters passed to the adapter

        Returns:
            Model adapter instance
        """
        # Check the cache
        if not force_reload and self.cache:
            cached_adapter = self.cache.get(model_name)
            if cached_adapter:
                logger.info(f"Using the cache model: {model_name}")
                return cached_adapter
        
        # Get model configuration
        model_entry = self._available_models.get(model_name)
        if not model_entry:
            raise ValueError(f"Model not found: {model_name}. Available models:{list(self._available_models.keys())}")
        
        model_config = model_entry['config'].copy()
        model_config.update(kwargs)  # Runtime parameters override configuration file parameters
        
        # Parsing model path
        model_path = model_config['path']
        if isinstance(model_path, str) and '{models_root}' in model_path:
            models_root = self.config_manager.get_models_root()
            model_path = model_path.format(models_root=models_root)
        
        # Get the adapter name
        adapter_name = model_config.get('adapter')
        if not adapter_name:
            # Try automatic detection
            model_info = model_entry.get('model_info')
            adapter_name = self.registry.auto_detect_adapter(model_path, model_info.__dict__ if model_info else None)
            
            if not adapter_name:
                raise ValueError(f"Unable to determine adapter type for model {model_name}")
        
        # Creating an Adapter Instance
        try:
            logger.info(f"Load model: {model_name} (adapter: {adapter_name})")
            adapter = self.registry.create_adapter(
                model_path=model_path,
                adapter_name=adapter_name,
                **{k: v for k, v in model_config.items() if k not in ['path', 'adapter']}
            )
            
            # Load the model into memory
            adapter.load_model()
            
            # Cache Model
            if self.cache:
                self.cache.put(model_name, adapter)
            
            logger.info(f"Model loaded successfully: {model_name}")
            return adapter
            
        except Exception as e:
            logger.error(f"Model loaded failed {model_name}: {e}")
            raise
    
    def predict(self, 
                model_name: str, 
                input_data: Any,
                **kwargs) -> Any:
        """
       Use the model to make predictions

        Args:
            model_name: Model name
            input_data: Input data
            **kwargs: Prediction parameters

        Returns:
            Prediction results
        """
        adapter = self.load_model(model_name)
        return adapter.predict(input_data, **kwargs)
    
    def unload_model(self, model_name: str) -> None:
        """Unload model"""
        if self.cache:
            self.cache.remove(model_name)
        logger.info(f"Model Unloaded: {model_name}")
    
    def clear_cache(self) -> None:
        """Clear the model cache"""
        if self.cache:
            self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache:
            return self.cache.get_cache_stats()
        return {"cache_enabled": False}
    
    def reload_configs(self) -> None:
        """Reload configuration"""
        logger.info("Reload model configuration...")
        self.config_manager.reload_configs()
        self._model_configs = self.config_manager.get_models_config()
        self.clear_cache()
        self._discover_models()
    
    def validate_models(self) -> Dict[str, Dict[str, Any]]:
        """Verify the availability of all models"""
        validation_results = {}
        
        for model_name, model_entry in self._available_models.items():
            result = {
                'name': model_name,
                'status': 'unknown',
                'errors': [],
                'warnings': []
            }
            
            try:
                model_config = model_entry['config']
                model_path = model_config['path']
                
                # Parsing Path
                if isinstance(model_path, str) and '{models_root}' in model_path:
                    models_root = self.config_manager.get_models_root()
                    model_path = model_path.format(models_root=models_root)
                
                # Check if a file exists
                if not model_path.startswith('torchvision://') and not Path(model_path).exists():
                    result['errors'].append(f"Model file does not exist: {model_path}")
                    result['status'] = 'error'
                    continue
                
                # Check if the adapter is available
                adapter_name = model_config.get('adapter')
                if adapter_name and not self.registry.get_adapter_class(adapter_name):
                    result['errors'].append(f"Adapter not available: {adapter_name}")
                    result['status'] = 'error'
                    continue
                
                # Try creating the adapter (without loading the model)
                if adapter_name:
                    adapter_class = self.registry.get_adapter_class(adapter_name)
                    if adapter_class:
                        # Simple instantiation test
                        test_adapter = adapter_class(model_path=model_path, device='cpu')
                        result['status'] = 'valid'
                    
            except Exception as e:
                result['errors'].append(str(e))
                result['status'] = 'error'
            
            validation_results[model_name] = result
        
        return validation_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        import torch
        import psutil
        
        status = {
            'models': {
                'total': len(self._available_models),
                'cached': len(self.cache._cache) if self.cache else 0
            },
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
            status['gpu'] = gpu_info
        
        return status


# Global Model Manager Instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager