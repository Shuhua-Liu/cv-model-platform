"""
模型管理器 - 统一管理所有模型的加载、缓存和调用

提供模型的统一接口，处理模型加载、缓存、设备管理等。
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
    """模型缓存管理"""
    
    def __init__(self, max_size: int = 3, ttl: int = 3600):
        """
        初始化模型缓存
        
        Args:
            max_size: 最大缓存模型数量
            ttl: 缓存生存时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        
        logger.info(f"模型缓存初始化 - 最大数量: {max_size}, TTL: {ttl}s")
    
    def get(self, model_name: str) -> Optional[BaseModelAdapter]:
        """获取缓存的模型"""
        with self._lock:
            if model_name not in self._cache:
                return None
            
            cache_entry = self._cache[model_name]
            
            # 检查TTL
            if time.time() - cache_entry['timestamp'] > self.ttl:
                logger.debug(f"模型缓存过期: {model_name}")
                self._remove(model_name)
                return None
            
            # 移到最近使用位置
            self._cache.move_to_end(model_name)
            cache_entry['last_accessed'] = time.time()
            cache_entry['access_count'] += 1
            
            logger.debug(f"命中模型缓存: {model_name}")
            return cache_entry['adapter']
    
    def put(self, model_name: str, adapter: BaseModelAdapter) -> None:
        """缓存模型"""
        with self._lock:
            # 如果缓存已满，移除最久未使用的模型
            while len(self._cache) >= self.max_size:
                oldest_model = next(iter(self._cache))
                self._remove(oldest_model)
            
            # 添加到缓存
            self._cache[model_name] = {
                'adapter': adapter,
                'timestamp': time.time(),
                'last_accessed': time.time(),
                'access_count': 1
            }
            
            logger.info(f"模型已缓存: {model_name}")
    
    def remove(self, model_name: str) -> None:
        """从缓存中移除模型"""
        with self._lock:
            self._remove(model_name)
    
    def _remove(self, model_name: str) -> None:
        """内部移除方法"""
        if model_name in self._cache:
            cache_entry = self._cache.pop(model_name)
            adapter = cache_entry['adapter']
            
            try:
                adapter.unload_model()
            except Exception as e:
                logger.warning(f"卸载模型失败 {model_name}: {e}")
            
            logger.info(f"模型已从缓存移除: {model_name}")
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            model_names = list(self._cache.keys())
            for model_name in model_names:
                self._remove(model_name)
            logger.info("模型缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
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
    """模型管理器"""
    
    def __init__(self):
        """初始化模型管理器"""
        self.config_manager = get_config_manager()
        self.registry = get_registry()
        self.detector = ModelDetector()
        
        # 初始化缓存
        platform_config = self.config_manager.get_platform_config()
        cache_config = platform_config.get('cache', {})
        
        cache_enabled = cache_config.get('enabled', True)
        max_cache_size = cache_config.get('max_size', 3)
        cache_ttl = cache_config.get('ttl', 3600)
        
        if cache_enabled:
            # 解析缓存大小（如果是字符串格式如"4GB"）
            if isinstance(max_cache_size, str) and max_cache_size.endswith('GB'):
                max_cache_size = 3  # 默认缓存3个模型
            
            self.cache = ModelCache(max_size=max_cache_size, ttl=cache_ttl)
        else:
            self.cache = None
        
        # 模型配置
        self._model_configs = self.config_manager.get_models_config()
        self._available_models = {}
        
        # 发现和注册模型
        self._discover_models()
        
        logger.info("模型管理器初始化完成")
    
    def _discover_models(self) -> None:
        """发现并注册可用模型"""
        logger.info("开始发现模型...")
        
        # 从配置文件加载模型
        models_config = self._model_configs.get('models', {})
        for model_name, model_config in models_config.items():
            self._available_models[model_name] = {
                'name': model_name,
                'config': model_config,
                'source': 'config'
            }
        
        # 自动发现模型文件
        try:
            detected_models = self.detector.detect_models()
            for model_info in detected_models:
                if model_info.name not in self._available_models:
                    # 自动检测适配器
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
            logger.warning(f"自动发现模型失败: {e}")
        
        logger.info(f"发现 {len(self._available_models)} 个可用模型")
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """列出所有可用模型"""
        return self._available_models.copy()
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型信息"""
        return self._available_models.get(model_name)
    
    def load_model(self, 
                   model_name: str, 
                   force_reload: bool = False,
                   **kwargs) -> BaseModelAdapter:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            force_reload: 是否强制重新加载
            **kwargs: 传递给适配器的参数
            
        Returns:
            模型适配器实例
        """
        # 检查缓存
        if not force_reload and self.cache:
            cached_adapter = self.cache.get(model_name)
            if cached_adapter:
                logger.info(f"使用缓存模型: {model_name}")
                return cached_adapter
        
        # 获取模型配置
        model_entry = self._available_models.get(model_name)
        if not model_entry:
            raise ValueError(f"未找到模型: {model_name}. 可用模型: {list(self._available_models.keys())}")
        
        model_config = model_entry['config'].copy()
        model_config.update(kwargs)  # 运行时参数覆盖配置文件参数
        
        # 解析模型路径
        model_path = model_config['path']
        if isinstance(model_path, str) and '{models_root}' in model_path:
            models_root = self.config_manager.get_models_root()
            model_path = model_path.format(models_root=models_root)
        
        # 获取适配器名称
        adapter_name = model_config.get('adapter')
        if not adapter_name:
            # 尝试自动检测
            model_info = model_entry.get('model_info')
            adapter_name = self.registry.auto_detect_adapter(model_path, model_info.__dict__ if model_info else None)
            
            if not adapter_name:
                raise ValueError(f"无法确定模型 {model_name} 的适配器类型")
        
        # 创建适配器实例
        try:
            logger.info(f"加载模型: {model_name} (适配器: {adapter_name})")
            adapter = self.registry.create_adapter(
                model_path=model_path,
                adapter_name=adapter_name,
                **{k: v for k, v in model_config.items() if k not in ['path', 'adapter']}
            )
            
            # 加载模型到内存
            adapter.load_model()
            
            # 缓存模型
            if self.cache:
                self.cache.put(model_name, adapter)
            
            logger.info(f"模型加载成功: {model_name}")
            return adapter
            
        except Exception as e:
            logger.error(f"加载模型失败 {model_name}: {e}")
            raise
    
    def predict(self, 
                model_name: str, 
                input_data: Any,
                **kwargs) -> Any:
        """
        使用模型进行预测
        
        Args:
            model_name: 模型名称
            input_data: 输入数据
            **kwargs: 预测参数
            
        Returns:
            预测结果
        """
        adapter = self.load_model(model_name)
        return adapter.predict(input_data, **kwargs)
    
    def unload_model(self, model_name: str) -> None:
        """卸载模型"""
        if self.cache:
            self.cache.remove(model_name)
        logger.info(f"模型已卸载: {model_name}")
    
    def clear_cache(self) -> None:
        """清空模型缓存"""
        if self.cache:
            self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if self.cache:
            return self.cache.get_cache_stats()
        return {"cache_enabled": False}
    
    def reload_configs(self) -> None:
        """重新加载配置"""
        logger.info("重新加载模型配置...")
        self.config_manager.reload_configs()
        self._model_configs = self.config_manager.get_models_config()
        self.clear_cache()
        self._discover_models()
    
    def validate_models(self) -> Dict[str, Dict[str, Any]]:
        """验证所有模型的可用性"""
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
                
                # 解析路径
                if isinstance(model_path, str) and '{models_root}' in model_path:
                    models_root = self.config_manager.get_models_root()
                    model_path = model_path.format(models_root=models_root)
                
                # 检查文件是否存在
                if not model_path.startswith('torchvision://') and not Path(model_path).exists():
                    result['errors'].append(f"模型文件不存在: {model_path}")
                    result['status'] = 'error'
                    continue
                
                # 检查适配器是否可用
                adapter_name = model_config.get('adapter')
                if adapter_name and not self.registry.get_adapter_class(adapter_name):
                    result['errors'].append(f"适配器不可用: {adapter_name}")
                    result['status'] = 'error'
                    continue
                
                # 尝试创建适配器（不加载模型）
                if adapter_name:
                    adapter_class = self.registry.get_adapter_class(adapter_name)
                    if adapter_class:
                        # 简单的实例化测试
                        test_adapter = adapter_class(model_path=model_path, device='cpu')
                        result['status'] = 'valid'
                    
            except Exception as e:
                result['errors'].append(str(e))
                result['status'] = 'error'
            
            validation_results[model_name] = result
        
        return validation_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
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
        
        # GPU信息
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


# 全局模型管理器实例
_model_manager = None

def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager