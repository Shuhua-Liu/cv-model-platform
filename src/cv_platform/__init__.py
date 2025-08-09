"""
CV Model Platform - A unified platform for computer vision model management and deployment.

This package provides a unified interface for managing and deploying various computer vision models
including detection, segmentation, classification, generation, and multimodal models.
"""

__version__ = "0.1.0"
__author__ = "Shuhua Liu"
__email__ = "shuhua.liu0709@gmail.com"
__license__ = "MIT"

# 延迟导入避免循环导入问题
def get_model_manager():
    """获取模型管理器实例"""
    from .core.model_manager import get_model_manager
    return get_model_manager()

def get_config_manager():
    """获取配置管理器实例"""
    from .core.config_manager import get_config_manager
    return get_config_manager()

def get_adapter_registry():
    """获取适配器注册中心实例"""
    from .adapters.registry import get_registry
    return get_registry()

# 主要导出类
__all__ = [
    "__version__",
    "get_model_manager", 
    "get_config_manager",
    "get_adapter_registry",
]

# 平台配置
DEFAULT_CONFIG = {
    "models_root": "./cv_models",  # 改为相对路径，适配Mac系统
    "cache_dir": ".cv_platform_cache",
    "log_level": "INFO",
    "device": "auto",  # auto, cpu, cuda:0, etc.
}

# 简化导入
def setup_platform(models_root=None, cache_dir=None, device=None, log_level=None):
    """
    初始化CV平台
    
    Args:
        models_root: 模型根目录路径
        cache_dir: 缓存目录路径
        device: 计算设备 (cpu, cuda:0, auto等)
        log_level: 日志级别
    """
    config_updates = {}
    
    if models_root:
        config_updates["models_root"] = models_root
    if cache_dir:
        config_updates["cache_dir"] = cache_dir  
    if device:
        config_updates["device"] = device
    if log_level:
        config_updates["log_level"] = log_level
        
    # 更新全局配置
    DEFAULT_CONFIG.update(config_updates)
    
    # 设置日志
    from .utils.logger import setup_logger
    setup_logger(log_level or DEFAULT_CONFIG["log_level"])
    
    return DEFAULT_CONFIG