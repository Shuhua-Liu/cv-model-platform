"""
CV Platform 核心模块

包含配置管理、模型检测、模型管理等核心功能。
"""

from .config_manager import ConfigManager, get_config_manager
from .model_detector import ModelDetector, ModelInfo
from .model_manager import ModelManager, get_model_manager

__all__ = [
    'ConfigManager',
    'get_config_manager', 
    'ModelDetector',
    'ModelInfo',
    'ModelManager',
    'get_model_manager',
]
