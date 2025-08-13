"""
CV Platform Core Module

Contains configuration management, model detection, model management,
caching, GPU monitoring, scheduling, and base management functionality.
"""

from .config_manager import ConfigManager, get_config_manager
from .model_detector import ModelDetector, ModelInfo, get_model_detector
from .model_manager import ModelManager, get_model_manager
from .cache_manager import CacheManager, get_cache_manager, CacheStrategy, CacheStats
from .gpu_monitor import GPUMonitor, get_gpu_monitor, GPUInfo, GPUMemoryInfo, GPUUtilization, DeviceType
from .scheduler import (
    TaskScheduler, get_scheduler, TaskRequest, TaskResult, TaskStatus, 
    TaskPriority, SchedulingStrategy, submit_inference_task
)
from .base_manager import (
    BaseManager, ManagerState, HealthStatus, HealthCheckResult, 
    ManagerRegistry, get_manager_registry, register_manager
)

__all__ = [
    # Base Management
    'BaseManager',
    'ManagerState',
    'HealthStatus',
    'HealthCheckResult',
    'ManagerRegistry',
    'get_manager_registry',
    'register_manager',
    
    # Configuration Management
    'ConfigManager',
    'get_config_manager',
    
    # Model Detection and Management
    'ModelDetector',
    'ModelInfo',
    'get_model_detector'
    'ModelManager',
    'get_model_manager',
    
    # Caching System
    'CacheManager',
    'get_cache_manager',
    'CacheStrategy',
    'CacheStats',
    
    # GPU Monitoring
    'GPUMonitor',
    'get_gpu_monitor',
    'GPUInfo',
    'GPUMemoryInfo',
    'GPUUtilization',
    'DeviceType',
    
    # Task Scheduling
    'TaskScheduler',
    'get_scheduler',
    'TaskRequest',
    'TaskResult',
    'TaskStatus',
    'TaskPriority',
    'SchedulingStrategy',
    'submit_inference_task',
]