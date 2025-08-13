"""
Dependency injection package for API
"""

from .auth import get_current_user, get_admin_user, verify_permissions
from .components import (
    get_model_manager, get_model_detector, get_scheduler,
    get_gpu_monitor, get_cache_manager, get_components_dependencies
)

__all__ = [
    "get_current_user",
    "get_admin_user", 
    "verify_permissions",
    "get_model_manager",
    "get_model_detector",
    "get_scheduler",
    "get_gpu_monitor",
    "get_cache_manager",
    "get_components_dependencies"
]