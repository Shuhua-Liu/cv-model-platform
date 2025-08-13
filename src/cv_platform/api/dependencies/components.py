"""
Component Dependencies

Provides access to core platform components through dependency injection.
"""

from typing import Optional
from fastapi import HTTPException, Depends
from loguru import logger


def get_app_state():
    """Get application state from main module"""
    try:
        from ..main import get_app_state
        return get_app_state()
    except ImportError:
        logger.error("Failed to import app state")
        raise HTTPException(
            status_code=503,
            detail="Application state not available"
        )


def get_component(component_name: str):
    """Get a specific component from app state"""
    try:
        from ..main import get_component
        component = get_component(component_name)
        if component is None:
            raise HTTPException(
                status_code=503,
                detail=f"Component '{component_name}' not available"
            )
        return component
    except ImportError:
        logger.error(f"Failed to get component: {component_name}")
        raise HTTPException(
            status_code=503,
            detail=f"Component '{component_name}' not available"
        )


async def get_model_manager():
    """
    Get model manager component
    
    Returns:
        ModelManager instance
        
    Raises:
        HTTPException: If component not available
    """
    return get_component("model_manager")


async def get_model_detector():
    """
    Get model detector component
    
    Returns:
        ModelDetector instance
        
    Raises:
        HTTPException: If component not available
    """
    return get_component("model_detector")


async def get_scheduler():
    """
    Get task scheduler component
    
    Returns:
        TaskScheduler instance
        
    Raises:
        HTTPException: If component not available
    """
    return get_component("scheduler")


async def get_gpu_monitor():
    """
    Get GPU monitor component
    
    Returns:
        GPUMonitor instance
        
    Raises:
        HTTPException: If component not available
    """
    return get_component("gpu_monitor")


async def get_cache_manager():
    """
    Get cache manager component
    
    Returns:
        CacheManager instance
        
    Raises:
        HTTPException: If component not available
    """
    return get_component("cache_manager")


async def get_manager_registry():
    """
    Get manager registry
    
    Returns:
        ManagerRegistry instance
        
    Raises:
        HTTPException: If component not available
    """
    app_state = get_app_state()
    registry = app_state.get("manager_registry")
    if registry is None:
        raise HTTPException(
            status_code=503,
            detail="Manager registry not available"
        )
    return registry


class ComponentDependencies:
    """Container for all component dependencies"""
    
    def __init__(
        self,
        model_manager,
        model_detector,
        scheduler,
        gpu_monitor,
        cache_manager,
        manager_registry
    ):
        self.model_manager = model_manager
        self.model_detector = model_detector
        self.scheduler = scheduler
        self.gpu_monitor = gpu_monitor
        self.cache_manager = cache_manager
        self.manager_registry = manager_registry


async def get_components_dependencies() -> ComponentDependencies:
    """
    Get all core components as a single dependency
    
    Returns:
        ComponentDependencies container
    """
    try:
        return ComponentDependencies(
            model_manager=await get_model_manager(),
            model_detector=await get_model_detector(),
            scheduler=await get_scheduler(),
            gpu_monitor=await get_gpu_monitor(),
            cache_manager=await get_cache_manager(),
            manager_registry=await get_manager_registry()
        )
    except Exception as e:
        logger.error(f"Failed to get component dependencies: {e}")
        raise HTTPException(
            status_code=503,
            detail="Core components not available"
        )