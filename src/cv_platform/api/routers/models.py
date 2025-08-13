"""
Model Management Router

File location: src/cv_platform/api/routers/models.py

Handles all model-related API endpoints including listing, loading,
unloading, and model information retrieval.
"""

from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends
from loguru import logger

from ..models1.responses import APIResponse, ModelInfoResponse, ModelListResponse
from ..models1.requests import ModelLoadRequest
from ..dependencies.auth import get_current_user
from ..dependencies.components import get_model_manager, get_model_detector, get_cache_manager


# Create router
router = APIRouter()


@router.get("/", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[str] = Query(None, description="Filter by model type (detection, segmentation, etc.)"),
    framework: Optional[str] = Query(None, description="Filter by framework (pytorch, ultralytics, etc.)"),
    include_cached: bool = Query(True, description="Include cache information"),
    current_user: dict = Depends(get_current_user),
    model_manager = Depends(get_model_manager),
    cache_manager = Depends(get_cache_manager)
):
    """
    List all available models with filtering options
    
    Args:
        model_type: Optional model type filter
        framework: Optional framework filter
        include_cached: Whether to include cache information
        current_user: Current authenticated user
        model_manager: Model manager dependency
        cache_manager: Cache manager dependency
        
    Returns:
        List of available models with metadata
    """
    try:
        # Get available models
        available_models = model_manager.list_available_models()
        
        # Get cache information if requested
        cache_stats = cache_manager.get_stats() if include_cached else {}
        cached_models = cache_stats.get('models', {}) if include_cached else {}
        
        # Convert to API format and apply filters
        models_list = []
        for name, model_data in available_models.items():
            config = model_data.get('config', {})
            
            # Apply filters
            if model_type and config.get('type') != model_type:
                continue
            if framework and config.get('framework') != framework:
                continue
            
            # Get cache information for this model
            cache_info = cached_models.get(name, {}) if include_cached else {}
            
            model_info = {
                "name": name,
                "type": config.get('type', 'unknown'),
                "framework": config.get('framework', 'unknown'),
                "architecture": config.get('architecture', 'unknown'),
                "device": config.get('device', 'auto'),
                "source": model_data.get('source', 'unknown'),
                "is_loaded": bool(cache_info),
                "cache_info": {
                    "last_accessed": cache_info.get('last_accessed'),
                    "access_count": cache_info.get('access_count', 0),
                    "age_seconds": cache_info.get('age', 0)
                } if cache_info else None,
                "config": config
            }
            models_list.append(model_info)
        
        # Generate summary statistics
        summary = {
            "total_models": len(models_list),
            "loaded_models": len([m for m in models_list if m["is_loaded"]]),
            "available_types": list(set(m["type"] for m in models_list)),
            "available_frameworks": list(set(m["framework"] for m in models_list)),
            "cache_enabled": include_cached
        }
        
        return ModelListResponse(
            success=True,
            message=f"Found {len(models_list)} models",
            data={
                "models": models_list,
                "summary": summary
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {e}")


@router.get("/{model_name}", response_model=ModelInfoResponse)
async def get_model_details(
    model_name: str,
    include_detection_info: bool = Query(True, description="Include model detection metadata"),
    current_user: dict = Depends(get_current_user),
    model_manager = Depends(get_model_manager),
    model_detector = Depends(get_model_detector),
    cache_manager = Depends(get_cache_manager)
):
    """
    Get detailed information about a specific model
    
    Args:
        model_name: Name of the model
        include_detection_info: Whether to include detection metadata
        current_user: Current authenticated user
        model_manager: Model manager dependency
        model_detector: Model detector dependency
        cache_manager: Cache manager dependency
        
    Returns:
        Detailed model information
    """
    try:
        # Get basic model information
        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found"
            )
        
        # Get cache information
        cache_stats = cache_manager.get_stats()
        cache_info = cache_stats.get('models', {}).get(model_name, {})
        
        # Get detection information if requested
        detection_info = None
        if include_detection_info:
            detected_model = model_detector.get_model_info(model_name)
            if detected_model:
                detection_info = {
                    "file_path": str(detected_model.path),
                    "file_size_mb": detected_model.file_size_mb,
                    "last_modified": detected_model.last_modified,
                    "detection_confidence": detected_model.confidence,
                    "metadata": detected_model.metadata
                }
        
        config = model_info.get('config', {})
        
        detailed_info = {
            "name": model_name,
            "type": config.get('type', 'unknown'),
            "framework": config.get('framework', 'unknown'),
            "architecture": config.get('architecture', 'unknown'),
            "device": config.get('device', 'auto'),
            "source": model_info.get('source', 'unknown'),
            "is_loaded": bool(cache_info),
            "config": config,
            "cache_info": {
                "last_accessed": cache_info.get('last_accessed'),
                "access_count": cache_info.get('access_count', 0),
                "age_seconds": cache_info.get('age', 0),
                "timestamp": cache_info.get('timestamp')
            } if cache_info else None,
            "detection_info": detection_info
        }
        
        return ModelInfoResponse(
            success=True,
            message=f"Model information for '{model_name}'",
            data=detailed_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info for '{model_name}': {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model info: {e}"
        )


@router.post("/{model_name}/load", response_model=APIResponse)
async def load_model(
    model_name: str,
    request: Optional[ModelLoadRequest] = None,
    current_user: dict = Depends(get_current_user),
    model_manager = Depends(get_model_manager)
):
    """
    Load a specific model into memory
    
    Args:
        model_name: Name of the model to load
        request: Optional load parameters
        current_user: Current authenticated user
        model_manager: Model manager dependency
        
    Returns:
        Model loading result with performance metrics
    """
    try:
        # Check permissions
        if "write" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=403, 
                detail="Insufficient permissions to load models"
            )
        
        # Check if model exists
        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found"
            )
        
        # Prepare load parameters
        load_kwargs = {}
        if request:
            if request.device:
                load_kwargs['device'] = request.device
            if request.force_reload:
                # Clear from cache first
                model_manager.unload_model(model_name)
            # Add any other parameters from request
            load_kwargs.update(request.additional_params or {})
        
        # Load model with timing
        import time
        start_time = time.time()
        
        adapter = model_manager.load_model(model_name, **load_kwargs)
        
        load_time = time.time() - start_time
        
        # Get updated cache information
        cache_stats = model_manager.get_cache_stats()
        cache_info = cache_stats.get('models', {}).get(model_name, {})
        
        # Get device information from adapter
        device_used = getattr(adapter, 'device', None)
        if hasattr(adapter, 'model') and hasattr(adapter.model, 'device'):
            device_used = str(adapter.model.device)
        
        return APIResponse(
            success=True,
            message=f"Model '{model_name}' loaded successfully",
            data={
                "model_name": model_name,
                "adapter_type": type(adapter).__name__,
                "load_time_seconds": round(load_time, 3),
                "device_used": device_used,
                "cache_info": cache_info,
                "memory_usage": getattr(adapter, 'memory_usage_mb', None)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to load model: {e}"
        )


@router.post("/{model_name}/unload", response_model=APIResponse)
async def unload_model(
    model_name: str,
    current_user: dict = Depends(get_current_user),
    model_manager = Depends(get_model_manager)
):
    """
    Unload a specific model from memory
    
    Args:
        model_name: Name of the model to unload
        current_user: Current authenticated user
        model_manager: Model manager dependency
        
    Returns:
        Model unloading result
    """
    try:
        # Check permissions
        if "write" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=403, 
                detail="Insufficient permissions to unload models"
            )
        
        # Get cache info before unloading
        cache_stats = model_manager.get_cache_stats()
        was_loaded = model_name in cache_stats.get('models', {})
        
        # Unload model
        success = model_manager.unload_model(model_name)
        
        if success or was_loaded:
            return APIResponse(
                success=True,
                message=f"Model '{model_name}' unloaded successfully",
                data={
                    "model_name": model_name,
                    "was_loaded": was_loaded,
                    "unloaded": success
                }
            )
        else:
            return APIResponse(
                success=False,
                message=f"Model '{model_name}' was not loaded",
                data={
                    "model_name": model_name,
                    "was_loaded": False,
                    "unloaded": False
                }
            )
        
    except Exception as e:
        logger.error(f"Failed to unload model '{model_name}': {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to unload model: {e}"
        )


@router.post("/{model_name}/reload", response_model=APIResponse)
async def reload_model(
    model_name: str,
    request: Optional[ModelLoadRequest] = None,
    current_user: dict = Depends(get_current_user),
    model_manager = Depends(get_model_manager)
):
    """
    Reload a model (unload then load)
    
    Args:
        model_name: Name of the model to reload
        request: Optional load parameters
        current_user: Current authenticated user
        model_manager: Model manager dependency
        
    Returns:
        Model reload result
    """
    try:
        # Check permissions
        if "write" not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=403, 
                detail="Insufficient permissions to reload models"
            )
        
        # Check if model exists
        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found"
            )
        
        # Unload first
        was_loaded = model_manager.unload_model(model_name)
        
        # Prepare load parameters
        load_kwargs = {}
        if request:
            if request.device:
                load_kwargs['device'] = request.device
            load_kwargs.update(request.additional_params or {})
        
        # Load with timing
        import time
        start_time = time.time()
        
        adapter = model_manager.load_model(model_name, **load_kwargs)
        
        load_time = time.time() - start_time
        
        return APIResponse(
            success=True,
            message=f"Model '{model_name}' reloaded successfully",
            data={
                "model_name": model_name,
                "was_previously_loaded": was_loaded,
                "adapter_type": type(adapter).__name__,
                "reload_time_seconds": round(load_time, 3),
                "device_used": getattr(adapter, 'device', None)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload model '{model_name}': {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to reload model: {e}"
        )


@router.get("/{model_name}/status", response_model=APIResponse)
async def get_model_status(
    model_name: str,
    current_user: dict = Depends(get_current_user),
    model_manager = Depends(get_model_manager),
    cache_manager = Depends(get_cache_manager)
):
    """
    Get current status of a specific model
    
    Args:
        model_name: Name of the model
        current_user: Current authenticated user
        model_manager: Model manager dependency
        cache_manager: Cache manager dependency
        
    Returns:
        Current model status and metrics
    """
    try:
        # Check if model exists
        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            raise HTTPException(
                status_code=404, 
                detail=f"Model '{model_name}' not found"
            )
        
        # Get cache information
        cache_stats = cache_manager.get_stats()
        cache_info = cache_stats.get('models', {}).get(model_name, {})
        
        # Get manager metrics for this model
        manager_metrics = model_manager.metrics
        model_metrics = {
            key: value for key, value in manager_metrics.items()
            if model_name in str(key).lower()
        }
        
        status_info = {
            "model_name": model_name,
            "is_loaded": bool(cache_info),
            "load_status": "loaded" if cache_info else "not_loaded",
            "cache_info": cache_info,
            "metrics": model_metrics,
            "health_status": "healthy" if cache_info else "unloaded"
        }
        
        return APIResponse(
            success=True,
            message=f"Status for model '{model_name}'",
            data=status_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model status for '{model_name}': {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model status: {e}"
        )