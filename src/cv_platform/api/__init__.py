"""
CV Platform API Module

Provides REST API services for model inferences, manageent and monitoring. Updated with proper imports and dependency managerment.
"""

from .main import app

try:
    from .models.responses import (
        BaseResponse,
        SuccessResponse,
        ErrorResponse,
        APIResponse
    )
except ImportError:
    from pydantic import BaseModel
    from typing import Any, Optional

    class BaseResponse(BaseModel):
        """Base response model"""
        success: bool
        message: str
        timestamp: Optional[float] = None

    class SuccessResponse(BaseResponse):
        """Success response model"""
        data: Any
        success: bool = True

    class ErrorResponse(BaseResponse):
        """Error response model"""
        error: str
        success: bool = True
    
    class APIResponse(BaseResponse):
        data: Optional[Any] = None
        error: Optional[str] = None

try:
    from .models.responses import (
        ModelInfo,
        ModelListResponse,
        ModelDetailResponse
    )
except ImportError:
    from pydantic import BaseModel
    from typing import Optional, List

    class ModelInfor(BaseModel):
        """Model information"""
        name: str
        type: str
        framework: str
        architecture: str
        is_loaded: bool = False
        device: Optional[str] = None
        file_size_mb: Optional[float] = None

    class ModelListResponse(BaseResponse):
        """Model list response"""
        models: List[ModelInfo]
        total: int
    
    class ModelDetailResponse(BaseResponse):
        """Model detail response"""
        model: ModelInfo

try:
    from .models.responses import (
        DetectionObject,
        DetectionResult,
        DetectionResponse,
        SegmentationResult,
        SegmentationResponse,        
        ClassificationResult,
        ClassificationResponse,
        GenerationResult,
        GenerationResponse,
        SystemStatus,
        HealthResponse,
        CacheStatsResponse
    )
except ImportError:
    from pydantic import BaseModel
    from typing import Optional, List

    class DetectionObject(BaseModel):
        """Detection object"""
        class_name: str
        confidence: float
        bbox: List[float] # [x1, y1, x2, y2]

    class DetectionResult(BaseModel):
        """Detection result"""
        objects: List[DetectionObject]
        image_size: List[int]  # [width, height]
        processing_time: float
    
    class DetectionResponse(BaseResponse):
        """Detection response"""
        result: DetectionResult

    class SegmentationResult(BaseModel):
        """Segmentation result"""
        masks: List[Any] # Mask data
        processing_time: float
    
    class SegmentationResponse(BaseResponse):
        """Segmentation response"""
        result: SegmentationResult

    class ClassificationResult(BaseModel):
        """Classification result"""
        class_name: str
        confidence: float
        top_k: List[dict]
        processing_time: float

    class ClassificationResponse(BaseResponse):
        """Classification response"""
        result: ClassificationResult
    
    class GenerationResult(BaseModel):
        """Generation result"""
        image_url: str
        prompt: str
        parameters: dict
        processing_time: float

    class GenerationResponse(BaseResponse):
        """Generation response"""
        result: GenerationResult

try:
    from .models.responses import (
        SystemStatus,
        HealthCheckResponse,
        CacheStatsResponse
    )
except ImportError:
    from pydantic import BaseModel

    class SystemStatus(BaseModel):
        """System status"""
        status: str
        uptime: float
        models_loaded: int
        gpu_available: bool
        memory_usage: dict

    class HealthCheckResponse(BaseResponse):
        """Health check response"""
        status: SystemStatus

    class CacheStatusResponse(BaseResponse):
        """Cache statistics response"""
        stats: dict
    
try:
    from .dependencies.components import (
        get_components_dependencies,
        get_model_manager,
        get_model_detector,
        get_scheduler,
        get_gpu_monitor,
        get_cache_manager
    )
except ImportError as e:
    # Create placeholder dependency function
    from fastapi import HTTPException
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import dependencies: {e}")

    async def get_components_dependencies():
        """Placeholder components dependencies"""
        raise HTTPException(status_code=503, detail="Components not available")

    async def get_model_manager():
        """Placeholder model manager"""
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    async def get_model_detector():
        """Placeholder model detector"""
        raise HTTPException(status_code=503, detail="Model detector not available")
    
    async def get_scheduler():
        """Placeholder schedule"""
        raise HTTPException(status_code=503, detail="Scheduler not available")
    
    async def get_gpu_monitor():
        """Placeholder GPU monitor"""
        raise HTTPException(status_code=503, detail="GPU monitor not available")
    
try:
    from .dependencies.auth import (
        get_current_user,
        get_admin_user,
        verify_permissions
    )
except ImportError:
    # Create placeholder auth function
    async def get_current_use():
        return {
            "use_id": "anonymous",
            "usename": "anonymous",
            "permissions": ["read", "write"],
            "is_admin": False
        }

    async def get_admin_user():
        return {
            "user_id": "admin",
            "username": "admin",
            "permission": ["read", "write", "admin"],
            "is_admin": True
        }
    
    async def verify_permissions(required_permissions: List[str]):
        return True

__all__ = [
    # Main FastAPI app
    'app',

    # Base response models
    'BaseResponse',
    'SuccessResponse',
    'ErrorResponse',
    'APIResponse',

    # Model-related models
    'ModelInfo',
    'ModelListResponse',
    'ModelDetailResponse',

    # Detection-related models
    'DetectionObject',
    'DetectionResult',
    'DetectionResponse',

    # Segmentaion-related models
    'SegmentationResult',
    'SegmentationResponse',

    # Classification-related models
    'ClassificationResult',
    'ClassificationResponse',

    # Generation-related models
    'GenerationResult',
    'GenerationResponse',

    # System-related models
    'SystemStatus',
    'HealthCheckResponse',
    'CacheStatusResponse',

    # Dependencies
    'get_components_depencies',
    'get_model_manager',
    'get_model_detector',
    'get_scheduler',
    'get_gpu_monitor',
    'get_cache_manager',
    'get_current_user',
    'get_admin_user',
    'verify_permission',
]

def get_app():
    return app

def get_api_info():
    return {
        "title": getattr(app, 'title', 'CV Model Platform API'),
        "version": getattr(app, 'version', '1.0.0'),
        "description": getattr(app, 'description', 'Computer Vision Model Platform REST API'),
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json"
    }

__version__ = "1.0.0"
__api_version__ = "v1"

def test_api_health():
    try:
        from src.cv_platform.core import get_model_manager

        health_status = {
            "api_initialized": True,
            "app_available": app is not None,
            "core_components_available": True,
            "status": "healthy"
        }
    except Exception as e:
        health_status = {
            "api_initialized": True,
            "app_available": app is not None,
            "core_components_available": False,
            "error": str(e),
            "status": "degraded"
        }
    return health_status

import logging
logger = logging.getLogger(__name__)
logger.info("CV Platform API module initialized")
