"""
Response data models for API endpoints
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .common import APIResponse


class ModelInfo(BaseModel):
    """Model information structure"""
    
    name: str = Field(description="Model name")
    type: str = Field(description="Model type (detection, segmentation, etc.)")
    framework: str = Field(description="Framework used (pytorch, ultralytics, etc.)")
    architecture: str = Field(description="Model architecture")
    device: Optional[str] = Field(None, description="Currently assigned device")
    source: str = Field(description="Source of model definition (config, auto_detected)")
    is_loaded: bool = Field(description="Whether model is currently loaded")
    config: Dict[str, Any] = Field(description="Model configuration")
    cache_info: Optional[Dict[str, Any]] = Field(None, description="Cache information")
    detection_info: Optional[Dict[str, Any]] = Field(None, description="Detection metadata")


class ModelInfoResponse(APIResponse):
    """Response for individual model information"""
    
    data: ModelInfo = Field(description="Model information")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Model information for 'yolov8n'",
                "data": {
                    "name": "yolov8n",
                    "type": "detection",
                    "framework": "ultralytics",
                    "architecture": "yolov8n",
                    "device": "cuda:0",
                    "source": "auto_detected",
                    "is_loaded": True,
                    "config": {
                        "path": "/models/yolov8n.pt",
                        "adapter": "ultralytics"
                    }
                }
            }
        }


class ModelListSummary(BaseModel):
    """Summary information for model list"""
    
    total_models: int = Field(description="Total number of models")
    loaded_models: int = Field(description="Number of currently loaded models")
    available_types: List[str] = Field(description="Available model types")
    available_frameworks: List[str] = Field(description="Available frameworks")
    cache_enabled: bool = Field(description="Whether caching is enabled")


class ModelListResponse(APIResponse):
    """Response for model list endpoint"""
    
    data: Dict[str, Any] = Field(description="Model list data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Found 5 models",
                "data": {
                    "models": [
                        {
                            "name": "yolov8n",
                            "type": "detection",
                            "framework": "ultralytics",
                            "is_loaded": True
                        }
                    ],
                    "summary": {
                        "total_models": 5,
                        "loaded_models": 2,
                        "available_types": ["detection", "segmentation"],
                        "available_frameworks": ["ultralytics", "pytorch"]
                    }
                }
            }
        }


class TaskResult(BaseModel):
    """Task execution result structure"""
    
    task_id: str = Field(description="Unique task identifier")
    status: str = Field(description="Task status")
    result: Optional[Any] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if task failed")
    start_time: Optional[float] = Field(None, description="Task start timestamp")
    end_time: Optional[float] = Field(None, description="Task completion timestamp")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    device_used: Optional[str] = Field(None, description="Device used for execution")
    memory_used_mb: float = Field(default=0, description="Memory used in MB")


class TaskResponse(APIResponse):
    """Response for task-related endpoints"""
    
    data: TaskResult = Field(description="Task result")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Task completed successfully",
                "data": {
                    "task_id": "task_123456",
                    "status": "completed",
                    "result": {"detections": []},
                    "execution_time": 0.15,
                    "device_used": "cuda:0"
                }
            }
        }


class SystemStatus(BaseModel):
    """System status information"""
    
    api_version: str = Field(description="API version")
    uptime_seconds: float = Field(description="API uptime in seconds")
    total_managers: int = Field(description="Total number of managers")
    healthy_managers: int = Field(description="Number of healthy managers")
    models_available: int = Field(description="Number of available models")
    models_cached: int = Field(description="Number of cached models")
    active_tasks: int = Field(description="Number of active tasks")
    completed_tasks: int = Field(description="Number of completed tasks")
    gpu_devices: int = Field(description="Number of available GPU devices")
    cache_size_mb: float = Field(description="Current cache size in MB")


class SystemStatusResponse(APIResponse):
    """Response for system status endpoint"""
    
    data: SystemStatus = Field(description="System status")


class ComponentHealth(BaseModel):
    """Health information for individual components"""
    
    status: str = Field(description="Component health status")
    message: str = Field(description="Health status message")
    details: Dict[str, Any] = Field(description="Detailed health information")


class HealthCheckResponse(APIResponse):
    """Response for health check endpoint"""
    
    data: Dict[str, Any] = Field(description="Health check data")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "All systems operational",
                "data": {
                    "overall_status": "healthy",
                    "components": {
                        "ModelManager": {
                            "status": "healthy",
                            "message": "5 models available"
                        }
                    },
                    "metrics": {
                        "uptime": 3600,
                        "total_requests": 1000,
                        "error_rate": 0.01
                    }
                }
            }
        }