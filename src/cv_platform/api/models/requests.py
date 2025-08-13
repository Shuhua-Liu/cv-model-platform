"""
Request data models for API endpoints
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator


class ModelLoadRequest(BaseModel):
    """Request model for loading models"""
    
    device: Optional[str] = Field(
        None,
        description="Device to load model on (e.g., 'cuda:0', 'cpu', 'auto')"
    )
    force_reload: bool = Field(
        False,
        description="Force reload even if model is already loaded"
    )
    cache_enabled: bool = Field(
        True,
        description="Whether to cache the loaded model"
    )
    additional_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional parameters to pass to model loader"
    )
    
    @validator('device')
    def validate_device(cls, v):
        if v is not None:
            valid_devices = ['auto', 'cpu', 'mps']
            if not (v in valid_devices or v.startswith('cuda:')):
                raise ValueError(
                    f"Device must be one of {valid_devices} or 'cuda:N' format"
                )
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "device": "cuda:0",
                "force_reload": False,
                "cache_enabled": True,
                "additional_params": {
                    "precision": "fp16",
                    "optimization": "tensorrt"
                }
            }
        }


class TaskSubmissionRequest(BaseModel):
    """Request model for submitting inference tasks"""
    
    model_name: str = Field(description="Name of the model to use")
    method: str = Field(
        default="predict",
        description="Method to call on the model adapter"
    )
    inputs: Dict[str, Any] = Field(
        description="Input data for the inference task"
    )
    priority: str = Field(
        default="normal",
        pattern="^(low|normal|high|critical)$",
        description="Task priority level"
    )
    timeout: Optional[float] = Field(
        None,
        gt=0,
        description="Task timeout in seconds"
    )
    device_preference: Optional[str] = Field(
        None,
        description="Preferred device for execution"
    )
    memory_requirement_mb: float = Field(
        default=0,
        ge=0,
        description="Estimated memory requirement in MB"
    )
    callback_url: Optional[str] = Field(
        None,
        description="URL to send results to when task completes"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata"
    )
    
    @validator('inputs')
    def validate_inputs(cls, v):
        if not v:
            raise ValueError("Inputs cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "yolov8n",
                "method": "predict",
                "inputs": {
                    "image": "base64_encoded_image_data",
                    "confidence": 0.5,
                    "iou_threshold": 0.45
                },
                "priority": "normal",
                "timeout": 30.0,
                "device_preference": "cuda:0",
                "memory_requirement_mb": 500,
                "metadata": {
                    "source": "api",
                    "user_id": "12345"
                }
            }
        }


class FileUploadMetadata(BaseModel):
    """Metadata for file uploads"""
    
    filename: str = Field(description="Original filename")
    content_type: str = Field(description="MIME content type")
    description: Optional[str] = Field(None, description="File description")
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing the file"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional file metadata"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "test_image.jpg",
                "content_type": "image/jpeg",
                "description": "Test image for object detection",
                "tags": ["test", "detection", "demo"],
                "metadata": {
                    "width": 640,
                    "height": 480,
                    "channels": 3
                }
            }
        }