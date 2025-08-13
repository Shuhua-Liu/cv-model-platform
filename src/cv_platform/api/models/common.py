"""
Common API data models shared across endpoints
"""

import time
import uuid
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """Standard API response wrapper"""
    
    success: bool = Field(description="Whether the request was successful")
    message: str = Field(description="Human-readable response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data payload")
    error: Optional[str] = Field(None, description="Error message if request failed")
    timestamp: float = Field(default_factory=time.time, description="Response timestamp")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Operation completed successfully",
                "data": {"result": "example_data"},
                "error": None,
                "timestamp": 1640995200.0,
                "request_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints"""
    
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of items to return"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of items to skip"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "limit": 50,
                "offset": 0
            }
        }


class FilterParams(BaseModel):
    """Common filtering parameters"""
    
    search: Optional[str] = Field(None, description="Search query string")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field(
        "asc",
        pattern="^(asc|desc)$",
        description="Sort order (asc or desc)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "search": "yolo",
                "sort_by": "name",
                "sort_order": "asc"
            }
        }
