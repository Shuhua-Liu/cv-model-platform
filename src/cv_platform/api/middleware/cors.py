"""
CORS Middleware Configuration

Enhanced CORS handling with environment-based configuration.
"""

from typing import List, Optional
import os
from fastapi.middleware.cors import CORSMiddleware


def get_cors_middleware_config() -> dict:
    """
    Get CORS middleware configuration from environment
    
    Returns:
        Dictionary of CORS configuration parameters
    """
    # Default development configuration
    default_origins = ["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"]
    
    # Get allowed origins from environment
    origins_env = os.getenv("CORS_ORIGINS", "")
    if origins_env:
        allowed_origins = [origin.strip() for origin in origins_env.split(",")]
    else:
        allowed_origins = default_origins
    
    # In development, allow all origins
    if os.getenv("ENVIRONMENT", "development") == "development":
        allowed_origins = ["*"]
    
    return {
        "allow_origins": allowed_origins,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "Accept",
            "Origin",
            "User-Agent",
            "Cache-Control",
            "Pragma"
        ],
        "expose_headers": [
            "X-Request-ID",
            "X-Processing-Time",
            "X-Rate-Limit-Limit",
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset"
        ]
    }


def add_cors_middleware(app):
    """
    Add CORS middleware to FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    config = get_cors_middleware_config()
    
    app.add_middleware(
        CORSMiddleware,
        **config
    )
    
    return app