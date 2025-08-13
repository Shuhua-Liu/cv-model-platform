"""
CV Model Platform API - Main Entry Point

File location: src/cv_platform/api/main.py

Main FastAPI application with modular router structure and 
comprehensive middleware integration.
"""

import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import core components
    from src.cv_platform.core import (
        get_model_manager, get_model_detector, get_scheduler,
        get_gpu_monitor, get_cache_manager, get_manager_registry,
        SchedulingStrategy
    )
    from src.cv_platform.utils.logger import setup_logger
    
    # Import API components
    from .routers import models, tasks, files, monitoring, websocket
    from .middleware.logging import RequestLoggingMiddleware
    from .middleware.auth import AuthenticationMiddleware
    from .dependencies.components import get_components_dependencies
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Global application state
app_state = {
    "start_time": None,
    "components": {},
    "temp_dir": Path("temp_api_files")
}

# Ensure temp directory exists
app_state["temp_dir"].mkdir(exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle management
    
    Handles startup and shutdown of all core components
    """
    # Startup
    logger.info("🚀 Starting CV Model Platform API...")
    
    try:
        # Initialize components in correct order
        logger.info("Initializing core components...")
        
        # Get manager registry
        manager_registry = get_manager_registry()
        
        # Initialize components
        components = {
            "gpu_monitor": get_gpu_monitor(),
            "cache_manager": get_cache_manager(
                max_size_bytes=2*1024*1024*1024,  # 2GB
                persistence_dir=app_state["temp_dir"] / "cache"
            ),
            "model_detector": get_model_detector(),
            "model_manager": get_model_manager(),
        }
        
        # Register all components
        for name, component in components.items():
            manager_registry.register(component)
        
        # Add scheduler after model_manager is ready
        components["scheduler"] = get_scheduler(
            model_manager=components["model_manager"],
            strategy=SchedulingStrategy.RESOURCE_AWARE
        )
        manager_registry.register(components["scheduler"])
        
        # Start all components
        start_results = manager_registry.start_all()
        
        failed_components = [name for name, success in start_results.items() if not success]
        if failed_components:
            logger.error(f"Failed to start components: {failed_components}")
            raise RuntimeError(f"Failed to start components: {failed_components}")
        
        # Store components in app state
        app_state["components"] = components
        app_state["manager_registry"] = manager_registry
        app_state["start_time"] = time.time()
        
        logger.info("✅ All components started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down CV Model Platform API...")
    
    try:
        if "manager_registry" in app_state:
            app_state["manager_registry"].stop_all()
        logger.info("✅ All components stopped gracefully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="CV Model Platform API",
    description="Unified computer vision model service API with advanced scheduling and monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Configure middleware (order matters!)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(AuthenticationMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory=str(app_state["temp_dir"])), name="static")

# Include routers with prefixes
app.include_router(
    models.router,
    prefix="/api/v1/models",
    tags=["models"],
    dependencies=[Depends(get_components_dependencies)]
)

app.include_router(
    tasks.router,
    prefix="/api/v1/tasks",
    tags=["tasks"],
    dependencies=[Depends(get_components_dependencies)]
)

app.include_router(
    files.router,
    prefix="/api/v1/files",
    tags=["files"],
    dependencies=[Depends(get_components_dependencies)]
)

app.include_router(
    monitoring.router,
    prefix="/api/v1/monitor",
    tags=["monitoring"],
    dependencies=[Depends(get_components_dependencies)]
)

app.include_router(
    websocket.router,
    prefix="/ws",
    tags=["websocket"]
)

# Root endpoints
@app.get("/")
async def root():
    """API root endpoint with basic information"""
    return {
        "message": "CV Model Platform API",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": time.time() - app_state["start_time"] if app_state["start_time"] else 0,
        "docs": "/docs",
        "health": "/api/v1/monitor/health"
    }


@app.get("/version")
async def get_version():
    """Get API version information"""
    return {
        "api_version": "1.0.0",
        "platform_version": "0.1.0",
        "python_version": sys.version,
        "build_time": "2024-01-01T00:00:00Z"  # Could be set during build
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc),
            "timestamp": time.time()
        }
    )


# Application state access functions
def get_app_state():
    """Get application state for use in other modules"""
    return app_state


def get_component(component_name: str):
    """Get a specific component from app state"""
    return app_state["components"].get(component_name)


# Development server runner
if __name__ == "__main__":
    # Setup logging
    setup_logger("INFO")
    
    # Configuration from environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    workers = int(os.getenv("API_WORKERS", "1"))
    
    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Reload: {reload}, Workers: {workers}")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Single worker for reload mode
        log_level="info",
        access_log=True
    )