"""
Task Management Router

File location: src/cv_platform/api/routers/tasks.py

Handles all task-related API endpoints including submission, status checking,
cancellation, and result retrieval with comprehensive error handling.
"""

import asyncio
import json
import base64
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from loguru import logger
import numpy as np
from PIL import Image
import io

from ..models.responses import APIResponse, TaskResponse
from ..models.requests import TaskSubmissionRequest
from ..dependencies.auth import get_current_user, verify_permissions
from ..dependencies.components import get_scheduler, get_model_manager, get_gpu_monitor

# Import core components for type hints
try:
    from src.cv_platform.core import TaskPriority, TaskStatus, SchedulingStrategy
except ImportError:
    logger.warning("Core components not available for type hints")


# Create router
router = APIRouter()


@router.post("/submit", response_model=APIResponse)
async def submit_inference_task(
    request: TaskSubmissionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    scheduler = Depends(get_scheduler),
    model_manager = Depends(get_model_manager)
):
    """
    Submit a new inference task to the scheduler
    
    Args:
        request: Task submission request with model name, inputs, and parameters
        background_tasks: FastAPI background tasks for async processing
        current_user: Current authenticated user
        scheduler: Task scheduler dependency
        model_manager: Model manager dependency
        
    Returns:
        Task submission result with task ID and estimated processing time
    """
    try:
        # Validate model exists
        model_info = model_manager.get_model_info(request.model_name)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model_name}' not found"
            )
        
        # Convert priority string to enum
        priority_map = {
            'low': TaskPriority.LOW,
            'normal': TaskPriority.NORMAL,
            'high': TaskPriority.HIGH,
            'critical': TaskPriority.CRITICAL
        }
        priority = priority_map.get(request.priority, TaskPriority.NORMAL)
        
        # Process input data
        processed_inputs = await _process_task_inputs(request.inputs)
        
        # Add user information to metadata
        task_metadata = request.metadata.copy()
        task_metadata.update({
            'user_id': current_user.get('user_id'),
            'submitted_at': time.time(),
            'api_version': '1.0.0'
        })
        
        # Submit task to scheduler
        task_id = scheduler.submit_task(
            model_name=request.model_name,
            method=request.method,
            args=(),  # Pass inputs as kwargs instead
            kwargs=processed_inputs,
            priority=priority,
            timeout=request.timeout,
            device_preference=request.device_preference,
            memory_requirement_mb=request.memory_requirement_mb,
            metadata=task_metadata
        )
        
        # Get queue status for estimation
        queue_status = scheduler.get_queue_status()
        estimated_wait_time = _estimate_wait_time(queue_status, priority)
        
        # Schedule callback if provided
        if request.callback_url:
            background_tasks.add_task(
                _schedule_result_callback,
                task_id,
                request.callback_url,
                scheduler
            )
        
        return APIResponse(
            success=True,
            message=f"Task submitted successfully",
            data={
                "task_id": task_id,
                "model_name": request.model_name,
                "method": request.method,
                "priority": request.priority,
                "estimated_wait_time_seconds": estimated_wait_time,
                "queue_position": queue_status.get('queue_size', 0) + 1,
                "callback_scheduled": bool(request.callback_url)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit task: {e}"
        )


@router.post("/submit/image", response_model=APIResponse)
async def submit_image_inference_task(
    model_name: str = Form(..., description="Name of the model to use"),
    method: str = Form(default="predict", description="Method to call on model"),
    priority: str = Form(default="normal", regex="^(low|normal|high|critical)$"),
    confidence: float = Form(default=0.5, ge=0.0, le=1.0, description="Confidence threshold"),
    timeout: Optional[float] = Form(None, gt=0, description="Task timeout in seconds"),
    device_preference: Optional[str] = Form(None, description="Preferred device"),
    image: UploadFile = File(..., description="Image file for inference"),
    additional_params: Optional[str] = Form(None, description="Additional parameters as JSON"),
    current_user: dict = Depends(get_current_user),
    scheduler = Depends(get_scheduler),
    model_manager = Depends(get_model_manager)
):
    """
    Submit image inference task with file upload
    
    Args:
        model_name: Name of the model to use
        method: Method to call on model adapter
        priority: Task priority level
        confidence: Confidence threshold for detection/classification
        timeout: Task timeout in seconds
        device_preference: Preferred execution device
        image: Image file for inference
        additional_params: Additional parameters as JSON string
        current_user: Current authenticated user
        scheduler: Task scheduler dependency
        model_manager: Model manager dependency
        
    Returns:
        Task submission result
    """
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {image.content_type}. Only image files are allowed."
            )
        
        # Read and process image
        image_data = await image.read()
        processed_image = await _process_image_upload(image_data, image.content_type)
        
        # Parse additional parameters
        extra_params = {}
        if additional_params:
            try:
                extra_params = json.loads(additional_params)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON format in additional_params"
                )
        
        # Prepare task inputs
        task_inputs = {
            'image': processed_image,
            'confidence': confidence,
            **extra_params
        }
        
        # Create task submission request
        task_request = TaskSubmissionRequest(
            model_name=model_name,
            method=method,
            inputs=task_inputs,
            priority=priority,
            timeout=timeout,
            device_preference=device_preference,
            metadata={
                'input_type': 'image_upload',
                'filename': image.filename,
                'content_type': image.content_type,
                'file_size_bytes': len(image_data)
            }
        )
        
        # Submit task using the main submission endpoint logic
        return await submit_inference_task(
            request=task_request,
            background_tasks=BackgroundTasks(),
            current_user=current_user,
            scheduler=scheduler,
            model_manager=model_manager
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit image inference task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit image task: {e}"
        )


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task_result(
    task_id: str,
    timeout: Optional[float] = Query(None, gt=0, description="Timeout in seconds to wait for result"),
    include_result_data: bool = Query(True, description="Whether to include full result data"),
    current_user: dict = Depends(get_current_user),
    scheduler = Depends(get_scheduler)
):
    """
    Get task execution result by task ID
    
    Args:
        task_id: Unique task identifier
        timeout: Maximum time to wait for result (None for immediate return)
        include_result_data: Whether to include full result data in response
        current_user: Current authenticated user
        scheduler: Task scheduler dependency
        
    Returns:
        Task execution result with status and data
    """
    try:
        # Get task result from scheduler
        result = scheduler.get_task_result(task_id, timeout=timeout)
        
        if result is None:
            # Check if task exists in queue or is unknown
            queue_status = scheduler.get_queue_status()
            if task_id in queue_status.get('running_tasks', []):
                status = "running"
                message = f"Task '{task_id}' is still running"
            else:
                status = "not_found"
                message = f"Task '{task_id}' not found or timed out"
            
            return TaskResponse(
                success=False,
                message=message,
                data={
                    "task_id": task_id,
                    "status": status,
                    "result": None,
                    "error": None if status == "running" else "Task not found",
                    "execution_time": None,
                    "device_used": None
                }
            )
        
        # Process result data
        result_data = result.result
        if not include_result_data and result_data:
            # Provide summary instead of full data
            if isinstance(result_data, dict):
                result_data = {
                    "summary": f"Result contains {len(result_data)} fields",
                    "keys": list(result_data.keys()) if len(result_data) < 10 else list(result_data.keys())[:10]
                }
            elif isinstance(result_data, (list, tuple)):
                result_data = {
                    "summary": f"Result contains {len(result_data)} items",
                    "length": len(result_data)
                }
            else:
                result_data = {"summary": f"Result type: {type(result_data).__name__}"}
        
        # Create task response
        task_response_data = {
            "task_id": task_id,
            "status": result.status.value,
            "result": result_data,
            "error": str(result.error) if result.error else None,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "execution_time": result.execution_time,
            "device_used": result.device_used,
            "memory_used_mb": result.memory_used_mb
        }
        
        return TaskResponse(
            success=result.success,
            message=f"Task result for '{task_id}'" + (f" - {result.status.value}" if not result.success else ""),
            data=task_response_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get task result '{task_id}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task result: {e}"
        )


@router.delete("/{task_id}", response_model=APIResponse)
async def cancel_task(
    task_id: str,
    current_user: dict = Depends(verify_permissions(["write"])),
    scheduler = Depends(get_scheduler)
):
    """
    Cancel a pending or running task
    
    Args:
        task_id: Task identifier to cancel
        current_user: Current authenticated user with write permissions
        scheduler: Task scheduler dependency
        
    Returns:
        Task cancellation result
    """
    try:
        # Attempt to cancel task
        success = scheduler.cancel_task(task_id)
        
        if success:
            return APIResponse(
                success=True,
                message=f"Task '{task_id}' cancelled successfully",
                data={
                    "task_id": task_id,
                    "cancelled": True,
                    "cancelled_by": current_user.get('user_id'),
                    "cancelled_at": time.time()
                }
            )
        else:
            return APIResponse(
                success=False,
                message=f"Task '{task_id}' could not be cancelled (may have already completed or not found)",
                data={
                    "task_id": task_id,
                    "cancelled": False,
                    "reason": "Task not found or already completed"
                }
            )
        
    except Exception as e:
        logger.error(f"Failed to cancel task '{task_id}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {e}"
        )


@router.get("/", response_model=APIResponse)
async def list_tasks(
    status: Optional[str] = Query(None, regex="^(pending|running|completed|failed|cancelled)$", description="Filter by task status"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    include_results: bool = Query(False, description="Whether to include task results"),
    current_user: dict = Depends(get_current_user),
    scheduler = Depends(get_scheduler)
):
    """
    List tasks with filtering and pagination
    
    Args:
        status: Optional status filter
        model_name: Optional model name filter
        user_id: Optional user ID filter (admin only)
        limit: Maximum number of results
        offset: Number of results to skip
        include_results: Whether to include full task results
        current_user: Current authenticated user
        scheduler: Task scheduler dependency
        
    Returns:
        Paginated list of tasks
    """
    try:
        # Check permissions for user_id filter
        if user_id and user_id != current_user.get('user_id'):
            if 'admin' not in current_user.get('permissions', []):
                raise HTTPException(
                    status_code=403,
                    detail="Admin permissions required to filter by other users"
                )
        
        # Get queue status for summary
        queue_status = scheduler.get_queue_status()
        
        # Note: This is a simplified implementation
        # In production, you'd want a proper task database with indexing
        tasks_summary = {
            "total_tasks": queue_status.get('total_scheduled', 0),
            "running_tasks": queue_status.get('running_tasks', 0),
            "completed_tasks": queue_status.get('completed_tasks', 0),
            "failed_tasks": queue_status.get('failed_tasks', 0),
            "queue_size": queue_status.get('queue_size', 0),
            "average_wait_time": queue_status.get('average_queue_wait_time', 0)
        }
        
        # Get executor statistics for additional context
        executor_stats = scheduler.get_executor_stats()
        
        return APIResponse(
            success=True,
            message=f"Task list retrieved (showing up to {limit} tasks)",
            data={
                "tasks": [],  # Would be populated from task database
                "summary": tasks_summary,
                "executor_stats": executor_stats,
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": tasks_summary["total_tasks"]
                },
                "filters": {
                    "status": status,
                    "model_name": model_name,
                    "user_id": user_id if 'admin' in current_user.get('permissions', []) else current_user.get('user_id')
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tasks: {e}"
        )


@router.get("/{task_id}/stream")
async def stream_task_progress(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    scheduler = Depends(get_scheduler)
):
    """
    Stream task progress updates via Server-Sent Events (SSE)
    
    Args:
        task_id: Task identifier to stream
        current_user: Current authenticated user
        scheduler: Task scheduler dependency
        
    Returns:
        Server-sent events stream with task progress
    """
    async def generate_progress_stream():
        """Generate progress updates for the task"""
        try:
            # Check if task exists
            result = scheduler.get_task_result(task_id, timeout=0.1)
            if result is None:
                yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
                return
            
            # Stream progress updates
            while True:
                result = scheduler.get_task_result(task_id, timeout=1.0)
                
                if result is None:
                    # Task still pending/running
                    queue_status = scheduler.get_queue_status()
                    progress_data = {
                        "task_id": task_id,
                        "status": "running",
                        "queue_size": queue_status.get('queue_size', 0),
                        "timestamp": time.time()
                    }
                else:
                    # Task completed
                    progress_data = {
                        "task_id": task_id,
                        "status": result.status.value,
                        "success": result.success,
                        "execution_time": result.execution_time,
                        "timestamp": time.time(),
                        "final": True
                    }
                    
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    break
                
                yield f"data: {json.dumps(progress_data)}\n\n"
                await asyncio.sleep(1)  # Update every second
                
        except Exception as e:
            error_data = {
                "error": str(e),
                "timestamp": time.time(),
                "final": True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


# =============================================================================
# Utility Functions
# =============================================================================

async def _process_task_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and validate task inputs
    
    Args:
        inputs: Raw input data from request
        
    Returns:
        Processed inputs ready for model execution
    """
    processed = {}
    
    for key, value in inputs.items():
        if key == 'image' and isinstance(value, str):
            # Handle base64 encoded images
            try:
                if value.startswith('data:image'):
                    # Extract base64 data from data URL
                    header, data = value.split(',', 1)
                    image_data = base64.b64decode(data)
                else:
                    # Assume raw base64
                    image_data = base64.b64decode(value)
                
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_data))
                # Convert to numpy array for model processing
                processed[key] = np.array(image)
                
            except Exception as e:
                logger.error(f"Failed to process image input: {e}")
                raise ValueError(f"Invalid image data: {e}")
        else:
            # Pass through other inputs as-is
            processed[key] = value
    
    return processed


async def _process_image_upload(image_data: bytes, content_type: str) -> np.ndarray:
    """
    Process uploaded image file
    
    Args:
        image_data: Raw image bytes
        content_type: MIME content type
        
    Returns:
        Processed image as numpy array
    """
    try:
        # Open image with PIL
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        return np.array(image)
        
    except Exception as e:
        logger.error(f"Failed to process uploaded image: {e}")
        raise ValueError(f"Invalid image file: {e}")


def _estimate_wait_time(queue_status: Dict[str, Any], priority: 'TaskPriority') -> float:
    """
    Estimate task wait time based on queue status and priority
    
    Args:
        queue_status: Current queue status
        priority: Task priority
        
    Returns:
        Estimated wait time in seconds
    """
    base_wait = queue_status.get('average_queue_wait_time', 10.0)
    queue_size = queue_status.get('queue_size', 0)
    running_tasks = queue_status.get('running_tasks', 0)
    
    # Adjust based on priority
    priority_multipliers = {
        TaskPriority.CRITICAL: 0.1,
        TaskPriority.HIGH: 0.5,
        TaskPriority.NORMAL: 1.0,
        TaskPriority.LOW: 2.0
    }
    
    multiplier = priority_multipliers.get(priority, 1.0)
    estimated_wait = (base_wait * queue_size / max(running_tasks, 1)) * multiplier
    
    return max(0.1, estimated_wait)  # Minimum 0.1 seconds


async def _schedule_result_callback(task_id: str, callback_url: str, scheduler):
    """
    Schedule callback when task completes
    
    Args:
        task_id: Task identifier
        callback_url: URL to POST results to
        scheduler: Task scheduler instance
    """
    try:
        # Wait for task completion
        result = scheduler.get_task_result(task_id, timeout=3600)  # 1 hour max
        
        if result:
            # Prepare callback payload
            payload = {
                "task_id": task_id,
                "status": result.status.value,
                "success": result.success,
                "result": result.result,
                "error": str(result.error) if result.error else None,
                "execution_time": result.execution_time,
                "completed_at": time.time()
            }
            
            # Send callback (would use aiohttp in production)
            logger.info(f"Callback scheduled for task {task_id} to {callback_url}")
            # TODO: Implement actual HTTP POST to callback_url
            
    except Exception as e:
        logger.error(f"Failed to execute callback for task {task_id}: {e}")