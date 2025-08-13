"""
WebSocket Router - Real-time model inference and system monitoring

Provides WebSocket endpoints for:
- Real-time model inference
- System status monitoring
- Task progress tracking
- Live performance metrics
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.websockets import WebSocketState
from loguru import logger
from pydantic import BaseModel, ValidationError

from ..dependencies.components import get_components_dependencies
from ..models.responses import APIResponse


# WebSocket message models
class WSMessage(BaseModel):
    """Base WebSocket message"""
    type: str
    id: Optional[str] = None
    timestamp: Optional[float] = None


class WSRequest(WSMessage):
    """WebSocket request message"""
    data: Dict[str, Any]


class WSResponse(WSMessage):
    """WebSocket response message"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class InferenceRequest(BaseModel):
    """Model inference request"""
    model_name: str
    method: str = "predict"
    image_data: Optional[str] = None  # Base64 encoded image
    image_url: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class MonitoringRequest(BaseModel):
    """System monitoring request"""
    components: List[str] = ["all"]  # system, gpu, cache, models, tasks
    interval: float = 1.0  # Update interval in seconds


# Create router
router = APIRouter()


class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        # Active connections organized by endpoint type
        self.connections: Dict[str, List[WebSocket]] = {
            "inference": [],
            "monitoring": [],
            "tasks": []
        }
        # Connection metadata
        self.connection_data: Dict[WebSocket, Dict[str, Any]] = {}
        # Active monitoring tasks
        self.monitoring_tasks: Dict[WebSocket, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, endpoint_type: str, client_id: str = None):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        
        if endpoint_type not in self.connections:
            self.connections[endpoint_type] = []
        
        self.connections[endpoint_type].append(websocket)
        
        # Store connection metadata
        self.connection_data[websocket] = {
            "type": endpoint_type,
            "client_id": client_id or str(uuid.uuid4()),
            "connected_at": time.time(),
            "last_activity": time.time()
        }
        
        logger.info(f"WebSocket client connected: {endpoint_type} ({client_id})")
        
        # Send welcome message
        await self.send_message(websocket, WSResponse(
            type="connection",
            success=True,
            data={
                "message": f"Connected to {endpoint_type} endpoint",
                "client_id": self.connection_data[websocket]["client_id"],
                "server_time": time.time()
            }
        ))
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        # Stop monitoring task if exists
        if websocket in self.monitoring_tasks:
            self.monitoring_tasks[websocket].cancel()
            del self.monitoring_tasks[websocket]
        
        # Remove from all connection lists
        for endpoint_type, connections in self.connections.items():
            if websocket in connections:
                connections.remove(websocket)
                logger.info(f"WebSocket client disconnected: {endpoint_type}")
                break
        
        # Remove metadata
        if websocket in self.connection_data:
            del self.connection_data[websocket]
    
    async def send_message(self, websocket: WebSocket, message: WSResponse):
        """Send message to a specific WebSocket"""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                # Set timestamp if not provided
                if message.timestamp is None:
                    message.timestamp = time.time()
                
                await websocket.send_text(message.model_dump_json())
                
                # Update last activity
                if websocket in self.connection_data:
                    self.connection_data[websocket]["last_activity"] = time.time()
        
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_type(self, endpoint_type: str, message: WSResponse):
        """Broadcast message to all connections of a specific type"""
        connections = self.connections.get(endpoint_type, [])
        
        # Send to all active connections
        disconnected = []
        for websocket in connections:
            try:
                await self.send_message(websocket, message)
            except:
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            self.disconnect(websocket)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        stats = {
            "total_connections": sum(len(conns) for conns in self.connections.values()),
            "connections_by_type": {
                endpoint_type: len(connections) 
                for endpoint_type, connections in self.connections.items()
            },
            "active_monitoring_tasks": len(self.monitoring_tasks)
        }
        return stats


# Global connection manager
manager = ConnectionManager()


@router.websocket("/inference")
async def websocket_inference(
    websocket: WebSocket,
    client_id: Optional[str] = None,
    components = Depends(get_components_dependencies)
):
    """
    WebSocket endpoint for real-time model inference
    
    Supports:
    - Real-time image processing
    - Multiple model types
    - Streaming results
    """
    await manager.connect(websocket, "inference", client_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                # Parse message
                message_data = json.loads(data)
                request = WSRequest(**message_data)
                
                # Handle different request types
                if request.type == "inference":
                    await handle_inference_request(websocket, request, components)
                elif request.type == "ping":
                    await manager.send_message(websocket, WSResponse(
                        type="pong",
                        success=True,
                        data={"timestamp": time.time()}
                    ))
                else:
                    await manager.send_message(websocket, WSResponse(
                        type="error",
                        success=False,
                        error=f"Unknown request type: {request.type}"
                    ))
            
            except json.JSONDecodeError:
                await manager.send_message(websocket, WSResponse(
                    type="error",
                    success=False,
                    error="Invalid JSON format"
                ))
            
            except ValidationError as e:
                await manager.send_message(websocket, WSResponse(
                    type="error",
                    success=False,
                    error=f"Invalid message format: {e}"
                ))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket inference error: {e}")
        manager.disconnect(websocket)


@router.websocket("/monitoring")
async def websocket_monitoring(
    websocket: WebSocket,
    client_id: Optional[str] = None,
    components = Depends(get_components_dependencies)
):
    """
    WebSocket endpoint for real-time system monitoring
    
    Provides:
    - System metrics
    - GPU utilization
    - Cache statistics
    - Model status
    - Task queue status
    """
    await manager.connect(websocket, "monitoring", client_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                # Parse message
                message_data = json.loads(data)
                request = WSRequest(**message_data)
                
                # Handle different request types
                if request.type == "start_monitoring":
                    await start_monitoring(websocket, request, components)
                elif request.type == "stop_monitoring":
                    await stop_monitoring(websocket)
                elif request.type == "get_status":
                    await send_current_status(websocket, components)
                elif request.type == "ping":
                    await manager.send_message(websocket, WSResponse(
                        type="pong",
                        success=True,
                        data={"timestamp": time.time()}
                    ))
                else:
                    await manager.send_message(websocket, WSResponse(
                        type="error",
                        success=False,
                        error=f"Unknown request type: {request.type}"
                    ))
            
            except json.JSONDecodeError:
                await manager.send_message(websocket, WSResponse(
                    type="error",
                    success=False,
                    error="Invalid JSON format"
                ))
            
            except ValidationError as e:
                await manager.send_message(websocket, WSResponse(
                    type="error",
                    success=False,
                    error=f"Invalid message format: {e}"
                ))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket monitoring error: {e}")
        manager.disconnect(websocket)


@router.websocket("/tasks")
async def websocket_tasks(
    websocket: WebSocket,
    client_id: Optional[str] = None,
    components = Depends(get_components_dependencies)
):
    """
    WebSocket endpoint for real-time task tracking
    
    Provides:
    - Task submission
    - Progress updates
    - Completion notifications
    - Queue status
    """
    await manager.connect(websocket, "tasks", client_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                # Parse message
                message_data = json.loads(data)
                request = WSRequest(**message_data)
                
                # Handle different request types
                if request.type == "submit_task":
                    await handle_task_submission(websocket, request, components)
                elif request.type == "get_task_status":
                    await handle_task_status_request(websocket, request, components)
                elif request.type == "cancel_task":
                    await handle_task_cancellation(websocket, request, components)
                elif request.type == "get_queue_status":
                    await send_queue_status(websocket, components)
                elif request.type == "ping":
                    await manager.send_message(websocket, WSResponse(
                        type="pong",
                        success=True,
                        data={"timestamp": time.time()}
                    ))
                else:
                    await manager.send_message(websocket, WSResponse(
                        type="error",
                        success=False,
                        error=f"Unknown request type: {request.type}"
                    ))
            
            except json.JSONDecodeError:
                await manager.send_message(websocket, WSResponse(
                    type="error",
                    success=False,
                    error="Invalid JSON format"
                ))
            
            except ValidationError as e:
                await manager.send_message(websocket, WSResponse(
                    type="error",
                    success=False,
                    error=f"Invalid message format: {e}"
                ))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket tasks error: {e}")
        manager.disconnect(websocket)


# Handler functions
async def handle_inference_request(websocket: WebSocket, request: WSRequest, components):
    """Handle model inference request"""
    try:
        # Parse inference request
        inference_req = InferenceRequest(**request.data)
        
        # Get components
        model_manager = components.get("model_manager")
        scheduler = components.get("scheduler")
        
        if not model_manager or not scheduler:
            await manager.send_message(websocket, WSResponse(
                type="inference_error",
                success=False,
                error="Model manager or scheduler not available"
            ))
            return
        
        # Send acknowledgment
        await manager.send_message(websocket, WSResponse(
            type="inference_started",
            success=True,
            data={
                "model_name": inference_req.model_name,
                "method": inference_req.method,
                "request_id": request.id
            }
        ))
        
        # Prepare inference data
        inference_data = None
        if inference_req.image_data:
            # Decode base64 image
            import base64
            import io
            from PIL import Image
            import numpy as np
            
            image_bytes = base64.b64decode(inference_req.image_data)
            image = Image.open(io.BytesIO(image_bytes))
            inference_data = np.array(image)
        
        elif inference_req.image_url:
            # Download image from URL
            import requests
            from PIL import Image
            import numpy as np
            
            response = requests.get(inference_req.image_url)
            image = Image.open(io.BytesIO(response.content))
            inference_data = np.array(image)
        
        if inference_data is None:
            await manager.send_message(websocket, WSResponse(
                type="inference_error",
                success=False,
                error="No valid image data provided"
            ))
            return
        
        # Submit task to scheduler
        task_id = scheduler.submit_task(
            model_name=inference_req.model_name,
            method=inference_req.method,
            args=(inference_data,),
            kwargs=inference_req.parameters or {}
        )
        
        # Send task submitted confirmation
        await manager.send_message(websocket, WSResponse(
            type="task_submitted",
            success=True,
            data={
                "task_id": task_id,
                "request_id": request.id
            }
        ))
        
        # Wait for result (with timeout)
        result = scheduler.get_task_result(task_id, timeout=30.0)
        
        if result and result.success:
            # Send successful result
            await manager.send_message(websocket, WSResponse(
                type="inference_complete",
                success=True,
                data={
                    "request_id": request.id,
                    "task_id": task_id,
                    "result": result.result,
                    "execution_time": result.execution_time,
                    "device_used": result.device_used
                }
            ))
        else:
            # Send error result
            await manager.send_message(websocket, WSResponse(
                type="inference_error",
                success=False,
                error=str(result.error) if result else "Task timeout",
                data={
                    "request_id": request.id,
                    "task_id": task_id
                }
            ))
    
    except Exception as e:
        logger.error(f"Inference request error: {e}")
        await manager.send_message(websocket, WSResponse(
            type="inference_error",
            success=False,
            error=str(e),
            data={"request_id": request.id}
        ))


async def start_monitoring(websocket: WebSocket, request: WSRequest, components):
    """Start system monitoring for a WebSocket connection"""
    try:
        # Parse monitoring request
        monitoring_req = MonitoringRequest(**request.data)
        
        # Stop existing monitoring task if any
        if websocket in manager.monitoring_tasks:
            manager.monitoring_tasks[websocket].cancel()
        
        # Start new monitoring task
        task = asyncio.create_task(
            monitoring_loop(websocket, monitoring_req, components)
        )
        manager.monitoring_tasks[websocket] = task
        
        # Send confirmation
        await manager.send_message(websocket, WSResponse(
            type="monitoring_started",
            success=True,
            data={
                "components": monitoring_req.components,
                "interval": monitoring_req.interval
            }
        ))
    
    except Exception as e:
        await manager.send_message(websocket, WSResponse(
            type="error",
            success=False,
            error=f"Failed to start monitoring: {e}"
        ))


async def stop_monitoring(websocket: WebSocket):
    """Stop monitoring for a WebSocket connection"""
    if websocket in manager.monitoring_tasks:
        manager.monitoring_tasks[websocket].cancel()
        del manager.monitoring_tasks[websocket]
        
        await manager.send_message(websocket, WSResponse(
            type="monitoring_stopped",
            success=True,
            data={"message": "Monitoring stopped"}
        ))


async def monitoring_loop(websocket: WebSocket, monitoring_req: MonitoringRequest, components):
    """Continuous monitoring loop"""
    try:
        while True:
            # Collect monitoring data
            monitoring_data = await collect_monitoring_data(monitoring_req.components, components)
            
            # Send monitoring update
            await manager.send_message(websocket, WSResponse(
                type="monitoring_update",
                success=True,
                data=monitoring_data
            ))
            
            # Wait for next update
            await asyncio.sleep(monitoring_req.interval)
    
    except asyncio.CancelledError:
        # Monitoring stopped
        pass
    except Exception as e:
        logger.error(f"Monitoring loop error: {e}")
        await manager.send_message(websocket, WSResponse(
            type="monitoring_error",
            success=False,
            error=str(e)
        ))


async def collect_monitoring_data(components_to_monitor: List[str], components) -> Dict[str, Any]:
    """Collect monitoring data from specified components"""
    data = {
        "timestamp": time.time(),
        "server_time": datetime.now().isoformat()
    }
    
    # Check if we should monitor all components
    monitor_all = "all" in components_to_monitor
    
    try:
        # System metrics
        if monitor_all or "system" in components_to_monitor:
            import psutil
            data["system"] = {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "uptime": time.time() - psutil.boot_time()
            }
        
        # GPU metrics
        if monitor_all or "gpu" in components_to_monitor:
            gpu_monitor = components.get("gpu_monitor")
            if gpu_monitor:
                data["gpu"] = gpu_monitor.get_device_utilization_summary()
        
        # Cache metrics
        if monitor_all or "cache" in components_to_monitor:
            cache_manager = components.get("cache_manager")
            if cache_manager:
                data["cache"] = cache_manager.get_stats()
        
        # Model metrics
        if monitor_all or "models" in components_to_monitor:
            model_manager = components.get("model_manager")
            if model_manager:
                data["models"] = {
                    "available": len(model_manager.list_available_models()),
                    "cache_stats": model_manager.get_cache_stats(),
                    "metrics": model_manager.get_performance_summary()
                }
        
        # Task metrics
        if monitor_all or "tasks" in components_to_monitor:
            scheduler = components.get("scheduler")
            if scheduler:
                data["tasks"] = scheduler.get_queue_status()
        
        # WebSocket connection stats
        if monitor_all or "websocket" in components_to_monitor:
            data["websocket"] = manager.get_connection_stats()
    
    except Exception as e:
        data["error"] = str(e)
        logger.error(f"Error collecting monitoring data: {e}")
    
    return data


async def send_current_status(websocket: WebSocket, components):
    """Send current system status"""
    try:
        status_data = await collect_monitoring_data(["all"], components)
        
        await manager.send_message(websocket, WSResponse(
            type="status_update",
            success=True,
            data=status_data
        ))
    
    except Exception as e:
        await manager.send_message(websocket, WSResponse(
            type="error",
            success=False,
            error=f"Failed to get status: {e}"
        ))


async def handle_task_submission(websocket: WebSocket, request: WSRequest, components):
    """Handle task submission via WebSocket"""
    try:
        scheduler = components.get("scheduler")
        if not scheduler:
            await manager.send_message(websocket, WSResponse(
                type="error",
                success=False,
                error="Scheduler not available"
            ))
            return
        
        # Extract task parameters
        task_params = request.data
        
        # Submit task
        task_id = scheduler.submit_task(**task_params)
        
        # Send confirmation
        await manager.send_message(websocket, WSResponse(
            type="task_submitted",
            success=True,
            data={
                "task_id": task_id,
                "request_id": request.id
            }
        ))
        
        # Start monitoring this task
        asyncio.create_task(monitor_task_progress(websocket, task_id, scheduler))
    
    except Exception as e:
        await manager.send_message(websocket, WSResponse(
            type="error",
            success=False,
            error=f"Failed to submit task: {e}"
        ))


async def monitor_task_progress(websocket: WebSocket, task_id: str, scheduler):
    """Monitor task progress and send updates"""
    try:
        while True:
            result = scheduler.get_task_result(task_id, timeout=1.0)
            
            if result:
                # Task completed
                await manager.send_message(websocket, WSResponse(
                    type="task_completed",
                    success=result.success,
                    data={
                        "task_id": task_id,
                        "result": result.to_dict()
                    }
                ))
                break
            
            # Task still running, send progress update
            await manager.send_message(websocket, WSResponse(
                type="task_progress",
                success=True,
                data={
                    "task_id": task_id,
                    "status": "running",
                    "timestamp": time.time()
                }
            ))
            
            await asyncio.sleep(1.0)
    
    except Exception as e:
        logger.error(f"Task monitoring error: {e}")


async def handle_task_status_request(websocket: WebSocket, request: WSRequest, components):
    """Handle task status request"""
    try:
        task_id = request.data.get("task_id")
        if not task_id:
            await manager.send_message(websocket, WSResponse(
                type="error",
                success=False,
                error="Task ID required"
            ))
            return
        
        scheduler = components.get("scheduler")
        if not scheduler:
            await manager.send_message(websocket, WSResponse(
                type="error",
                success=False,
                error="Scheduler not available"
            ))
            return
        
        # Get task result
        result = scheduler.get_task_result(task_id, timeout=0.1)
        
        if result:
            await manager.send_message(websocket, WSResponse(
                type="task_status",
                success=True,
                data={
                    "task_id": task_id,
                    "result": result.to_dict()
                }
            ))
        else:
            await manager.send_message(websocket, WSResponse(
                type="task_status",
                success=True,
                data={
                    "task_id": task_id,
                    "status": "running"
                }
            ))
    
    except Exception as e:
        await manager.send_message(websocket, WSResponse(
            type="error",
            success=False,
            error=f"Failed to get task status: {e}"
        ))


async def handle_task_cancellation(websocket: WebSocket, request: WSRequest, components):
    """Handle task cancellation request"""
    try:
        task_id = request.data.get("task_id")
        if not task_id:
            await manager.send_message(websocket, WSResponse(
                type="error",
                success=False,
                error="Task ID required"
            ))
            return
        
        scheduler = components.get("scheduler")
        if not scheduler:
            await manager.send_message(websocket, WSResponse(
                type="error",
                success=False,
                error="Scheduler not available"
            ))
            return
        
        # Cancel task
        cancelled = scheduler.cancel_task(task_id)
        
        await manager.send_message(websocket, WSResponse(
            type="task_cancelled" if cancelled else "task_cancel_failed",
            success=cancelled,
            data={
                "task_id": task_id,
                "cancelled": cancelled
            }
        ))
    
    except Exception as e:
        await manager.send_message(websocket, WSResponse(
            type="error",
            success=False,
            error=f"Failed to cancel task: {e}"
        ))


async def send_queue_status(websocket: WebSocket, components):
    """Send current queue status"""
    try:
        scheduler = components.get("scheduler")
        if not scheduler:
            await manager.send_message(websocket, WSResponse(
                type="error",
                success=False,
                error="Scheduler not available"
            ))
            return
        
        queue_status = scheduler.get_queue_status()
        
        await manager.send_message(websocket, WSResponse(
            type="queue_status",
            success=True,
            data=queue_status
        ))
    
    except Exception as e:
        await manager.send_message(websocket, WSResponse(
            type="error",
            success=False,
            error=f"Failed to get queue status: {e}"
        ))


# Utility endpoints
@router.get("/connections/stats")
async def get_connection_stats():
    """Get WebSocket connection statistics"""
    return APIResponse(
        success=True,
        data=manager.get_connection_stats()
    )


@router.post("/broadcast/{endpoint_type}")
async def broadcast_message(
    endpoint_type: str,
    message: Dict[str, Any]
):
    """Broadcast message to all connections of a specific type"""
    try:
        ws_message = WSResponse(
            type="broadcast",
            success=True,
            data=message
        )
        
        await manager.broadcast_to_type(endpoint_type, ws_message)
        
        return APIResponse(
            success=True,
            data={
                "message": f"Broadcasted to {endpoint_type} connections",
                "connections": len(manager.connections.get(endpoint_type, []))
            }
        )
    
    except Exception as e:
        return APIResponse(
            success=False,
            error=str(e)
        )
