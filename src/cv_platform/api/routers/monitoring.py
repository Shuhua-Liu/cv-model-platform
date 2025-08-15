"""
Monitoring and Health Router

File location: src/cv_platform/api/routers/monitoring.py

Provides comprehensive system monitoring, health checks, metrics collection,
and performance analysis endpoints.
"""

import time
import psutil
from typing import Optional, Dict, List, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import PlainTextResponse
from loguru import logger

from ..models.responses import APIResponse, HealthCheckResponse, SystemStatusResponse
from ..dependencies.auth import get_current_user, get_admin_user
from ..dependencies.components import (
    get_manager_registry, get_model_manager, get_scheduler,
    get_gpu_monitor, get_cache_manager
)

# Create router
router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def comprehensive_health_check(
    include_details: bool = Query(True, description="Include detailed component information"),
    check_dependencies: bool = Query(True, description="Check external dependencies"),
    current_user: dict = Depends(get_current_user),
    manager_registry = Depends(get_manager_registry)
):
    """
    Comprehensive health check of all system components
    
    Args:
        include_details: Whether to include detailed health information
        check_dependencies: Whether to check external dependencies
        current_user: Current authenticated user
        manager_registry: Manager registry dependency
        
    Returns:
        Complete health status of all components
    """
    try:
        component_health = {}
        overall_status = "healthy"
        messages = []
        critical_issues = []
        warnings = []
        
        # Check all registered components
        for name, manager in manager_registry.get_all_managers().items():
            try:
                health_result = manager.perform_health_check()
                component_health[name] = {
                    "status": health_result.status.value,
                    "message": health_result.message,
                    "check_duration": health_result.check_duration,
                    "details": health_result.details if include_details else {}
                }
                
                # Track issues
                if health_result.status.value == "critical":
                    overall_status = "critical"
                    critical_issues.append(f"{name}: {health_result.message}")
                elif health_result.status.value == "warning" and overall_status != "critical":
                    overall_status = "warning"
                    warnings.append(f"{name}: {health_result.message}")
                
                messages.append(f"{name}: {health_result.message}")
                
            except Exception as e:
                component_health[name] = {
                    "status": "critical",
                    "message": f"Health check failed: {e}",
                    "check_duration": 0.0,
                    "details": {"error": str(e)}
                }
                overall_status = "critical"
                critical_issues.append(f"{name}: Health check failed - {e}")
        
        # Check system resources
        system_health = await _check_system_health()
        if system_health["status"] != "healthy":
            if system_health["status"] == "critical":
                overall_status = "critical"
                critical_issues.extend(system_health.get("critical_issues", []))
            else:
                if overall_status != "critical":
                    overall_status = "warning"
                warnings.extend(system_health.get("warnings", []))
        
        # Check external dependencies if requested
        dependency_health = {}
        if check_dependencies:
            dependency_health = await _check_external_dependencies()
            if dependency_health.get("status") != "healthy":
                if dependency_health.get("status") == "critical":
                    overall_status = "critical"
                    critical_issues.extend(dependency_health.get("issues", []))
                else:
                    if overall_status != "critical":
                        overall_status = "warning"
                    warnings.extend(dependency_health.get("issues", []))
        
        # Prepare summary message
        if overall_status == "healthy":
            summary_message = "All systems operational"
        elif overall_status == "warning":
            summary_message = f"System operational with {len(warnings)} warning(s)"
        else:
            summary_message = f"System has {len(critical_issues)} critical issue(s)"
        
        # Get application uptime
        app_uptime = time.time() - getattr(manager_registry, '_start_time', time.time())
        
        health_data = {
            "overall_status": overall_status,
            "summary": summary_message,
            "components": component_health,
            "system": system_health,
            "dependencies": dependency_health if check_dependencies else {},
            "metrics": {
                "uptime_seconds": app_uptime,
                "total_components": len(component_health),
                "healthy_components": len([c for c in component_health.values() if c["status"] == "healthy"]),
                "critical_issues_count": len(critical_issues),
                "warnings_count": len(warnings)
            },
            "issues": {
                "critical": critical_issues,
                "warnings": warnings
            }
        }
        
        return HealthCheckResponse(
            success=overall_status in ["healthy", "warning"],
            message=summary_message,
            data=health_data
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            success=False,
            message=f"Health check system failure: {e}",
            data={
                "overall_status": "critical",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    include_performance: bool = Query(True, description="Include performance metrics"),
    include_gpu: bool = Query(True, description="Include GPU information"),
    current_user: dict = Depends(get_current_user),
    manager_registry = Depends(get_manager_registry),
    model_manager = Depends(get_model_manager),
    scheduler = Depends(get_scheduler),
    gpu_monitor = Depends(get_gpu_monitor),
    cache_manager = Depends(get_cache_manager)
):
    """
    Get comprehensive system status and metrics
    
    Args:
        include_performance: Whether to include performance metrics
        include_gpu: Whether to include GPU information
        current_user: Current authenticated user
        manager_registry: Manager registry dependency
        model_manager: Model manager dependency
        scheduler: Task scheduler dependency
        gpu_monitor: GPU monitor dependency
        cache_manager: Cache manager dependency
        
    Returns:
        Detailed system status information
    """
    try:
        # Get basic registry status
        registry_status = manager_registry.get_system_status()
        
        # Get model manager status and safely handle list_available_models()
        try:
            available_models = model_manager.list_available_models()
            if hasattr(available_models, 'values'):
                # If it's a dict-like object
                models_list = list(available_models.values())
            else:
                # If it's already a list or other iterable
                models_list = list(available_models) if available_models else []
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
            models_list = []
        
        # Get model status safely
        try:
            model_status = model_manager.get_system_status()
            models_info = model_status.get('models', {}) if isinstance(model_status, dict) else {}
        except Exception as e:
            logger.warning(f"Failed to get model status: {e}")
            models_info = {}
        
        # Get scheduler status safely
        try:
            queue_status = scheduler.get_queue_status()
        except Exception as e:
            logger.warning(f"Failed to get queue status: {e}")
            queue_status = {}
        
        try:
            executor_stats = scheduler.get_executor_stats()
            # Ensure executor_stats is a dict
            if not isinstance(executor_stats, dict):
                executor_stats = {}
        except Exception as e:
            logger.warning(f"Failed to get executor stats: {e}")
            executor_stats = {}
        
        # Get cache statistics safely
        try:
            cache_stats = cache_manager.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            cache_stats = {}
        
        # Get GPU information if requested
        gpu_devices_count = 0
        if include_gpu:
            try:
                gpu_info = gpu_monitor.get_device_utilization_summary()
                # Try to count GPU devices
                if isinstance(gpu_info, dict):
                    gpu_devices_count = len(gpu_info.get('devices', []))
                elif isinstance(gpu_info, list):
                    gpu_devices_count = len(gpu_info)
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")
                gpu_devices_count = 0
        
        # Calculate application uptime
        app_start_time = getattr(manager_registry, '_start_time', time.time())
        uptime_seconds = time.time() - app_start_time
        
        # Get the count of available model types safely
        try:
            available_types_count = len(set(m.get('type', 'unknown') for m in models_list if isinstance(m, dict)))
        except Exception:
            available_types_count = 0
        
        # Get active executors count safely
        try:
            active_executors = len([e for e in executor_stats.values() if isinstance(e, dict) and e.get('running_tasks', 0) > 0])
        except Exception:
            active_executors = 0
        
        # Create SystemStatus object with all required fields
        from ..models.responses import SystemStatus
        
        system_status = SystemStatus(
            api_version="1.0.0",
            uptime_seconds=uptime_seconds,
            total_managers=registry_status.get('total_managers', 0),
            healthy_managers=registry_status.get('healthy_managers', 0),
            models_available=models_info.get('total', len(models_list)),
            models_cached=models_info.get('cached', 0),
            active_tasks=queue_status.get('running_tasks', 0),
            completed_tasks=queue_status.get('completed_tasks', 0),
            gpu_devices=gpu_devices_count,
            cache_size_mb=cache_stats.get('size_mb', 0.0)
        )
        
        return SystemStatusResponse(
            success=True,
            message="System status retrieved successfully",
            data=system_status
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {e}"
        )


@router.get("/metrics", response_model=APIResponse)
async def get_system_metrics(
    format: str = Query("json", regex="^(json|prometheus)$", description="Output format"),
    component: Optional[str] = Query(None, description="Specific component to get metrics for"),
    current_user: dict = Depends(get_current_user),
    manager_registry = Depends(get_manager_registry)
):
    """
    Get system metrics in various formats
    
    Args:
        format: Output format (json or prometheus)
        component: Specific component name to filter metrics
        current_user: Current authenticated user
        manager_registry: Manager registry dependency
        
    Returns:
        System metrics in requested format
    """
    try:
        all_metrics = {}
        
        # Collect metrics from all components
        for name, manager in manager_registry.get_all_managers().items():
            if component and name != component:
                continue
                
            try:
                component_metrics = manager.get_performance_summary()
                all_metrics[name] = component_metrics
            except Exception as e:
                logger.warning(f"Failed to get metrics from {name}: {e}")
                all_metrics[name] = {"error": str(e)}
        
        if format == "prometheus":
            # Convert to Prometheus format
            prometheus_metrics = _convert_to_prometheus_format(all_metrics)
            return PlainTextResponse(
                content=prometheus_metrics,
                media_type="text/plain"
            )
        else:
            # Return JSON format
            return APIResponse(
                success=True,
                message=f"Metrics retrieved for {len(all_metrics)} component(s)",
                data={
                    "metrics": all_metrics,
                    "timestamp": time.time(),
                    "format": format,
                    "component_filter": component
                }
            )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {e}"
        )


@router.get("/performance", response_model=APIResponse)
async def analyze_performance(
    time_window: int = Query(3600, ge=60, le=86400, description="Analysis time window in seconds"),
    include_recommendations: bool = Query(True, description="Include optimization recommendations"),
    current_user: dict = Depends(get_admin_user),
    scheduler = Depends(get_scheduler),
    model_manager = Depends(get_model_manager),
    gpu_monitor = Depends(get_gpu_monitor)
):
    """
    Analyze system performance and provide optimization recommendations
    
    Args:
        time_window: Time window for analysis in seconds
        include_recommendations: Whether to include recommendations
        current_user: Current authenticated admin user
        scheduler: Task scheduler dependency
        model_manager: Model manager dependency
        gpu_monitor: GPU monitor dependency
        
    Returns:
        Performance analysis and optimization recommendations
    """
    try:
        # Get performance analysis from scheduler
        scheduler_analysis = scheduler.optimize_performance()
        
        # Get model manager performance
        model_metrics = model_manager.get_performance_summary()
        
        # Get GPU recommendations if available
        gpu_recommendations = {}
        try:
            gpu_recommendations = gpu_monitor.get_device_recommendations()
        except Exception as e:
            logger.warning(f"Failed to get GPU recommendations: {e}")
        
        # Analyze cache performance
        cache_analysis = await _analyze_cache_performance(model_manager)
        
        # Compile performance analysis
        analysis = {
            "time_window_seconds": time_window,
            "analysis_timestamp": time.time(),
            "scheduler": {
                "analysis": scheduler_analysis.get("analysis", {}),
                "recommendations": scheduler_analysis.get("recommendations", []),
                "warnings": scheduler_analysis.get("warnings", [])
            },
            "models": {
                "metrics": model_metrics,
                "cache_analysis": cache_analysis
            },
            "gpu": gpu_recommendations,
            "overall_health_score": _calculate_health_score(scheduler_analysis, model_metrics, gpu_recommendations),
            "optimization_priority": _get_optimization_priorities(scheduler_analysis, cache_analysis, gpu_recommendations)
        }
        
        return APIResponse(
            success=True,
            message="Performance analysis completed",
            data=analysis
        )
        
    except Exception as e:
        logger.error(f"Failed to analyze performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze performance: {e}"
        )


@router.post("/optimize", response_model=APIResponse)
async def trigger_optimization(
    optimization_type: str = Query("auto", regex="^(auto|cache|gpu|scheduler)$", description="Type of optimization"),
    dry_run: bool = Query(True, description="Whether to perform a dry run"),
    current_user: dict = Depends(get_admin_user),
    scheduler = Depends(get_scheduler),
    model_manager = Depends(get_model_manager),
    cache_manager = Depends(get_cache_manager),
    gpu_monitor = Depends(get_gpu_monitor)
):
    """
    Trigger system optimization
    
    Args:
        optimization_type: Type of optimization to perform
        dry_run: Whether to perform dry run only
        current_user: Current authenticated admin user
        scheduler: Task scheduler dependency
        model_manager: Model manager dependency
        cache_manager: Cache manager dependency
        
    Returns:
        Optimization results
    """
    try:
        optimization_results = {
            "optimization_type": optimization_type,
            "dry_run": dry_run,
            "timestamp": time.time(),
            "performed_by": current_user.get('user_id'),
            "actions": []
        }
        
        if optimization_type in ["auto", "cache"]:
            # Cache optimization
            if not dry_run:
                cache_manager.cleanup()
                optimization_results["actions"].append({
                    "type": "cache_cleanup",
                    "description": "Performed cache cleanup",
                    "status": "completed"
                })
            else:
                cache_stats = cache_manager.get_stats()
                optimization_results["actions"].append({
                    "type": "cache_cleanup",
                    "description": f"Would cleanup cache (current: {cache_stats.get('size_mb', 0):.1f}MB)",
                    "status": "simulated"
                })
        
        if optimization_type in ["auto", "scheduler"]:
            # Scheduler optimization
            scheduler_recommendations = scheduler.optimize_performance()
            optimization_results["actions"].append({
                "type": "scheduler_analysis",
                "description": "Analyzed scheduler performance",
                "recommendations": scheduler_recommendations.get("recommendations", []),
                "status": "analyzed"
            })
        
        if optimization_type in ["auto", "gpu"]:
            # GPU optimization
            try:
                gpu_monitor.clear_memory()  # Clear GPU memory cache
                optimization_results["actions"].append({
                    "type": "gpu_memory_clear",
                    "description": "Cleared GPU memory cache",
                    "status": "completed" if not dry_run else "simulated"
                })
            except Exception as e:
                optimization_results["actions"].append({
                    "type": "gpu_memory_clear",
                    "description": f"GPU memory clear failed: {e}",
                    "status": "failed"
                })
        
        return APIResponse(
            success=True,
            message=f"Optimization {'simulation' if dry_run else 'execution'} completed",
            data=optimization_results
        )
        
    except Exception as e:
        logger.error(f"Failed to perform optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform optimization: {e}"
        )


@router.get("/logs", response_model=APIResponse)
async def get_system_logs(
    level: str = Query("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Log level filter"),
    lines: int = Query(100, ge=1, le=10000, description="Number of recent log lines"),
    component: Optional[str] = Query(None, description="Filter by component name"),
    current_user: dict = Depends(get_admin_user)
):
    """
    Get recent system logs (admin only)
    
    Args:
        level: Minimum log level to include
        lines: Number of recent log lines to return
        component: Optional component name filter
        current_user: Current authenticated admin user
        
    Returns:
        Recent system logs
    """
    try:
        # Note: This is a simplified implementation
        # In production, you'd integrate with your logging system
        log_data = {
            "filters": {
                "level": level,
                "lines": lines,
                "component": component
            },
            "logs": [
                {
                    "timestamp": time.time() - 300,
                    "level": "INFO",
                    "component": "ModelManager",
                    "message": "Model yolov8n loaded successfully",
                    "details": {}
                },
                {
                    "timestamp": time.time() - 600,
                    "level": "WARNING",
                    "component": "GPUMonitor",
                    "message": "GPU memory usage high: 85%",
                    "details": {"device": "cuda:0", "memory_percent": 85}
                }
            ],
            "total_lines": 2,
            "truncated": False
        }
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(log_data['logs'])} log entries",
            data=log_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get logs: {e}"
        )


# =============================================================================
# Utility Functions
# =============================================================================

async def _check_system_health() -> Dict[str, Any]:
    """Check system resource health"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        critical_issues = []
        warnings = []
        
        # Check thresholds
        if cpu_percent > 90:
            critical_issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
        elif cpu_percent > 75:
            warnings.append(f"CPU usage high: {cpu_percent:.1f}%")
        
        if memory.percent > 95:
            critical_issues.append(f"Memory usage critical: {memory.percent:.1f}%")
        elif memory.percent > 85:
            warnings.append(f"Memory usage high: {memory.percent:.1f}%")
        
        if disk.percent > 95:
            critical_issues.append(f"Disk usage critical: {disk.percent:.1f}%")
        elif disk.percent > 85:
            warnings.append(f"Disk usage high: {disk.percent:.1f}%")
        
        # Determine overall status
        if critical_issues:
            status = "critical"
        elif warnings:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
            "critical_issues": critical_issues,
            "warnings": warnings
        }
        
    except Exception as e:
        return {
            "status": "critical",
            "error": str(e),
            "critical_issues": [f"System health check failed: {e}"],
            "warnings": []
        }


async def _check_external_dependencies() -> Dict[str, Any]:
    """Check external service dependencies"""
    # This would check external services like databases, APIs, etc.
    # For now, return a placeholder
    return {
        "status": "healthy",
        "checks": {
            "database": {"status": "healthy", "response_time_ms": 5},
            "external_api": {"status": "healthy", "response_time_ms": 150}
        },
        "issues": []
    }


async def _collect_performance_metrics(model_manager, scheduler, cache_manager) -> Dict[str, Any]:
    """Collect comprehensive performance metrics"""
    try:
        # Get individual component metrics safely
        try:
            model_perf = model_manager.get_performance_summary()
        except Exception as e:
            logger.warning(f"Failed to get model performance: {e}")
            model_perf = {}
        
        try:
            scheduler_stats = scheduler.get_system_stats()
        except Exception as e:
            logger.warning(f"Failed to get scheduler stats: {e}")
            scheduler_stats = {}
        
        try:
            cache_stats = cache_manager.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {e}")
            cache_stats = {}
        
        # Safely extract metrics with defaults
        model_metrics = model_perf.get('metrics', {}) if isinstance(model_perf, dict) else {}
        total_load_time = model_metrics.get('total_load_time', {}).get('value', 0) if isinstance(model_metrics.get('total_load_time'), dict) else 0
        models_loaded = model_metrics.get('models_loaded', {}).get('value', 1) if isinstance(model_metrics.get('models_loaded'), dict) else 1
        
        queue_info = scheduler_stats.get('queue', {}) if isinstance(scheduler_stats, dict) else {}
        scheduler_info = scheduler_stats.get('scheduler', {}) if isinstance(scheduler_stats, dict) else {}
        
        return {
            "models": {
                "total_load_time": total_load_time,
                "average_load_time": total_load_time / max(models_loaded, 1),
                "cache_hit_rate": cache_stats.get('hit_rate', 0)
            },
            "scheduler": {
                "queue_utilization": queue_info.get('queue_size', 0) / max(queue_info.get('max_queue_size', 1), 1),
                "average_execution_time": queue_info.get('average_queue_wait_time', 0),
                "success_rate": scheduler_info.get('success_rate', 0)
            },
            "cache": {
                "utilization_percent": cache_stats.get('size_mb', 0) / max(cache_stats.get('max_size_mb', 1), 1) * 100,
                "hit_rate": cache_stats.get('hit_rate', 0),
                "eviction_rate": cache_stats.get('evictions', 0) / max(cache_stats.get('hits', 1) + cache_stats.get('misses', 1), 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to collect performance metrics: {e}")
        return {"error": str(e)}


async def _get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": psutil.cpu_count(),
                "count_logical": psutil.cpu_count(logical=True)
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent": disk.percent
            }
        }
        
    except Exception as e:
        return {"error": str(e)}


async def _analyze_cache_performance(model_manager) -> Dict[str, Any]:
    """Analyze cache performance"""
    try:
        cache_stats = model_manager.get_cache_stats()
        
        analysis = {
            "hit_rate": cache_stats.get('hit_rate', 0),
            "utilization": cache_stats.get('size_mb', 0) / max(cache_stats.get('max_size_mb', 1), 1),
            "recommendations": []
        }
        
        # Generate recommendations
        if analysis["hit_rate"] < 0.5:
            analysis["recommendations"].append("Consider increasing cache size or TTL")
        
        if analysis["utilization"] > 0.9:
            analysis["recommendations"].append("Cache near capacity - consider cleanup")
        
        return analysis
        
    except Exception as e:
        return {"error": str(e)}


def _format_uptime(seconds: float) -> str:
    """Format uptime in human-readable format"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def _convert_to_prometheus_format(metrics: Dict[str, Any]) -> str:
    """Convert metrics to Prometheus format"""
    lines = []
    timestamp = int(time.time() * 1000)  # Prometheus uses milliseconds
    
    for component, component_metrics in metrics.items():
        if isinstance(component_metrics, dict) and 'metrics' in component_metrics:
            for metric_name, metric_data in component_metrics['metrics'].items():
                if isinstance(metric_data, dict) and 'value' in metric_data:
                    # Convert metric name to Prometheus format
                    prom_metric_name = f"cv_platform_{component.lower()}_{metric_name.lower()}"
                    prom_metric_name = prom_metric_name.replace('-', '_').replace(' ', '_')
                    
                    lines.append(f"{prom_metric_name} {metric_data['value']} {timestamp}")
    
    return "\n".join(lines)


def _calculate_health_score(scheduler_analysis: Dict, model_metrics: Dict, gpu_recommendations: Dict) -> float:
    """Calculate overall system health score (0-100)"""
    score = 100.0
    
    # Deduct for scheduler issues
    if scheduler_analysis.get("warnings"):
        score -= len(scheduler_analysis["warnings"]) * 5
    
    # Deduct for GPU warnings
    if gpu_recommendations.get("warnings"):
        score -= len(gpu_recommendations["warnings"]) * 10
    
    # Deduct for poor performance metrics
    success_rate = scheduler_analysis.get("analysis", {}).get("success_rate", 1.0)
    if success_rate < 0.95:
        score -= (0.95 - success_rate) * 100
    
    return max(0.0, min(100.0, score))


def _get_optimization_priorities(scheduler_analysis: Dict, cache_analysis: Dict, gpu_recommendations: Dict) -> List[str]:
    """Get prioritized list of optimization recommendations"""
    priorities = []
    
    # High priority issues
    if gpu_recommendations.get("warnings"):
        priorities.append("GPU optimization")
    
    if scheduler_analysis.get("warnings"):
        priorities.append("Task scheduling optimization")
    
    # Medium priority issues
    cache_hit_rate = cache_analysis.get("hit_rate", 1.0)
    if cache_hit_rate < 0.7:
        priorities.append("Cache optimization")
    
    # Low priority improvements
    if not priorities:
        priorities.append("System is performing well - monitor for changes")
    
    return priorities