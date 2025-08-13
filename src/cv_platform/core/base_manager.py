"""
Base Manager - Abstract base class for all manager components

Provides common manager functionality including state management, health checks, 
event handling, performance monitoring, and lifecycle management.
"""

import abc
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class ManagerState(Enum):
    """Manager state enumeration"""
    INITIALIZING = "initializing"
    READY = "ready" 
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result data structure"""
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: float
    check_duration: float


class BaseManager(abc.ABC):
    """Abstract base class for all managers"""
    
    def __init__(self, name: str, enable_health_checks: bool = True):
        """
        Initialize base manager
        
        Args:
            name: Manager name
            enable_health_checks: Enable automatic health checking
        """
        self.name = name
        self._state = ManagerState.INITIALIZING
        self._last_activity = time.time()
        self._metrics = {}
        self._event_listeners = {}
        self._lock = threading.RLock()
        
        # Health check configuration
        self._enable_health_checks = enable_health_checks
        self._health_check_interval = 60  # 60 seconds
        self._last_health_check = time.time()
        self._health_status = HealthStatus.UNKNOWN
        self._health_details = {}
        self._health_history = []
        self._max_health_history = 100
        
        # Background health monitoring
        self._health_monitor_active = False
        self._health_monitor_thread = None
        
        logger.info(f"Base manager initialized: {name}")
    
    @property
    def state(self) -> ManagerState:
        """Get current state"""
        return self._state
    
    @property
    def health_status(self) -> HealthStatus:
        """Get health status"""
        return self._health_status
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        with self._lock:
            return self._metrics.copy()
    
    @property
    def is_healthy(self) -> bool:
        """Check if manager is in healthy state"""
        return self._health_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]
    
    @property
    def is_running(self) -> bool:
        """Check if manager is running"""
        return self._state == ManagerState.RUNNING
    
    def set_state(self, new_state: ManagerState, reason: str = "") -> None:
        """
        Set manager state
        
        Args:
            new_state: New state to set
            reason: Reason for state change
        """
        with self._lock:
            old_state = self._state
            self._state = new_state
            self._last_activity = time.time()
            
            log_msg = f"{self.name} state changed: {old_state.value} -> {new_state.value}"
            if reason:
                log_msg += f" (reason: {reason})"
            logger.info(log_msg)
            
            # Trigger state change event
            self._emit_event('state_changed', {
                'old_state': old_state,
                'new_state': new_state,
                'reason': reason,
                'timestamp': time.time()
            })
    
    def update_metric(self, key: str, value: Any, metadata: Dict[str, Any] = None) -> None:
        """
        Update performance metric
        
        Args:
            key: Metric name
            value: Metric value
            metadata: Additional metric metadata
        """
        with self._lock:
            self._metrics[key] = {
                'value': value,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self._last_activity = time.time()
    
    def increment_metric(self, key: str, delta: Union[int, float] = 1) -> None:
        """
        Increment performance metric
        
        Args:
            key: Metric name
            delta: Increment amount
        """
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = {'value': 0, 'timestamp': time.time(), 'metadata': {}}
            
            self._metrics[key]['value'] += delta
            self._metrics[key]['timestamp'] = time.time()
            self._last_activity = time.time()
    
    def get_metric(self, key: str, default: Any = None) -> Any:
        """
        Get specific metric value
        
        Args:
            key: Metric name
            default: Default value if metric not found
            
        Returns:
            Metric value or default
        """
        with self._lock:
            metric = self._metrics.get(key)
            return metric['value'] if metric else default
    
    def add_event_listener(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add event listener
        
        Args:
            event_type: Type of event to listen for
            callback: Callback function to execute
        """
        with self._lock:
            if event_type not in self._event_listeners:
                self._event_listeners[event_type] = []
            self._event_listeners[event_type].append(callback)
        
        logger.debug(f"Added event listener for {event_type} in {self.name}")
    
    def remove_event_listener(self, event_type: str, callback: Callable) -> None:
        """
        Remove event listener
        
        Args:
            event_type: Event type
            callback: Callback function to remove
        """
        with self._lock:
            if event_type in self._event_listeners:
                try:
                    self._event_listeners[event_type].remove(callback)
                    logger.debug(f"Removed event listener for {event_type} in {self.name}")
                except ValueError:
                    logger.warning(f"Event listener not found for {event_type} in {self.name}")
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Emit event to all registered listeners
        
        Args:
            event_type: Type of event
            event_data: Event data payload
        """
        with self._lock:
            listeners = self._event_listeners.get(event_type, []).copy()
            
        for callback in listeners:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in event listener for {event_type}: {e}")
    
    @abc.abstractmethod
    def initialize(self) -> bool:
        """
        Initialize manager - must be implemented by subclasses
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def cleanup(self) -> None:
        """
        Cleanup manager resources - must be implemented by subclasses
        """
        pass
    
    def perform_health_check(self) -> HealthCheckResult:
        """
        Perform health check - can be overridden by subclasses
        
        Returns:
            Health check result
        """
        start_time = time.time()
        
        try:
            # Basic health check implementation
            if self._state == ManagerState.ERROR:
                status = HealthStatus.CRITICAL
                message = "Manager is in error state"
            elif self._state == ManagerState.SHUTDOWN:
                status = HealthStatus.CRITICAL
                message = "Manager is shutdown"
            elif time.time() - self._last_activity > 3600:  # 1 hour
                status = HealthStatus.WARNING
                message = "No activity for over 1 hour"
            else:
                status = HealthStatus.HEALTHY
                message = "Manager is operating normally"
            
            details = {
                'state': self._state.value,
                'last_activity': self._last_activity,
                'uptime': time.time() - self._metrics.get('start_time', {}).get('value', time.time()),
                'metrics_count': len(self._metrics)
            }
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Health check failed: {e}"
            details = {'error': str(e)}
        
        check_duration = time.time() - start_time
        
        result = HealthCheckResult(
            status=status,
            message=message,
            details=details,
            timestamp=time.time(),
            check_duration=check_duration
        )
        
        # Update health status
        with self._lock:
            self._health_status = status
            self._health_details = details
            self._last_health_check = time.time()
            
            # Add to history
            self._health_history.append(result)
            if len(self._health_history) > self._max_health_history:
                self._health_history.pop(0)
        
        # Emit health status event
        self._emit_event('health_check', {
            'status': status.value,
            'message': message,
            'details': details,
            'timestamp': result.timestamp
        })
        
        return result
    
    def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self._health_monitor_active:
            try:
                self.perform_health_check()
                time.sleep(self._health_check_interval)
            except Exception as e:
                logger.error(f"Error in health monitor loop for {self.name}: {e}")
                time.sleep(self._health_check_interval)
    
    def start_health_monitoring(self):
        """Start background health monitoring"""
        if not self._enable_health_checks or self._health_monitor_active:
            return
        
        self._health_monitor_active = True
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop, 
            daemon=True,
            name=f"{self.name}_health_monitor"
        )
        self._health_monitor_thread.start()
        logger.info(f"Health monitoring started for {self.name}")
    
    def stop_health_monitoring(self):
        """Stop background health monitoring"""
        if not self._health_monitor_active:
            return
        
        self._health_monitor_active = False
        if self._health_monitor_thread and self._health_monitor_thread.is_alive():
            self._health_monitor_thread.join(timeout=5.0)
        logger.info(f"Health monitoring stopped for {self.name}")
    
    def get_health_history(self, limit: Optional[int] = None) -> List[HealthCheckResult]:
        """
        Get health check history
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of health check results
        """
        with self._lock:
            history = self._health_history.copy()
            if limit:
                history = history[-limit:]
            return history
    
    def start(self) -> bool:
        """
        Start manager operations
        
        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self._state == ManagerState.INITIALIZING:
                logger.info(f"Initializing {self.name}...")
                if not self.initialize():
                    self.set_state(ManagerState.ERROR, "Initialization failed")
                    return False
                self.set_state(ManagerState.READY, "Initialization completed")
            
            self.set_state(ManagerState.RUNNING, "Manager started")
            
            # Record start time
            self.update_metric('start_time', time.time())
            
            # Start health monitoring
            self.start_health_monitoring()
            
            logger.info(f"{self.name} manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {self.name} manager: {e}")
            self.set_state(ManagerState.ERROR, f"Start failed: {e}")
            return False
    
    def stop(self) -> None:
        """Stop manager operations"""
        try:
            logger.info(f"Stopping {self.name} manager...")
            
            # Stop health monitoring
            self.stop_health_monitoring()
            
            self.set_state(ManagerState.SHUTDOWN, "Manager stopped")
            self.cleanup()
            
            # Record stop time
            self.update_metric('stop_time', time.time())
            
            logger.info(f"{self.name} manager stopped")
            
        except Exception as e:
            logger.error(f"Error stopping {self.name} manager: {e}")
            self.set_state(ManagerState.ERROR, f"Stop failed: {e}")
    
    def pause(self) -> None:
        """Pause manager operations"""
        if self._state == ManagerState.RUNNING:
            self.set_state(ManagerState.PAUSED, "Manager paused")
            logger.info(f"{self.name} manager paused")
    
    def resume(self) -> None:
        """Resume manager operations"""
        if self._state == ManagerState.PAUSED:
            self.set_state(ManagerState.RUNNING, "Manager resumed")
            logger.info(f"{self.name} manager resumed")
    
    def restart(self) -> bool:
        """
        Restart manager
        
        Returns:
            True if restart successful
        """
        logger.info(f"Restarting {self.name} manager...")
        self.stop()
        time.sleep(1.0)  # Brief delay
        return self.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive manager status"""
        with self._lock:
            uptime = time.time() - self.get_metric('start_time', time.time())
            
            status = {
                'name': self.name,
                'state': self._state.value,
                'health_status': self._health_status.value,
                'uptime_seconds': uptime,
                'last_activity': self._last_activity,
                'last_health_check': self._last_health_check,
                'health_monitoring_enabled': self._enable_health_checks,
                'health_monitoring_active': self._health_monitor_active,
                'metrics_count': len(self._metrics),
                'event_listeners': {event_type: len(listeners) 
                                  for event_type, listeners in self._event_listeners.items()},
                'health_details': self._health_details.copy()
            }
            
            return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        with self._lock:
            metrics_summary = {}
            
            for key, metric in self._metrics.items():
                metrics_summary[key] = {
                    'value': metric['value'],
                    'age_seconds': time.time() - metric['timestamp']
                }
            
            # Calculate health metrics
            recent_health = [h for h in self._health_history 
                           if time.time() - h.timestamp < 3600]  # Last hour
            
            health_summary = {
                'total_checks': len(self._health_history),
                'recent_checks': len(recent_health),
                'healthy_percentage': (len([h for h in recent_health 
                                          if h.status == HealthStatus.HEALTHY]) / 
                                     len(recent_health) * 100 
                                     if recent_health else 0),
                'average_check_duration': (sum(h.check_duration for h in recent_health) / 
                                         len(recent_health) 
                                         if recent_health else 0)
            }
            
            return {
                'metrics': metrics_summary,
                'health': health_summary,
                'uptime': time.time() - self.get_metric('start_time', time.time()),
                'activity_age': time.time() - self._last_activity
            }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics"""
        with self._lock:
            self._metrics.clear()
            logger.info(f"Metrics reset for {self.name}")
    
    def export_metrics(self, format: str = 'dict') -> Union[Dict, str]:
        """
        Export metrics in various formats
        
        Args:
            format: Export format ('dict', 'json', 'csv')
            
        Returns:
            Metrics in requested format
        """
        metrics_data = {
            'manager_name': self.name,
            'timestamp': time.time(),
            'state': self._state.value,
            'health_status': self._health_status.value,
            'metrics': self.metrics
        }
        
        if format == 'dict':
            return metrics_data
        elif format == 'json':
            import json
            return json.dumps(metrics_data, indent=2, default=str)
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['metric_name', 'value', 'timestamp', 'metadata'])
            
            # Write metrics
            for key, metric in metrics_data['metrics'].items():
                writer.writerow([
                    key, 
                    metric['value'], 
                    metric['timestamp'],
                    str(metric.get('metadata', {}))
                ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """
        Validate manager configuration - can be overridden by subclasses
        
        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        validation_result = {
            'errors': [],
            'warnings': []
        }
        
        # Basic validation
        if not self.name:
            validation_result['errors'].append("Manager name is required")
        
        if self._health_check_interval <= 0:
            validation_result['errors'].append("Health check interval must be positive")
        
        return validation_result
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
    
    def __repr__(self) -> str:
        """String representation"""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"state={self._state.value}, health={self._health_status.value})")


class ManagerRegistry:
    """Registry for managing multiple manager instances"""
    
    def __init__(self):
        self._managers: Dict[str, BaseManager] = {}
        self._lock = threading.RLock()
    
    def register(self, manager: BaseManager) -> None:
        """Register a manager"""
        with self._lock:
            # check if BaseManager instance
            if hasattr(manager, 'name'):
                self._managers[manager.name] = manager
                logger.info(f"Registered manager: {manager.name}")
            else:
                # If not BaseManager，use class name as name
                manager_name = manager.__class__.__name__
                self._managers[manager_name] = manager
                logger.info(f"Registered component: {manager_name} (not a BaseManager)")
    
    def unregister(self, name: str) -> None:
        """Unregister a manager"""
        with self._lock:
            if name in self._managers:
                del self._managers[name]
                logger.info(f"Unregistered manager: {name}")
    
    def get_manager(self, name: str) -> Optional[BaseManager]:
        """Get a manager by name"""
        with self._lock:
            return self._managers.get(name)
    
    def get_all_managers(self) -> Dict[str, BaseManager]:
        """Get all registered managers"""
        with self._lock:
            return self._managers.copy()
    
    def start_all(self) -> Dict[str, bool]:
        """Start all registered managers"""
        results = {}
        with self._lock:
            for name, manager in self._managers.items():
                try:
                    results[name] = manager.start()
                except Exception as e:
                    logger.error(f"Failed to start manager {name}: {e}")
                    results[name] = False
        return results
    
    def stop_all(self) -> None:
        """Stop all registered managers"""
        with self._lock:
            for manager in self._managers.values():
                try:
                    manager.stop()
                except Exception as e:
                    logger.error(f"Failed to stop manager {manager.name}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system-wide status of all managers"""
        with self._lock:
            status = {
                'total_managers': len(self._managers),
                'running_managers': 0,
                'healthy_managers': 0,
                'managers': {}
            }
            
            for name, manager in self._managers.items():
                manager_status = manager.get_status()
                status['managers'][name] = manager_status
                
                if manager.is_running:
                    status['running_managers'] += 1
                
                if manager.is_healthy:
                    status['healthy_managers'] += 1
            
            return status


# Global manager registry
_manager_registry = ManagerRegistry()

def get_manager_registry() -> ManagerRegistry:
    """Get the global manager registry"""
    return _manager_registry


def register_manager(manager: BaseManager) -> None:
    """Register a manager with the global registry"""
    _manager_registry.register(manager)