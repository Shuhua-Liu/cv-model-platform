"""
GPU Monitor - GPU resource monitoring and management

Provides real-time GPU monitoring, memory tracking, device selection,
and performance optimization for model execution.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union, NamedTuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU monitoring limited")

try:
    import pynvml
    # Try to initialize NVML
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception) as e:
    NVML_AVAILABLE = False
    logger.info("NVIDIA Management Library not available - detailed GPU info unavailable")
    logger.debug(f"NVML error: {e}")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - system monitoring limited")


class DeviceType(Enum):
    """Device type enumeration"""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Metal Performance Shaders
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """GPU device information"""
    device_id: int
    name: str
    compute_capability: Optional[tuple] = None
    total_memory_mb: float = 0.0
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    is_available: bool = False


@dataclass
class GPUMemoryInfo:
    """GPU memory usage information"""
    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    utilization_percent: float


@dataclass
class GPUUtilization:
    """GPU utilization metrics"""
    gpu_percent: float
    memory_percent: float
    temperature_c: Optional[float] = None
    power_usage_w: Optional[float] = None
    fan_speed_percent: Optional[float] = None


class GPUMonitor:
    """GPU resource monitor and manager"""
    
    def __init__(self, 
                 monitoring_interval: float = 1.0,
                 enable_background_monitoring: bool = True,
                 memory_threshold: float = 0.9):
        """
        Initialize GPU monitor
        
        Args:
            monitoring_interval: Monitoring update interval in seconds
            enable_background_monitoring: Enable continuous monitoring
            memory_threshold: Memory usage threshold for warnings (0.0-1.0)
        """
        self.monitoring_interval = monitoring_interval
        self.memory_threshold = memory_threshold
        
        # Device information
        self._devices: Dict[int, GPUInfo] = {}
        self._current_device: Optional[int] = None
        
        # Monitoring data
        self._memory_history: Dict[int, List[GPUMemoryInfo]] = {}
        self._utilization_history: Dict[int, List[GPUUtilization]] = {}
        self._max_history_length = 100  # Keep last 100 measurements
        
        # Background monitoring
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        
        # Initialize
        self._discover_devices()
        
        if enable_background_monitoring:
            self.start_monitoring()
        
        logger.info(f"GPU monitor initialized - Found {len(self._devices)} GPU(s)")
    
    def _discover_devices(self):
        """Discover available GPU devices"""
        self._devices.clear()
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - GPU discovery skipped")
            return
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            
            for device_id in range(device_count):
                try:
                    props = torch.cuda.get_device_properties(device_id)
                    
                    gpu_info = GPUInfo(
                        device_id=device_id,
                        name=props.name,
                        compute_capability=(props.major, props.minor),
                        total_memory_mb=props.total_memory / (1024 * 1024),
                        is_available=True
                    )
                    
                    # Add CUDA version info
                    if hasattr(torch.version, 'cuda') and torch.version.cuda:
                        gpu_info.cuda_version = torch.version.cuda
                    
                    self._devices[device_id] = gpu_info
                    self._memory_history[device_id] = []
                    self._utilization_history[device_id] = []
                    
                    logger.debug(f"Discovered GPU {device_id}: {props.name}")
                    
                except Exception as e:
                    logger.error(f"Error discovering GPU {device_id}: {e}")
        
        # Check MPS availability (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS is treated as device ID -1 for identification
            mps_info = GPUInfo(
                device_id=-1,
                name="Apple MPS",
                total_memory_mb=self._get_mps_memory_info(),
                is_available=True
            )
            self._devices[-1] = mps_info
            self._memory_history[-1] = []
            self._utilization_history[-1] = []
            logger.info("Discovered Apple MPS device")
        else:
            # No GPU devices available
            logger.info("No GPU devices found - running in CPU-only mode")
            # Add a virtual CPU device for consistency
            cpu_info = GPUInfo(
                device_id=-2,
                name="CPU",
                total_memory_mb=self._get_system_memory_mb(),
                is_available=True
            )
            self._devices[-2] = cpu_info
            self._memory_history[-2] = []
            self._utilization_history[-2] = []
    
    def _get_mps_memory_info(self) -> float:
        """Get MPS memory information"""
        try:
            # Apple MPS memory is shared with system memory
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                return memory.total / (1024 * 1024)  # Convert to MB
            return 8192.0  # Default assumption of 8GB
        except Exception:
            return 8192.0
    
    def _get_system_memory_mb(self) -> float:
        """Get system memory in MB"""
        try:
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                return memory.total / (1024 * 1024)  # Convert to MB
            return 8192.0  # Default assumption of 8GB
        except Exception:
            return 8192.0
    
    def _get_nvidia_info(self, device_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed NVIDIA GPU information using nvidia-ml-py"""
        if not NVML_AVAILABLE:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                memory_util = utilization.memory
            except:
                gpu_util = 0
                memory_util = 0
            
            # Get temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None
            
            # Get power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power_usage = None
            
            # Get fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except:
                fan_speed = None
            
            return {
                'memory_total': memory_info.total,
                'memory_used': memory_info.used,
                'memory_free': memory_info.free,
                'gpu_utilization': gpu_util,
                'memory_utilization': memory_util,
                'temperature': temperature,
                'power_usage': power_usage,
                'fan_speed': fan_speed
            }
            
        except Exception as e:
            logger.warning(f"Failed to get NVIDIA info for device {device_id}: {e}")
            return None
    
    def _collect_memory_info(self, device_id: int) -> Optional[GPUMemoryInfo]:
        """Collect memory information for a specific device"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            if device_id == -1:  # MPS device
                if torch.backends.mps.is_available():
                    # MPS memory tracking is limited
                    return GPUMemoryInfo(
                        total_mb=self._devices[device_id].total_memory_mb,
                        allocated_mb=0.0,  # Not easily available for MPS
                        reserved_mb=0.0,
                        free_mb=self._devices[device_id].total_memory_mb,
                        utilization_percent=0.0
                    )
                return None
            
            # CUDA device
            if device_id >= torch.cuda.device_count():
                return None
            
            # Get PyTorch memory info
            allocated = torch.cuda.memory_allocated(device_id) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
            total = self._devices[device_id].total_memory_mb
            
            # Try to get more detailed info from nvidia-ml-py
            nvidia_info = self._get_nvidia_info(device_id)
            if nvidia_info:
                total = nvidia_info['memory_total'] / (1024 * 1024)
                allocated = nvidia_info['memory_used'] / (1024 * 1024)
                free = nvidia_info['memory_free'] / (1024 * 1024)
                reserved = total - free  # Approximate
            else:
                free = total - reserved
            
            utilization = (allocated / total * 100) if total > 0 else 0
            
            return GPUMemoryInfo(
                total_mb=total,
                allocated_mb=allocated,
                reserved_mb=reserved,
                free_mb=free,
                utilization_percent=utilization
            )
            
        except Exception as e:
            logger.warning(f"Failed to collect memory info for device {device_id}: {e}")
            return None
    
    def _collect_utilization_info(self, device_id: int) -> Optional[GPUUtilization]:
        """Collect utilization information for a specific device"""
        try:
            nvidia_info = self._get_nvidia_info(device_id)
            if nvidia_info:
                return GPUUtilization(
                    gpu_percent=nvidia_info['gpu_utilization'],
                    memory_percent=nvidia_info['memory_utilization'],
                    temperature_c=nvidia_info['temperature'],
                    power_usage_w=nvidia_info['power_usage'],
                    fan_speed_percent=nvidia_info['fan_speed']
                )
            else:
                # Fallback for non-NVIDIA or when nvidia-ml-py unavailable
                memory_info = self._collect_memory_info(device_id)
                memory_percent = memory_info.utilization_percent if memory_info else 0
                
                return GPUUtilization(
                    gpu_percent=0.0,  # Not available without nvidia-ml-py
                    memory_percent=memory_percent
                )
                
        except Exception as e:
            logger.warning(f"Failed to collect utilization info for device {device_id}: {e}")
            return None
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self._monitoring_active:
            try:
                with self._lock:
                    for device_id in self._devices.keys():
                        # Collect memory info
                        memory_info = self._collect_memory_info(device_id)
                        if memory_info:
                            history = self._memory_history[device_id]
                            history.append(memory_info)
                            if len(history) > self._max_history_length:
                                history.pop(0)
                            
                            # Check for memory warnings
                            if memory_info.utilization_percent > self.memory_threshold * 100:
                                logger.warning(f"GPU {device_id} memory usage high: "
                                             f"{memory_info.utilization_percent:.1f}%")
                        
                        # Collect utilization info
                        util_info = self._collect_utilization_info(device_id)
                        if util_info:
                            history = self._utilization_history[device_id]
                            history.append(util_info)
                            if len(history) > self._max_history_length:
                                history.pop(0)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        logger.info("GPU monitoring stopped")
    
    def get_devices(self) -> Dict[int, GPUInfo]:
        """Get all available GPU devices"""
        return self._devices.copy()
    
    def get_device_info(self, device_id: int) -> Optional[GPUInfo]:
        """Get information for a specific device"""
        return self._devices.get(device_id)
    
    def get_memory_info(self, device_id: Optional[int] = None) -> Union[GPUMemoryInfo, Dict[int, GPUMemoryInfo]]:
        """
        Get current memory information
        
        Args:
            device_id: Specific device ID, or None for all devices
            
        Returns:
            Memory info for specified device or all devices
        """
        if device_id is not None:
            return self._collect_memory_info(device_id)
        
        return {did: self._collect_memory_info(did) 
                for did in self._devices.keys()}
    
    def get_utilization_info(self, device_id: Optional[int] = None) -> Union[GPUUtilization, Dict[int, GPUUtilization]]:
        """
        Get current utilization information
        
        Args:
            device_id: Specific device ID, or None for all devices
            
        Returns:
            Utilization info for specified device or all devices
        """
        if device_id is not None:
            return self._collect_utilization_info(device_id)
        
        return {did: self._collect_utilization_info(did) 
                for did in self._devices.keys()}
    
    def get_memory_history(self, device_id: int, samples: Optional[int] = None) -> List[GPUMemoryInfo]:
        """
        Get memory usage history for a device
        
        Args:
            device_id: Device ID
            samples: Number of recent samples (None for all)
            
        Returns:
            List of memory info samples
        """
        with self._lock:
            history = self._memory_history.get(device_id, [])
            if samples is None:
                return history.copy()
            return history[-samples:] if samples > 0 else []
    
    def get_utilization_history(self, device_id: int, samples: Optional[int] = None) -> List[GPUUtilization]:
        """
        Get utilization history for a device
        
        Args:
            device_id: Device ID
            samples: Number of recent samples (None for all)
            
        Returns:
            List of utilization samples
        """
        with self._lock:
            history = self._utilization_history.get(device_id, [])
            if samples is None:
                return history.copy()
            return history[-samples:] if samples > 0 else []
    
    def select_best_device(self, 
                          memory_required_mb: float = 0,
                          prefer_empty: bool = True,
                          exclude_devices: Optional[List[int]] = None) -> Optional[int]:
        """
        Select the best available GPU device
        
        Args:
            memory_required_mb: Minimum memory required in MB
            prefer_empty: Prefer devices with less memory usage
            exclude_devices: List of device IDs to exclude
            
        Returns:
            Best device ID or None if no suitable device found
        """
        exclude_devices = exclude_devices or []
        available_devices = []
        
        for device_id, device_info in self._devices.items():
            if device_id in exclude_devices:
                continue
            
            if not device_info.is_available:
                continue
            
            memory_info = self._collect_memory_info(device_id)
            if not memory_info:
                continue
            
            # Check if device has enough memory
            if memory_info.free_mb < memory_required_mb:
                continue
            
            available_devices.append((device_id, memory_info))
        
        if not available_devices:
            logger.warning("No suitable GPU device found")
            return None
        
        # Sort by preference
        if prefer_empty:
            # Prefer devices with more free memory
            available_devices.sort(key=lambda x: x[1].free_mb, reverse=True)
        else:
            # Prefer devices with less free memory (better utilization)
            available_devices.sort(key=lambda x: x[1].free_mb)
        
        best_device_id = available_devices[0][0]
        logger.info(f"Selected GPU device {best_device_id}")
        return best_device_id
    
    def set_device(self, device_id: int) -> bool:
        """
        Set the current PyTorch device
        
        Args:
            device_id: Device ID to set
            
        Returns:
            True if device was set successfully
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available - cannot set device")
            return False
        
        if device_id not in self._devices:
            logger.error(f"Device {device_id} not available")
            return False
        
        try:
            if device_id == -1:  # MPS device
                if torch.backends.mps.is_available():
                    # MPS doesn't need explicit device setting like CUDA
                    self._current_device = device_id
                    logger.info("Set device to MPS")
                    return True
                else:
                    logger.error("MPS not available")
                    return False
            else:
                torch.cuda.set_device(device_id)
                self._current_device = device_id
                logger.info(f"Set CUDA device to {device_id}")
                return True
        
        except Exception as e:
            logger.error(f"Failed to set device {device_id}: {e}")
            return False
    
    def get_current_device(self) -> Optional[int]:
        """Get current active device"""
        if not TORCH_AVAILABLE:
            return None
        
        try:
            if torch.cuda.is_available():
                return torch.cuda.current_device()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return -1  # MPS device identifier
        except Exception:
            pass
        
        return self._current_device
    
    def clear_memory(self, device_id: Optional[int] = None):
        """
        Clear GPU memory cache
        
        Args:
            device_id: Device ID to clear, or None for current device
        """
        if not TORCH_AVAILABLE:
            logger.info("PyTorch not available - memory clear skipped")
            return
        
        try:
            if device_id == -1:  # MPS device
                # MPS doesn't have explicit cache clearing
                logger.info("MPS cache clearing not available")
                return
            elif device_id == -2:  # CPU device
                logger.info("CPU memory clear not applicable")
                return
            
            if device_id is None:
                device_id = self.get_current_device()
            
            if device_id is not None and device_id >= 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if device_id < torch.cuda.device_count():
                    torch.cuda.synchronize(device_id)
                logger.info(f"Cleared memory cache for device {device_id}")
            else:
                logger.info("No CUDA devices available for memory clearing")
        
        except Exception as e:
            logger.error(f"Failed to clear memory cache: {e}")
    
    def get_device_utilization_summary(self) -> Dict[str, Any]:
        """Get comprehensive device utilization summary"""
        summary = {
            'total_devices': len(self._devices),
            'available_devices': sum(1 for d in self._devices.values() if d.is_available),
            'current_device': self.get_current_device(),
            'devices': {}
        }
        
        for device_id, device_info in self._devices.items():
            memory_info = self._collect_memory_info(device_id)
            util_info = self._collect_utilization_info(device_id)
            
            device_summary = {
                'name': device_info.name,
                'available': device_info.is_available,
                'total_memory_mb': device_info.total_memory_mb,
                'compute_capability': device_info.compute_capability,
            }
            
            if memory_info:
                device_summary.update({
                    'memory_allocated_mb': memory_info.allocated_mb,
                    'memory_free_mb': memory_info.free_mb,
                    'memory_utilization_percent': memory_info.utilization_percent
                })
            
            if util_info:
                device_summary.update({
                    'gpu_utilization_percent': util_info.gpu_percent,
                    'temperature_c': util_info.temperature_c,
                    'power_usage_w': util_info.power_usage_w
                })
            
            summary['devices'][device_id] = device_summary
        
        return summary
    
    def get_device_recommendations(self) -> Dict[str, Any]:
        """Get device usage recommendations"""
        recommendations = {
            'suggestions': [],
            'warnings': [],
            'optimal_device': None
        }
        
        device_scores = []
        
        for device_id, device_info in self._devices.items():
            if not device_info.is_available:
                continue
            
            memory_info = self._collect_memory_info(device_id)
            util_info = self._collect_utilization_info(device_id)
            
            if not memory_info:
                continue
            
            # Calculate device score (higher is better)
            score = 0
            
            # Memory availability score (0-40 points)
            memory_ratio = memory_info.free_mb / memory_info.total_mb
            score += memory_ratio * 40
            
            # Low utilization is better for new tasks (0-30 points)
            if util_info:
                util_penalty = util_info.gpu_percent / 100.0
                score += (1 - util_penalty) * 30
            else:
                score += 30  # Assume low utilization if unknown
            
            # Temperature consideration (0-20 points)
            if util_info and util_info.temperature_c:
                if util_info.temperature_c < 70:
                    score += 20
                elif util_info.temperature_c < 80:
                    score += 10
                # No points for high temperature
            else:
                score += 15  # Neutral score if unknown
            
            # Compute capability bonus (0-10 points)
            if device_info.compute_capability:
                major, minor = device_info.compute_capability
                score += min(major * 2 + minor * 0.5, 10)
            
            device_scores.append((device_id, score, device_info, memory_info, util_info))
            
            # Generate warnings
            if memory_info.utilization_percent > 90:
                recommendations['warnings'].append(
                    f"Device {device_id} ({device_info.name}) memory usage critical: "
                    f"{memory_info.utilization_percent:.1f}%"
                )
            
            if util_info and util_info.temperature_c and util_info.temperature_c > 85:
                recommendations['warnings'].append(
                    f"Device {device_id} ({device_info.name}) temperature high: "
                    f"{util_info.temperature_c}°C"
                )
        
        # Sort by score and select optimal device
        if device_scores:
            device_scores.sort(key=lambda x: x[1], reverse=True)
            optimal_device_id = device_scores[0][0]
            recommendations['optimal_device'] = optimal_device_id
            
            # Generate suggestions
            best_device = device_scores[0]
            recommendations['suggestions'].append(
                f"Recommended device: {optimal_device_id} ({best_device[2].name}) "
                f"with {best_device[3].free_mb:.0f}MB free memory"
            )
            
            # Memory optimization suggestions
            total_allocated = sum(d[3].allocated_mb for d in device_scores)
            if total_allocated > 1000:  # More than 1GB allocated
                recommendations['suggestions'].append(
                    "Consider clearing GPU memory cache if experiencing memory issues"
                )
        
        return recommendations
    
    def export_metrics(self, device_id: int, format: str = 'dict') -> Union[Dict, str]:
        """
        Export device metrics in various formats
        
        Args:
            device_id: Device ID
            format: Output format ('dict', 'json', 'csv')
            
        Returns:
            Metrics in requested format
        """
        if device_id not in self._devices:
            raise ValueError(f"Device {device_id} not found")
        
        # Collect current data
        device_info = self._devices[device_id]
        memory_info = self._collect_memory_info(device_id)
        util_info = self._collect_utilization_info(device_id)
        
        metrics = {
            'device_id': device_id,
            'device_name': device_info.name,
            'timestamp': time.time(),
            'memory_total_mb': memory_info.total_mb if memory_info else 0,
            'memory_allocated_mb': memory_info.allocated_mb if memory_info else 0,
            'memory_free_mb': memory_info.free_mb if memory_info else 0,
            'memory_utilization_percent': memory_info.utilization_percent if memory_info else 0,
            'gpu_utilization_percent': util_info.gpu_percent if util_info else 0,
            'temperature_c': util_info.temperature_c if util_info else None,
            'power_usage_w': util_info.power_usage_w if util_info else None
        }
        
        if format == 'dict':
            return metrics
        elif format == 'json':
            import json
            return json.dumps(metrics, indent=2)
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def shutdown(self):
        """Shutdown GPU monitor"""
        self.stop_monitoring()
        logger.info("GPU monitor shutdown complete")


# Global GPU monitor instance
_gpu_monitor = None

def get_gpu_monitor(**kwargs) -> GPUMonitor:
    """
    Get global GPU monitor instance
    
    Args:
        **kwargs: Parameters for GPU monitor initialization
        
    Returns:
        Global GPU monitor instance
    """
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor(**kwargs)
    return _gpu_monitor


def get_optimal_device(memory_required_mb: float = 0) -> Optional[str]:
    """
    Convenience function to get optimal device string
    
    Args:
        memory_required_mb: Required memory in MB
        
    Returns:
        Device string (e.g., 'cuda:0', 'mps', 'cpu') or None
    """
    monitor = get_gpu_monitor()
    device_id = monitor.select_best_device(memory_required_mb)
    
    if device_id is None:
        return 'cpu'
    elif device_id == -1:
        return 'mps'
    else:
        return f'cuda:{device_id}'