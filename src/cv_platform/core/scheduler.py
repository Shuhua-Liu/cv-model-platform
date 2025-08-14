"""
Enhanced Task Scheduler - Inheriting from BaseManager

Task scheduling and resource management system with full BaseManager integration
for state management, health monitoring, and lifecycle management.
"""

import time
import uuid
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from queue import PriorityQueue, Queue, Empty
from concurrent.futures import ThreadPoolExecutor, Future
from loguru import logger

from .base_manager import BaseManager, ManagerState, HealthStatus, HealthCheckResult

try:
    from .gpu_monitor import get_gpu_monitor, GPUMonitor
    from .cache_manager import get_cache_manager, CacheManager
except ImportError:
    # Handle relative imports for standalone usage
    logger.warning("Core modules not available - some features may be limited")


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class SchedulingStrategy(Enum):
    """Scheduling strategies"""
    FIFO = "fifo"  # First In First Out
    PRIORITY = "priority"  # Priority-based
    ROUND_ROBIN = "round_robin"  # Round-robin across devices
    LOAD_BALANCED = "load_balanced"  # Load-balanced allocation
    RESOURCE_AWARE = "resource_aware"  # Resource-aware scheduling


@dataclass
class TaskRequest:
    """Task request data structure"""
    task_id: str
    model_name: str
    method: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    device_preference: Optional[str] = None
    memory_requirement_mb: float = 0
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.created_at < other.created_at


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    device_used: Optional[str] = None
    memory_used_mb: float = 0
    execution_time: float = 0
    
    @property
    def success(self) -> bool:
        """Check if task completed successfully"""
        return self.status == TaskStatus.COMPLETED
    
    def to_dict(self) -> dict:
        """Convert result to dictionary"""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'success': self.success,
            'error': str(self.error) if self.error else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'device_used': self.device_used,
            'memory_used_mb': self.memory_used_mb,
            'execution_time': self.execution_time
        }


class TaskExecutor:
    """Individual task executor with resource management"""
    
    def __init__(self, 
                 executor_id: str,
                 model_manager: Any,
                 device_id: Optional[str] = None,
                 max_concurrent: int = 1):
        """
        Initialize task executor
        
        Args:
            executor_id: Unique executor identifier
            model_manager: Model manager instance
            device_id: Device to use for execution
            max_concurrent: Maximum concurrent tasks
        """
        self.executor_id = executor_id
        self.model_manager = model_manager
        self.device_id = device_id
        self.max_concurrent = max_concurrent
        
        # Execution state
        self._running_tasks: Dict[str, TaskRequest] = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=max_concurrent)
        self._lock = threading.RLock()
        
        # Statistics
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
        
        logger.info(f"Task executor {executor_id} initialized for device {device_id}")
    
    @property
    def is_available(self) -> bool:
        """Check if executor can accept new tasks"""
        with self._lock:
            return len(self._running_tasks) < self.max_concurrent
    
    @property
    def current_load(self) -> float:
        """Get current load as ratio (0.0 to 1.0)"""
        with self._lock:
            return len(self._running_tasks) / self.max_concurrent
    
    def execute_task(self, task: TaskRequest) -> Future[TaskResult]:
        """
        Execute a task asynchronously
        
        Args:
            task: Task to execute
            
        Returns:
            Future containing task result
        """
        if not self.is_available:
            raise RuntimeError(f"Executor {self.executor_id} is at capacity")
        
        with self._lock:
            self._running_tasks[task.task_id] = task
        
        future = self._thread_pool.submit(self._execute_task_impl, task)
        return future
    
    def _execute_task_impl(self, task: TaskRequest) -> TaskResult:
        """Internal task execution implementation"""
        start_time = time.time()
        result = TaskResult(
            task_id=task.task_id,
            status=TaskStatus.RUNNING,
            start_time=start_time,
            device_used=self.device_id
        )
        
        try:
            logger.debug(f"Executing task {task.task_id} on {self.executor_id}")
            
            # Load model
            model_adapter = self.model_manager.load_model(task.model_name)
            if not model_adapter:
                raise RuntimeError(f"Failed to load model: {task.model_name}")
            
            # Get method from adapter
            method = getattr(model_adapter, task.method, None)
            if not method:
                raise AttributeError(f"Method {task.method} not found in model adapter")
            
            # Execute with timeout
            if task.timeout:
                # Simple timeout implementation
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Task {task.task_id} timed out after {task.timeout}s")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(task.timeout))
            
            try:
                # Execute the method
                task_result = method(*task.args, **task.kwargs)
                
                if task.timeout:
                    signal.alarm(0)  # Cancel timeout
                
                result.result = task_result
                result.status = TaskStatus.COMPLETED
                self.completed_tasks += 1
                
            except TimeoutError:
                result.status = TaskStatus.TIMEOUT
                result.error = TimeoutError(f"Task timed out after {task.timeout}s")
                self.failed_tasks += 1
                
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            result.status = TaskStatus.FAILED
            result.error = e
            self.failed_tasks += 1
        
        finally:
            end_time = time.time()
            result.end_time = end_time
            result.execution_time = end_time - start_time
            self.total_execution_time += result.execution_time
            
            # Clean up
            with self._lock:
                self._running_tasks.pop(task.task_id, None)
            
            logger.debug(f"Task {task.task_id} completed in {result.execution_time:.2f}s")
        
        return result
    
    def get_stats(self) -> dict:
        """Get executor statistics"""
        with self._lock:
            avg_execution_time = (self.total_execution_time / self.completed_tasks 
                                 if self.completed_tasks > 0 else 0)
            
            return {
                'executor_id': self.executor_id,
                'device_id': self.device_id,
                'running_tasks': len(self._running_tasks),
                'max_concurrent': self.max_concurrent,
                'current_load': self.current_load,
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'total_execution_time': self.total_execution_time,
                'average_execution_time': avg_execution_time,
                'success_rate': (self.completed_tasks / 
                               (self.completed_tasks + self.failed_tasks)
                               if (self.completed_tasks + self.failed_tasks) > 0 else 0)
            }
    
    def shutdown(self):
        """Shutdown executor"""
        logger.info(f"Shutting down executor {self.executor_id}")
        self._thread_pool.shutdown(wait=True)


class TaskScheduler(BaseManager):
    """Enhanced task scheduler inheriting from BaseManager"""
    
    def __init__(self,
                 model_manager: Any,
                 strategy: SchedulingStrategy = SchedulingStrategy.RESOURCE_AWARE,
                 max_queue_size: int = 1000,
                 enable_gpu_monitoring: bool = True,
                 enable_caching: bool = True):
        """
        Initialize task scheduler with BaseManager capabilities
        
        Args:
            model_manager: Model manager instance
            strategy: Scheduling strategy
            max_queue_size: Maximum queue size
            enable_gpu_monitoring: Enable GPU monitoring
            enable_caching: Enable result caching
        """
        super().__init__("TaskScheduler")
        
        self.model_manager = model_manager
        self.strategy = strategy
        self.max_queue_size = max_queue_size
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.enable_caching = enable_caching
        
        # Task management (will be initialized in initialize() method)
        self._task_queue = None
        self._task_futures: Dict[str, Future] = {}
        self._task_results: Dict[str, TaskResult] = {}
        self._scheduler_lock = threading.RLock()
        
        # Executors
        self._executors: Dict[str, TaskExecutor] = {}
        self._round_robin_index = 0
        
        # External components
        self.gpu_monitor = None
        self.cache_manager = None
        
        # Scheduler state
        self._scheduler_running = False
        self._scheduler_thread = None
        
        # Statistics
        self.total_tasks_scheduled = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.queue_wait_times = []
        
        logger.info("TaskScheduler initialized with BaseManager capabilities")
    
    def initialize(self) -> bool:
        """
        Initialize task scheduler - implements BaseManager abstract method
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize task queue
            self._task_queue = PriorityQueue(maxsize=self.max_queue_size)
            
            # Initialize monitoring and caching if enabled
            if self.enable_gpu_monitoring:
                try:
                    self.gpu_monitor = get_gpu_monitor()
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU monitor: {e}")
            
            if self.enable_caching:
                try:
                    self.cache_manager = get_cache_manager()
                except Exception as e:
                    logger.warning(f"Failed to initialize cache manager: {e}")
            
            # Initialize executors
            self._initialize_executors()
            
            # Set up initial metrics
            self.update_metric('strategy', self.strategy.value)
            self.update_metric('max_queue_size', self.max_queue_size)
            self.update_metric('executors_count', len(self._executors))
            self.update_metric('gpu_monitoring_enabled', self.enable_gpu_monitoring)
            self.update_metric('caching_enabled', self.enable_caching)
            self.update_metric('initialization_time', time.time())
            
            # Start scheduler loop
            self._start_scheduler_loop()
            
            logger.info(f"TaskScheduler initialization completed with {self.strategy.value} strategy")
            return True
            
        except Exception as e:
            logger.error(f"TaskScheduler initialization failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """
        Cleanup task scheduler resources - implements BaseManager abstract method
        """
        try:
            # Stop scheduler loop
            self._stop_scheduler_loop()
            
            # Shutdown all executors
            for executor in self._executors.values():
                executor.shutdown()
            
            # Clear task data
            self._task_futures.clear()
            self._task_results.clear()
            self._executors.clear()
            
            # Update final metrics
            self.update_metric('cleanup_time', time.time())
            
            logger.info("TaskScheduler cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during TaskScheduler cleanup: {e}")
    
    def perform_health_check(self) -> HealthCheckResult:
        """
        Perform comprehensive health check
        
        Returns:
            Health check result with detailed status
        """
        start_time = time.time()
        
        try:
            # Check basic state
            if self.state not in [ManagerState.RUNNING, ManagerState.READY]:
                return HealthCheckResult(
                    status=HealthStatus.CRITICAL,
                    message=f"TaskScheduler not running (state: {self.state.value})",
                    details={'state': self.state.value},
                    timestamp=time.time(),
                    check_duration=time.time() - start_time
                )
            
            # Check scheduler loop
            scheduler_healthy = self._scheduler_running and (
                self._scheduler_thread and self._scheduler_thread.is_alive()
            )
            
            # Check task queue
            queue_size = self._task_queue.qsize() if self._task_queue else 0
            queue_full = queue_size >= self.max_queue_size * 0.9
            
            # Check executors
            available_executors = sum(1 for e in self._executors.values() if e.is_available)
            total_executors = len(self._executors)
            
            # Check average queue wait time
            avg_wait_time = (sum(self.queue_wait_times) / len(self.queue_wait_times)
                           if self.queue_wait_times else 0)
            
            # Check success rate
            total_processed = self.total_tasks_completed + self.total_tasks_failed
            success_rate = (self.total_tasks_completed / total_processed 
                          if total_processed > 0 else 1.0)
            
            # Determine health status
            if not scheduler_healthy:
                status = HealthStatus.CRITICAL
                message = "Scheduler loop not running"
            elif total_executors == 0:
                status = HealthStatus.CRITICAL
                message = "No executors available"
            elif queue_full:
                status = HealthStatus.WARNING
                message = f"Queue nearly full: {queue_size}/{self.max_queue_size}"
            elif available_executors == 0:
                status = HealthStatus.WARNING
                message = "No available executors"
            elif avg_wait_time > 10.0:  # 10 seconds
                status = HealthStatus.WARNING
                message = f"High queue wait time: {avg_wait_time:.1f}s"
            elif success_rate < 0.9:  # Less than 90% success
                status = HealthStatus.WARNING
                message = f"Low success rate: {success_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Scheduler healthy - {available_executors}/{total_executors} executors available"
            
            details = {
                'scheduler_running': scheduler_healthy,
                'queue_size': queue_size,
                'max_queue_size': self.max_queue_size,
                'available_executors': available_executors,
                'total_executors': total_executors,
                'avg_wait_time': avg_wait_time,
                'success_rate': success_rate,
                'total_scheduled': self.total_tasks_scheduled,
                'total_completed': self.total_tasks_completed,
                'total_failed': self.total_tasks_failed
            }
            
            return HealthCheckResult(
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                check_duration=time.time() - start_time
            )
            
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {e}",
                details={'error': str(e)},
                timestamp=time.time(),
                check_duration=time.time() - start_time
            )
    
    def _initialize_executors(self):
        """Initialize task executors for available devices"""
        if self.gpu_monitor:
            devices = self.gpu_monitor.get_devices()
            for device_id in devices.keys():
                if device_id == -1:  # MPS device
                    executor_id = "mps_executor"
                    device_str = "mps"
                else:
                    executor_id = f"cuda_{device_id}_executor"
                    device_str = f"cuda:{device_id}"
                
                executor = TaskExecutor(
                    executor_id=executor_id,
                    model_manager=self.model_manager,
                    device_id=device_str,
                    max_concurrent=2  # Allow 2 concurrent tasks per GPU
                )
                self._executors[executor_id] = executor
        
        # Always have a CPU executor as fallback
        cpu_executor = TaskExecutor(
            executor_id="cpu_executor",
            model_manager=self.model_manager,
            device_id="cpu",
            max_concurrent=4  # More concurrent tasks for CPU
        )
        self._executors["cpu_executor"] = cpu_executor
        
        logger.info(f"Initialized {len(self._executors)} task executors")
    
    def _start_scheduler_loop(self):
        """Start the scheduler loop"""
        if self._scheduler_running:
            return
        
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        
        self.update_metric('scheduler_started', time.time())
        logger.info("Task scheduler loop started")
    
    def _stop_scheduler_loop(self):
        """Stop the scheduler loop"""
        if not self._scheduler_running:
            return
        
        self._scheduler_running = False
        
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=10.0)
        
        self.update_metric('scheduler_stopped', time.time())
        logger.info("Task scheduler loop stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Task scheduler loop started")
        
        while self._scheduler_running:
            try:
                # Get next task from queue (with timeout to allow shutdown)
                try:
                    task = self._task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Select executor
                executor = self._select_executor(task)
                if not executor:
                    # No available executor, put task back in queue
                    self._task_queue.put(task)
                    time.sleep(0.1)  # Brief delay before retry
                    continue
                
                # Calculate queue wait time
                queue_wait_time = time.time() - task.created_at
                self.queue_wait_times.append(queue_wait_time)
                if len(self.queue_wait_times) > 1000:  # Keep last 1000 measurements
                    self.queue_wait_times.pop(0)
                
                # Execute task
                try:
                    future = executor.execute_task(task)
                    
                    with self._scheduler_lock:
                        self._task_futures[task.task_id] = future
                    
                    # Set up completion callback
                    def task_complete_callback(task_id: str, fut: Future):
                        try:
                            result = fut.result()
                            with self._scheduler_lock:
                                self._task_results[task_id] = result
                                self._task_futures.pop(task_id, None)
                            
                            if result.success:
                                self.total_tasks_completed += 1
                                self.increment_metric('tasks_completed')
                            else:
                                self.total_tasks_failed += 1
                                self.increment_metric('tasks_failed')
                            
                            logger.debug(f"Task {task_id} completed with status {result.status.value}")
                            
                        except Exception as e:
                            logger.error(f"Error in task completion callback: {e}")
                    
                    # Add callback with partial to capture task_id
                    from functools import partial
                    callback = partial(task_complete_callback, task.task_id)
                    future.add_done_callback(lambda f: callback(f))
                    
                    logger.debug(f"Task {task.task_id} assigned to executor {executor.executor_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to execute task {task.task_id}: {e}")
                    # Create failed result
                    failed_result = TaskResult(
                        task_id=task.task_id,
                        status=TaskStatus.FAILED,
                        error=e,
                        end_time=time.time()
                    )
                    with self._scheduler_lock:
                        self._task_results[task.task_id] = failed_result
                    self.total_tasks_failed += 1
                    self.increment_metric('tasks_failed')
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                self.increment_metric('scheduler_errors')
                time.sleep(1.0)  # Brief delay on error
        
        logger.info("Task scheduler loop stopped")
    
    def _select_executor(self, task: TaskRequest) -> Optional[TaskExecutor]:
        """
        Select best executor for a task
        
        Args:
            task: Task to schedule
            
        Returns:
            Selected executor or None if none available
        """
        available_executors = [e for e in self._executors.values() if e.is_available]
        
        if not available_executors:
            return None
        
        # Handle device preference
        if task.device_preference:
            preferred_executors = [e for e in available_executors 
                                 if e.device_id == task.device_preference]
            if preferred_executors:
                available_executors = preferred_executors
        
        # Apply scheduling strategy
        if self.strategy == SchedulingStrategy.FIFO:
            return available_executors[0]
        
        elif self.strategy == SchedulingStrategy.PRIORITY:
            # For priority scheduling, just return first available
            return available_executors[0]
        
        elif self.strategy == SchedulingStrategy.ROUND_ROBIN:
            # Round-robin selection
            executor = available_executors[self._round_robin_index % len(available_executors)]
            self._round_robin_index += 1
            return executor
        
        elif self.strategy == SchedulingStrategy.LOAD_BALANCED:
            # Select executor with lowest current load
            return min(available_executors, key=lambda e: e.current_load)
        
        elif self.strategy == SchedulingStrategy.RESOURCE_AWARE:
            # Resource-aware selection considering GPU memory and utilization
            if self.gpu_monitor and task.memory_requirement_mb > 0:
                # Filter executors by memory requirement
                suitable_executors = []
                for executor in available_executors:
                    if executor.device_id.startswith('cuda:'):
                        device_id = int(executor.device_id.split(':')[1])
                        memory_info = self.gpu_monitor.get_memory_info(device_id)
                        if memory_info and memory_info.free_mb >= task.memory_requirement_mb:
                            suitable_executors.append(executor)
                    elif executor.device_id == 'cpu':
                        suitable_executors.append(executor)  # CPU always suitable
                
                if suitable_executors:
                    available_executors = suitable_executors
            
            # Select based on lowest load among suitable executors
            return min(available_executors, key=lambda e: e.current_load)
        
        # Default fallback
        return available_executors[0]
    
    def submit_task(self,
                   model_name: str,
                   method: str,
                   args: tuple = (),
                   kwargs: dict = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[float] = None,
                   device_preference: Optional[str] = None,
                   memory_requirement_mb: float = 0,
                   task_id: Optional[str] = None,
                   metadata: dict = None) -> str:
        """
        Submit a task for execution
        
        Args:
            model_name: Name of model to use
            method: Method to call on model adapter
            args: Positional arguments for method
            kwargs: Keyword arguments for method
            priority: Task priority
            timeout: Task timeout in seconds
            device_preference: Preferred device (e.g., 'cuda:0', 'cpu')
            memory_requirement_mb: Memory requirement in MB
            task_id: Custom task ID (auto-generated if None)
            metadata: Additional metadata
            
        Returns:
            Task ID
        """
        if kwargs is None:
            kwargs = {}
        if metadata is None:
            metadata = {}
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = str(uuid.uuid4())
        
        # Check if result is cached
        if self.cache_manager:
            cache_key = f"{model_name}_{method}_{hash(str(args))}_{hash(str(sorted(kwargs.items())))}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Task {task_id} result found in cache")
                # Create completed result from cache
                cached_task_result = TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    result=cached_result,
                    start_time=time.time(),
                    end_time=time.time(),
                    execution_time=0.0
                )
                with self._scheduler_lock:
                    self._task_results[task_id] = cached_task_result
                self.total_tasks_completed += 1
                self.increment_metric('cache_hits')
                return task_id
        
        # Create task request
        task = TaskRequest(
            task_id=task_id,
            model_name=model_name,
            method=method,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            device_preference=device_preference,
            memory_requirement_mb=memory_requirement_mb,
            metadata=metadata
        )
        
        # Add to queue
        try:
            self._task_queue.put(task, block=False)
            self.total_tasks_scheduled += 1
            self.increment_metric('tasks_scheduled')
            logger.debug(f"Task {task_id} queued for execution")
            return task_id
            
        except:
            raise RuntimeError("Task queue is full")
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Get task result
        
        Args:
            task_id: Task ID
            timeout: Maximum time to wait for result
            
        Returns:
            Task result or None if not available
        """
        start_time = time.time()
        
        while True:
            with self._scheduler_lock:
                # Check if result is ready
                if task_id in self._task_results:
                    result = self._task_results[task_id]
                    
                    # Cache successful results
                    if (self.cache_manager and result.success and 
                        hasattr(result, 'result') and result.result is not None):
                        cache_key = f"result_{task_id}"
                        self.cache_manager.put(cache_key, result.result)
                    
                    return result
                
                # Check if task is still running
                if task_id in self._task_futures:
                    future = self._task_futures[task_id]
                    if future.done():
                        # Result should be available, wait a bit
                        time.sleep(0.01)
                        continue
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)  # Brief delay before checking again
    
    def get_queue_status(self) -> dict:
        """Get current queue status"""
        with self._scheduler_lock:
            running_tasks = len(self._task_futures)
            completed_tasks = len([r for r in self._task_results.values() 
                                 if r.status == TaskStatus.COMPLETED])
            failed_tasks = len([r for r in self._task_results.values() 
                              if r.status == TaskStatus.FAILED])
            
            avg_queue_wait = (sum(self.queue_wait_times) / len(self.queue_wait_times)
                            if self.queue_wait_times else 0)
            
            return {
                'queue_size': self._task_queue.qsize() if self._task_queue else 0,
                'max_queue_size': self.max_queue_size,
                'running_tasks': running_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'total_scheduled': self.total_tasks_scheduled,
                'average_queue_wait_time': avg_queue_wait,
                'scheduler_running': self._scheduler_running
            }
    
    def get_executor_stats(self) -> Dict[str, dict]:
        """Get statistics for all executors"""
        return {executor_id: executor.get_stats() 
                for executor_id, executor in self._executors.items()}

    def get_system_stats(self) -> dict:
        """Get comprehensive system statistics"""
        queue_status = self.get_queue_status()
        executor_stats = self.get_executor_stats

        # Calculate system-wide metrics
        total_executor_load = sum(stats['current_load'] for stats in executor_stats.values())
        avg_executor_load = total_executor_load / len(executor_stats) if executor_stats else 0

        total_success_rate = (queue_status['completed_tasks'] / (queue_status['completed_tasks'] + queue_status['failed_tasks']) if (queue_status['completed_tasks'] + queue_status['failed_tasks']) > 0 else 0)

        stats = {
            'scheduler': {
                'strategy': self.strategy.value,
                'running': self._scheduler_running,
                'total_executors': len(self._executors),
                'average_executor_load': avg_executor_load,
                'success_rate': total_success_rate
            },
            'queue': queue_status,
            'executors': executor_stats
        }

        # Add GPU information if available
        if self.gpu_monitor:
            try:
                stats['gpu'] = self.gpu_monitor.get_device_utilization_summary()
            except Exception as e:
                logger.warning(f"Failed to get GPU stats: {e}")
                stats['gpu'] = {'error': str(e)}
        
        return stats


# Global scheduler instance
_scheduler = None

def get_scheduler(model_manager: Any = None, **kwargs) -> TaskScheduler:
    """
    Get global task scheduler instance
    
    Args:
        model_manager: Model manager instance
        **kwargs: Additional scheduler parameters
        
    Returns:
        Global scheduler instance
    """
    global _scheduler
    if _scheduler is None:
        if model_manager is None:
            raise ValueError("model_manager is required for scheduler initialization")
        _scheduler = TaskScheduler(model_manager, **kwargs)
        # Auto-start the scheduler
        if not _scheduler.start():
            logger.error("Failed to start TaskScheduler")
    return _scheduler


def submit_inference_task(model_name: str, 
                         method: str = 'predict',
                         *args, 
                         priority: TaskPriority = TaskPriority.NORMAL,
                         **kwargs) -> str:
    """
    Convenience function to submit an inference task
    
    Args:
        model_name: Name of model to use
        method: Method to call (default: 'predict')
        *args: Arguments for the method
        priority: Task priority
        **kwargs: Keyword arguments for the method
        
    Returns:
        Task ID
    """
    scheduler = get_scheduler()
    return scheduler.submit_task(
        model_name=model_name,
        method=method,
        args=args,
        kwargs=kwargs,
        priority=priority
    )