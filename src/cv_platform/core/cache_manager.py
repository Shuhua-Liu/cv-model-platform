"""
Enhanced Cache Manager - Inheriting from BaseManager

Advanced caching system with BaseManager integration for state management,
health monitoring, persistence, and automatic cleanup mechanisms.
"""

import os
import pickle
import time
import threading
import hashlib
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from .base_manager import BaseManager, ManagerState, HealthStatus, HealthCheckResult

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory monitoring disabled")


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based


@dataclass
class CacheEntry:
    """Cache entry metadata"""
    key: str
    data: Any
    size_bytes: int
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired based on TTL"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    @property
    def age(self) -> float:
        """Get entry age in seconds"""
        return time.time() - self.created_at


class CacheStats:
    """Cache statistics tracker"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_bytes = 0
        self.entry_count = 0
        self._lock = threading.Lock()
    
    def record_hit(self):
        """Record cache hit"""
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        """Record cache miss"""
        with self._lock:
            self.misses += 1
    
    def record_eviction(self, size_bytes: int = 0):
        """Record cache eviction"""
        with self._lock:
            self.evictions += 1
            self.size_bytes -= size_bytes
    
    def update_size(self, delta_bytes: int, delta_entries: int = 0):
        """Update cache size statistics"""
        with self._lock:
            self.size_bytes += delta_bytes
            self.entry_count += delta_entries
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'size_bytes': self.size_bytes,
            'size_mb': self.size_bytes / (1024 * 1024),
            'entry_count': self.entry_count
        }


class CacheManager(BaseManager):
    """Enhanced cache manager inheriting from BaseManager"""
    
    def __init__(self,
                 max_size_bytes: int = 2 * 1024 * 1024 * 1024,  # 2GB default
                 max_entries: int = 1000,
                 strategy: CacheStrategy = CacheStrategy.LRU,
                 default_ttl: Optional[float] = None,
                 persistence_dir: Optional[Union[str, Path]] = None,
                 memory_threshold: float = 0.8,
                 cleanup_interval: int = 300):  # 5 minutes
        """
        Initialize cache manager with BaseManager capabilities
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            max_entries: Maximum number of entries
            strategy: Cache eviction strategy
            default_ttl: Default TTL for entries (seconds)
            persistence_dir: Directory for cache persistence
            memory_threshold: Memory usage threshold for eviction (0.0-1.0)
            cleanup_interval: Automatic cleanup interval in seconds
        """
        super().__init__("CacheManager")
        
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.memory_threshold = memory_threshold
        self.cleanup_interval = cleanup_interval
        
        # Internal storage (will be initialized in initialize() method)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._cache_lock = threading.RLock()
        self._stats = CacheStats()
        
        # Persistence
        self.persistence_dir = Path(persistence_dir) if persistence_dir else None
        
        # Background cleanup
        self._cleanup_timer = None
        self._cleanup_active = False
        
        logger.info("CacheManager initialized with BaseManager capabilities")
    
    def initialize(self) -> bool:
        """
        Initialize cache manager - implements BaseManager abstract method
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Setup persistence directory
            if self.persistence_dir:
                self.persistence_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cache persistence directory: {self.persistence_dir}")
            
            # Set up initial metrics
            self.update_metric('max_size_bytes', self.max_size_bytes)
            self.update_metric('max_entries', self.max_entries)
            self.update_metric('strategy', self.strategy.value)
            self.update_metric('memory_threshold', self.memory_threshold)
            self.update_metric('cleanup_interval', self.cleanup_interval)
            self.update_metric('initialization_time', time.time())
            
            # Start background cleanup
            self._start_cleanup_timer()
            
            logger.info(f"CacheManager initialization completed - Strategy: {self.strategy.value}, "
                       f"Max size: {self.max_size_bytes / (1024*1024):.1f}MB, "
                       f"Max entries: {self.max_entries}")
            return True
            
        except Exception as e:
            logger.error(f"CacheManager initialization failed: {e}")
            return False
    
    def cleanup(self) -> None:
        """
        Cleanup cache manager resources - implements BaseManager abstract method
        """
        try:
            # Stop background cleanup
            self._stop_cleanup_timer()
            
            # Persist all entries if persistence is enabled
            if self.persistence_dir:
                with self._cache_lock:
                    for entry in self._cache.values():
                        self._persist_entry(entry)
            
            # Clear cache
            with self._cache_lock:
                self._cache.clear()
            
            # Update final metrics
            self.update_metric('cleanup_time', time.time())
            
            logger.info("CacheManager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during CacheManager cleanup: {e}")
    
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
                    message=f"CacheManager not running (state: {self.state.value})",
                    details={'state': self.state.value},
                    timestamp=time.time(),
                    check_duration=time.time() - start_time
                )
            
            # Check cache statistics
            stats = self._stats.to_dict()
            size_ratio = stats['size_bytes'] / self.max_size_bytes
            entry_ratio = stats['entry_count'] / self.max_entries
            hit_rate = stats['hit_rate']
            
            # Check memory pressure
            memory_pressure = self._check_memory_pressure()
            
            # Check persistence directory
            persistence_healthy = True
            if self.persistence_dir:
                persistence_healthy = self.persistence_dir.exists() and os.access(self.persistence_dir, os.W_OK)
            
            # Determine health status
            if not persistence_healthy:
                status = HealthStatus.CRITICAL
                message = "Persistence directory not accessible"
            elif size_ratio > 0.95 or entry_ratio > 0.95:
                status = HealthStatus.CRITICAL
                message = "Cache nearly full"
            elif memory_pressure:
                status = HealthStatus.WARNING
                message = "System memory pressure detected"
            elif size_ratio > 0.8 or entry_ratio > 0.8:
                status = HealthStatus.WARNING
                message = "Cache usage high"
            elif hit_rate < 0.3 and stats['hits'] + stats['misses'] > 100:
                status = HealthStatus.WARNING
                message = f"Low cache hit rate: {hit_rate:.1%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Cache healthy - {stats['entry_count']} entries, {hit_rate:.1%} hit rate"
            
            details = {
                'cache_stats': stats,
                'size_ratio': size_ratio,
                'entry_ratio': entry_ratio,
                'memory_pressure': memory_pressure,
                'persistence_healthy': persistence_healthy,
                'cleanup_active': self._cleanup_active
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
    
    def _generate_key(self, key: Union[str, bytes]) -> str:
        """Generate cache key from input"""
        if isinstance(key, bytes):
            return hashlib.md5(key).hexdigest()
        elif isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()
    
    def _get_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation for unpicklable objects
            if hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            return 1024  # Default size estimate
    
    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        if not PSUTIL_AVAILABLE:
            return False
            
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0 > self.memory_threshold
        except Exception:
            return False
    
    def _evict_entries(self, target_bytes: int = 0) -> int:
        """
        Evict entries based on strategy
        
        Args:
            target_bytes: Target bytes to free (0 = evict one entry)
            
        Returns:
            Number of bytes freed
        """
        freed_bytes = 0
        evicted_count = 0
        
        with self._cache_lock:
            while self._cache and (target_bytes == 0 or freed_bytes < target_bytes):
                if self.strategy == CacheStrategy.LRU:
                    # Evict least recently used
                    key = next(iter(self._cache))
                elif self.strategy == CacheStrategy.LFU:
                    # Evict least frequently used
                    key = min(self._cache.keys(), 
                             key=lambda k: self._cache[k].access_count)
                elif self.strategy == CacheStrategy.FIFO:
                    # Evict oldest entry
                    key = min(self._cache.keys(), 
                             key=lambda k: self._cache[k].created_at)
                else:  # TTL or fallback to LRU
                    key = next(iter(self._cache))
                
                entry = self._cache[key]
                freed_bytes += entry.size_bytes
                evicted_count += 1
                
                # Save to persistence if configured
                self._persist_entry(entry)
                
                del self._cache[key]
                self._stats.record_eviction(entry.size_bytes)
                
                logger.debug(f"Evicted cache entry: {key} ({entry.size_bytes} bytes)")
                
                if target_bytes == 0:  # Only evict one entry
                    break
        
        self._stats.update_size(-freed_bytes, -evicted_count)
        
        if evicted_count > 0:
            self.update_metric('total_evictions', self.get_metric('total_evictions', 0) + evicted_count)
            logger.info(f"Evicted {evicted_count} entries, freed {freed_bytes / (1024*1024):.1f}MB")
        
        return freed_bytes
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries"""
        expired_keys = []
        freed_bytes = 0
        
        with self._cache_lock:
            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
                    freed_bytes += entry.size_bytes
            
            for key in expired_keys:
                del self._cache[key]
                self._stats.record_eviction()
        
        if expired_keys:
            self._stats.update_size(-freed_bytes, -len(expired_keys))
            self.update_metric('expired_entries', self.get_metric('expired_entries', 0) + len(expired_keys))
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")
        
        return len(expired_keys)
    
    def _start_cleanup_timer(self):
        """Start background cleanup timer"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        self._cleanup_active = True
        self._cleanup_timer = threading.Timer(
            self.cleanup_interval, 
            self._background_cleanup
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def _stop_cleanup_timer(self):
        """Stop background cleanup timer"""
        self._cleanup_active = False
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
    
    def _background_cleanup(self):
        """Background cleanup task"""
        try:
            # Clean expired entries
            expired_count = self._cleanup_expired()
            
            # Check memory pressure and evict if needed
            if self._check_memory_pressure():
                target_bytes = self._stats.size_bytes * 0.2  # Free 20%
                self._evict_entries(int(target_bytes))
                self.increment_metric('memory_pressure_cleanups')
            
            # Update cleanup metrics
            self.increment_metric('cleanup_cycles')
            self.update_metric('last_cleanup_time', time.time())
        
        except Exception as e:
            logger.error(f"Error in background cleanup: {e}")
            self.increment_metric('cleanup_errors')
        
        finally:
            # Restart timer if still active
            if self._cleanup_active:
                self._start_cleanup_timer()
    
    def _persist_entry(self, entry: CacheEntry):
        """Persist cache entry to disk"""
        if not self.persistence_dir:
            return
        
        try:
            file_path = self.persistence_dir / f"{entry.key}.cache"
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'data': entry.data,
                    'metadata': {
                        'created_at': entry.created_at,
                        'access_count': entry.access_count,
                        'size_bytes': entry.size_bytes
                    }
                }, f)
        except Exception as e:
            logger.warning(f"Failed to persist cache entry {entry.key}: {e}")
    
    def put(self, 
            key: Union[str, bytes], 
            data: Any, 
            ttl: Optional[float] = None) -> bool:
        """
        Store data in cache
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds (overrides default)
            
        Returns:
            True if stored successfully
        """
        cache_key = self._generate_key(key)
        size_bytes = self._get_object_size(data)
        
        # Check if entry would exceed size limits
        if size_bytes > self.max_size_bytes:
            logger.warning(f"Entry too large for cache: {size_bytes} bytes")
            self.increment_metric('oversized_entries')
            return False
        
        with self._cache_lock:
            # Make room if needed
            while (len(self._cache) >= self.max_entries or 
                   self._stats.size_bytes + size_bytes > self.max_size_bytes):
                if not self._cache:  # No entries to evict
                    break
                self._evict_entries()
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                data=data,
                size_bytes=size_bytes,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl or self.default_ttl
            )
            
            # Store in cache
            self._cache[cache_key] = entry
            self._stats.update_size(size_bytes, 1)
            
            # Update metrics
            self.increment_metric('entries_added')
            
            logger.debug(f"Cached entry: {cache_key} ({size_bytes} bytes)")
            return True
    
    def get(self, key: Union[str, bytes]) -> Optional[Any]:
        """
        Retrieve data from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        cache_key = self._generate_key(key)
        
        with self._cache_lock:
            if cache_key not in self._cache:
                self._stats.record_miss()
                
                # Try loading from persistence
                if self.persistence_dir:
                    persisted_data = self._load_persisted(cache_key)
                    if persisted_data:
                        self.increment_metric('persistence_hits')
                        return persisted_data
                
                return None
            
            entry = self._cache[cache_key]
            
            # Check if expired
            if entry.is_expired:
                del self._cache[cache_key]
                self._stats.record_eviction(entry.size_bytes)
                self._stats.update_size(-entry.size_bytes, -1)
                self._stats.record_miss()
                return None
            
            # Update access info
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end for LRU
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(cache_key)
            
            self._stats.record_hit()
            return entry.data
    
    def _load_persisted(self, cache_key: str) -> Optional[Any]:
        """Load persisted cache entry"""
        try:
            file_path = self.persistence_dir / f"{cache_key}.cache"
            if not file_path.exists():
                return None
            
            with open(file_path, 'rb') as f:
                cached = pickle.load(f)
                return cached['data']
        
        except Exception as e:
            logger.warning(f"Failed to load persisted cache entry {cache_key}: {e}")
            return None
    
    def delete(self, key: Union[str, bytes]) -> bool:
        """
        Delete entry from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was deleted
        """
        cache_key = self._generate_key(key)
        
        with self._cache_lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                del self._cache[cache_key]
                self._stats.update_size(-entry.size_bytes, -1)
                
                # Remove from persistence
                if self.persistence_dir:
                    try:
                        file_path = self.persistence_dir / f"{cache_key}.cache"
                        if file_path.exists():
                            file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove persisted entry: {e}")
                
                self.increment_metric('entries_deleted')
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self._cache_lock:
            self._cache.clear()
            self._stats = CacheStats()
            
            # Clear persistence directory
            if self.persistence_dir and self.persistence_dir.exists():
                try:
                    for file_path in self.persistence_dir.glob("*.cache"):
                        file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clear persistence directory: {e}")
        
        self.increment_metric('cache_clears')
        logger.info("Cache cleared")
    
    def exists(self, key: Union[str, bytes]) -> bool:
        """Check if key exists in cache"""
        cache_key = self._generate_key(key)
        
        with self._cache_lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                return not entry.is_expired
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self._stats.to_dict()
        
        with self._cache_lock:
            stats.update({
                'current_entries': len(self._cache),
                'max_entries': self.max_entries,
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'strategy': self.strategy.value,
                'memory_pressure': self._check_memory_pressure(),
                'oldest_entry_age': min([e.age for e in self._cache.values()]) 
                                   if self._cache else 0,
                'newest_entry_age': max([e.age for e in self._cache.values()]) 
                                   if self._cache else 0
            })
        
        return stats
    
    def list_keys(self) -> List[str]:
        """List all cache keys"""
        with self._cache_lock:
            return list(self._cache.keys())
    
    def manual_cleanup(self):
        """Perform manual cleanup"""
        with self._cache_lock:
            expired_count = self._cleanup_expired()
            
            # Force memory pressure cleanup if needed
            if self._check_memory_pressure():
                target_bytes = self._stats.size_bytes * 0.3  # Free 30%
                freed_bytes = self._evict_entries(int(target_bytes))
                logger.info(f"Memory pressure cleanup freed {freed_bytes / (1024*1024):.1f}MB")
        
        self.increment_metric('manual_cleanups')
        return expired_count


# Global cache manager instance
_cache_manager = None

def get_cache_manager(**kwargs) -> CacheManager:
    """
    Get global cache manager instance
    
    Args:
        **kwargs: Parameters for cache manager initialization
        
    Returns:
        Global cache manager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(**kwargs)
        # Auto-start the manager
        if not _cache_manager.start():
            logger.error("Failed to start CacheManager")
    return _cache_manager