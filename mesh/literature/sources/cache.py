"""
TTL-based caching for literature search sources.

Provides per-source caching with configurable TTL (time-to-live) to reduce
redundant API calls and improve response times.

Features:
- In-memory cache with TTL expiration
- Optional disk persistence
- Per-source cache isolation
- Thread-safe operations
- Statistics tracking
"""

import json
import hashlib
import time
import threading
from pathlib import Path
from dataclasses import dataclass, field

from ...paths import resolve_path
from typing import Any, Optional, Callable


@dataclass
class CacheEntry:
    """A single cache entry with expiration."""
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class TTLCache:
    """
    In-memory cache with TTL expiration.

    Thread-safe implementation with automatic cleanup of expired entries.

    Example:
        cache = TTLCache(ttl_seconds=3600, max_size=1000)

        # Store a value
        cache.set("key", {"data": "value"})

        # Retrieve with default
        result = cache.get("key", default=None)

        # Check if exists
        if cache.has("key"):
            ...
    """

    def __init__(
        self,
        ttl_seconds: float = 3600,  # 1 hour default
        max_size: int = 1000,
        cleanup_interval: float = 300,  # 5 minutes
    ):
        """
        Initialize the cache.

        Args:
            ttl_seconds: Time-to-live for entries in seconds
            max_size: Maximum number of entries (oldest evicted when full)
            cleanup_interval: How often to run cleanup (seconds)
        """
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval

        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self._last_cleanup = time.time()

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self.cleanup_interval:
            self._cleanup()
            self._last_cleanup = now

    def _cleanup(self) -> None:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                k for k, v in self._cache.items()
                if v.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
                self._stats.evictions += 1

    def _evict_oldest(self) -> None:
        """Evict oldest entries if over max size."""
        with self._lock:
            while len(self._cache) >= self.max_size:
                # Find oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].created_at
                )
                del self._cache[oldest_key]
                self._stats.evictions += 1

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store (must be JSON-serializable for disk persistence)
            ttl: Optional custom TTL for this entry (defaults to cache TTL)
        """
        self._maybe_cleanup()

        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict_oldest()

            effective_ttl = ttl if ttl is not None else self.ttl
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=time.time() + effective_ttl,
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key
            default: Value to return if not found or expired

        Returns:
            Cached value or default
        """
        self._maybe_cleanup()

        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return default

            if entry.is_expired:
                del self._cache[key]
                self._stats.evictions += 1
                self._stats.misses += 1
                return default

            self._stats.hits += 1
            return entry.value

    def has(self, key: str) -> bool:
        """Check if a non-expired entry exists."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                self._stats.evictions += 1
                return False
            return True

    def delete(self, key: str) -> bool:
        """Delete an entry. Returns True if existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)


class DiskCache:
    """
    Disk-persistent cache with TTL expiration.

    Stores cache entries as JSON files for persistence across restarts.
    Uses file modification time for TTL tracking.

    Example:
        cache = DiskCache(
            cache_dir="~/.cache/literature_search",
            ttl_seconds=86400,  # 24 hours
        )

        cache.set("search:transformer", results)
        results = cache.get("search:transformer")
    """

    def __init__(
        self,
        cache_dir: str = "~/.cache/literature_search",
        ttl_seconds: float = 86400,  # 24 hours default
        max_size_mb: float = 100,  # Max cache size in MB
    ):
        """
        Initialize disk cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_seconds: Time-to-live for entries
            max_size_mb: Maximum total cache size in megabytes
        """
        self.cache_dir = Path(resolve_path(str(cache_dir)))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._stats = CacheStats()
        self._lock = threading.RLock()

    def _key_to_path(self, key: str) -> Path:
        """Convert cache key to file path."""
        # Use hash to handle special characters and long keys
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.cache_dir / f"{key_hash}.json"

    def _is_expired(self, path: Path) -> bool:
        """Check if a cache file is expired based on mtime."""
        if not path.exists():
            return True
        mtime = path.stat().st_mtime
        return time.time() - mtime > self.ttl

    def set(self, key: str, value: Any) -> None:
        """Store a value to disk."""
        with self._lock:
            path = self._key_to_path(key)
            try:
                # Store value with metadata
                data = {
                    "key": key,
                    "value": value,
                    "created_at": time.time(),
                }
                path.write_text(json.dumps(data, default=str))
            except (OSError, TypeError) as e:
                # Silently fail on disk errors
                pass

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from disk."""
        with self._lock:
            path = self._key_to_path(key)

            if not path.exists():
                self._stats.misses += 1
                return default

            if self._is_expired(path):
                try:
                    path.unlink()
                except OSError:
                    pass
                self._stats.evictions += 1
                self._stats.misses += 1
                return default

            try:
                data = json.loads(path.read_text())
                self._stats.hits += 1
                return data.get("value", default)
            except (OSError, json.JSONDecodeError):
                self._stats.misses += 1
                return default

    def has(self, key: str) -> bool:
        """Check if a non-expired entry exists."""
        path = self._key_to_path(key)
        return path.exists() and not self._is_expired(path)

    def delete(self, key: str) -> bool:
        """Delete a cache entry."""
        with self._lock:
            path = self._key_to_path(key)
            if path.exists():
                try:
                    path.unlink()
                    return True
                except OSError:
                    return False
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            for path in self.cache_dir.glob("*.json"):
                try:
                    path.unlink()
                except OSError:
                    pass

    def cleanup(self) -> int:
        """Remove expired entries. Returns number removed."""
        count = 0
        with self._lock:
            for path in self.cache_dir.glob("*.json"):
                if self._is_expired(path):
                    try:
                        path.unlink()
                        count += 1
                        self._stats.evictions += 1
                    except OSError:
                        pass
        return count

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def size_bytes(self) -> int:
        """Total cache size in bytes."""
        total = 0
        for path in self.cache_dir.glob("*.json"):
            try:
                total += path.stat().st_size
            except OSError:
                pass
        return total


class LiteratureCache:
    """
    Unified cache for all literature search sources.

    Provides namespaced caching for different sources with configurable
    TTLs per source type.

    Example:
        cache = LiteratureCache()

        # Cache a search result
        cache.set_search("arxiv", "transformer attention", results)

        # Get cached result
        results = cache.get_search("arxiv", "transformer attention")

        # Cache paper metadata
        cache.set_paper("arxiv", "1706.03762", paper_data)
    """

    # Default TTLs per source (in seconds)
    DEFAULT_TTLS = {
        "arxiv": 3600,           # 1 hour (papers don't change often)
        "pubmed": 3600,          # 1 hour
        "semantic_scholar": 1800, # 30 minutes (metadata updates)
        "google_scholar": 900,    # 15 minutes (results vary)
        "fulltext": 86400,       # 24 hours (full text rarely changes)
    }

    def __init__(
        self,
        use_disk_cache: bool = True,
        disk_cache_dir: str = "~/.cache/literature_search",
        memory_max_size: int = 500,
        custom_ttls: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the literature cache.

        Args:
            use_disk_cache: Whether to persist cache to disk
            disk_cache_dir: Directory for disk cache
            memory_max_size: Max entries in memory cache
            custom_ttls: Override default TTLs per source
        """
        self.ttls = {**self.DEFAULT_TTLS, **(custom_ttls or {})}

        # Memory cache for fast lookups
        self._memory = TTLCache(
            ttl_seconds=max(self.ttls.values()),
            max_size=memory_max_size,
        )

        # Disk cache for persistence
        self._disk: Optional[DiskCache] = None
        if use_disk_cache:
            self._disk = DiskCache(
                cache_dir=disk_cache_dir,
                ttl_seconds=max(self.ttls.values()) * 2,  # Disk TTL longer
            )

    def _make_key(self, namespace: str, *parts: str) -> str:
        """Create a cache key from namespace and parts."""
        return f"{namespace}:" + ":".join(str(p) for p in parts)

    def set_search(
        self,
        source: str,
        query: str,
        results: list[dict],
        max_results: int = 10,
    ) -> None:
        """Cache search results."""
        key = self._make_key("search", source, query, str(max_results))
        ttl = self.ttls.get(source, 1800)

        self._memory.set(key, results, ttl=ttl)
        if self._disk:
            self._disk.set(key, results)

    def get_search(
        self,
        source: str,
        query: str,
        max_results: int = 10,
    ) -> Optional[list[dict]]:
        """Get cached search results."""
        key = self._make_key("search", source, query, str(max_results))

        # Try memory first
        result = self._memory.get(key)
        if result is not None:
            return result

        # Fall back to disk
        if self._disk:
            result = self._disk.get(key)
            if result is not None:
                # Promote to memory
                ttl = self.ttls.get(source, 1800)
                self._memory.set(key, result, ttl=ttl)
                return result

        return None

    def set_paper(self, source: str, paper_id: str, data: dict) -> None:
        """Cache paper metadata."""
        key = self._make_key("paper", source, paper_id)
        ttl = self.ttls.get(source, 3600)

        self._memory.set(key, data, ttl=ttl)
        if self._disk:
            self._disk.set(key, data)

    def get_paper(self, source: str, paper_id: str) -> Optional[dict]:
        """Get cached paper metadata."""
        key = self._make_key("paper", source, paper_id)

        result = self._memory.get(key)
        if result is not None:
            return result

        if self._disk:
            result = self._disk.get(key)
            if result is not None:
                ttl = self.ttls.get(source, 3600)
                self._memory.set(key, result, ttl=ttl)
                return result

        return None

    def set_fulltext(self, paper_id: str, text: str) -> None:
        """Cache full text (disk only due to size)."""
        key = self._make_key("fulltext", paper_id)
        # Full text only in disk cache
        if self._disk:
            self._disk.set(key, text)

    def get_fulltext(self, paper_id: str) -> Optional[str]:
        """Get cached full text."""
        key = self._make_key("fulltext", paper_id)
        if self._disk:
            return self._disk.get(key)
        return None

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "memory": {
                "hits": self._memory.stats.hits,
                "misses": self._memory.stats.misses,
                "hit_rate": self._memory.stats.hit_rate,
                "size": self._memory.size,
            }
        }
        if self._disk:
            stats["disk"] = {
                "hits": self._disk.stats.hits,
                "misses": self._disk.stats.misses,
                "hit_rate": self._disk.stats.hit_rate,
                "size_mb": self._disk.size_bytes / (1024 * 1024),
            }
        return stats

    def clear(self, source: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            source: If provided, only clear entries for this source.
                   If None, clear everything.
        """
        if source is None:
            self._memory.clear()
            if self._disk:
                self._disk.clear()
        else:
            # Selective clearing not implemented for simplicity
            # Would require key prefix scanning
            pass
