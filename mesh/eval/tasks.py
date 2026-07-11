"""
Task definitions for model evaluation.

Each task defines:
- Setup files (initial state)
- Prompt (what to ask the model)
- Validation (how to check success)
- Scoring criteria
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional
import json
import subprocess
import tempfile
import shutil


class TaskCategory(Enum):
    """Categories of evaluation tasks."""
    BUGFIX = "bugfix"              # Find and fix bugs
    REFACTOR = "refactor"         # Multi-file refactoring
    AMBIGUOUS = "ambiguous"       # Requires clarification
    ITERATIVE = "iterative"       # Multi-round refinement
    LONG_CONTEXT = "long_context" # Large codebase navigation
    GREENFIELD = "greenfield"     # Build from scratch


@dataclass
class TaskResult:
    """Result of running a task."""
    task_id: str
    model: str
    success: bool
    score: float  # 0.0 to 1.0
    elapsed_seconds: float
    tool_calls: int
    tools_used: dict[str, int]
    tests_passed: int
    tests_total: int
    error: Optional[str] = None
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "success": self.success,
            "score": self.score,
            "elapsed_seconds": self.elapsed_seconds,
            "tool_calls": self.tool_calls,
            "tools_used": self.tools_used,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "error": self.error,
            "details": self.details,
        }


@dataclass
class EvalTask:
    """
    A single evaluation task.

    Attributes:
        id: Unique task identifier
        name: Human-readable task name
        category: Task category for filtering
        description: Full task description for the model
        setup_files: Dict of {relative_path: content} for initial state
        validation_script: Python script that returns (passed, total, details)
        max_iterations: Maximum tool call iterations
        timeout_seconds: Task timeout
        partial_credit: Whether to award partial credit for partial completion
    """
    id: str
    name: str
    category: TaskCategory
    description: str
    setup_files: dict[str, str]
    validation_script: str
    max_iterations: int = 50
    timeout_seconds: int = 600
    partial_credit: bool = True

    def setup_sandbox(self, sandbox_dir: Path) -> None:
        """Create sandbox directory with initial files."""
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        for rel_path, content in self.setup_files.items():
            file_path = sandbox_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

    def validate(self, sandbox_dir: Path) -> tuple[int, int, dict]:
        """
        Run validation script and return (passed, total, details).

        The validation script should be a Python script that:
        1. Accepts sandbox_dir as first argument
        2. Prints JSON to stdout: {"passed": N, "total": M, "details": {...}}
        3. Returns exit code 0 on success
        """
        # Write validation script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.validation_script)
            script_path = f.name

        try:
            result = subprocess.run(
                ['python3', script_path, str(sandbox_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                return 0, 1, {"error": result.stderr or "Validation failed"}

            try:
                data = json.loads(result.stdout.strip())
                return data.get("passed", 0), data.get("total", 1), data.get("details", {})
            except json.JSONDecodeError:
                return 0, 1, {"error": f"Invalid validation output: {result.stdout[:200]}"}

        except subprocess.TimeoutExpired:
            return 0, 1, {"error": "Validation timed out"}
        finally:
            Path(script_path).unlink(missing_ok=True)


# =============================================================================
# Task Library
# =============================================================================

TASK_LIBRARY: dict[str, EvalTask] = {}


def register_task(task: EvalTask) -> EvalTask:
    """Register a task in the library."""
    TASK_LIBRARY[task.id] = task
    return task


def get_task(task_id: str) -> EvalTask:
    """Get a task by ID."""
    return TASK_LIBRARY[task_id]


def get_tasks_by_category(category: TaskCategory) -> list[EvalTask]:
    """Get all tasks in a category."""
    return [t for t in TASK_LIBRARY.values() if t.category == category]


def get_all_tasks() -> list[EvalTask]:
    """Get all registered tasks."""
    return list(TASK_LIBRARY.values())


# =============================================================================
# Task Definitions
# =============================================================================

# -----------------------------------------------------------------------------
# Category A: Bug Fixing
# -----------------------------------------------------------------------------

register_task(EvalTask(
    id="bugfix_pagination",
    name="Fix Pagination Off-by-One",
    category=TaskCategory.BUGFIX,
    description="""Fix the pagination bug in `paginator.py`.

The `Paginator` class has an off-by-one error causing:
1. The last item on each page appears as the first item on the next page
2. The total page count is incorrect for certain item counts

The test file `test_paginator.py` has failing tests that demonstrate the bug.

Your task:
1. Read the code and tests to understand the bug
2. Fix the bug in `paginator.py`
3. Ensure all tests pass

Do NOT modify the test file - only fix the implementation.""",
    setup_files={
        "paginator.py": '''"""Paginator for splitting items into pages."""

from dataclasses import dataclass
from typing import TypeVar, Generic, Sequence

T = TypeVar("T")


@dataclass
class Page(Generic[T]):
    """A single page of items."""
    items: Sequence[T]
    page_number: int
    total_pages: int
    total_items: int

    @property
    def has_next(self) -> bool:
        return self.page_number < self.total_pages

    @property
    def has_previous(self) -> bool:
        return self.page_number > 1


class Paginator(Generic[T]):
    """Split a sequence of items into pages."""

    def __init__(self, items: Sequence[T], page_size: int = 10):
        if page_size < 1:
            raise ValueError("page_size must be at least 1")
        self.items = items
        self.page_size = page_size

    @property
    def total_pages(self) -> int:
        """Return total number of pages."""
        # BUG: Should round up, not down
        return len(self.items) // self.page_size

    def get_page(self, page_number: int) -> Page[T]:
        """Get a specific page (1-indexed)."""
        if page_number < 1:
            raise ValueError("page_number must be at least 1")
        if page_number > self.total_pages and len(self.items) > 0:
            raise ValueError(f"page_number {page_number} exceeds total_pages {self.total_pages}")

        # BUG: Off-by-one in slice calculation
        start = (page_number - 1) * self.page_size
        end = start + self.page_size + 1  # Should not have +1

        return Page(
            items=self.items[start:end],
            page_number=page_number,
            total_pages=self.total_pages,
            total_items=len(self.items),
        )

    def __iter__(self):
        """Iterate over all pages."""
        for i in range(1, self.total_pages + 1):
            yield self.get_page(i)
''',
        "test_paginator.py": '''"""Tests for Paginator."""

import pytest
from paginator import Paginator, Page


class TestPaginator:
    def test_total_pages_exact_fit(self):
        """Items that exactly fill pages."""
        items = list(range(20))
        p = Paginator(items, page_size=10)
        assert p.total_pages == 2

    def test_total_pages_partial_last_page(self):
        """Items that don't exactly fill last page."""
        items = list(range(25))
        p = Paginator(items, page_size=10)
        assert p.total_pages == 3  # Should be 3, not 2

    def test_total_pages_single_item(self):
        """Single item should be 1 page."""
        p = Paginator([1], page_size=10)
        assert p.total_pages == 1

    def test_get_page_correct_items(self):
        """Each page should have exactly page_size items (except last)."""
        items = list(range(25))
        p = Paginator(items, page_size=10)

        page1 = p.get_page(1)
        assert len(page1.items) == 10
        assert list(page1.items) == list(range(10))

        page2 = p.get_page(2)
        assert len(page2.items) == 10
        assert list(page2.items) == list(range(10, 20))

        page3 = p.get_page(3)
        assert len(page3.items) == 5
        assert list(page3.items) == list(range(20, 25))

    def test_no_duplicate_items_across_pages(self):
        """Items should not appear on multiple pages."""
        items = list(range(15))
        p = Paginator(items, page_size=5)

        all_items = []
        for page in p:
            all_items.extend(page.items)

        assert len(all_items) == len(items)
        assert sorted(all_items) == sorted(items)

    def test_has_next_has_previous(self):
        """Navigation properties should be correct."""
        items = list(range(30))
        p = Paginator(items, page_size=10)

        page1 = p.get_page(1)
        assert not page1.has_previous
        assert page1.has_next

        page2 = p.get_page(2)
        assert page2.has_previous
        assert page2.has_next

        page3 = p.get_page(3)
        assert page3.has_previous
        assert not page3.has_next

    def test_empty_items(self):
        """Empty items should have 0 pages."""
        p = Paginator([], page_size=10)
        assert p.total_pages == 0

    def test_invalid_page_size(self):
        """page_size < 1 should raise."""
        with pytest.raises(ValueError):
            Paginator([1, 2, 3], page_size=0)

    def test_invalid_page_number(self):
        """page_number < 1 or > total should raise."""
        p = Paginator(list(range(10)), page_size=5)
        with pytest.raises(ValueError):
            p.get_page(0)
        with pytest.raises(ValueError):
            p.get_page(3)
''',
    },
    validation_script='''#!/usr/bin/env python3
"""Validate bugfix_pagination task."""
import sys
import subprocess
import json

sandbox_dir = sys.argv[1]

result = subprocess.run(
    ["python3", "-m", "pytest", "-v", "--tb=short", f"{sandbox_dir}/test_paginator.py"],
    capture_output=True,
    text=True,
    cwd=sandbox_dir,
)

# Parse pytest output for pass/fail counts
lines = result.stdout + result.stderr
passed = lines.count(" PASSED")
failed = lines.count(" FAILED")
total = passed + failed

print(json.dumps({
    "passed": passed,
    "total": total,
    "details": {
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "exit_code": result.returncode,
    }
}))
''',
))


register_task(EvalTask(
    id="bugfix_cache_invalidation",
    name="Fix Cache Invalidation Bug",
    category=TaskCategory.BUGFIX,
    description="""Fix the cache invalidation bug in `cache.py`.

The `TTLCache` class has a bug where:
1. Expired entries are not properly removed during get operations
2. The `cleanup()` method has a bug causing valid entries to be deleted

The test file `test_cache.py` demonstrates the failures.

Your task:
1. Read the code and tests to understand the bugs
2. Fix both bugs in `cache.py`
3. Ensure all tests pass

Do NOT modify the test file - only fix the implementation.""",
    setup_files={
        "cache.py": '''"""TTL-based cache implementation."""

import time
from threading import Lock
from typing import TypeVar, Generic, Optional, Any

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Cache with time-to-live expiration."""

    def __init__(self, default_ttl: float = 60.0):
        self._store: dict[K, tuple[V, float]] = {}
        self._default_ttl = default_ttl
        self._lock = Lock()

    def set(self, key: K, value: V, ttl: Optional[float] = None) -> None:
        """Set a value with optional custom TTL."""
        if ttl is None:
            ttl = self._default_ttl
        expiry = time.time() + ttl
        with self._lock:
            self._store[key] = (value, expiry)

    def get(self, key: K, default: Any = None) -> Optional[V]:
        """Get a value, returning default if missing or expired."""
        with self._lock:
            if key not in self._store:
                return default
            value, expiry = self._store[key]
            # BUG: Should check if expired and remove
            # Currently returns expired values
            return value

    def delete(self, key: K) -> bool:
        """Delete a key, return True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def cleanup(self) -> int:
        """Remove all expired entries. Return count removed."""
        now = time.time()
        removed = 0
        with self._lock:
            # BUG: Iterating while modifying, and wrong comparison
            for key in self._store:
                value, expiry = self._store[key]
                if expiry < now:  # Should be <=, and can't delete during iteration
                    del self._store[key]
                    removed += 1
        return removed

    def __len__(self) -> int:
        """Return number of entries (including expired)."""
        return len(self._store)

    def __contains__(self, key: K) -> bool:
        """Check if key exists and is not expired."""
        with self._lock:
            if key not in self._store:
                return False
            _, expiry = self._store[key]
            return expiry > time.time()
''',
        "test_cache.py": '''"""Tests for TTLCache."""

import pytest
import time
from cache import TTLCache


class TestTTLCache:
    def test_set_and_get(self):
        """Basic set/get works."""
        cache = TTLCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_get_missing_returns_default(self):
        """Missing key returns default."""
        cache = TTLCache()
        assert cache.get("missing") is None
        assert cache.get("missing", "default") == "default"

    def test_expired_entry_returns_default(self):
        """Expired entry should return default, not stale value."""
        cache = TTLCache()
        cache.set("key", "value", ttl=0.1)
        time.sleep(0.15)
        assert cache.get("key") is None
        assert cache.get("key", "default") == "default"

    def test_expired_entry_removed_on_get(self):
        """Expired entry should be removed when accessed."""
        cache = TTLCache()
        cache.set("key", "value", ttl=0.1)
        time.sleep(0.15)
        cache.get("key")  # Should trigger removal
        assert "key" not in cache._store

    def test_contains_checks_expiry(self):
        """__contains__ should check expiry."""
        cache = TTLCache()
        cache.set("key", "value", ttl=0.1)
        assert "key" in cache
        time.sleep(0.15)
        assert "key" not in cache

    def test_cleanup_removes_expired(self):
        """cleanup() should remove only expired entries."""
        cache = TTLCache()
        cache.set("short", "value", ttl=0.1)
        cache.set("long", "value", ttl=10.0)
        time.sleep(0.15)

        removed = cache.cleanup()

        assert removed == 1
        assert "short" not in cache._store
        assert "long" in cache._store

    def test_cleanup_preserves_valid_entries(self):
        """cleanup() should not remove valid entries."""
        cache = TTLCache()
        cache.set("a", 1, ttl=10.0)
        cache.set("b", 2, ttl=10.0)
        cache.set("c", 3, ttl=10.0)

        removed = cache.cleanup()

        assert removed == 0
        assert len(cache._store) == 3

    def test_delete_existing(self):
        """delete() returns True for existing key."""
        cache = TTLCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert "key" not in cache._store

    def test_delete_missing(self):
        """delete() returns False for missing key."""
        cache = TTLCache()
        assert cache.delete("missing") is False
''',
    },
    validation_script='''#!/usr/bin/env python3
"""Validate bugfix_cache_invalidation task."""
import sys
import subprocess
import json

sandbox_dir = sys.argv[1]

result = subprocess.run(
    ["python3", "-m", "pytest", "-v", "--tb=short", f"{sandbox_dir}/test_cache.py"],
    capture_output=True,
    text=True,
    cwd=sandbox_dir,
)

lines = result.stdout + result.stderr
passed = lines.count(" PASSED")
failed = lines.count(" FAILED")
total = passed + failed

print(json.dumps({
    "passed": passed,
    "total": total,
    "details": {
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "exit_code": result.returncode,
    }
}))
''',
))


# -----------------------------------------------------------------------------
# Category B: Multi-File Refactoring
# -----------------------------------------------------------------------------

register_task(EvalTask(
    id="refactor_extract_module",
    name="Extract Logger Module",
    category=TaskCategory.REFACTOR,
    description="""Refactor the monolithic `app.py` to extract the logging functionality.

The file `app.py` contains both application logic and logging utilities mixed together.
Your task is to:

1. Extract all logging-related code into a new `logger.py` module
2. Update `app.py` to import from `logger.py`
3. Ensure `test_app.py` still passes without modification

The logging code includes:
- The `Logger` class
- The `LogLevel` enum
- The `log_decorator` function
- Any helper functions used only by logging

Keep the Application class and its tests in `app.py`.
Do NOT modify `test_app.py`.""",
    setup_files={
        "app.py": '''"""Application with mixed logging code."""

from enum import Enum
from functools import wraps
from datetime import datetime
from typing import Callable, Any
import json


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


def _format_timestamp() -> str:
    """Format current time for logs."""
    return datetime.now().isoformat()


def _format_message(level: LogLevel, message: str, **context) -> str:
    """Format a log message with context."""
    base = {
        "timestamp": _format_timestamp(),
        "level": level.name,
        "message": message,
    }
    if context:
        base["context"] = context
    return json.dumps(base)


class Logger:
    """Simple logger with levels and formatting."""

    def __init__(self, name: str, min_level: LogLevel = LogLevel.INFO):
        self.name = name
        self.min_level = min_level
        self._handlers: list[Callable[[str], None]] = []

    def add_handler(self, handler: Callable[[str], None]) -> None:
        """Add a log handler."""
        self._handlers.append(handler)

    def _log(self, level: LogLevel, message: str, **context) -> None:
        """Internal log method."""
        if level.value < self.min_level.value:
            return
        formatted = _format_message(level, message, logger=self.name, **context)
        for handler in self._handlers:
            handler(formatted)

    def debug(self, message: str, **context) -> None:
        self._log(LogLevel.DEBUG, message, **context)

    def info(self, message: str, **context) -> None:
        self._log(LogLevel.INFO, message, **context)

    def warning(self, message: str, **context) -> None:
        self._log(LogLevel.WARNING, message, **context)

    def error(self, message: str, **context) -> None:
        self._log(LogLevel.ERROR, message, **context)

    def critical(self, message: str, **context) -> None:
        self._log(LogLevel.CRITICAL, message, **context)


def log_decorator(logger: Logger):
    """Decorator to log function calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.debug(f"Calling {func.__name__}", args=str(args), kwargs=str(kwargs))
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func.__name__}", result=str(result)[:100])
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}", error=str(e))
                raise
        return wrapper
    return decorator


# Application code that should stay in app.py

class Application:
    """Main application class."""

    def __init__(self, name: str):
        self.name = name
        self.logger = Logger(name)
        self._data: dict[str, Any] = {}

    def configure_logging(self, handler: Callable[[str], None], level: LogLevel = LogLevel.INFO) -> None:
        """Configure application logging."""
        self.logger.add_handler(handler)
        self.logger.min_level = level

    @property
    def decorated_process(self):
        """Get the decorated process method."""
        return log_decorator(self.logger)(self._process)

    def _process(self, key: str, value: Any) -> bool:
        """Process a key-value pair."""
        if not key:
            raise ValueError("key cannot be empty")
        self._data[key] = value
        return True

    def process(self, key: str, value: Any) -> bool:
        """Process with logging."""
        self.logger.info(f"Processing {key}")
        return self._process(key, value)

    def get(self, key: str) -> Any:
        """Get a value."""
        return self._data.get(key)

    def list_keys(self) -> list[str]:
        """List all keys."""
        return list(self._data.keys())
''',
        "test_app.py": '''"""Tests for Application."""

import pytest
from app import Application, LogLevel


class TestApplication:
    def test_create_app(self):
        """Can create an application."""
        app = Application("test")
        assert app.name == "test"

    def test_process_stores_value(self):
        """process() stores the value."""
        app = Application("test")
        result = app.process("key", "value")
        assert result is True
        assert app.get("key") == "value"

    def test_process_empty_key_raises(self):
        """Empty key raises ValueError."""
        app = Application("test")
        with pytest.raises(ValueError):
            app.process("", "value")

    def test_list_keys(self):
        """list_keys() returns all keys."""
        app = Application("test")
        app.process("a", 1)
        app.process("b", 2)
        assert sorted(app.list_keys()) == ["a", "b"]

    def test_get_missing_returns_none(self):
        """get() returns None for missing key."""
        app = Application("test")
        assert app.get("missing") is None

    def test_logging_handler_receives_messages(self):
        """Configured handler receives log messages."""
        app = Application("test")
        messages = []
        app.configure_logging(messages.append, LogLevel.INFO)
        app.process("key", "value")
        assert len(messages) >= 1
        assert "Processing key" in messages[0]

    def test_log_level_filtering(self):
        """Log level filters messages."""
        app = Application("test")
        messages = []
        app.configure_logging(messages.append, LogLevel.WARNING)
        app.process("key", "value")  # INFO level - should be filtered
        assert len(messages) == 0
''',
    },
    validation_script='''#!/usr/bin/env python3
"""Validate refactor_extract_module task."""
import sys
import subprocess
import json
from pathlib import Path

sandbox_dir = Path(sys.argv[1])

# Check that logger.py exists
logger_exists = (sandbox_dir / "logger.py").exists()

# Check that app.py imports from logger
app_content = (sandbox_dir / "app.py").read_text()
imports_logger = "from logger import" in app_content or "import logger" in app_content

# Run tests
result = subprocess.run(
    ["python3", "-m", "pytest", "-v", "--tb=short", f"{sandbox_dir}/test_app.py"],
    capture_output=True,
    text=True,
    cwd=sandbox_dir,
)

lines = result.stdout + result.stderr
passed = lines.count(" PASSED")
failed = lines.count(" FAILED")
total = passed + failed

# Check that Logger class is in logger.py, not app.py
logger_has_class = False
if logger_exists:
    logger_content = (sandbox_dir / "logger.py").read_text()
    logger_has_class = "class Logger" in logger_content

app_has_logger_class = "class Logger" in app_content

# Scoring: tests + structure
structure_score = 0
if logger_exists:
    structure_score += 1
if imports_logger:
    structure_score += 1
if logger_has_class and not app_has_logger_class:
    structure_score += 2

total_score = passed + structure_score
max_score = total + 4  # tests + 4 structure points

print(json.dumps({
    "passed": total_score,
    "total": max_score,
    "details": {
        "tests_passed": passed,
        "tests_total": total,
        "logger_py_exists": logger_exists,
        "imports_logger": imports_logger,
        "logger_has_class": logger_has_class,
        "app_has_logger_class": app_has_logger_class,
        "stdout": result.stdout[-1500:],
    }
}))
''',
))


# -----------------------------------------------------------------------------
# Category C: Greenfield Implementation
# -----------------------------------------------------------------------------

register_task(EvalTask(
    id="greenfield_rate_limiter",
    name="Implement Rate Limiter",
    category=TaskCategory.GREENFIELD,
    description="""Implement a concurrent rate limiter using the token bucket algorithm.

Create a file `rate_limiter.py` with:

1. `TokenBucket` class:
   - `__init__(self, rate: float, capacity: float)` - tokens per second, max tokens
   - `acquire(self, tokens: float = 1.0) -> bool` - try to consume tokens (non-blocking)
   - `tokens_available` property - current token count
   - Thread-safe using locks

2. `ConcurrentRateLimiter` class:
   - `__init__(self, rate: float, capacity: float)` - creates internal TokenBucket
   - `acquire(self, key: Any, tokens: float = 1.0) -> bool` - per-key rate limiting
   - `async_acquire(self, key: Any, tokens: float = 1.0) -> bool` - async version
   - Context manager support: `async with limiter.limit(key):`
   - Each unique key gets its own bucket

3. Tests in `test_rate_limiter.py`:
   - Test TokenBucket basic acquire
   - Test capacity limits
   - Test refill over time
   - Test ConcurrentRateLimiter per-key isolation
   - Test async acquire
   - Test context manager

Requirements:
- Pure Python, no external dependencies except pytest for tests
- Thread-safe for sync operations
- Asyncio-compatible for async operations""",
    setup_files={
        ".gitkeep": "# Placeholder - implement rate_limiter.py and test_rate_limiter.py",
    },
    validation_script='''#!/usr/bin/env python3
"""Validate greenfield_rate_limiter task."""
import sys
import subprocess
import json
from pathlib import Path

sandbox_dir = Path(sys.argv[1])

# Check files exist
rate_limiter_exists = (sandbox_dir / "rate_limiter.py").exists()
test_exists = (sandbox_dir / "test_rate_limiter.py").exists()

if not rate_limiter_exists or not test_exists:
    print(json.dumps({
        "passed": 0,
        "total": 10,
        "details": {
            "error": f"Missing files: rate_limiter.py={rate_limiter_exists}, test_rate_limiter.py={test_exists}"
        }
    }))
    sys.exit(0)

# Check for required classes
content = (sandbox_dir / "rate_limiter.py").read_text()
has_token_bucket = "class TokenBucket" in content
has_rate_limiter = "class ConcurrentRateLimiter" in content or "class RateLimiter" in content
has_async = "async def" in content or "async with" in content

# Run tests
result = subprocess.run(
    ["python3", "-m", "pytest", "-v", "--tb=short", f"{sandbox_dir}/test_rate_limiter.py"],
    capture_output=True,
    text=True,
    cwd=sandbox_dir,
    timeout=60,
)

lines = result.stdout + result.stderr
passed = lines.count(" PASSED")
failed = lines.count(" FAILED")
total = passed + failed if (passed + failed) > 0 else 10

# Structure points
structure_score = 0
if has_token_bucket:
    structure_score += 1
if has_rate_limiter:
    structure_score += 1
if has_async:
    structure_score += 1

print(json.dumps({
    "passed": passed + structure_score,
    "total": total + 3,
    "details": {
        "tests_passed": passed,
        "tests_failed": failed,
        "has_token_bucket": has_token_bucket,
        "has_rate_limiter": has_rate_limiter,
        "has_async": has_async,
        "stdout": result.stdout[-2000:],
    }
}))
''',
))


# -----------------------------------------------------------------------------
# Category D: Iterative Refinement
# -----------------------------------------------------------------------------

register_task(EvalTask(
    id="iterative_add_features",
    name="Iterative Feature Addition",
    category=TaskCategory.ITERATIVE,
    description="""This is a multi-round task. You will receive additional requirements after completing each phase.

PHASE 1: Create a basic `Counter` class in `counter.py`:
- `__init__(self, initial: int = 0)`
- `increment(self) -> int` - add 1, return new value
- `decrement(self) -> int` - subtract 1, return new value
- `value` property - current count
- Tests in `test_counter.py`

Complete Phase 1, then wait for Phase 2 instructions.

[PHASE 2 will be sent after Phase 1 completion]
[PHASE 3 will be sent after Phase 2 completion]""",
    setup_files={
        ".gitkeep": "# Implement counter.py and test_counter.py",
        "PHASES.md": """# Task Phases

## Phase 1 (Initial)
Basic Counter class with increment/decrement.

## Phase 2 (After Phase 1)
Add these features:
- `add(n: int)` and `subtract(n: int)` methods
- `reset()` method to set back to initial value
- `history` property returning list of all values

## Phase 3 (After Phase 2)
Add these features:
- Thread-safe operations using locks
- `max_value` and `min_value` properties (from history)
- Context manager: `with counter.transaction(): ...` that can be rolled back
""",
    },
    validation_script='''#!/usr/bin/env python3
"""Validate iterative_add_features task."""
import sys
import subprocess
import json
from pathlib import Path

sandbox_dir = Path(sys.argv[1])

# Check files exist
counter_exists = (sandbox_dir / "counter.py").exists()
test_exists = (sandbox_dir / "test_counter.py").exists()

if not counter_exists or not test_exists:
    print(json.dumps({
        "passed": 0,
        "total": 10,
        "details": {"error": "Missing required files"}
    }))
    sys.exit(0)

# Check for features across phases
content = (sandbox_dir / "counter.py").read_text()

# Phase 1 features
phase1 = {
    "class": "class Counter" in content,
    "increment": "def increment" in content,
    "decrement": "def decrement" in content,
}

# Phase 2 features
phase2 = {
    "add": "def add" in content,
    "subtract": "def subtract" in content,
    "reset": "def reset" in content,
    "history": "history" in content,
}

# Phase 3 features
phase3 = {
    "lock": "Lock" in content or "lock" in content.lower(),
    "max_value": "max_value" in content,
    "min_value": "min_value" in content,
    "transaction": "transaction" in content or "Transaction" in content,
}

# Run tests
result = subprocess.run(
    ["python3", "-m", "pytest", "-v", "--tb=short", f"{sandbox_dir}/test_counter.py"],
    capture_output=True,
    text=True,
    cwd=sandbox_dir,
    timeout=60,
)

lines = result.stdout + result.stderr
passed = lines.count(" PASSED")
failed = lines.count(" FAILED")

# Calculate phase completion
phase1_complete = all(phase1.values())
phase2_complete = all(phase2.values())
phase3_complete = all(phase3.values())

feature_score = sum(phase1.values()) + sum(phase2.values()) + sum(phase3.values())
max_features = len(phase1) + len(phase2) + len(phase3)

print(json.dumps({
    "passed": passed + feature_score,
    "total": (passed + failed if passed + failed > 0 else 5) + max_features,
    "details": {
        "tests_passed": passed,
        "tests_failed": failed,
        "phase1": phase1,
        "phase2": phase2,
        "phase3": phase3,
        "phase1_complete": phase1_complete,
        "phase2_complete": phase2_complete,
        "phase3_complete": phase3_complete,
    }
}))
''',
    max_iterations=75,  # More iterations for multi-phase
))


# -----------------------------------------------------------------------------
# Category E: Long Context
# -----------------------------------------------------------------------------

register_task(EvalTask(
    id="long_context_find_bug",
    name="Find Bug in Large Codebase",
    category=TaskCategory.LONG_CONTEXT,
    description="""A bug has been reported in the e-commerce system. Users report that
their cart totals are sometimes incorrect when using discount codes.

The codebase has multiple files:
- `models.py` - Data models (Product, CartItem, Cart, Discount)
- `cart.py` - Shopping cart logic
- `discounts.py` - Discount code processing
- `checkout.py` - Checkout flow
- `test_checkout.py` - Tests (some are failing)

Your task:
1. Run the tests to see which ones fail
2. Investigate the codebase to find the bug
3. Fix the bug
4. Ensure all tests pass

The bug is subtle - it only manifests in certain conditions.""",
    setup_files={
        "models.py": '''"""Data models for e-commerce system."""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
from datetime import datetime


@dataclass
class Product:
    """A product that can be purchased."""
    id: str
    name: str
    price: Decimal
    category: str
    in_stock: bool = True

    def __post_init__(self):
        if isinstance(self.price, (int, float)):
            self.price = Decimal(str(self.price))


@dataclass
class CartItem:
    """An item in a shopping cart."""
    product: Product
    quantity: int

    @property
    def subtotal(self) -> Decimal:
        return self.product.price * self.quantity


@dataclass
class Discount:
    """A discount code."""
    code: str
    discount_type: str  # "percentage" or "fixed"
    value: Decimal
    min_purchase: Decimal = Decimal("0")
    max_discount: Optional[Decimal] = None
    applicable_categories: list[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        if isinstance(self.value, (int, float)):
            self.value = Decimal(str(self.value))
        if isinstance(self.min_purchase, (int, float)):
            self.min_purchase = Decimal(str(self.min_purchase))
        if self.max_discount and isinstance(self.max_discount, (int, float)):
            self.max_discount = Decimal(str(self.max_discount))

    def is_valid(self) -> bool:
        """Check if discount is still valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True


@dataclass
class Cart:
    """Shopping cart."""
    items: list[CartItem] = field(default_factory=list)
    discount: Optional[Discount] = None

    def add_item(self, product: Product, quantity: int = 1) -> None:
        """Add item to cart."""
        for item in self.items:
            if item.product.id == product.id:
                item.quantity += quantity
                return
        self.items.append(CartItem(product=product, quantity=quantity))

    def remove_item(self, product_id: str) -> bool:
        """Remove item from cart."""
        for i, item in enumerate(self.items):
            if item.product.id == product_id:
                self.items.pop(i)
                return True
        return False

    @property
    def subtotal(self) -> Decimal:
        """Total before discount."""
        return sum((item.subtotal for item in self.items), Decimal("0"))

    def apply_discount(self, discount: Discount) -> bool:
        """Apply a discount code."""
        if not discount.is_valid():
            return False
        if self.subtotal < discount.min_purchase:
            return False
        self.discount = discount
        return True
''',
        "discounts.py": '''"""Discount processing logic."""

from decimal import Decimal
from models import Cart, Discount


def calculate_discount_amount(cart: Cart) -> Decimal:
    """Calculate the discount amount for a cart."""
    if not cart.discount:
        return Decimal("0")

    discount = cart.discount

    # Calculate base amount to apply discount to
    if discount.applicable_categories:
        # Only apply to items in specific categories
        applicable_subtotal = sum(
            (item.subtotal for item in cart.items
             if item.product.category in discount.applicable_categories),
            Decimal("0")
        )
    else:
        applicable_subtotal = cart.subtotal

    # Calculate discount based on type
    if discount.discount_type == "percentage":
        # BUG: Using integer division instead of proper decimal percentage
        discount_amount = applicable_subtotal * discount.value / 100
    elif discount.discount_type == "fixed":
        discount_amount = discount.value
    else:
        return Decimal("0")

    # Apply max discount cap if set
    if discount.max_discount:
        discount_amount = min(discount_amount, discount.max_discount)

    # Don't let discount exceed applicable subtotal
    discount_amount = min(discount_amount, applicable_subtotal)

    return discount_amount


def validate_discount_code(code: str, available_discounts: list[Discount]) -> Discount | None:
    """Find and validate a discount code."""
    for discount in available_discounts:
        if discount.code.upper() == code.upper():
            if discount.is_valid():
                return discount
    return None
''',
        "cart.py": '''"""Shopping cart operations."""

from decimal import Decimal
from models import Cart, Product, CartItem
from discounts import calculate_discount_amount


def get_cart_total(cart: Cart) -> Decimal:
    """Get the final cart total after discounts."""
    subtotal = cart.subtotal
    discount = calculate_discount_amount(cart)
    return subtotal - discount


def get_cart_summary(cart: Cart) -> dict:
    """Get a summary of the cart for display."""
    return {
        "items": [
            {
                "name": item.product.name,
                "price": str(item.product.price),
                "quantity": item.quantity,
                "subtotal": str(item.subtotal),
            }
            for item in cart.items
        ],
        "subtotal": str(cart.subtotal),
        "discount_code": cart.discount.code if cart.discount else None,
        "discount_amount": str(calculate_discount_amount(cart)),
        "total": str(get_cart_total(cart)),
    }


def merge_carts(cart1: Cart, cart2: Cart) -> Cart:
    """Merge two carts into a new cart."""
    merged = Cart()
    for item in cart1.items:
        merged.add_item(item.product, item.quantity)
    for item in cart2.items:
        merged.add_item(item.product, item.quantity)
    # Keep discount from first cart if present
    if cart1.discount:
        merged.discount = cart1.discount
    elif cart2.discount:
        merged.discount = cart2.discount
    return merged
''',
        "checkout.py": '''"""Checkout flow."""

from decimal import Decimal
from models import Cart, Product, Discount
from cart import get_cart_total
from discounts import validate_discount_code, calculate_discount_amount


class CheckoutError(Exception):
    """Error during checkout."""
    pass


class Checkout:
    """Manages the checkout process."""

    def __init__(self, cart: Cart, available_discounts: list[Discount] = None):
        self.cart = cart
        self.available_discounts = available_discounts or []
        self._completed = False

    def apply_discount_code(self, code: str) -> bool:
        """Try to apply a discount code."""
        if self._completed:
            raise CheckoutError("Checkout already completed")

        discount = validate_discount_code(code, self.available_discounts)
        if not discount:
            return False

        return self.cart.apply_discount(discount)

    def get_total(self) -> Decimal:
        """Get the final total."""
        return get_cart_total(self.cart)

    def get_breakdown(self) -> dict:
        """Get detailed price breakdown."""
        subtotal = self.cart.subtotal
        discount_amount = calculate_discount_amount(self.cart)
        total = self.get_total()

        return {
            "subtotal": subtotal,
            "discount": discount_amount,
            "total": total,
            "savings_percent": (discount_amount / subtotal * 100) if subtotal > 0 else Decimal("0"),
        }

    def complete(self) -> dict:
        """Complete the checkout."""
        if self._completed:
            raise CheckoutError("Checkout already completed")

        if not self.cart.items:
            raise CheckoutError("Cart is empty")

        # Verify all items in stock
        for item in self.cart.items:
            if not item.product.in_stock:
                raise CheckoutError(f"Product {item.product.name} is out of stock")

        total = self.get_total()
        self._completed = True

        return {
            "success": True,
            "total": total,
            "items_count": sum(item.quantity for item in self.cart.items),
        }
''',
        "test_checkout.py": '''"""Tests for checkout system."""

import pytest
from decimal import Decimal
from models import Product, Cart, Discount
from cart import get_cart_total, get_cart_summary
from checkout import Checkout, CheckoutError
from discounts import calculate_discount_amount


@pytest.fixture
def products():
    """Sample products."""
    return {
        "laptop": Product("1", "Laptop", Decimal("999.99"), "electronics"),
        "mouse": Product("2", "Mouse", Decimal("29.99"), "electronics"),
        "book": Product("3", "Python Book", Decimal("49.99"), "books"),
        "shirt": Product("4", "T-Shirt", Decimal("24.99"), "clothing"),
    }


@pytest.fixture
def discounts():
    """Sample discount codes."""
    return [
        Discount("SAVE10", "percentage", Decimal("10"), min_purchase=Decimal("50")),
        Discount("FLAT20", "fixed", Decimal("20"), min_purchase=Decimal("100")),
        Discount("ELECTRONICS15", "percentage", Decimal("15"),
                 applicable_categories=["electronics"]),
        Discount("MEGA50", "percentage", Decimal("50"), max_discount=Decimal("100")),
    ]


class TestCartBasics:
    def test_empty_cart_total(self):
        """Empty cart has zero total."""
        cart = Cart()
        assert get_cart_total(cart) == Decimal("0")

    def test_single_item_total(self, products):
        """Single item cart total."""
        cart = Cart()
        cart.add_item(products["laptop"])
        assert get_cart_total(cart) == Decimal("999.99")

    def test_multiple_items_total(self, products):
        """Multiple items sum correctly."""
        cart = Cart()
        cart.add_item(products["laptop"])
        cart.add_item(products["mouse"], 2)
        # 999.99 + 29.99*2 = 1059.97
        assert get_cart_total(cart) == Decimal("1059.97")


class TestPercentageDiscount:
    def test_10_percent_discount(self, products, discounts):
        """10% discount applied correctly."""
        cart = Cart()
        cart.add_item(products["laptop"])  # 999.99
        cart.apply_discount(discounts[0])  # SAVE10 - 10%

        # 999.99 * 0.10 = 99.999, total should be 899.991
        total = get_cart_total(cart)
        expected = Decimal("999.99") - (Decimal("999.99") * Decimal("10") / Decimal("100"))
        assert total == expected

    def test_15_percent_electronics_only(self, products, discounts):
        """Category-specific discount only applies to matching items."""
        cart = Cart()
        cart.add_item(products["laptop"])  # 999.99 electronics
        cart.add_item(products["book"])     # 49.99 books
        cart.apply_discount(discounts[2])   # ELECTRONICS15 - 15% electronics only

        # Discount only on laptop: 999.99 * 0.15 = 149.9985
        # Total: 999.99 + 49.99 - 149.9985 = 899.9815
        electronics_discount = Decimal("999.99") * Decimal("15") / Decimal("100")
        expected = Decimal("999.99") + Decimal("49.99") - electronics_discount
        assert get_cart_total(cart) == expected

    def test_50_percent_with_cap(self, products, discounts):
        """50% discount capped at $100."""
        cart = Cart()
        cart.add_item(products["laptop"])  # 999.99
        cart.apply_discount(discounts[3])  # MEGA50 - 50% max $100

        # 999.99 * 0.50 = 499.995, but capped at 100
        # Total: 999.99 - 100 = 899.99
        assert get_cart_total(cart) == Decimal("899.99")


class TestFixedDiscount:
    def test_flat_20_discount(self, products, discounts):
        """Fixed $20 discount."""
        cart = Cart()
        cart.add_item(products["laptop"])  # 999.99
        cart.apply_discount(discounts[1])  # FLAT20 - $20 off

        assert get_cart_total(cart) == Decimal("979.99")


class TestDiscountValidation:
    def test_min_purchase_not_met(self, products, discounts):
        """Discount not applied if min purchase not met."""
        cart = Cart()
        cart.add_item(products["mouse"])  # 29.99 - below $50 minimum

        applied = cart.apply_discount(discounts[0])  # SAVE10 requires $50
        assert applied is False
        assert get_cart_total(cart) == Decimal("29.99")


class TestCheckoutFlow:
    def test_checkout_with_discount(self, products, discounts):
        """Full checkout flow with discount."""
        cart = Cart()
        cart.add_item(products["laptop"])
        cart.add_item(products["mouse"])

        checkout = Checkout(cart, discounts)
        assert checkout.apply_discount_code("SAVE10") is True

        # 999.99 + 29.99 = 1029.98
        # 10% off = 102.998
        # Total = 926.982
        breakdown = checkout.get_breakdown()
        assert breakdown["subtotal"] == Decimal("1029.98")

        result = checkout.complete()
        assert result["success"] is True

    def test_checkout_empty_cart_fails(self):
        """Cannot checkout empty cart."""
        cart = Cart()
        checkout = Checkout(cart)

        with pytest.raises(CheckoutError):
            checkout.complete()

    def test_checkout_out_of_stock_fails(self, products):
        """Cannot checkout with out of stock item."""
        products["laptop"].in_stock = False
        cart = Cart()
        cart.add_item(products["laptop"])

        checkout = Checkout(cart)
        with pytest.raises(CheckoutError):
            checkout.complete()


class TestCartSummary:
    def test_summary_format(self, products, discounts):
        """Cart summary has correct format."""
        cart = Cart()
        cart.add_item(products["laptop"])
        cart.apply_discount(discounts[0])  # SAVE10

        summary = get_cart_summary(cart)

        assert len(summary["items"]) == 1
        assert summary["discount_code"] == "SAVE10"
        assert Decimal(summary["discount_amount"]) > 0
''',
    },
    validation_script='''#!/usr/bin/env python3
"""Validate long_context_find_bug task."""
import sys
import subprocess
import json
from pathlib import Path

sandbox_dir = Path(sys.argv[1])

# Run tests
result = subprocess.run(
    ["python3", "-m", "pytest", "-v", "--tb=short", f"{sandbox_dir}/test_checkout.py"],
    capture_output=True,
    text=True,
    cwd=sandbox_dir,
    timeout=60,
)

lines = result.stdout + result.stderr
passed = lines.count(" PASSED")
failed = lines.count(" FAILED")
total = passed + failed if (passed + failed) > 0 else 15

# Check if the bug was fixed (percentage calculation)
discounts_content = (sandbox_dir / "discounts.py").read_text()
# The bug was integer division - fix should use Decimal properly
has_fix = "Decimal" in discounts_content and "100" in discounts_content

print(json.dumps({
    "passed": passed,
    "total": total,
    "details": {
        "tests_passed": passed,
        "tests_failed": failed,
        "stdout": result.stdout[-2000:],
        "likely_fixed": has_fix and failed == 0,
    }
}))
''',
    max_iterations=50,
))


# Summary of tasks
print(f"Registered {len(TASK_LIBRARY)} evaluation tasks:")
for task_id, task in TASK_LIBRARY.items():
    print(f"  - {task_id}: {task.name} ({task.category.value})")
