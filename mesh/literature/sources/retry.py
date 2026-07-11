"""
Retry utilities with exponential backoff for literature search.

Provides decorators and utilities for handling transient failures
gracefully across all API sources.

Features:
- Exponential backoff with jitter
- Configurable retry conditions
- Rate limit (429) handling
- Timeout handling
- Error aggregation
"""

import time
import random
import logging
import functools
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar, Any
from enum import Enum

import requests

logger = logging.getLogger(__name__)


class RetryableError(Exception):
    """An error that should trigger a retry."""
    pass


class NonRetryableError(Exception):
    """An error that should NOT trigger a retry."""
    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded (429 status)."""
    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limited. Retry after: {retry_after}s")


class BlockedError(NonRetryableError):
    """Source is blocking requests (CAPTCHA, IP ban, etc.)."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 60.0  # Maximum delay
    exponential_base: float = 2.0  # Multiplier for exponential backoff
    jitter: float = 0.1  # Random jitter factor (0.1 = ±10%)

    # Which HTTP status codes to retry
    retryable_status_codes: set[int] = field(default_factory=lambda: {
        408,  # Request Timeout
        429,  # Too Many Requests
        500,  # Internal Server Error
        502,  # Bad Gateway
        503,  # Service Unavailable
        504,  # Gateway Timeout
    })

    # Which exceptions to retry
    retryable_exceptions: tuple = field(default_factory=lambda: (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        RateLimitError,
    ))


def calculate_backoff(
    attempt: int,
    config: RetryConfig,
    retry_after: Optional[float] = None,
) -> float:
    """
    Calculate backoff delay for a retry attempt.

    Args:
        attempt: The retry attempt number (0-indexed)
        config: Retry configuration
        retry_after: Optional server-specified delay (from 429 response)

    Returns:
        Delay in seconds before next retry
    """
    if retry_after is not None:
        # Honor server's retry-after header
        return min(retry_after, config.max_delay)

    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = config.base_delay * (config.exponential_base ** attempt)

    # Apply jitter
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)

    # Cap at max delay
    return min(delay, config.max_delay)


def is_retryable_response(response: requests.Response, config: RetryConfig) -> bool:
    """Check if an HTTP response should trigger a retry."""
    return response.status_code in config.retryable_status_codes


def get_retry_after(response: requests.Response) -> Optional[float]:
    """Extract retry-after value from response headers."""
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            # Could be a date string, ignore for simplicity
            pass
    return None


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator to add retry logic to a function.

    Example:
        @with_retry(RetryConfig(max_retries=3))
        def fetch_data(url):
            return requests.get(url).json()

    Args:
        config: Retry configuration
        on_retry: Optional callback(attempt, exception) called before each retry
    """
    config = config or RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        # Calculate delay
                        retry_after = None
                        if isinstance(e, RateLimitError):
                            retry_after = e.retry_after

                        delay = calculate_backoff(attempt, config, retry_after)

                        if on_retry:
                            on_retry(attempt, e)
                        else:
                            logger.debug(
                                f"Retry {attempt + 1}/{config.max_retries} "
                                f"for {func.__name__} after {delay:.1f}s: {e}"
                            )

                        time.sleep(delay)
                    else:
                        raise

                except NonRetryableError:
                    # Don't retry these
                    raise

            raise last_exception

        return wrapper
    return decorator


class RetryableRequest:
    """
    Helper class for making retryable HTTP requests.

    Handles common patterns for API requests with proper error handling.

    Example:
        requester = RetryableRequest(
            session=requests.Session(),
            config=RetryConfig(max_retries=3),
        )

        response = requester.get("https://api.example.com/data")
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        config: Optional[RetryConfig] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the requester.

        Args:
            session: Requests session (created if not provided)
            config: Retry configuration
            timeout: Default request timeout
        """
        self.session = session or requests.Session()
        self.config = config or RetryConfig()
        self.timeout = timeout

    def _handle_response(self, response: requests.Response) -> requests.Response:
        """Check response and raise appropriate exceptions."""
        if response.status_code == 429:
            retry_after = get_retry_after(response)
            raise RateLimitError(retry_after)

        if is_retryable_response(response, self.config):
            raise RetryableError(f"HTTP {response.status_code}")

        return response

    @with_retry()
    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> requests.Response:
        """Make a GET request with retry logic."""
        response = self.session.get(
            url,
            params=params,
            timeout=timeout or self.timeout,
            **kwargs,
        )
        return self._handle_response(response)

    @with_retry()
    def post(
        self,
        url: str,
        data: Optional[dict] = None,
        json: Optional[dict] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> requests.Response:
        """Make a POST request with retry logic."""
        response = self.session.post(
            url,
            data=data,
            json=json,
            timeout=timeout or self.timeout,
            **kwargs,
        )
        return self._handle_response(response)


@dataclass
class SourceResult:
    """Result from a source query, including error info."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    source: Optional[str] = None
    attempts: int = 1
    blocked: bool = False


def query_with_fallback(
    sources: list[tuple[str, Callable]],
    config: Optional[RetryConfig] = None,
    stop_on_first_success: bool = True,
) -> list[SourceResult]:
    """
    Query multiple sources with fallback on failure.

    Args:
        sources: List of (source_name, query_function) tuples
        config: Retry configuration
        stop_on_first_success: If True, stop after first successful query

    Returns:
        List of SourceResult objects (one per queried source)

    Example:
        results = query_with_fallback([
            ("arxiv", lambda: arxiv_search(query)),
            ("pubmed", lambda: pubmed_search(query)),
        ])

        successful = [r for r in results if r.success]
    """
    config = config or RetryConfig()
    results: list[SourceResult] = []

    for source_name, query_func in sources:
        last_error = None
        attempts = 0

        for attempt in range(config.max_retries + 1):
            attempts += 1
            try:
                data = query_func()
                results.append(SourceResult(
                    success=True,
                    data=data,
                    source=source_name,
                    attempts=attempts,
                ))

                if stop_on_first_success:
                    return results

                break  # Success, move to next source

            except BlockedError as e:
                results.append(SourceResult(
                    success=False,
                    error=str(e),
                    source=source_name,
                    attempts=attempts,
                    blocked=True,
                ))
                break  # Don't retry blocked sources

            except config.retryable_exceptions as e:
                last_error = e
                if attempt < config.max_retries:
                    delay = calculate_backoff(attempt, config)
                    time.sleep(delay)
                else:
                    results.append(SourceResult(
                        success=False,
                        error=str(last_error),
                        source=source_name,
                        attempts=attempts,
                    ))

            except Exception as e:
                results.append(SourceResult(
                    success=False,
                    error=str(e),
                    source=source_name,
                    attempts=attempts,
                ))
                break  # Non-retryable error

    return results


class CircuitBreaker:
    """
    Circuit breaker pattern for failing sources.

    Prevents repeated calls to a failing source, allowing it time to recover.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Source is failing, requests are blocked
    - HALF_OPEN: Testing if source has recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=3, recovery_time=60)

        if breaker.allow_request():
            try:
                result = make_request()
                breaker.record_success()
            except Exception as e:
                breaker.record_failure()
                raise
    """

    class State(Enum):
        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_time: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_time: Seconds to wait before testing recovery
            half_open_max_calls: Max test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.half_open_max_calls = half_open_max_calls

        self._state = self.State.CLOSED
        self._failures = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> State:
        """Get current state, updating if recovery time has passed."""
        if self._state == self.State.OPEN:
            if time.time() - self._last_failure_time > self.recovery_time:
                self._state = self.State.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state

        if state == self.State.CLOSED:
            return True

        if state == self.State.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

        # OPEN
        return False

    def record_success(self) -> None:
        """Record a successful request."""
        if self._state == self.State.HALF_OPEN:
            # Recovery successful, close circuit
            self._state = self.State.CLOSED
            self._failures = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self._failures += 1
        self._last_failure_time = time.time()

        if self._state == self.State.HALF_OPEN:
            # Still failing, reopen circuit
            self._state = self.State.OPEN

        elif self._failures >= self.failure_threshold:
            self._state = self.State.OPEN

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = self.State.CLOSED
        self._failures = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == self.State.OPEN
