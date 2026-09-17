"""Small, testable policies for talking to rate-limited, occasionally flaky HTTP APIs.

Both classes take their clock and sleep functions as parameters. Production
passes ``time.monotonic`` and ``time.sleep``; tests pass fakes and assert on
the exact delays requested, so the suite stays instant and deterministic.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff: wait ``base_delay * 2**n`` after the n-th failure, capped.

    ``max_attempts`` counts the first try, so 3 means one try and two retries.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            msg = "max_attempts must be at least 1"
            raise ValueError(msg)

    def delay_before_retry(self, failed_attempt: int) -> float:
        """Seconds to wait after ``failed_attempt`` (1-based) before the next try."""
        return min(self.base_delay_seconds * 2.0 ** (failed_attempt - 1), self.max_delay_seconds)


@dataclass
class MinIntervalLimiter:
    """Ensure at least ``1 / requests_per_second`` seconds between calls to ``wait``."""

    requests_per_second: float
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    _last_call: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.requests_per_second <= 0:
            msg = "requests_per_second must be positive"
            raise ValueError(msg)

    @property
    def interval_seconds(self) -> float:
        return 1.0 / self.requests_per_second

    def wait(self) -> None:
        """Block until the next request is allowed, then record it."""
        now = self.clock()
        if self._last_call is not None:
            remaining = self.interval_seconds - (now - self._last_call)
            if remaining > 0:
                self.sleep(remaining)
                now = self.clock()
        self._last_call = now
