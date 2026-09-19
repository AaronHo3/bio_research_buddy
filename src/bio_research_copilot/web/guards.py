"""What stands between a public URL and an unbounded bill.

Three independent mechanisms, deliberately separate because they fail
differently and a reader should be able to reason about each alone.

``TokenBudget`` is the hard ceiling. It counts tokens actually spent and, once
the day's allowance is gone, reports that the language model is unavailable so
the caller falls back to the deterministic provider. The worst case for a day is
therefore the number configured, whatever happens. It charges *after* a call
using real usage rather than estimating beforehand, which can overshoot a single
call's worth; that is accepted, because the alternative is refusing work on a
guess and still being wrong.

``AnswerCache`` makes repetition free. A demo's traffic is mostly the same
handful of questions, so normalising and caching turns a popular link into one
paid call rather than hundreds. It bounds its own size so memory cannot grow
without limit.

``RateLimiter`` slows one visitor down. It is a fixed-window counter per client,
which is coarse: a caller can send two full windows' worth across a boundary.
That is fine here, because the budget above is the real protection and this only
has to stop casual hammering.

All three keep state in memory. A restart forgets the cache and the limiter,
which is harmless, and forgets the budget, which is not: a host that restarts
frequently could spend more than a day's allowance in a day. That is recorded as
a known limitation rather than solved, because solving it means a database and
this demo deliberately has none.
"""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

Clock = Callable[[], float]

DEFAULT_DAILY_TOKENS = 200_000
DEFAULT_CACHE_SIZE = 256
DEFAULT_RATE_LIMIT = 10
DEFAULT_RATE_WINDOW = 60.0
SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class BudgetStatus:
    """What remains, in a form an endpoint can report to a caller."""

    limit: int
    spent: int
    remaining: int
    exhausted: bool


class TokenBudget:
    """A daily ceiling on tokens spent with a paid provider.

    Args:
        daily_tokens: Total input plus output tokens allowed per day. A value of
            zero disables the language model entirely, which is the correct
            configuration for a deployment with no key.
        clock: Monotonic-enough source of seconds, injected so a test can move
            a day without waiting one.
    """

    def __init__(
        self,
        daily_tokens: int = DEFAULT_DAILY_TOKENS,
        *,
        clock: Clock | None = None,
    ) -> None:
        if daily_tokens < 0:
            msg = "daily_tokens cannot be negative"
            raise ValueError(msg)
        self._limit = daily_tokens
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._spent = 0
        self._day = self._today()

    def _today(self) -> int:
        return int(self._clock() // SECONDS_PER_DAY)

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day = today
            self._spent = 0

    def allows(self) -> bool:
        """Whether a paid call may be attempted now."""
        with self._lock:
            self._roll()
            return self._spent < self._limit

    def charge(self, tokens: int) -> None:
        """Record tokens actually spent. Negative counts are ignored."""
        if tokens <= 0:
            return
        with self._lock:
            self._roll()
            self._spent += tokens

    def status(self) -> BudgetStatus:
        with self._lock:
            self._roll()
            return BudgetStatus(
                limit=self._limit,
                spent=self._spent,
                remaining=max(0, self._limit - self._spent),
                exhausted=self._spent >= self._limit,
            )


def normalize_question(question: str) -> str:
    """A cache key that ignores spacing, case, and trailing punctuation.

    Deliberately crude. Two questions that differ only in wording are different
    keys and will each be paid for; the goal is to collapse the identical
    resubmissions a demo actually receives, not to cluster paraphrases.
    """
    return " ".join(question.lower().split()).rstrip("?!. ")


@dataclass
class AnswerCache:
    """A bounded, least-recently-used cache of answers by normalised question."""

    max_entries: int = DEFAULT_CACHE_SIZE
    _entries: OrderedDict[str, object] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    hits: int = 0
    misses: int = 0

    def get(self, question: str) -> object | None:
        key = normalize_question(question)
        with self._lock:
            if key not in self._entries:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return self._entries[key]

    def put(self, question: str, value: object) -> None:
        key = normalize_question(question)
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class RateLimiter:
    """A fixed-window request counter per client.

    Args:
        limit: Requests allowed per window.
        window_seconds: Length of the window.
        clock: Injected time source.
    """

    def __init__(
        self,
        limit: int = DEFAULT_RATE_LIMIT,
        window_seconds: float = DEFAULT_RATE_WINDOW,
        *,
        clock: Clock | None = None,
    ) -> None:
        if limit < 1:
            msg = "limit must be at least 1"
            raise ValueError(msg)
        if window_seconds <= 0:
            msg = "window_seconds must be positive"
            raise ValueError(msg)
        self._limit = limit
        self._window = window_seconds
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._counts: dict[str, tuple[int, int]] = {}

    def allows(self, client: str) -> bool:
        """Whether this client may make a request now, counting it if so."""
        window = int(self._clock() // self._window)
        with self._lock:
            current, count = self._counts.get(client, (window, 0))
            if current != window:
                current, count = window, 0
            if count >= self._limit:
                self._counts[client] = (current, count)
                return False
            self._counts[client] = (current, count + 1)
            # Drop clients from older windows so the table cannot grow forever.
            if len(self._counts) > 4096:
                self._counts = {
                    key: value for key, value in self._counts.items() if value[0] == window
                }
            return True
