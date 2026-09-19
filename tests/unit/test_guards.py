"""Tests for the spend guards.

These protect a public URL from an unbounded bill, so their edge cases are
asserted rather than assumed: the day boundary that resets the budget, the
window boundary the rate limiter is deliberately coarse about, and the eviction
that stops the cache growing forever.
"""

import pytest

from bio_research_copilot.web.guards import (
    SECONDS_PER_DAY,
    AnswerCache,
    RateLimiter,
    TokenBudget,
    normalize_question,
)


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestTokenBudget:
    def test_allows_spending_until_the_limit(self) -> None:
        budget = TokenBudget(1000)
        assert budget.allows()
        budget.charge(999)
        assert budget.allows()
        budget.charge(1)
        assert not budget.allows()

    def test_zero_disables_the_model_entirely(self) -> None:
        """The correct configuration for a deployment with no key."""
        assert not TokenBudget(0).allows()

    def test_resets_on_a_new_day(self) -> None:
        clock = FakeClock()
        budget = TokenBudget(1000, clock=clock)
        budget.charge(1000)
        assert not budget.allows()
        clock.advance(SECONDS_PER_DAY)
        assert budget.allows()
        assert budget.status().spent == 0

    def test_does_not_reset_within_a_day(self) -> None:
        clock = FakeClock()
        budget = TokenBudget(1000, clock=clock)
        budget.charge(1000)
        clock.advance(SECONDS_PER_DAY - 1)
        assert not budget.allows()

    def test_reports_what_remains(self) -> None:
        budget = TokenBudget(1000)
        budget.charge(300)
        status = budget.status()
        assert (status.spent, status.remaining, status.exhausted) == (300, 700, False)

    def test_remaining_never_goes_negative(self) -> None:
        budget = TokenBudget(100)
        budget.charge(500)
        assert budget.status().remaining == 0
        assert budget.status().exhausted

    def test_ignores_nonpositive_charges(self) -> None:
        budget = TokenBudget(100)
        budget.charge(0)
        budget.charge(-50)
        assert budget.status().spent == 0

    def test_a_negative_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            TokenBudget(-1)


class TestNormalizeQuestion:
    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("What causes X?", "what causes x"),
            ("  spaced   out  ", "spaced out"),
            ("Ends with period.", "ends with period"),
            ("Shouting!", "shouting"),
        ],
    )
    def test_collapses_incidental_differences(self, left: str, right: str) -> None:
        assert normalize_question(left) == normalize_question(right)

    def test_keeps_genuinely_different_questions_apart(self) -> None:
        assert normalize_question("what causes X") != normalize_question("what prevents X")


class TestAnswerCache:
    def test_returns_what_was_stored(self) -> None:
        cache = AnswerCache()
        cache.put("A question?", "answer")
        assert cache.get("a question") == "answer"

    def test_reports_a_miss_as_none(self) -> None:
        assert AnswerCache().get("never seen") is None

    def test_counts_hits_and_misses(self) -> None:
        cache = AnswerCache()
        cache.get("x")
        cache.put("x", 1)
        cache.get("x")
        assert (cache.hits, cache.misses) == (1, 1)

    def test_evicts_the_least_recently_used(self) -> None:
        cache = AnswerCache(max_entries=2)
        cache.put("one", 1)
        cache.put("two", 2)
        cache.get("one")  # makes "two" the least recently used
        cache.put("three", 3)
        assert len(cache) == 2
        assert cache.get("two") is None
        assert cache.get("one") == 1

    def test_never_exceeds_its_bound(self) -> None:
        cache = AnswerCache(max_entries=3)
        for n in range(50):
            cache.put(f"question {n}", n)
        assert len(cache) == 3


class TestRateLimiter:
    def test_allows_up_to_the_limit(self) -> None:
        limiter = RateLimiter(2, 60.0, clock=FakeClock())
        assert limiter.allows("1.2.3.4")
        assert limiter.allows("1.2.3.4")
        assert not limiter.allows("1.2.3.4")

    def test_clients_are_counted_separately(self) -> None:
        limiter = RateLimiter(1, 60.0, clock=FakeClock())
        assert limiter.allows("1.2.3.4")
        assert not limiter.allows("1.2.3.4")
        assert limiter.allows("5.6.7.8")

    def test_a_new_window_resets_the_count(self) -> None:
        clock = FakeClock()
        limiter = RateLimiter(1, 60.0, clock=clock)
        assert limiter.allows("1.2.3.4")
        assert not limiter.allows("1.2.3.4")
        clock.advance(60.0)
        assert limiter.allows("1.2.3.4")

    def test_a_fixed_window_permits_a_burst_across_the_boundary(self) -> None:
        """The known coarseness: the budget is the real protection, not this."""
        clock = FakeClock(59.0)
        limiter = RateLimiter(2, 60.0, clock=clock)
        assert limiter.allows("1.2.3.4")
        assert limiter.allows("1.2.3.4")
        clock.advance(1.0)
        assert limiter.allows("1.2.3.4")
        assert limiter.allows("1.2.3.4")

    @pytest.mark.parametrize(
        ("limit", "window", "message"),
        [(0, 60.0, "limit must be at least 1"), (1, 0.0, "window_seconds must be positive")],
    )
    def test_invalid_configuration(self, limit: int, window: float, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            RateLimiter(limit, window)
