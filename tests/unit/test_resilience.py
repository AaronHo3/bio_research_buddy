"""Tests for the retry and rate-limit policies, with a fake clock."""

import pytest

from bio_research_copilot.adapters.resilience import MinIntervalLimiter, RetryPolicy
from tests.unit.fakes import FakeClock


class TestRetryPolicy:
    def test_delays_double_and_cap(self) -> None:
        policy = RetryPolicy(max_attempts=6, base_delay_seconds=0.5, max_delay_seconds=3.0)
        assert [policy.delay_before_retry(n) for n in range(1, 6)] == [0.5, 1.0, 2.0, 3.0, 3.0]

    def test_rejects_zero_attempts(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            RetryPolicy(max_attempts=0)


class TestMinIntervalLimiter:
    def test_first_call_never_waits(self) -> None:
        clock = FakeClock()
        MinIntervalLimiter(3, clock=clock, sleep=clock.sleep).wait()
        assert clock.sleeps == []

    def test_rapid_second_call_sleeps_the_remainder(self) -> None:
        clock = FakeClock()
        limiter = MinIntervalLimiter(4, clock=clock, sleep=clock.sleep)  # 0.25 s interval
        limiter.wait()
        clock.now += 0.1
        limiter.wait()
        assert clock.sleeps == [pytest.approx(0.15)]

    def test_slow_second_call_does_not_sleep(self) -> None:
        clock = FakeClock()
        limiter = MinIntervalLimiter(4, clock=clock, sleep=clock.sleep)
        limiter.wait()
        clock.now += 1.0
        limiter.wait()
        assert clock.sleeps == []

    def test_interval_is_measured_from_the_last_allowed_call(self) -> None:
        clock = FakeClock()
        limiter = MinIntervalLimiter(2, clock=clock, sleep=clock.sleep)  # 0.5 s interval
        limiter.wait()
        limiter.wait()
        limiter.wait()
        assert clock.sleeps == [pytest.approx(0.5), pytest.approx(0.5)]

    def test_rejects_non_positive_rate(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MinIntervalLimiter(0)
