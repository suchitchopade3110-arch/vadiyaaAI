"""Unit tests for app/utils/retry.py — retry/backoff/fallback behavior."""
from __future__ import annotations

import pytest

from app.utils import retry as retry_module
from app.utils.retry import with_retry


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Retry backoff must not slow the test suite down."""
    monkeypatch.setattr(retry_module.time, "sleep", lambda _seconds: None)


def test_succeeds_on_first_call_without_retrying():
    calls = []

    @with_retry(max_retries=2, backoff_seconds=0.01)
    def flaky():
        calls.append(1)
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 1


def test_retries_then_succeeds():
    calls = []

    @with_retry(max_retries=3, backoff_seconds=0.01)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("transient")
        return "recovered"

    assert flaky() == "recovered"
    assert len(calls) == 3


def test_exhausts_retries_and_raises_last_exception_without_fallback():
    calls = []

    @with_retry(max_retries=2, backoff_seconds=0.01)
    def always_fails():
        calls.append(1)
        raise ValueError(f"attempt {len(calls)}")

    with pytest.raises(ValueError, match="attempt 3"):
        always_fails()
    assert len(calls) == 3  # initial attempt + 2 retries


def test_exhausts_retries_and_calls_fallback_with_same_args():
    def fallback(x, y=None):
        return f"fallback-{x}-{y}"

    @with_retry(max_retries=1, backoff_seconds=0.01, fallback=fallback)
    def always_fails(x, y=None):
        raise RuntimeError("boom")

    result = always_fails("a", y="b")
    assert result == "fallback-a-b"


def test_only_catches_specified_exception_types():
    @with_retry(max_retries=2, backoff_seconds=0.01, exceptions=(ValueError,))
    def raises_type_error():
        raise TypeError("not retried")

    with pytest.raises(TypeError):
        raises_type_error()


def test_backoff_uses_exponential_growth(monkeypatch):
    waits = []
    monkeypatch.setattr(retry_module.time, "sleep", lambda seconds: waits.append(seconds))

    @with_retry(max_retries=3, backoff_seconds=1.0)
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        always_fails()

    assert waits == [1.0, 2.0, 4.0]
