"""Tests for autopilot.ai.runtime."""

from autopilot.ai.runtime import ParallelRunner, RateLimiter, SlidingWindowLimiter
from unittest.mock import AsyncMock, patch
import asyncio
import pytest
import time


class TestRateLimiter:
  def test_base_acquire_not_implemented(self) -> None:
    class _Bare(RateLimiter):
      pass

    with pytest.raises(NotImplementedError):
      _Bare().acquire()

  @pytest.mark.asyncio
  async def test_base_async_acquire_not_implemented(self) -> None:
    class _Bare(RateLimiter):
      def acquire(self) -> None:
        return None

    with pytest.raises(NotImplementedError):
      await _Bare().async_acquire()


class TestSlidingWindowLimiter:
  @pytest.mark.asyncio
  async def test_no_delay_under_limit(self) -> None:
    limiter = SlidingWindowLimiter(100)
    t0 = time.monotonic()
    for _ in range(5):
      await limiter.async_acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0

  @pytest.mark.asyncio
  async def test_safety_margin_reduces_effective(self) -> None:
    limiter = SlidingWindowLimiter(100, safety_margin=0.5)
    assert limiter._effective_rpm == 50

  @pytest.mark.asyncio
  async def test_window_purges_old_entries(self) -> None:
    limiter = SlidingWindowLimiter(10)
    mono_calls = iter([0.0, 0.0, 61.0, 61.0])

    def fake_monotonic() -> float:
      return next(mono_calls)

    with patch('autopilot.ai.runtime.time.monotonic', side_effect=fake_monotonic):
      await limiter.async_acquire()
      await limiter.async_acquire()
    assert len(limiter._window) == 1
    assert limiter._window[0] == 61.0

  @pytest.mark.asyncio
  async def test_concurrent_async_acquires(self) -> None:
    limiter = SlidingWindowLimiter(20)

    async def worker() -> None:
      await limiter.async_acquire()

    await asyncio.gather(*[worker() for _ in range(10)])

  def test_sync_acquire_under_limit(self) -> None:
    limiter = SlidingWindowLimiter(1000)
    t0 = time.monotonic()
    for _ in range(5):
      limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0


class TestParallelRunner:
  @pytest.mark.asyncio
  async def test_processes_all_items(self) -> None:
    limiter = SlidingWindowLimiter(1000)
    runner = ParallelRunner(5, limiter=limiter)

    async def fn(x: int) -> int:
      return x * 2

    results = await runner.run([1, 2, 3, 4, 5], fn)
    assert len(results) == 5

  @pytest.mark.asyncio
  async def test_no_limiter_concurrency_only(self) -> None:
    runner = ParallelRunner(3, limiter=None)

    async def fn(x: int) -> int:
      return x + 1

    results = await runner.run([1, 2, 3], fn)
    assert set(results) == {2, 3, 4}

  @pytest.mark.asyncio
  async def test_results_complete(self) -> None:
    limiter = SlidingWindowLimiter(1000)
    runner = ParallelRunner(5, limiter=limiter)

    async def fn(x: int) -> int:
      return x + 10

    results = await runner.run([1, 2, 3], fn)
    assert set(results) == {11, 12, 13}

  @pytest.mark.asyncio
  async def test_concurrency_limit_enforced(self) -> None:
    limiter = SlidingWindowLimiter(1000)
    runner = ParallelRunner(2, limiter=limiter)
    concurrent = 0
    max_concurrent = 0
    counter_lock = asyncio.Lock()

    async def fn(x: int) -> int:
      nonlocal concurrent, max_concurrent
      async with counter_lock:
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
      await asyncio.sleep(0.01)
      async with counter_lock:
        concurrent -= 1
      return x

    await runner.run(list(range(10)), fn)
    assert max_concurrent == 2

  @pytest.mark.asyncio
  async def test_on_complete_callback(self) -> None:
    limiter = SlidingWindowLimiter(1000)
    runner = ParallelRunner(5, limiter=limiter)
    seen: list[int] = []

    async def fn(x: int) -> int:
      return x * 2

    def on_complete(r: int) -> None:
      seen.append(r)

    await runner.run([1, 2, 3], fn, on_complete=on_complete)
    assert sorted(seen) == [2, 4, 6]

  @pytest.mark.asyncio
  async def test_exception_propagates(self) -> None:
    limiter = SlidingWindowLimiter(1000)
    runner = ParallelRunner(5, limiter=limiter)

    async def fn(x: int) -> int:
      if x == 2:
        raise ValueError('bad')
      return x

    with pytest.raises(ValueError, match='bad'):
      await runner.run([1, 2, 3], fn)

  @pytest.mark.asyncio
  async def test_empty_items(self) -> None:
    limiter = SlidingWindowLimiter(1000)
    runner = ParallelRunner(5, limiter=limiter)

    async def fn(x: int) -> int:
      return x

    assert await runner.run([], fn) == []

  @pytest.mark.asyncio
  async def test_single_item(self) -> None:
    limiter = SlidingWindowLimiter(1000)
    runner = ParallelRunner(5, limiter=limiter)

    async def fn(x: int) -> int:
      return x + 1

    assert await runner.run([42], fn) == [43]

  @pytest.mark.asyncio
  async def test_limiter_async_acquire_integrated(self) -> None:
    limiter = SlidingWindowLimiter(100)
    mock_async = AsyncMock(return_value=None)
    with patch.object(limiter, 'async_acquire', mock_async):
      runner = ParallelRunner(3, limiter=limiter)

      async def fn(x: int) -> int:
        return x

      await runner.run([1, 2, 3], fn)
    assert mock_async.await_count == 3
