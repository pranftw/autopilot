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

  def test_sync_acquire_triggers_sleep_when_full(self) -> None:
    """When window is full, acquire sleeps until a slot opens."""
    limiter = SlidingWindowLimiter(max_rpm=1, safety_margin=1.0)
    # first acquire: now=10.0, window empty, appends 10.0
    # second acquire: now=11.0, window[0]=10.0 not expired, len=1>=1
    # sleep_time = 60 - (11 - 10) = 59
    # post-sleep: now=70.0, prune 10.0 <= 70-60=10 (yes), appends 70.0
    mono_values = iter([10.0, 10.0, 11.0, 70.0, 70.0])
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
      return next(mono_values)

    def fake_sleep(t: float) -> None:
      sleep_calls.append(t)

    with (
      patch('autopilot.ai.runtime.time.monotonic', side_effect=fake_monotonic),
      patch('autopilot.ai.runtime.time.sleep', side_effect=fake_sleep),
    ):
      limiter.acquire()
      limiter.acquire()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(59.0)
    # first entry (10.0) pruned post-sleep (10.0 <= 70-60), only new append remains
    assert len(limiter._window) == 1
    assert limiter._window[0] == 70.0

  def test_sync_acquire_post_sleep_prunes_old(self) -> None:
    """After sleeping, the post-sleep prune loop removes expired entries."""
    limiter = SlidingWindowLimiter(max_rpm=2, safety_margin=1.0)
    # pre-fill window with 2 entries at t=0 and t=1
    limiter._window.append(0.0)
    limiter._window.append(1.0)
    # acquire: now=2.0, window[0]=0.0, 0.0 <= 2.0-60=-58? No. Not expired.
    # len=2 >= 2, sleep_time = 60 - (2.0 - 0.0) = 58
    # after sleep: now=65.0, prune: 0.0 <= 65-60=5? Yes. 1.0 <= 5? Yes.
    # appends 65.0
    mono_values = iter([2.0, 65.0, 65.0])
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
      return next(mono_values)

    def fake_sleep(t: float) -> None:
      sleep_calls.append(t)

    with (
      patch('autopilot.ai.runtime.time.monotonic', side_effect=fake_monotonic),
      patch('autopilot.ai.runtime.time.sleep', side_effect=fake_sleep),
    ):
      limiter.acquire()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(58.0)
    # old entries pruned, only new one remains
    assert len(limiter._window) == 1
    assert limiter._window[0] == 65.0

  @pytest.mark.asyncio
  async def test_async_acquire_triggers_sleep_when_full(self) -> None:
    """async_acquire sleeps when window is at capacity."""
    limiter = SlidingWindowLimiter(max_rpm=1, safety_margin=1.0)
    # same logic as sync: sleep_time = 60 - (11 - 10) = 59
    mono_values = iter([10.0, 10.0, 11.0, 70.0, 70.0])
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
      return next(mono_values)

    async def fake_async_sleep(t: float) -> None:
      sleep_calls.append(t)

    with (
      patch('autopilot.ai.runtime.time.monotonic', side_effect=fake_monotonic),
      patch('autopilot.ai.runtime.asyncio.sleep', side_effect=fake_async_sleep),
    ):
      await limiter.async_acquire()
      await limiter.async_acquire()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(59.0)
    # first entry pruned post-sleep, only the new append remains
    assert len(limiter._window) == 1
    assert limiter._window[0] == 70.0

  @pytest.mark.asyncio
  async def test_async_acquire_post_sleep_prunes(self) -> None:
    """async_acquire prunes expired entries after sleeping."""
    limiter = SlidingWindowLimiter(max_rpm=1, safety_margin=1.0)
    limiter._window.append(5.0)
    # now=6.0, window[0]=5.0 not expired (5.0 <= 6-60=-54? No), len=1>=1
    # sleep_time = 60 - (6-5) = 59, post-sleep now=70.0, 5.0<=70-60=10 -> pruned
    mono_values = iter([6.0, 70.0, 70.0])
    sleep_calls: list[float] = []

    def fake_monotonic() -> float:
      return next(mono_values)

    async def fake_async_sleep(t: float) -> None:
      sleep_calls.append(t)

    with (
      patch('autopilot.ai.runtime.time.monotonic', side_effect=fake_monotonic),
      patch('autopilot.ai.runtime.asyncio.sleep', side_effect=fake_async_sleep),
    ):
      await limiter.async_acquire()

    assert len(sleep_calls) == 1
    assert sleep_calls[0] == pytest.approx(59.0)
    assert len(limiter._window) == 1
    assert limiter._window[0] == 70.0


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
        msg = 'bad'
        raise ValueError(msg)
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

  @pytest.mark.asyncio
  async def test_parallel_runner_deterministic_order(self) -> None:
    runner = ParallelRunner(3, limiter=None)
    delays = {'a': 0.03, 'b': 0.02, 'c': 0.01}

    async def fn(x: str) -> str:
      await asyncio.sleep(delays[x])
      return x

    results = await runner.run(['a', 'b', 'c'], fn)
    assert results == ['a', 'b', 'c']

  @pytest.mark.asyncio
  async def test_parallel_runner_partial_failure(self) -> None:
    runner = ParallelRunner(3, limiter=None)

    async def fn(x: str) -> str:
      await asyncio.sleep(0.01)
      if x == 'b':
        msg = 'middle'
        raise ValueError(msg)
      return f'ok_{x}'

    with pytest.raises(ValueError, match='middle'):
      await runner.run(['a', 'b', 'c'], fn)
