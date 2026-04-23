"""Rate limiting and parallel execution for AI workflows."""

from collections import deque
from typing import Awaitable, Callable, TypeVar
import asyncio
import threading
import time

T = TypeVar('T')
R = TypeVar('R')


class RateLimiter:
  """Base rate limiter. Backend-agnostic, works in both sync and async contexts.

  Subclass for different strategies: sliding window RPM, token bucket,
  cost-based, or no-op.
  """

  def acquire(self) -> None:
    """Sync: wait until a request slot is available. Override in subclasses."""
    raise NotImplementedError

  async def async_acquire(self) -> None:
    """Async: wait until a request slot is available. Override in subclasses."""
    raise NotImplementedError


class SlidingWindowLimiter(RateLimiter):
  """Sliding-window RPM limiter. Built-in default.

  Deque of timestamps, 60-second window.
  Provides both sync (threading.Lock + time.sleep) and
  async (asyncio.Lock + asyncio.sleep) implementations.
  """

  def __init__(self, max_rpm: int, safety_margin: float = 1.0) -> None:
    self._effective_rpm = int(max_rpm * safety_margin)
    self._window: deque[float] = deque()
    self._sync_lock = threading.Lock()
    self._async_lock = asyncio.Lock()

  def acquire(self) -> None:
    """Sync acquire: blocks with time.sleep() under threading.Lock."""
    with self._sync_lock:
      now = time.monotonic()
      while self._window and self._window[0] <= now - 60.0:
        self._window.popleft()
      if len(self._window) >= self._effective_rpm:
        sleep_time = 60.0 - (now - self._window[0])
        if sleep_time > 0:
          time.sleep(sleep_time)
          now = time.monotonic()
          while self._window and self._window[0] <= now - 60.0:
            self._window.popleft()
      self._window.append(time.monotonic())

  async def async_acquire(self) -> None:
    """Async acquire: awaits with asyncio.sleep() under asyncio.Lock."""
    async with self._async_lock:
      now = time.monotonic()
      while self._window and self._window[0] <= now - 60.0:
        self._window.popleft()
      if len(self._window) >= self._effective_rpm:
        sleep_time = 60.0 - (now - self._window[0])
        if sleep_time > 0:
          await asyncio.sleep(sleep_time)
          now = time.monotonic()
          while self._window and self._window[0] <= now - 60.0:
            self._window.popleft()
      self._window.append(time.monotonic())


class ParallelRunner:
  """Run async tasks with concurrency limit and optional rate limiting.

  Takes a RateLimiter (any implementation). Uses async_acquire() since
  ParallelRunner is inherently async. When limiter is None, rate limiting
  is skipped (concurrency only).
  """

  def __init__(self, num_parallel: int, limiter: RateLimiter | None = None) -> None:
    self._num_parallel = num_parallel
    self._limiter = limiter

  async def run(
    self,
    items: list[T],
    fn: Callable[[T], Awaitable[R]],
    on_complete: Callable[[R], None] | None = None,
  ) -> list[R]:
    """Process all items with concurrency and optional RPM limits."""
    if not items:
      return []

    semaphore = asyncio.Semaphore(self._num_parallel)
    results: list[R] = []
    lock = asyncio.Lock()

    async def _process(item: T) -> R:
      async with semaphore:
        if self._limiter is not None:
          await self._limiter.async_acquire()
        result = await fn(item)
        async with lock:
          results.append(result)
        if on_complete is not None:
          on_complete(result)
        return result

    tasks = [asyncio.create_task(_process(item)) for item in items]
    await asyncio.gather(*tasks)
    return results
