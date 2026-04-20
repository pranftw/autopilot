"""Rate limiting and parallel execution for AI workflows."""

from collections import deque
from typing import Awaitable, Callable, TypeVar
import asyncio
import time

T = TypeVar('T')
R = TypeVar('R')


class RPMLimiter:
  """Sliding-window RPM (requests per minute) limiter.

  Tracks request timestamps in a 60-second sliding window.
  Delays acquire() when the effective limit is reached.
  """

  def __init__(self, max_rpm: int, safety_margin: float = 1.0) -> None:
    self._effective_rpm = int(max_rpm * safety_margin)
    self._window: deque[float] = deque()
    self._lock = asyncio.Lock()

  async def acquire(self) -> None:
    """Wait until a request slot is available within the RPM window."""
    async with self._lock:
      now = time.monotonic()
      # Purge entries older than 60 seconds
      while self._window and self._window[0] <= now - 60.0:
        self._window.popleft()
      if len(self._window) >= self._effective_rpm:
        # Wait until the oldest entry expires
        sleep_time = 60.0 - (now - self._window[0])
        if sleep_time > 0:
          await asyncio.sleep(sleep_time)
          # Re-purge after sleeping
          now = time.monotonic()
          while self._window and self._window[0] <= now - 60.0:
            self._window.popleft()
      self._window.append(time.monotonic())


class ParallelRunner:
  """Run async tasks with concurrency limit and RPM limiting.

  Items are processed concurrently up to num_parallel. Each task
  acquires an RPM slot before executing.
  """

  def __init__(self, num_parallel: int, rpm_limiter: RPMLimiter) -> None:
    self._num_parallel = num_parallel
    self._rpm_limiter = rpm_limiter

  async def run(
    self,
    items: list[T],
    fn: Callable[[T], Awaitable[R]],
    on_complete: Callable[[R], None] | None = None,
  ) -> list[R]:
    """Process all items with concurrency and RPM limits.

    Returns results in completion order (not input order).
    Exceptions in fn propagate after all tasks complete.
    """
    if not items:
      return []

    semaphore = asyncio.Semaphore(self._num_parallel)
    results: list[R] = []
    lock = asyncio.Lock()

    async def _process(item: T) -> R:
      async with semaphore:
        await self._rpm_limiter.acquire()
        result = await fn(item)
        async with lock:
          results.append(result)
        if on_complete is not None:
          on_complete(result)
        return result

    tasks = [asyncio.create_task(_process(item)) for item in items]
    # gather with return_exceptions=False to propagate first exception
    await asyncio.gather(*tasks)
    return results
