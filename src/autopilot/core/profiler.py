"""Wall-clock profiler for training section timing.

GPU/CUDA profiling is not relevant for non-differentiable optimization.
This module provides wall-clock timing via ``time.perf_counter()`` for
diagnosing bottlenecks in agent steps, store operations, and training
sections.

Classes:
  Profiler: Base protocol for wall-clock timing backends.
  SimpleProfiler: Built-in profiler recording durations per action.

Usage:
  trainer = Trainer(..., profiler=SimpleProfiler())
  # after fit completes, profiler_summary.json is written to experiment dir
"""

from collections.abc import Iterator
from typing import Any
import contextlib
import time


class Profiler:
  """Base profiler protocol for wall-clock timing of training sections.

  Override start/stop for custom profiling backends. The ``profile()``
  context manager delegates to start/stop and is the recommended entry
  point for section wrapping.
  """

  def start(self, action: str) -> None:
    """Begin timing an action.

    Args:
      action: Name of the section being timed (e.g. 'training_step').

    Raises:
      NotImplementedError: Subclasses must implement.
    """
    raise NotImplementedError

  def stop(self, action: str) -> None:
    """End timing an action.

    Args:
      action: Name of the section being stopped.

    Raises:
      NotImplementedError: Subclasses must implement.
    """
    raise NotImplementedError

  @contextlib.contextmanager
  def profile(self, action: str) -> Iterator[None]:
    """Context manager wrapping start/stop for a named action.

    Args:
      action: Name of the section to time.

    Yields:
      None -- the body of the ``with`` block runs between start and stop.
    """
    self.start(action)
    try:
      yield
    finally:
      self.stop(action)

  def describe(self) -> dict[str, Any]:
    """Return a JSON-serializable summary of recorded timings.

    Returns:
      Dict keyed by action name with timing statistics.

    Raises:
      NotImplementedError: Subclasses must implement.
    """
    raise NotImplementedError


class SimpleProfiler(Profiler):
  """Wall-clock profiler recording durations per action.

  Records wall-clock durations using ``time.perf_counter()``.
  ``describe()`` returns ``{action: {'count': int, 'total_ms': float,
  'mean_ms': float}}``. Nested same-action calls are not supported --
  calling ``start('a')`` while ``'a'`` is already active raises
  ``ValueError``.
  """

  def __init__(self) -> None:
    """Initialize empty timing state."""
    self._start_times: dict[str, float] = {}
    self._durations: dict[str, list[float]] = {}

  def start(self, action: str) -> None:
    """Begin timing an action.

    Args:
      action: Name of the section being timed.

    Raises:
      ValueError: When the action is already active (no nested same-action).
    """
    if action in self._start_times:
      msg = (
        f'Profiler action {action!r} already started. Call stop({action!r}) before starting again.'
      )
      raise ValueError(msg)
    self._start_times[action] = time.perf_counter()

  def stop(self, action: str) -> None:
    """End timing an action and record its duration.

    Args:
      action: Name of the section being stopped.

    Raises:
      ValueError: When the action was never started.
    """
    if action not in self._start_times:
      msg = f'Profiler action {action!r} was never started. Call start({action!r}) first.'
      raise ValueError(msg)
    elapsed = (time.perf_counter() - self._start_times.pop(action)) * 1000
    self._durations.setdefault(action, []).append(elapsed)

  def describe(self) -> dict[str, Any]:
    """Return timing statistics per action.

    Returns:
      Dict keyed by action with ``count``, ``total_ms``, and ``mean_ms``.
      Empty dict when no actions have been recorded.
    """
    result: dict[str, Any] = {}
    for action, durations in self._durations.items():
      total = sum(durations)
      result[action] = {
        'count': len(durations),
        'total_ms': round(total, 3),
        'mean_ms': round(total / len(durations), 3),
      }
    return result
