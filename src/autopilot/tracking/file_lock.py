"""File locking backed by the filelock library.

Provides AutopilotFileLock, a thin wrapper that maps filelock.Timeout
to ConcurrentMutationError for the autopilot error hierarchy. Uses
fcntl.flock on POSIX -- kernel-enforced advisory locking with automatic
release on process death.

Timeout semantics:

- ``timeout_s=None`` (default): fail-fast, raises immediately on contention.
- ``timeout_s > 0``: wait up to N seconds, then raise on timeout.
- ``timeout_s == -1.0``: block until the lock is acquired (infinite wait).

On contention or timeout, raises ``ConcurrentMutationError`` (a
``TrackingError`` subclass) with ``retry_after_ms`` hint and
``operation`` label so agents can retry with backoff.
"""

from autopilot.core.errors import TrackingError
from filelock import FileLock as _FileLock
from filelock import Timeout as _Timeout
from pathlib import Path

LOCK_RETRY_AFTER_MS = 100


class ConcurrentMutationError(TrackingError):
  """Raised when an exclusive workspace mutation cannot acquire its lock.

  Attributes:
    operation: Label identifying the mutation that was blocked.
    retry_after_ms: Suggested wait before retrying, in milliseconds.
  """

  def __init__(self, operation: str, *, retry_after_ms: int = LOCK_RETRY_AFTER_MS) -> None:
    """Create a concurrent mutation error with retry guidance.

    Args:
      operation: Short label for the blocked operation (e.g. ``'forest_save'``).
      retry_after_ms: Suggested retry delay in milliseconds.
    """
    self.retry_after_ms = retry_after_ms
    self.operation = operation
    super().__init__(
      f'concurrent mutation detected during {operation}. '
      f'Retry after {retry_after_ms}ms or pass --wait <milliseconds>.'
    )


class AutopilotFileLock:
  """Exclusive file lock backed by the filelock library.

  Uses fcntl.flock on POSIX (kernel-enforced, auto-released on crash).
  On contention, raises ``ConcurrentMutationError`` (``TrackingError``
  subclass) with a retry hint and operation label.

  Args:
    path: Lock file path.
    timeout_s: Seconds to wait. ``None`` means fail immediately on
      contention. Positive float waits up to that many seconds.
      ``-1.0`` blocks until acquired (infinite wait).
    operation: Label for the guarded mutation, embedded in the error
      message on contention.
  """

  def __init__(
    self,
    path: Path,
    timeout_s: float | None = None,
    *,
    operation: str = 'lock',
  ) -> None:
    """Create a lock targeting ``path``.

    Args:
      path: Lock file path.
      timeout_s: Seconds to wait. ``None`` means fail immediately (timeout=0).
        Positive float waits up to that many seconds. ``-1.0`` blocks
        indefinitely (maps to ``timeout=-1`` for filelock).
      operation: Label identifying the guarded mutation for error messages.
    """
    if timeout_s is None:
      effective_timeout = 0
    elif timeout_s == -1.0:
      effective_timeout = -1
    else:
      effective_timeout = timeout_s
    self._lock = _FileLock(path, timeout=effective_timeout)
    self._operation = operation

  def acquire(self) -> None:
    """Acquire the lock.

    Raises:
      ConcurrentMutationError: On contention or timeout.
    """
    try:
      self._lock.acquire()
    except _Timeout as exc:
      raise ConcurrentMutationError(self._operation) from exc

  def release(self) -> None:
    """Release the lock. Idempotent."""
    self._lock.release()

  @property
  def is_locked(self) -> bool:
    """Whether the lock is currently held."""
    return self._lock.is_locked

  @property
  def timeout_s(self) -> float | None:
    """Effective timeout in seconds (``None`` = fail-fast, ``-1.0`` = infinite)."""
    t = self._lock.timeout
    if t == 0:
      return None
    if t == -1:
      return -1.0
    return float(t)

  @timeout_s.setter
  def timeout_s(self, value: float | None) -> None:
    """Update the lock timeout dynamically.

    Args:
      value: ``None`` = fail-fast, positive = wait seconds, ``-1.0`` = block forever.
    """
    if value is None:
      self._lock.timeout = 0
    elif value == -1.0:
      self._lock.timeout = -1
    else:
      self._lock.timeout = value

  def __enter__(self) -> 'AutopilotFileLock':
    """Acquire lock on context entry.

    Returns:
      This lock instance.
    """
    self.acquire()
    return self

  def __exit__(self, *exc: object) -> None:
    """Release lock on context exit."""
    self.release()
