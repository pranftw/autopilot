"""Low-level storage backend for FileStore.

Handles content-addressed object storage, hash computation, and file
locking. Not intended for direct use -- use FileStore public API.

Locking is backed by AutopilotFileLock (filelock library, fcntl.flock
on POSIX). Crash recovery is automatic -- OS releases the advisory lock
on process death. Blob temp files use PID + thread ident suffixes to
avoid collision across concurrent writers.
"""

from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError, TrackingError
from autopilot.tracking.file_lock import AutopilotFileLock, ConcurrentMutationError
import hashlib
import os
import threading


def hash_content(text: str) -> str:
  """Return the SHA-256 hex digest of a UTF-8-encoded string.

  Args:
    text: The string to hash.

  Returns:
    64-character lowercase hex digest.
  """
  return hashlib.sha256(text.encode('utf-8')).hexdigest()


def hash_bytes(data: bytes) -> str:
  """Return the SHA-256 hex digest of raw bytes.

  Args:
    data: The bytes to hash.

  Returns:
    64-character lowercase hex digest.
  """
  return hashlib.sha256(data).hexdigest()


class StorageBackend:
  """Low-level content-addressed storage with file locking.

  SHA-256 hashing, 2-char prefix sharding for object blob I/O,
  and exclusive file locking for atomic snapshot writes. FileStore
  creates a StorageBackend in __init__ and delegates all low-level
  storage operations to it.
  """

  def __init__(self, config: AutoPilotConfig) -> None:
    """Prepare object directories and exclusive lock file path under the store root.

    Args:
      config: Supplies ``objects_path`` and ``store_path`` for layout creation.
    """
    self._objects_dir = config.objects_path
    self._lock_file = config.store_path / '.lock'
    config.store_path.mkdir(parents=True, exist_ok=True)
    self._objects_dir.mkdir(parents=True, exist_ok=True)
    self._lock = AutopilotFileLock(self._lock_file, operation='store')

  @property
  def lock_timeout_s(self) -> float | None:
    """Effective lock timeout in seconds (``None`` = fail-fast)."""
    return self._lock.timeout_s

  @lock_timeout_s.setter
  def lock_timeout_s(self, value: float | None) -> None:
    """Update the lock timeout dynamically.

    Args:
      value: ``None`` = fail-fast, positive = wait seconds, ``-1.0`` = block forever.
    """
    self._lock.timeout_s = value

  def store_object(self, content_hash: str, data: bytes) -> None:
    """Write a content-addressed blob to the object store.

    Uses a per-process, per-thread temp suffix to avoid collision when
    multiple writers store the same hash concurrently.

    Raises:
      StoreError: When temp write or replace fails with :class:`OSError`.
    """
    prefix = content_hash[:2]
    rest = content_hash[2:]
    obj_dir = self._objects_dir / prefix
    obj_path = obj_dir / rest
    if obj_path.exists():
      return
    obj_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'.{os.getpid()}.{threading.get_ident()}.tmp'
    tmp = obj_dir / f'{rest}{suffix}'
    try:
      tmp.write_bytes(data)
      tmp.replace(obj_path)
    except OSError as exc:
      tmp.unlink(missing_ok=True)
      msg = f'failed to store object {content_hash}: {exc}'
      raise StoreError(msg) from exc

  def object_exists(self, content_hash: str) -> bool:
    """Check whether a content-addressed blob exists in the object store.

    Args:
      content_hash: Full hex SHA-256 of the stored object.

    Returns:
      True when the blob file exists on disk.
    """
    prefix = content_hash[:2]
    rest = content_hash[2:]
    return (self._objects_dir / prefix / rest).exists()

  def read_object(self, content_hash: str) -> bytes:
    """Read a content-addressed blob from the object store.

    Returns:
      Raw bytes stored at ``content_hash``.

    Raises:
      StoreError: When the object path is missing or unreadable.
    """
    prefix = content_hash[:2]
    rest = content_hash[2:]
    obj_path = self._objects_dir / prefix / rest
    if not obj_path.exists():
      msg = f'object {content_hash} not found'
      raise StoreError(msg)
    return obj_path.read_bytes()

  def acquire_lock(self) -> None:
    """Acquire exclusive file lock for atomic snapshot writes.

    Backed by AutopilotFileLock (filelock library). Fail-fast: raises
    immediately on contention (no retry, no backoff). Crash-safe: OS
    releases the advisory lock on process death.

    Raises:
      ConcurrentMutationError: On lock contention (propagated directly).
      StoreError: On other lock acquisition failures.
    """
    try:
      self._lock.acquire()
    except ConcurrentMutationError:
      raise
    except TrackingError as exc:
      msg = f'store is locked by another operation: {exc}'
      raise StoreError(msg) from exc

  def release_lock(self) -> None:
    """Release the exclusive file lock. Idempotent."""
    self._lock.release()

  def __repr__(self) -> str:
    """Return a concise representation including the objects directory."""
    return f'StorageBackend(objects={self._objects_dir})'
