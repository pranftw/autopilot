"""Tests for autopilot.ai.store_lock.StorageBackend locking."""

from autopilot.ai.store_lock import StorageBackend
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.file_lock import ConcurrentMutationError
from pathlib import Path
import threading
import time


def _backend(tmp_path: Path) -> StorageBackend:
  cfg = AutoPilotConfig(workspace=tmp_path)
  cfg.store_path = tmp_path / '.store'
  return StorageBackend(cfg)


def test_second_thread_sees_concurrent_mutation_error_when_lock_held(tmp_path: Path) -> None:
  backend = _backend(tmp_path)
  barrier = threading.Barrier(2)
  err: list[BaseException | None] = [None]

  def holder() -> None:
    backend.acquire_lock()
    barrier.wait()
    time.sleep(0.15)
    backend.release_lock()

  def contender() -> None:
    barrier.wait()
    try:
      backend.acquire_lock()
    except ConcurrentMutationError as exc:
      err[0] = exc
    else:
      err[0] = None

  t1 = threading.Thread(target=holder)
  t2 = threading.Thread(target=contender)
  t1.start()
  t2.start()
  t1.join(timeout=5.0)
  t2.join(timeout=5.0)
  assert isinstance(err[0], ConcurrentMutationError)
  assert 'concurrent mutation' in str(err[0])


def test_stale_lock_file_does_not_block(tmp_path: Path) -> None:
  """With filelock/flock, a leftover lock file does not prevent acquisition."""
  backend = _backend(tmp_path)
  lock_path = backend._lock_file
  lock_path.parent.mkdir(parents=True, exist_ok=True)
  lock_path.write_text('stale', encoding='utf-8')
  backend.acquire_lock()
  backend.release_lock()


def test_release_allows_reacquire_after_successful_hold(tmp_path: Path) -> None:
  backend = _backend(tmp_path)
  backend.acquire_lock()
  backend.release_lock()
  backend.acquire_lock()
  backend.release_lock()
