"""Tests for AutopilotFileLock wrapper and ConcurrentMutationError."""

from autopilot.core.errors import TrackingError
from autopilot.tracking.file_lock import (
  LOCK_RETRY_AFTER_MS,
  AutopilotFileLock,
  ConcurrentMutationError,
)
from filelock import Timeout as _Timeout
import pytest
import threading
import time


class TestAutopilotFileLock:
  def test_file_lock_acquire_release(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    lock = AutopilotFileLock(lock_path)
    assert not lock.is_locked
    lock.acquire()
    assert lock.is_locked
    lock.release()
    assert not lock.is_locked

  def test_file_lock_context_manager(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    lock = AutopilotFileLock(lock_path)
    assert not lock.is_locked
    with lock:
      assert lock.is_locked
    assert not lock.is_locked

  def test_file_lock_contention_fail_fast(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    lock1 = AutopilotFileLock(lock_path)
    lock2 = AutopilotFileLock(lock_path)
    lock1.acquire()
    try:
      with pytest.raises(ConcurrentMutationError, match='concurrent mutation'):
        lock2.acquire()
    finally:
      lock1.release()

  def test_file_lock_contention_chains_timeout(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    lock1 = AutopilotFileLock(lock_path)
    lock2 = AutopilotFileLock(lock_path)
    lock1.acquire()
    try:
      with pytest.raises(ConcurrentMutationError) as exc_info:
        lock2.acquire()
      assert isinstance(exc_info.value.__cause__, _Timeout)
    finally:
      lock1.release()

  def test_file_lock_timeout_succeeds(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    acquired = threading.Event()

    def hold_briefly():
      lock = AutopilotFileLock(lock_path)
      lock.acquire()
      acquired.set()
      time.sleep(0.3)
      lock.release()

    thread = threading.Thread(target=hold_briefly)
    thread.start()
    acquired.wait(timeout=2.0)

    lock2 = AutopilotFileLock(lock_path, timeout_s=5.0)
    lock2.acquire()
    assert lock2.is_locked
    lock2.release()
    thread.join()

  def test_file_lock_idempotent_release(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    lock = AutopilotFileLock(lock_path)
    lock.acquire()
    lock.release()
    lock.release()

  def test_concurrent_mutation_error_is_tracking_error(self):
    exc = ConcurrentMutationError('forest_save')
    assert isinstance(exc, TrackingError)

  def test_concurrent_mutation_error_attributes(self):
    exc = ConcurrentMutationError('forest_save')
    assert exc.retry_after_ms == LOCK_RETRY_AFTER_MS
    assert exc.operation == 'forest_save'
    assert 'forest_save' in str(exc)
    assert '--wait' in str(exc)

  def test_concurrent_mutation_error_custom_retry(self):
    exc = ConcurrentMutationError('store', retry_after_ms=500)
    assert exc.retry_after_ms == 500
    assert exc.operation == 'store'

  def test_operation_label_propagated(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    lock1 = AutopilotFileLock(lock_path, operation='snapshot')
    lock2 = AutopilotFileLock(lock_path, operation='snapshot')
    lock1.acquire()
    try:
      with pytest.raises(ConcurrentMutationError) as exc_info:
        lock2.acquire()
      assert exc_info.value.operation == 'snapshot'
    finally:
      lock1.release()

  def test_infinite_wait_sentinel(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    acquired = threading.Event()

    def hold_briefly():
      lock = AutopilotFileLock(lock_path)
      lock.acquire()
      acquired.set()
      time.sleep(0.2)
      lock.release()

    thread = threading.Thread(target=hold_briefly)
    thread.start()
    acquired.wait(timeout=2.0)

    lock2 = AutopilotFileLock(lock_path, timeout_s=-1.0)
    lock2.acquire()
    assert lock2.is_locked
    lock2.release()
    thread.join()
