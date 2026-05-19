"""Tests for concurrent safety: ConcurrentMutationError, --wait flag, and JSON envelope."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import CLI
from autopilot.cli.context import CLIContext, build_context
from autopilot.cli.helpers import _wait_ms_to_timeout_s, load_forest
from autopilot.cli.main import build_parser
from autopilot.cli.output import Output
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.file_lock import (
  LOCK_RETRY_AFTER_MS,
  AutopilotFileLock,
  ConcurrentMutationError,
)
from pathlib import Path
import argparse
import contextlib
import io
import json
import pytest
import threading
import time


class TestConcurrentMutationErrorAttributes:
  """Test ConcurrentMutationError carries correct attributes."""

  def test_default_retry_after_ms(self):
    exc = ConcurrentMutationError('forest_save')
    assert exc.retry_after_ms == LOCK_RETRY_AFTER_MS
    assert exc.retry_after_ms == 100

  def test_custom_retry_after_ms(self):
    exc = ConcurrentMutationError('store', retry_after_ms=250)
    assert exc.retry_after_ms == 250

  def test_operation_label(self):
    exc = ConcurrentMutationError('forest_save')
    assert exc.operation == 'forest_save'

  def test_message_contains_operation(self):
    exc = ConcurrentMutationError('forest_save')
    assert 'forest_save' in str(exc)

  def test_message_contains_retry_hint(self):
    exc = ConcurrentMutationError('forest_save')
    assert '--wait' in str(exc)
    assert '100ms' in str(exc)


class TestConcurrentMutationJsonEnvelope:
  """Test JSON envelope output when ConcurrentMutationError is raised during dispatch."""

  def _build_ctx(self, workspace: Path, *, use_json: bool = False) -> CLIContext:
    config = AutoPilotConfig(workspace=workspace)
    return CLIContext(
      workspace=workspace,
      config=config,
      output=Output(use_json=use_json),
      context='test',
    )

  def test_json_envelope_keys(self, tmp_path):
    """JSON mode emits ok, error, error_code, retry_after_ms."""
    ctx = self._build_ctx(tmp_path, use_json=True)
    operation = 'forest_save'

    def handler(c, a):
      raise ConcurrentMutationError(operation)

    args = argparse.Namespace(handler=handler, command='tree create')

    cli = CLI()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
      cli.dispatch(ctx, args)

    output = buf.getvalue().strip()
    envelope = json.loads(output)
    assert envelope['ok'] is False
    assert envelope['error_code'] == 'concurrent_mutation'
    assert envelope['retry_after_ms'] == LOCK_RETRY_AFTER_MS
    assert 'forest_save' in envelope['error']

  def test_non_json_mode_stderr(self, tmp_path):
    """Non-JSON mode prints error to stderr and exits non-zero."""
    ctx = self._build_ctx(tmp_path, use_json=False)
    operation = 'forest_save'

    def handler(c, a):
      raise ConcurrentMutationError(operation)

    args = argparse.Namespace(handler=handler, command='tree create')

    cli = CLI()
    with pytest.raises(SystemExit) as exc_info:
      cli.dispatch(ctx, args)
    assert exc_info.value.code == 1


class TestWaitFlag:
  """Test --wait global flag wiring and behavior."""

  def test_wait_flag_parsed(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--wait', '5000', '--workspace', str(tmp_path), 'status'])
    assert args.wait == 5000

  def test_wait_flag_absent_is_none(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--workspace', str(tmp_path), 'status'])
    assert args.wait is None

  def test_wait_flag_zero(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--wait', '0', '--workspace', str(tmp_path), 'status'])
    assert args.wait == 0

  def test_wait_wired_to_context(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--wait', '3000', '--workspace', str(tmp_path), 'status'])
    ctx = build_context(args)
    assert ctx.wait_timeout_ms == 3000

  def test_wait_absent_context_is_none(self, tmp_path):
    parser = build_parser()
    args = parser.parse_args(['--workspace', str(tmp_path), 'status'])
    ctx = build_context(args)
    assert ctx.wait_timeout_ms is None


class TestWaitMsToTimeoutS:
  """Test the millisecond to seconds conversion helper."""

  def test_none_returns_none(self):
    assert _wait_ms_to_timeout_s(None) is None

  def test_zero_returns_negative_one(self):
    assert _wait_ms_to_timeout_s(0) == -1.0

  def test_positive_converts(self):
    assert _wait_ms_to_timeout_s(5000) == 5.0
    assert _wait_ms_to_timeout_s(100) == 0.1

  def test_small_value(self):
    assert _wait_ms_to_timeout_s(1) == 0.001

  def test_negative_raises(self):
    with pytest.raises(ValueError, match='non-negative'):
      _wait_ms_to_timeout_s(-1)


class TestWaitFlagEventuallyAcquires:
  """Test that --wait allows eventually acquiring a contended lock."""

  def test_wait_flag_eventually_acquires(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    barrier = threading.Barrier(2, timeout=5.0)
    acquired_by_waiter = threading.Event()

    def hold_briefly():
      lock = AutopilotFileLock(lock_path, operation='test')
      lock.acquire()
      barrier.wait()
      time.sleep(0.3)
      lock.release()

    thread = threading.Thread(target=hold_briefly)
    thread.start()
    barrier.wait()

    lock2 = AutopilotFileLock(lock_path, timeout_s=5.0, operation='test')
    lock2.acquire()
    acquired_by_waiter.set()
    assert lock2.is_locked
    lock2.release()
    thread.join()
    assert acquired_by_waiter.is_set()


class TestNoWaitFailsFast:
  """Test that absent --wait causes immediate failure on contention."""

  def test_no_wait_fails_fast(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    barrier = threading.Barrier(2, timeout=5.0)

    def hold_lock():
      lock = AutopilotFileLock(lock_path, operation='holder')
      lock.acquire()
      barrier.wait()
      time.sleep(1.0)
      lock.release()

    thread = threading.Thread(target=hold_lock)
    thread.start()
    barrier.wait()

    lock2 = AutopilotFileLock(lock_path, operation='contender')
    start = time.monotonic()
    with pytest.raises(ConcurrentMutationError):
      lock2.acquire()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5

    lock_holder = AutopilotFileLock(lock_path, timeout_s=5.0)
    lock_holder.acquire()
    lock_holder.release()
    thread.join()


class TestWaitTimeoutFailureMessage:
  """Test that a --wait with insufficient budget produces a failure."""

  def test_wait_timeout_failure(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    barrier = threading.Barrier(2, timeout=5.0)

    def hold_long():
      lock = AutopilotFileLock(lock_path, operation='holder')
      lock.acquire()
      barrier.wait()
      time.sleep(2.0)
      lock.release()

    thread = threading.Thread(target=hold_long)
    thread.start()
    barrier.wait()

    lock2 = AutopilotFileLock(lock_path, timeout_s=0.1, operation='store')
    with pytest.raises(ConcurrentMutationError) as exc_info:
      lock2.acquire()
    assert exc_info.value.operation == 'store'
    assert exc_info.value.retry_after_ms == LOCK_RETRY_AFTER_MS

    lock_cleanup = AutopilotFileLock(lock_path, timeout_s=5.0)
    lock_cleanup.acquire()
    lock_cleanup.release()
    thread.join()


class TestWaitFlagZeroBlocksUntilRelease:
  """Test that --wait 0 (infinite) blocks until the lock is released."""

  def test_wait_zero_blocks(self, tmp_path):
    lock_path = tmp_path / 'test.lock'
    barrier = threading.Barrier(2, timeout=5.0)

    def hold_briefly():
      lock = AutopilotFileLock(lock_path, operation='holder')
      lock.acquire()
      barrier.wait()
      time.sleep(0.3)
      lock.release()

    thread = threading.Thread(target=hold_briefly)
    thread.start()
    barrier.wait()

    lock2 = AutopilotFileLock(lock_path, timeout_s=-1.0, operation='waiter')
    lock2.acquire()
    assert lock2.is_locked
    lock2.release()
    thread.join()


class TestLoadForestThreadsTimeout:
  """Test that load_forest threads --wait timeout to forest and store."""

  def test_load_forest_sets_timeout(self, tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    ctx = CLIContext(
      workspace=ws,
      config=config,
      output=Output(),
      wait_timeout_ms=5000,
    )
    forest = load_forest(ctx)
    assert forest.lock_timeout_s == 5.0
    assert forest.store.lock_timeout_s == 5.0

  def test_load_forest_none_timeout(self, tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    ctx = CLIContext(
      workspace=ws,
      config=config,
      output=Output(),
      wait_timeout_ms=None,
    )
    forest = load_forest(ctx)
    assert forest.lock_timeout_s is None
    assert forest.store.lock_timeout_s is None

  def test_load_forest_zero_timeout(self, tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    ctx = CLIContext(
      workspace=ws,
      config=config,
      output=Output(),
      wait_timeout_ms=0,
    )
    forest = load_forest(ctx)
    assert forest.lock_timeout_s == -1.0
    assert forest.store.lock_timeout_s == -1.0


class TestForestSaveConcurrentMutation:
  """Test that FileForest.save raises ConcurrentMutationError on contention."""

  def test_forest_save_contention(self, tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    lock_path = config.store_path / 'forest.lock'
    blocker = AutopilotFileLock(lock_path, timeout_s=5.0, operation='blocker')
    blocker.acquire()
    try:
      with pytest.raises(ConcurrentMutationError) as exc_info:
        forest.save()
      assert exc_info.value.operation == 'forest_save'
    finally:
      blocker.release()


class TestStoreLockConcurrentMutation:
  """Test that FileStore operations raise ConcurrentMutationError on lock contention."""

  def test_store_snapshot_contention(self, tmp_path):
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    lock_path = config.store_path / '.lock'
    blocker = AutopilotFileLock(lock_path, timeout_s=5.0, operation='blocker')
    blocker.acquire()
    try:
      with pytest.raises(ConcurrentMutationError) as exc_info:
        store.snapshot('test-exp', 0)
    finally:
      blocker.release()
    assert exc_info.value.operation == 'store'
