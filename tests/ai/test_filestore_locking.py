"""Tests for FileStore backend locking on ref-mutating operations (BUG-036, BUG-037)."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store_lock import StorageBackend
from autopilot.core.errors import StoreError
from autopilot.core.store.types import MergeStrategy
from autopilot.tracking.file_lock import ConcurrentMutationError
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
from unittest.mock import patch
import json
import threading


def _make_store(
  tmp_path: Path,
  files: dict[str, str] | None = None,
) -> tuple[FileStore, Path, PathParameter]:
  if files is None:
    files = {'main.py': 'print("hello")\n'}
  src = make_source_dir(tmp_path, files=files)
  config = make_store_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('root', 0)
  return store, src, param


def _assert_lock_used(store: FileStore, operation: str, **kwargs: object) -> None:
  """Run a store operation and assert that backend lock was acquired and released."""
  call_log: list[str] = []
  orig_acquire = StorageBackend.acquire_lock
  orig_release = StorageBackend.release_lock

  def patched_acquire(self: StorageBackend) -> None:
    call_log.append('acquire')
    orig_acquire(self)

  def patched_release(self: StorageBackend) -> None:
    call_log.append('release')
    orig_release(self)

  with (
    patch.object(StorageBackend, 'acquire_lock', patched_acquire),
    patch.object(StorageBackend, 'release_lock', patched_release),
  ):
    getattr(store, operation)(**kwargs)

  assert 'acquire' in call_log, f'{operation} did not acquire lock'
  assert 'release' in call_log, f'{operation} did not release lock'


def test_branch_acquires_backend_lock(tmp_path: Path) -> None:
  store, _src, _param = _make_store(tmp_path)
  _assert_lock_used(store, 'branch', experiment_id='child')


def test_checkout_acquires_backend_lock(tmp_path: Path) -> None:
  store, _src, _param = _make_store(tmp_path)
  store.branch('other')
  _assert_lock_used(store, 'checkout', experiment_id='other', epoch=0)


def test_materialize_acquires_backend_lock(tmp_path: Path) -> None:
  store, src, _param = _make_store(tmp_path)
  (src / 'main.py').write_text('epoch 1\n')
  store.snapshot('root', 1)
  _assert_lock_used(store, 'materialize', experiment_id='root', epoch=0)


def test_merge_apply_acquires_backend_lock(tmp_path: Path) -> None:
  store, src, _param = _make_store(tmp_path)
  store.branch('exp-a')
  store.branch('exp-b')

  store.checkout('exp-a', 0)
  (src / 'main.py').write_text('ours change\n')
  store.snapshot('exp-a', 1)

  store.checkout('exp-b', 0)
  (src / 'main.py').write_text('theirs change\n')
  store.snapshot('exp-b', 1)

  mi = store.merge_preview('exp-a', 'exp-b', strategy=MergeStrategy.ours)
  _assert_lock_used(store, 'merge_apply', merge_index=mi)


def test_concurrent_snapshot_and_branch_serializes_refs(tmp_path: Path) -> None:
  store, src, _param = _make_store(tmp_path)
  errs: list[Exception] = []
  barrier = threading.Barrier(2)

  def run_snapshot() -> None:
    barrier.wait()
    try:
      (src / 'main.py').write_text('concurrent edit\n')
      store.snapshot('root', 1)
    except (StoreError, ConcurrentMutationError) as exc:
      errs.append(exc)

  def run_branch() -> None:
    barrier.wait()
    try:
      store.branch('concurrent-branch')
    except (StoreError, ConcurrentMutationError) as exc:
      errs.append(exc)

  t1 = threading.Thread(target=run_snapshot)
  t2 = threading.Thread(target=run_branch)
  t1.start()
  t2.start()
  t1.join(timeout=10.0)
  t2.join(timeout=10.0)

  refs_path = store.config.refs_file
  payload = json.loads(refs_path.read_text(encoding='utf-8'))
  assert isinstance(payload, dict)
  assert 'branches' in payload
