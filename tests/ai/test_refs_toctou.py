"""Tests for store refs TOCTOU fixes, StorageBackend migration, and blob safety.

Covers plan 04 subplans:
  2.1 -- StorageBackend migrated to AutopilotFileLock
  2.2 -- snapshot/branch/merge_apply/materialize reload refs inside lock
  2.3 -- create_worktree/remove_worktree/prune_orphans hold store lock
  2.4 -- blob temp file uses PID + thread ident suffix
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store import doctor as doctor_mod
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store_lock import StorageBackend, hash_content
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.store.types import MergeStrategy
from autopilot.tracking.file_lock import ConcurrentMutationError
from pathlib import Path
from typing import Any
from unittest.mock import patch
import os
import threading


def _make_config(tmp_path: Path) -> AutoPilotConfig:
  ws = tmp_path / 'project'
  ws.mkdir()
  return AutoPilotConfig(workspace=ws)


def _make_store(config: AutoPilotConfig, content: str = 'hello') -> FileStore:
  """Build a FileStore with a single PathParameter containing one file."""
  ws = config.workspace
  src = ws / 'data'
  src.mkdir(exist_ok=True)
  (src / 'a.txt').write_text(content, encoding='utf-8')
  store = FileStore(config)
  param = PathParameter(source=str(src), pattern='*')
  store.register_parameters({'data': param})
  return store


def _make_backend(tmp_path: Path) -> StorageBackend:
  cfg = AutoPilotConfig(workspace=tmp_path)
  return StorageBackend(cfg)


class _LockTracker:
  """Tracks whether lock is held during calls to load_refs / save_refs / etc."""

  def __init__(self, store: FileStore) -> None:
    self._store = store
    self._original_acquire = store._backend._lock.acquire
    self._original_release = store._backend._lock.release
    self.lock_held = False
    self.load_refs_calls: list[bool] = []
    self.save_refs_calls: list[bool] = []
    self.collect_calls: list[bool] = []

  def wrap_acquire(self) -> None:
    self._original_acquire()
    self.lock_held = True

  def wrap_release(self) -> None:
    self.lock_held = False
    self._original_release()

  def wrap_load_refs(self) -> dict[str, Any]:
    self.load_refs_calls.append(self.lock_held)
    return FileStore.load_refs(self._store)

  def wrap_save_refs(self, refs: dict[str, Any]) -> None:
    self.save_refs_calls.append(self.lock_held)
    return FileStore.save_refs(self._store, refs)

  def wrap_collect(self) -> set[str]:
    self.collect_calls.append(self.lock_held)
    return FileStore._collect_reachable_digests(self._store)


# -- 2.1: StorageBackend migration to AutopilotFileLock --


class TestBackendLockMigration:
  """StorageBackend acquire/release uses AutopilotFileLock."""

  def test_backend_acquire_holds_lock(self, tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    backend.acquire_lock()
    assert backend._lock.is_locked
    backend.release_lock()

  def test_backend_release_frees_lock(self, tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    backend.acquire_lock()
    backend.release_lock()
    assert not backend._lock.is_locked
    backend.acquire_lock()
    backend.release_lock()

  def test_backend_contention_raises_store_error(self, tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    barrier = threading.Barrier(2)
    errors: list[BaseException | None] = [None]

    def holder() -> None:
      backend.acquire_lock()
      barrier.wait()
      import time

      time.sleep(0.15)
      backend.release_lock()

    def contender() -> None:
      barrier.wait()
      try:
        backend.acquire_lock()
      except ConcurrentMutationError as exc:
        errors[0] = exc
      else:
        backend.release_lock()

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=contender)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert isinstance(errors[0], ConcurrentMutationError)
    assert 'concurrent mutation' in str(errors[0])


# -- 2.2: TOCTOU fixes (refs reloaded inside lock) --


class TestSnapshotFreshRefs:
  """snapshot uses refs loaded inside the lock."""

  def test_snapshot_reloads_refs_inside_lock(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = _make_store(config)
    tracker = _LockTracker(store)

    with (
      patch.object(store._backend._lock, 'acquire', tracker.wrap_acquire),
      patch.object(store._backend._lock, 'release', tracker.wrap_release),
      patch.object(store, 'load_refs', tracker.wrap_load_refs),
    ):
      store.snapshot('exp-1', 0)

    assert any(tracker.load_refs_calls), 'load_refs was never called'
    assert tracker.load_refs_calls[-1] is True, 'load_refs called outside lock'


class TestBranchFreshRefs:
  """branch uses refs loaded inside the lock."""

  def test_branch_reloads_refs_inside_lock(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = _make_store(config)
    store.snapshot('exp-1', 0)

    tracker = _LockTracker(store)
    with (
      patch.object(store._backend._lock, 'acquire', tracker.wrap_acquire),
      patch.object(store._backend._lock, 'release', tracker.wrap_release),
      patch.object(store, 'load_refs', tracker.wrap_load_refs),
    ):
      store.branch('exp-2')

    locked_calls = [c for c in tracker.load_refs_calls if c]
    assert len(locked_calls) >= 1, 'branch must reload refs inside lock'


class TestMergeApplyFreshRefs:
  """merge_apply uses refs loaded inside the lock."""

  def test_merge_apply_reloads_refs_inside_lock(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = _make_store(config)

    store.snapshot('exp-1', 0)
    store.branch('exp-2')

    ws = config.workspace
    (ws / 'data' / 'a.txt').write_text('modified', encoding='utf-8')
    store.snapshot('exp-2', 1)

    merge_index = store.merge_preview('exp-1', 'exp-2', strategy=MergeStrategy.theirs)

    tracker = _LockTracker(store)
    with (
      patch.object(store._backend._lock, 'acquire', tracker.wrap_acquire),
      patch.object(store._backend._lock, 'release', tracker.wrap_release),
      patch.object(store, 'load_refs', tracker.wrap_load_refs),
    ):
      store.merge_apply(merge_index)

    locked_calls = [c for c in tracker.load_refs_calls if c]
    assert len(locked_calls) >= 1, 'merge_apply must reload refs inside lock'


class TestMaterializeFreshRefs:
  """materialize uses refs loaded inside the lock."""

  def test_materialize_reloads_refs_inside_lock(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = _make_store(config, content='v0')

    store.snapshot('exp-1', 0)
    ws = config.workspace
    (ws / 'data' / 'a.txt').write_text('v1', encoding='utf-8')
    store.snapshot('exp-1', 1)

    tracker = _LockTracker(store)
    with (
      patch.object(store._backend._lock, 'acquire', tracker.wrap_acquire),
      patch.object(store._backend._lock, 'release', tracker.wrap_release),
      patch.object(store, 'load_refs', tracker.wrap_load_refs),
    ):
      store.materialize('exp-1', 0)

    locked_calls = [c for c in tracker.load_refs_calls if c]
    assert len(locked_calls) >= 1, 'materialize must reload refs inside lock'


# -- 2.3: Protected operations hold store lock --


class TestCreateWorktreeStoreLocked:
  """create_worktree holds the store lock for refs update."""

  def test_create_worktree_holds_store_lock(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = _make_store(config)
    store.snapshot('exp-1', 0)

    tracker = _LockTracker(store)
    with (
      patch.object(store._backend._lock, 'acquire', tracker.wrap_acquire),
      patch.object(store._backend._lock, 'release', tracker.wrap_release),
      patch.object(store, 'save_refs', tracker.wrap_save_refs),
    ):
      store.create_worktree('exp-1')

    assert any(tracker.save_refs_calls), 'save_refs was never called'
    assert tracker.save_refs_calls[-1] is True, 'save_refs called outside store lock'


class TestRemoveWorktreeStoreLocked:
  """remove_worktree holds the store lock for refs update."""

  def test_remove_worktree_holds_store_lock(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = _make_store(config)
    store.snapshot('exp-1', 0)
    store.create_worktree('exp-1')

    tracker = _LockTracker(store)
    with (
      patch.object(store._backend._lock, 'acquire', tracker.wrap_acquire),
      patch.object(store._backend._lock, 'release', tracker.wrap_release),
      patch.object(store, 'save_refs', tracker.wrap_save_refs),
    ):
      store.remove_worktree('exp-1')

    assert any(tracker.save_refs_calls), 'save_refs was never called'
    assert tracker.save_refs_calls[-1] is True, 'save_refs called outside store lock'


class TestPruneOrphansStoreLocked:
  """prune_orphans holds the store lock."""

  def test_prune_orphans_holds_store_lock(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = _make_store(config)
    store.snapshot('exp-1', 0)

    tracker = _LockTracker(store)
    original_collect = doctor_mod.collect_reachable_digests

    def patched_collect(s):
      tracker.collect_calls.append(tracker.lock_held)
      return original_collect(s)

    with (
      patch.object(store._backend._lock, 'acquire', tracker.wrap_acquire),
      patch.object(store._backend._lock, 'release', tracker.wrap_release),
      patch(
        'autopilot.ai.store.doctor.collect_reachable_digests',
        side_effect=patched_collect,
      ),
    ):
      store.prune_orphans()

    assert any(tracker.collect_calls), '_collect_reachable_digests was never called'
    assert tracker.collect_calls[0] is True, 'prune_orphans ran outside store lock'


# -- 2.4: Blob safety --


class TestBlobTempUniqueSuffix:
  """Blob temp file includes PID and thread ident for uniqueness."""

  def test_blob_temp_unique_per_process(self, tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    content = b'test blob data'
    content_hash = hash_content(content.decode('utf-8'))

    write_calls: list[Path] = []
    real_write_bytes = Path.write_bytes

    def tracking_write(self_path: Path, data: bytes) -> int:
      if '.tmp' in self_path.name:
        write_calls.append(self_path)
      return real_write_bytes(self_path, data)

    with patch.object(Path, 'write_bytes', tracking_write):
      backend.store_object(content_hash, content)

    assert len(write_calls) == 1
    tmp_name = write_calls[0].name
    assert str(os.getpid()) in tmp_name
    assert str(threading.get_ident()) in tmp_name


class TestBlobConcurrentWriteSafe:
  """Two threads writing the same blob do not corrupt data."""

  def test_blob_concurrent_write_safe(self, tmp_path: Path) -> None:
    backend = _make_backend(tmp_path)
    content = b'shared blob content'
    content_hash = hash_content(content.decode('utf-8'))

    errors: list[StoreError | OSError] = []
    barrier = threading.Barrier(2)

    def writer() -> None:
      barrier.wait()
      try:
        backend.store_object(content_hash, content)
      except (StoreError, OSError) as exc:
        errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
      t.start()
    for t in threads:
      t.join(timeout=5.0)

    assert not errors, f'concurrent blob write raised: {errors}'
    stored = backend.read_object(content_hash)
    assert stored == content
