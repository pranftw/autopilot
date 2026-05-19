"""Tests for StoreTransaction: atomic multi-resource context manager for FileStore."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.transaction import StoreTransaction
from autopilot.core.errors import StoreError
from autopilot.core.snapshot import SnapshotManifest
from autopilot.tracking.io import read_jsonl, utc_now_iso
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
from unittest.mock import patch
import pytest


def _make_store_with_snapshot(tmp_path: Path, slug: str = 'exp-001') -> tuple[FileStore, Path]:
  """Create a store with one snapshot at epoch 0."""
  src = make_source_dir(tmp_path)
  config = make_store_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot(slug, 0)
  return store, src


def _empty_manifest(epoch: int = 1) -> SnapshotManifest:
  return SnapshotManifest(epoch=epoch, timestamp=utc_now_iso(), entries={})


def _run_txn_that_raises(store: FileStore, exc: BaseException) -> None:
  """Run a transaction that buffers writes then raises ``exc``."""
  with store.transaction(context='will fail') as txn:
    txn.write_manifest('exp-001', 1, _empty_manifest())
    txn.update_refs('exp-001', 1)
    raise exc


class TestTransactionSuccessCommits:
  """test_transaction_success_commits: state visible after with block."""

  def test_manifest_persisted_after_commit(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)

    with store.transaction(context='test commit') as txn:
      txn.write_manifest('exp-001', 1, _empty_manifest())
      txn.update_refs('exp-001', 1)
      txn.append_reflog(
        {
          'operation': 'test',
          'experiment_id': 'exp-001',
          'old_epoch': 0,
          'new_epoch': 1,
        }
      )

    loaded = store.load_snapshot('exp-001', 1)
    assert loaded.epoch == 1

    refs = store.load_refs()
    assert refs['branches']['exp-001']['latest_epoch'] == 1
    assert refs['HEAD'] == 'exp-001'

    reflog = read_jsonl(store.reflog_path)
    ops = [r['operation'] for r in reflog]
    assert 'test' in ops

  def test_context_threaded(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    txn = store.transaction(context='my reason')
    assert txn.context == 'my reason'

  def test_context_none_allowed(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    txn = store.transaction()
    assert txn.context is None


class TestTransactionFailureRollsBack:
  """test_transaction_failure_rolls_back: exception leaves store as before enter."""

  def test_exception_during_block_preserves_prior_state(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    refs_before = store.load_refs()
    original_epoch = refs_before['branches']['exp-001']['latest_epoch']

    msg = 'simulated failure'
    with pytest.raises(ValueError, match='simulated'):
      _run_txn_that_raises(store, ValueError(msg))

    refs_after = store.load_refs()
    assert refs_after['branches']['exp-001']['latest_epoch'] == original_epoch


class TestTransactionAcquiresLock:
  """test_transaction_acquires_lock: lock held during block."""

  def test_lock_held_inside_block(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    lock_was_held = False

    with store.transaction(context='lock test'):
      lock_was_held = store._backend._lock.is_locked

    assert lock_was_held

  def test_lock_released_after_block(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)

    with store.transaction(context='lock test'):
      pass

    assert not store._backend._lock.is_locked


class TestTransactionReleasesLockOnFailure:
  """test_transaction_releases_lock_on_failure: lock not leaked after exception."""

  def test_lock_released_on_exception(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)

    msg = 'boom'
    with pytest.raises(RuntimeError, match='boom'), store.transaction(context='fail test'):
      raise RuntimeError(msg)

    assert not store._backend._lock.is_locked


class TestTransactionNestedRaises:
  """test_transaction_nested_raises: nested with store.transaction() -> RuntimeError."""

  def test_nested_transaction_raises(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)

    with (
      store.transaction(
        context='outer',
      ),
      pytest.raises(
        RuntimeError,
        match='nested transactions are not supported',
      ),
      store.transaction(context='inner'),
    ):
      pass

  def test_active_transaction_cleared_after_exit(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)

    with store.transaction(context='first'):
      pass

    with store.transaction(context='second'):
      pass


class TestTransactionConstructorInvalidStore:
  """test_transaction_constructor_invalid_store: wrong type raises."""

  def test_non_filestore_raises_type_error(self) -> None:
    with pytest.raises(TypeError, match='requires a Store instance'):
      StoreTransaction('not a store')  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

  def test_none_raises_type_error(self) -> None:
    with pytest.raises(TypeError, match='requires a Store instance'):
      StoreTransaction(None)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

  def test_dict_raises_type_error(self) -> None:
    with pytest.raises(TypeError, match='requires a Store instance'):
      StoreTransaction({})  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]


class TestTransactionManifestAndRefsAtomic:
  """test_transaction_manifest_and_refs_atomic: simulated crash does not leave half refs."""

  def test_crash_mid_commit_no_partial_refs(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    refs_before = store.load_refs()
    original_epoch = refs_before['branches']['exp-001']['latest_epoch']

    msg = 'disk full'
    with pytest.raises(OSError, match='disk full'):
      _run_txn_that_raises(store, OSError(msg))

    refs_after = store.load_refs()
    assert refs_after['branches']['exp-001']['latest_epoch'] == original_epoch


class TestTransactionReflogAppended:
  """test_transaction_reflog_appended: new reflog line appears only after successful commit."""

  def test_reflog_present_after_success(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    reflog_before = read_jsonl(store.reflog_path)

    with store.transaction(context='reflog test') as txn:
      txn.append_reflog(
        {
          'timestamp': utc_now_iso(),
          'operation': 'txn_test',
          'experiment_id': 'exp-001',
          'old_epoch': 0,
          'new_epoch': 1,
          'context': 'reflog test',
        }
      )

    reflog_after = read_jsonl(store.reflog_path)
    new_entries = reflog_after[len(reflog_before) :]
    assert len(new_entries) == 1
    assert new_entries[0]['operation'] == 'txn_test'

  def test_reflog_not_appended_on_failure(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    reflog_before = read_jsonl(store.reflog_path)

    msg = 'abort'
    with pytest.raises(ValueError, match='abort'):
      _run_txn_reflog_then_raise(store, ValueError(msg))

    reflog_after = read_jsonl(store.reflog_path)
    assert len(reflog_after) == len(reflog_before)


def _run_txn_reflog_then_raise(store: FileStore, exc: BaseException) -> None:
  """Buffer a reflog entry inside a transaction then raise."""
  with store.transaction(context='will fail') as txn:
    txn.append_reflog(
      {
        'timestamp': utc_now_iso(),
        'operation': 'should_not_persist',
        'experiment_id': 'exp-001',
      }
    )
    raise exc


class TestMergeApplyUsesTransaction:
  """test_merge_apply_uses_transaction: spy ensures transaction context entered."""

  def test_merge_apply_enters_transaction(self, tmp_path: Path) -> None:
    src = make_source_dir(tmp_path, files={'main.py': 'v1'})
    config = make_store_config(tmp_path)
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})

    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    (src / 'main.py').write_text('v2-a')
    store.snapshot('exp-a', 1)

    (src / 'main.py').write_text('v2-b')
    store.snapshot('exp-b', 1)

    (src / 'main.py').write_text('v3-a')
    store.snapshot('exp-a', 2)

    merge_index = store.merge_preview('exp-a', 'exp-b')
    for key in list(merge_index.conflicts):
      merge_index.resolve_ours(key)

    transaction_entered = []
    original_init = StoreTransaction.__init__

    def tracking_init(self_txn, store_arg, context=None):
      transaction_entered.append(True)
      original_init(self_txn, store_arg, context=context)

    with patch.object(StoreTransaction, '__init__', tracking_init):
      store.merge_apply(merge_index)

    assert len(transaction_entered) == 1


class TestTransactionCommitWithoutContextManager:
  """test_transaction_commit_without_context_manager: using without with raises RuntimeError."""

  def test_commit_before_enter_raises(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    txn = store.transaction(context='test')
    with pytest.raises(RuntimeError, match='must be used as a context manager'):
      txn.commit()

  def test_rollback_before_enter_raises(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    txn = store.transaction(context='test')
    with pytest.raises(RuntimeError, match='must be used as a context manager'):
      txn.rollback()

  def test_write_manifest_before_enter_raises(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    txn = store.transaction(context='test')
    with pytest.raises(RuntimeError, match='must be used as a context manager'):
      txn.write_manifest('exp-001', 1, _empty_manifest())

  def test_update_refs_before_enter_raises(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    txn = store.transaction(context='test')
    with pytest.raises(RuntimeError, match='must be used as a context manager'):
      txn.update_refs('exp-001', 1)

  def test_append_reflog_before_enter_raises(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    txn = store.transaction(context='test')
    with pytest.raises(RuntimeError, match='must be used as a context manager'):
      txn.append_reflog({'operation': 'test'})


class TestTransactionRollbackOnException:
  """test_transaction_rollback_on_exception: exception -> no durable partial artifacts."""

  def test_exception_during_write_rolls_back(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)

    msg = 'mid-write failure'
    with pytest.raises(RuntimeError, match='mid-write failure'):
      _run_txn_manifest_then_raise(store, RuntimeError(msg))

    refs = store.load_refs()
    assert refs['branches']['exp-001']['latest_epoch'] == 0

  def test_no_reflog_on_exception(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    reflog_before_count = len(read_jsonl(store.reflog_path))

    msg = 'abort'
    with pytest.raises(RuntimeError, match='abort'):
      _run_txn_reflog_then_raise(store, RuntimeError(msg))

    reflog_after_count = len(read_jsonl(store.reflog_path))
    assert reflog_after_count == reflog_before_count


def _run_txn_manifest_then_raise(store: FileStore, exc: BaseException) -> None:
  """Buffer a manifest write inside a transaction then raise."""
  with store.transaction(context='rollback test') as txn:
    txn.write_manifest('exp-001', 1, _empty_manifest())
    raise exc


class TestTransactionDiskFullSimulation:
  """test_transaction_disk_full_simulation: OSError on commit -> rollback path."""

  def test_oserror_on_save_refs_triggers_rollback(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)
    refs_before = store.load_refs()
    expected = refs_before['branches']['exp-001']['latest_epoch']

    with (
      patch.object(
        store,
        'save_refs',
        side_effect=OSError('No space left on device'),
      ),
      pytest.raises(
        OSError,
        match='No space left on device',
      ),
      store.transaction(context='disk full test') as txn,
    ):
      txn.update_refs('exp-001', 1)

    refs_after = store.load_refs()
    assert refs_after['branches']['exp-001']['latest_epoch'] == expected
    assert not store._backend._lock.is_locked

  def test_oserror_on_rename_during_manifest_write(self, tmp_path: Path) -> None:
    store, _ = _make_store_with_snapshot(tmp_path)

    with (
      patch.object(
        store,
        'persist_manifest',
        side_effect=StoreError('disk full'),
      ),
      pytest.raises(
        StoreError,
        match='disk full',
      ),
      store.transaction(context='disk full manifest') as txn,
    ):
      txn.write_manifest('exp-001', 1, _empty_manifest())

    assert not store._backend._lock.is_locked
    refs = store.load_refs()
    assert refs['branches']['exp-001']['latest_epoch'] == 0


def _run_txn_with_manifest_and_refs_fail(
  store: FileStore,
  branch: str,
  epoch: int,
  manifest: SnapshotManifest,
) -> None:
  """Buffer manifest + refs inside a transaction; save_refs patched to fail."""
  with (
    patch.object(store, 'save_refs', side_effect=OSError('disk full')),
    store.transaction(context='rollback test') as txn,
  ):
    txn.write_manifest(branch, epoch, manifest)
    txn.update_refs(branch, epoch)


def _run_txn_multi_manifest_refs_fail(
  store: FileStore,
  branches: list[tuple[str, int, SnapshotManifest]],
  refs_branch: str,
  refs_epoch: int,
) -> None:
  """Buffer multiple manifests + refs inside a transaction; save_refs patched to fail."""
  with (
    patch.object(store, 'save_refs', side_effect=OSError('disk full')),
    store.transaction(context='multi-branch rollback') as txn,
  ):
    for branch, epoch, manifest in branches:
      txn.write_manifest(branch, epoch, manifest)
    txn.update_refs(refs_branch, refs_epoch)


class TestRollbackCleansWrittenManifests:
  """Transaction rollback deletes manifest files already written to disk."""

  def test_rollback_cleans_written_manifest(self, tmp_path: Path) -> None:
    """Patch save_refs to raise; verify manifest file cleaned up."""
    store, _ = _make_store_with_snapshot(tmp_path)
    snap_path = store.snapshot_manifest_path('exp-001', 1)

    with pytest.raises(OSError, match='disk full'):
      _run_txn_with_manifest_and_refs_fail(store, 'exp-001', 1, _empty_manifest())

    assert not snap_path.exists()

  def test_rollback_cleans_multiple_manifests(self, tmp_path: Path) -> None:
    """Two manifests buffered, refs fails, both cleaned up."""
    src = make_source_dir(tmp_path)
    config = make_store_config(tmp_path)
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    path_a = store.snapshot_manifest_path('exp-a', 1)
    path_b = store.snapshot_manifest_path('exp-b', 1)

    with pytest.raises(OSError, match='disk full'):
      _run_txn_multi_manifest_refs_fail(
        store,
        [('exp-a', 1, _empty_manifest()), ('exp-b', 1, _empty_manifest())],
        'exp-a',
        1,
      )

    assert not path_a.exists()
    assert not path_b.exists()

  def test_successful_commit_keeps_manifest(self, tmp_path: Path) -> None:
    """Normal commit leaves manifest on disk."""
    store, _ = _make_store_with_snapshot(tmp_path)
    snap_path = store.snapshot_manifest_path('exp-001', 1)

    with store.transaction(context='success test') as txn:
      txn.write_manifest('exp-001', 1, _empty_manifest())
      txn.update_refs('exp-001', 1)

    assert snap_path.exists()
    refs = store.load_refs()
    assert refs['branches']['exp-001']['latest_epoch'] == 1

  def test_rollback_does_not_delete_preexisting_manifest(self, tmp_path: Path) -> None:
    """Pre-existing epoch_0.json not deleted by rollback of a failed epoch_1 write."""
    store, _ = _make_store_with_snapshot(tmp_path)
    epoch_0_path = store.snapshot_manifest_path('exp-001', 0)
    assert epoch_0_path.exists()

    with pytest.raises(OSError, match='disk full'):
      _run_txn_with_manifest_and_refs_fail(store, 'exp-001', 1, _empty_manifest())

    assert epoch_0_path.exists()
