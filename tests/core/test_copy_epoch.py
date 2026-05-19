"""Tests for FileStore copy_epoch (cherry-pick) (Plan 20).

Covers cross-branch copy, self-copy, uninitialized target (epoch -1),
missing source/target branches, missing source epoch, missing blob
integrity, blob sharing (no duplication), source unchanged after copy,
HEAD movement, concurrent copy lock contention, reflog entries, and
CLI smoke test.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.tracking.file_lock import ConcurrentMutationError
from autopilot.tracking.io import read_jsonl
from pathlib import Path
from tests.cli.conftest import make_cli_workspace, run_cli
import pytest
import threading


def _make_store(
  tmp_path: Path,
  files: dict[str, str] | None = None,
) -> tuple[FileStore, Path, PathParameter]:
  """Create a FileStore with a single PathParameter for testing."""
  if files is None:
    files = {'main.py': 'print("hello")\n'}
  src = tmp_path / 'src'
  src.mkdir(parents=True, exist_ok=True)
  for name, content in files.items():
    (src / name).write_text(content)
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  return store, src, param


def _read_reflog(store: FileStore) -> list[dict]:
  """Read all reflog entries from the store."""
  path = store.config.store_path / 'reflog.jsonl'
  return read_jsonl(path, strict=False)


def _count_object_files(store: FileStore) -> int:
  """Count blob files in the object store."""
  objects_dir = store.config.objects_path
  if not objects_dir.exists():
    return 0
  count = 0
  for shard in objects_dir.iterdir():
    if shard.is_dir():
      count += sum(1 for f in shard.iterdir() if f.is_file())
  return count


class TestCrossBranchCopy:
  """Copy epoch from one branch to another."""

  def test_happy_path_cross_branch_copy(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    (src / 'main.py').write_text('print("v2")\n')
    store.snapshot('exp-a', 1)

    store.branch('exp-b')
    (src / 'main.py').write_text('print("b1")\n')
    store.snapshot('exp-b', 1)

    result = store.copy_epoch('exp-a', 1, 'exp-b')

    assert result.epoch == 2
    source_snap = store.load_snapshot('exp-a', 1)
    assert result.entries == source_snap.entries

  def test_target_latest_epoch_increments(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    store.copy_epoch('exp-a', 0, 'exp-b')

    refs = store.load_refs()
    assert refs['branches']['exp-b']['latest_epoch'] == 1

  def test_digest_pointers_shared(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    source_snap = store.load_snapshot('exp-a', 0)
    result = store.copy_epoch('exp-a', 0, 'exp-b')

    for key in source_snap.entries:
      assert result.entries[key].digest == source_snap.entries[key].digest


class TestSelfCopy:
  """Copy epoch from a branch to itself."""

  def test_self_copy_advances_tip(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)

    result = store.copy_epoch('exp-a', 0, 'exp-a')

    assert result.epoch == 1
    refs = store.load_refs()
    assert refs['branches']['exp-a']['latest_epoch'] == 1

  def test_self_copy_identical_digests(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)

    source_snap = store.load_snapshot('exp-a', 0)
    result = store.copy_epoch('exp-a', 0, 'exp-a')

    for key in source_snap.entries:
      assert result.entries[key].digest == source_snap.entries[key].digest


class TestUninitializedTarget:
  """Copy to a branch with latest_epoch == -1 (after reset or fresh branch)."""

  def test_copy_to_reset_branch_writes_epoch_zero(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')
    store.reset_branch('exp-b')

    refs_before = store.load_refs()
    assert refs_before['branches']['exp-b']['latest_epoch'] == -1

    result = store.copy_epoch('exp-a', 0, 'exp-b')

    assert result.epoch == 0
    refs = store.load_refs()
    assert refs['branches']['exp-b']['latest_epoch'] == 0


class TestBlobSharing:
  """No new object files created when copying between branches."""

  def test_no_new_blobs_on_cross_branch_copy(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    count_before = _count_object_files(store)
    store.copy_epoch('exp-a', 0, 'exp-b')
    count_after = _count_object_files(store)

    assert count_after == count_before

  def test_no_new_blobs_on_self_copy(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)

    count_before = _count_object_files(store)
    store.copy_epoch('exp-a', 0, 'exp-a')
    count_after = _count_object_files(store)

    assert count_after == count_before


class TestSourceUnchanged:
  """Source branch is not modified by copy_epoch."""

  def test_source_tip_unchanged(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    refs_before = store.load_refs()
    source_tip_before = refs_before['branches']['exp-a']['latest_epoch']

    store.copy_epoch('exp-a', 0, 'exp-b')

    refs_after = store.load_refs()
    assert refs_after['branches']['exp-a']['latest_epoch'] == source_tip_before

  def test_source_snapshot_unchanged(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    source_snap_before = store.load_snapshot('exp-a', 0)

    store.branch('exp-b')
    store.copy_epoch('exp-a', 0, 'exp-b')

    source_snap_after = store.load_snapshot('exp-a', 0)
    assert source_snap_after.entries == source_snap_before.entries
    assert source_snap_after.timestamp == source_snap_before.timestamp


class TestHeadMovement:
  """HEAD should point to target after copy_epoch."""

  def test_head_moves_to_target(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    store.copy_epoch('exp-a', 0, 'exp-b')

    refs = store.load_refs()
    assert refs['HEAD'] == 'exp-b'


class TestMissingSourceBranch:
  """Missing source branch raises StoreError."""

  def test_missing_source_branch(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)

    with pytest.raises(StoreError, match='not found'):
      store.copy_epoch('nonexistent', 0, 'exp-a')


class TestMissingTargetBranch:
  """Missing target branch raises StoreError."""

  def test_missing_target_branch(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)

    with pytest.raises(StoreError, match='not found'):
      store.copy_epoch('exp-a', 0, 'nonexistent')


class TestMissingSourceEpoch:
  """Missing source epoch raises StoreError."""

  def test_missing_source_epoch(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    with pytest.raises(StoreError, match='snapshot not found'):
      store.copy_epoch('exp-a', 99, 'exp-b')


class TestMissingBlob:
  """Missing blob in object store raises StoreError."""

  def test_missing_blob_raises(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    snap = store.load_snapshot('exp-a', 0)
    for entry in snap.entries.values():
      digest = entry.digest
      prefix = digest[:2]
      rest = digest[2:]
      blob_path = store.config.objects_path / prefix / rest
      blob_path.unlink()
      break

    with pytest.raises(StoreError, match=r'blob.*missing'):
      store.copy_epoch('exp-a', 0, 'exp-b')


class TestSchemaPreserved:
  """Schema from source manifest is copied to the new manifest."""

  def test_schema_copied(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    source_snap = store.load_snapshot('exp-a', 0)
    result = store.copy_epoch('exp-a', 0, 'exp-b')

    if source_snap.schema is not None:
      assert result.schema is not None
      assert len(result.schema.parameters) == len(source_snap.schema.parameters)


class TestContextOnManifest:
  """Context string is set on the new manifest."""

  def test_explicit_context(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    result = store.copy_epoch('exp-a', 0, 'exp-b', context='cherry-pick for hotfix')

    assert result.context == 'cherry-pick for hotfix'

  def test_default_context(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    result = store.copy_epoch('exp-a', 0, 'exp-b')

    assert result.context is not None
    assert 'copy_epoch from exp-a@0' in result.context


class TestReflogEntry:
  """copy_epoch appends a reflog entry with correct fields."""

  def test_reflog_appended(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    store.copy_epoch('exp-a', 0, 'exp-b')

    entries = _read_reflog(store)
    copy_entries = [e for e in entries if e['operation'] == 'copy_epoch']
    assert len(copy_entries) == 1

    entry = copy_entries[0]
    assert entry['experiment_id'] == 'exp-b'
    assert entry['new_epoch'] == 1
    assert entry['source_experiment_id'] == 'exp-a'
    assert entry['source_epoch'] == 0
    assert 'timestamp' in entry

  def test_reflog_old_epoch_correct(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')
    (src / 'main.py').write_text('v2\n')
    store.snapshot('exp-b', 1)

    store.copy_epoch('exp-a', 0, 'exp-b')

    entries = _read_reflog(store)
    copy_entry = next(e for e in entries if e['operation'] == 'copy_epoch')
    assert copy_entry['old_epoch'] == 1
    assert copy_entry['new_epoch'] == 2


class TestConcurrentCopyEpoch:
  """Concurrent copy_epoch calls serialize under lock."""

  def test_concurrent_copy_no_corruption(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    errors: list[Exception] = []
    successes: list[bool] = []
    lock = threading.Lock()

    def copy_worker() -> None:
      try:
        store.copy_epoch('exp-a', 0, 'exp-b')
        with lock:
          successes.append(True)
      except (StoreError, ConcurrentMutationError) as exc:
        with lock:
          errors.append(exc)

    threads = [threading.Thread(target=copy_worker) for _ in range(5)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    refs = store.load_refs()
    tip = refs['branches']['exp-b']['latest_epoch']
    assert tip == len(successes)


class TestCopyEpochMultipleFiles:
  """Copy works correctly with multiple files."""

  def test_multiple_files_copied(self, tmp_path: Path) -> None:
    files = {
      'main.py': 'print("main")\n',
      'utils.py': 'def helper(): pass\n',
      'config.txt': 'key=value\n',
    }
    store, _src, _param = _make_store(tmp_path, files=files)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    result = store.copy_epoch('exp-a', 0, 'exp-b')

    source_snap = store.load_snapshot('exp-a', 0)
    assert len(result.entries) == len(source_snap.entries)
    for key in source_snap.entries:
      assert key in result.entries
      assert result.entries[key].digest == source_snap.entries[key].digest


class TestCopyEpochChained:
  """Multiple copy_epoch calls chain correctly."""

  def test_chained_copies(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-a', 0)
    (src / 'main.py').write_text('v2\n')
    store.snapshot('exp-a', 1)
    store.branch('exp-b')

    store.copy_epoch('exp-a', 0, 'exp-b')
    store.copy_epoch('exp-a', 1, 'exp-b')

    refs = store.load_refs()
    assert refs['branches']['exp-b']['latest_epoch'] == 2

    snap_1 = store.load_snapshot('exp-b', 1)
    snap_2 = store.load_snapshot('exp-b', 2)
    source_0 = store.load_snapshot('exp-a', 0)
    source_1 = store.load_snapshot('exp-a', 1)

    assert snap_1.entries == source_0.entries
    assert snap_2.entries == source_1.entries


# CLI smoke test


class TestCopyEpochCLI:
  """CLI smoke test for store copy-epoch."""

  def test_cli_copy_epoch(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    envelope = run_cli(
      ws,
      ['store', 'copy-epoch', 'exp-a', '0', 'exp-b'],
    )

    assert envelope['ok'] is True
    result = envelope['result']
    assert result['epoch'] == 1
    assert result['source_experiment_id'] == 'exp-a'
    assert result['source_epoch'] == 0
    assert result['target_experiment_id'] == 'exp-b'
    assert result['file_count'] > 0

  def test_cli_copy_epoch_missing_source(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-a', 0)

    with pytest.raises(SystemExit):
      run_cli(
        ws,
        ['store', 'copy-epoch', 'nonexistent', '0', 'exp-a'],
      )
