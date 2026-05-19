"""Tests for FileStore.repair_diagnostics and diagnostic pipeline.

Covers:
  - doctor returns DiagnosticEntry instances (4.2 #6)
  - orphan blob repair (4.2 #7)
  - stale lock repair only when PID dead (4.2 #8)
  - dry_run produces no mutations (4.2 #9)
  - corrupt manifest not destructively repaired (4.2 #10)
  - non-repairable entries produce no mutations (4.2 #11)
  - repair_diagnostics raises StoreError when context is None (4.2 #12)
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.diagnostic import DiagnosticEntry
from autopilot.core.errors import StoreError
from pathlib import Path
from unittest.mock import patch
import pytest


def _make_store_with_snapshot(tmp_path: Path) -> tuple[FileStore, AutoPilotConfig]:
  """Create a FileStore with a single snapshot for testing.

  Returns:
    Tuple of (store, config) after one snapshot at epoch 0.
  """
  prompts_dir = tmp_path / 'prompts'
  prompts_dir.mkdir()
  (prompts_dir / 'main.txt').write_text('hello world')

  config = AutoPilotConfig(workspace=tmp_path)
  param = PathParameter(source=str(prompts_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-001', 0)
  FileForest(store).save()
  return store, config


class TestDoctorReturnsDiagnostics:
  """test_store_doctor_returns_diagnostics -- doctor emits DiagnosticEntry instances."""

  def test_all_entries_are_diagnostic_entry(self, tmp_path: Path) -> None:
    """doctor() returns a list where every element is a DiagnosticEntry."""
    store, _ = _make_store_with_snapshot(tmp_path)
    entries = store.doctor()
    assert isinstance(entries, list)
    for entry in entries:
      assert isinstance(entry, DiagnosticEntry)

  def test_healthy_store_returns_empty_diagnostics(self, tmp_path: Path) -> None:
    """Healthy store produces no diagnostic entries."""
    store, _ = _make_store_with_snapshot(tmp_path)
    entries = store.doctor()
    assert entries == []

  def test_orphan_blob_detected_as_diagnostic(self, tmp_path: Path) -> None:
    """Orphan blob produces a DiagnosticEntry with code 'orphan_blob'."""
    store, config = _make_store_with_snapshot(tmp_path)
    shard = config.objects_path / 'ab'
    shard.mkdir(parents=True, exist_ok=True)
    (shard / 'cdef1234').write_bytes(b'orphan data')

    entries = store.doctor()
    orphans = [e for e in entries if e.code == 'orphan_blob']
    assert len(orphans) == 1
    assert orphans[0].repairable is True
    assert orphans[0].repair_action == 'delete'


class TestRepairOrphans:
  """test_store_doctor_repair_orphans -- orphan blobs removed after repair."""

  def test_orphan_removed_after_repair(self, tmp_path: Path) -> None:
    """Orphan blob is deleted from object store after repair."""
    store, config = _make_store_with_snapshot(tmp_path)

    shard = config.objects_path / 'ab'
    shard.mkdir(parents=True, exist_ok=True)
    orphan_file = shard / 'cdef1234'
    orphan_file.write_bytes(b'orphan data')
    assert orphan_file.exists()

    entries = store.doctor()
    orphans = [e for e in entries if e.code == 'orphan_blob']
    assert len(orphans) == 1

    repaired = store.repair_diagnostics(entries, context='test repair')
    assert len(repaired) == 1
    assert repaired[0].code == 'orphan_blob'
    assert not orphan_file.exists()

  def test_orphan_shard_dir_cleaned_when_empty(self, tmp_path: Path) -> None:
    """Shard directory removed when orphan blob was the only file."""
    store, config = _make_store_with_snapshot(tmp_path)

    shard = config.objects_path / 'ff'
    shard.mkdir(parents=True, exist_ok=True)
    (shard / 'aabb0011').write_bytes(b'junk')

    entries = store.doctor()
    store.repair_diagnostics(entries, context='test')
    assert not shard.exists()


class TestRepairStaleLock:
  """test_store_doctor_repair_stale_lock -- lock removed only when PID dead."""

  def test_stale_lock_removed_when_pid_dead(self, tmp_path: Path) -> None:
    """Lock file from a dead PID is detected and removed by repair."""
    store, config = _make_store_with_snapshot(tmp_path)
    lock_path = config.store_path / '.lock'
    lock_path.write_text('999999999')

    with patch('os.kill', side_effect=ProcessLookupError):
      entries = store.doctor()
      stale = [e for e in entries if e.code == 'stale_lock']
      assert len(stale) == 1
      assert stale[0].repairable is True

      repaired = store.repair_diagnostics(entries, context='remove stale lock')
      assert len(repaired) >= 1
      assert not lock_path.exists()

  def test_lock_not_removed_when_pid_alive(self, tmp_path: Path) -> None:
    """Lock file from a live PID is NOT detected as stale."""
    store, config = _make_store_with_snapshot(tmp_path)
    lock_path = config.store_path / '.lock'
    lock_path.write_text('12345')

    with patch('os.kill', return_value=None):
      entries = store.doctor()
      stale = [e for e in entries if e.code == 'stale_lock']
      assert stale == []


class TestRepairDryRun:
  """test_store_doctor_repair_dry_run -- no mutation when dry_run=True."""

  def test_dry_run_does_not_delete_orphan(self, tmp_path: Path) -> None:
    """dry_run=True reports would-repair but does not delete files."""
    store, config = _make_store_with_snapshot(tmp_path)

    shard = config.objects_path / 'ab'
    shard.mkdir(parents=True, exist_ok=True)
    orphan_file = shard / 'cdef1234'
    orphan_file.write_bytes(b'orphan data')

    entries = store.doctor()
    repaired = store.repair_diagnostics(entries, dry_run=True)
    assert len(repaired) == 1
    assert repaired[0].code == 'orphan_blob'
    assert orphan_file.exists()

  def test_dry_run_does_not_remove_stale_lock(self, tmp_path: Path) -> None:
    """dry_run=True does not delete stale lock files."""
    store, config = _make_store_with_snapshot(tmp_path)
    lock_path = config.store_path / '.lock'
    lock_path.write_text('999999999')

    with patch('os.kill', side_effect=ProcessLookupError):
      entries = store.doctor()
      repaired = store.repair_diagnostics(entries, dry_run=True)
      stale_repaired = [e for e in repaired if e.code == 'stale_lock']
      assert len(stale_repaired) == 1
      assert lock_path.exists()


class TestCorruptManifestNotRepaired:
  """test_store_doctor_repair_corrupt_manifest_quarantine -- fail-closed behavior."""

  def test_corrupt_manifest_not_repairable(self, tmp_path: Path) -> None:
    """Corrupt manifest entries are marked as non-repairable."""
    store, config = _make_store_with_snapshot(tmp_path)

    snap_dir = config.store_path / 'snapshots' / 'exp-001'
    snap_file = snap_dir / 'epoch_0.json'
    snap_file.write_text('not valid json')

    entries = store.doctor()
    manifest_errs = [e for e in entries if e.code == 'manifest_error']
    assert len(manifest_errs) >= 1
    for entry in manifest_errs:
      assert entry.repairable is False
      assert entry.repair_action is None

  def test_repair_skips_corrupt_manifest(self, tmp_path: Path) -> None:
    """repair_diagnostics does not mutate or delete corrupt manifest files."""
    store, config = _make_store_with_snapshot(tmp_path)

    snap_dir = config.store_path / 'snapshots' / 'exp-001'
    snap_file = snap_dir / 'epoch_0.json'
    snap_file.write_text('not valid json')

    entries = store.doctor()
    repaired = store.repair_diagnostics(entries, context='test')
    manifest_repaired = [e for e in repaired if e.code == 'manifest_error']
    assert manifest_repaired == []
    assert snap_file.exists()


class TestNonRepairableFlag:
  """test_diagnostic_entry_repairable_flag -- non-repairable entries skipped."""

  def test_non_repairable_entries_not_in_repaired_list(self, tmp_path: Path) -> None:
    """Entries with repairable=False are never included in repair output."""
    entries = [
      DiagnosticEntry(
        code='missing_blob',
        severity='error',
        path='abcdef123456',
        message='blob missing from object store',
        repairable=False,
      ),
      DiagnosticEntry(
        code='manifest_error',
        severity='error',
        path=None,
        message='corrupt manifest at epoch_1.json',
        repairable=False,
      ),
    ]
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})
    FileForest(store).save()

    repaired = store.repair_diagnostics(entries, context='test')
    assert repaired == []


class TestGhostEpochDiagnostic:
  """Ghost epoch detection and repair tests."""

  def test_ghost_epoch_diagnostic_entry_valid(self) -> None:
    """DiagnosticEntry with code='ghost_epoch' constructs without error."""
    entry = DiagnosticEntry(
      code='ghost_epoch',
      severity='warning',
      path='/tmp/epoch_1.json',
      message='ghost epoch found',
      repairable=True,
      repair_action='delete',
    )
    assert entry.code == 'ghost_epoch'
    assert entry.repairable is True

  def test_doctor_detects_ghost_epoch(self, tmp_path: Path) -> None:
    """Manually written epoch_1.json beyond latest_epoch=0 is detected."""
    store, config = _make_store_with_snapshot(tmp_path)

    snap_dir = config.store_path / 'snapshots' / 'exp-001'
    ghost_file = snap_dir / 'epoch_1.json'
    ghost_file.write_text('{"epoch": 1, "timestamp": "t", "entries": {}}')

    entries = store.doctor()
    ghosts = [e for e in entries if e.code == 'ghost_epoch']
    assert len(ghosts) == 1
    assert 'epoch 1' in ghosts[0].message
    assert ghosts[0].repairable is True
    assert ghosts[0].repair_action == 'delete'

  def test_repair_deletes_ghost_epoch(self, tmp_path: Path) -> None:
    """repair_diagnostics removes the ghost epoch file."""
    store, config = _make_store_with_snapshot(tmp_path)

    snap_dir = config.store_path / 'snapshots' / 'exp-001'
    ghost_file = snap_dir / 'epoch_1.json'
    ghost_file.write_text('{"epoch": 1, "timestamp": "t", "entries": {}}')
    assert ghost_file.exists()

    entries = store.doctor()
    repaired = store.repair_diagnostics(entries, context='remove ghost')
    ghost_repaired = [e for e in repaired if e.code == 'ghost_epoch']
    assert len(ghost_repaired) == 1
    assert not ghost_file.exists()

  def test_doctor_no_ghost_on_healthy_store(self, tmp_path: Path) -> None:
    """Healthy store with matching epochs produces no ghost_epoch diagnostics."""
    store, _ = _make_store_with_snapshot(tmp_path)
    entries = store.doctor()
    ghosts = [e for e in entries if e.code == 'ghost_epoch']
    assert ghosts == []


class TestRepairRequiresContext:
  """test_repair_diagnostics_requires_context -- StoreError when context is None."""

  def test_raises_when_context_none_and_repairable(self, tmp_path: Path) -> None:
    """repair_diagnostics raises StoreError when context is None and repairable entries exist."""
    store, config = _make_store_with_snapshot(tmp_path)

    shard = config.objects_path / 'ab'
    shard.mkdir(parents=True, exist_ok=True)
    (shard / 'cdef1234').write_bytes(b'orphan')

    entries = store.doctor()
    repairable = [e for e in entries if e.repairable]
    assert len(repairable) >= 1

    with pytest.raises(StoreError, match='context'):
      store.repair_diagnostics(entries, context=None)

  def test_no_error_when_context_none_and_nothing_repairable(self, tmp_path: Path) -> None:
    """repair_diagnostics does not raise when no repairable entries exist."""
    entries = [
      DiagnosticEntry(
        code='missing_blob',
        severity='error',
        path='abcdef',
        message='blob missing',
        repairable=False,
      ),
    ]
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})
    FileForest(store).save()

    repaired = store.repair_diagnostics(entries, context=None)
    assert repaired == []
