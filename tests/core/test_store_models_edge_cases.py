"""Edge-case tests for Store dataclass models: null entries, missing fields, round-trips."""

from autopilot.core.snapshot import FileEntry, SnapshotManifest
from autopilot.core.store.types import (
  ConflictEntry,
  DiffEntry,
  DiffResult,
  MergeAnalysisResult,
  MergeClassification,
  MergeIndex,
  MergeStrategy,
  SnapshotEntry,
  StatusEntry,
  StatusResult,
)


class TestSnapshotManifestNullEntries:
  def test_from_dict_with_null_entries(self) -> None:
    data = {'epoch': 0, 'timestamp': 'ts', 'entries': None}
    sm = SnapshotManifest.from_dict(data)
    assert sm.entries == {}

  def test_from_dict_with_missing_entries(self) -> None:
    data = {'epoch': 1, 'timestamp': 'ts'}
    sm = SnapshotManifest.from_dict(data)
    assert sm.entries == {}

  def test_from_dict_with_empty_entries(self) -> None:
    data = {'epoch': 2, 'timestamp': 'ts', 'entries': {}}
    sm = SnapshotManifest.from_dict(data)
    assert sm.entries == {}


class TestDiffResultNullEntries:
  def test_from_dict_with_null_entries(self) -> None:
    data = {'entries': None}
    dr = DiffResult.from_dict(data)
    assert dr.entries == []

  def test_from_dict_with_missing_entries(self) -> None:
    data = {}
    dr = DiffResult.from_dict(data)
    assert dr.entries == []

  def test_from_dict_with_empty_entries(self) -> None:
    data = {'entries': []}
    dr = DiffResult.from_dict(data)
    assert dr.entries == []


class TestMergeIndexRoundTrip:
  def test_resolved_index_round_trip(self) -> None:
    idx = MergeIndex(
      resolved={'k': FileEntry(digest='h', size=10, mtime=1.0)},
      experiment_id='exp-a',
      source_experiment_id='exp-b',
      strategy=MergeStrategy.normal,
      preview_token='tok',
    )
    data = idx.to_dict()
    idx2 = MergeIndex.from_dict(data)
    assert idx2.is_resolved() is True
    assert idx2.resolved['k'].digest == 'h'
    assert idx2.experiment_id == 'exp-a'
    assert idx2.strategy == MergeStrategy.normal
    assert idx2.preview_token == 'tok'

  def test_conflicted_index_round_trip(self) -> None:
    idx = MergeIndex(
      conflicts={
        'a.txt': ConflictEntry(key='a.txt'),
        'b.txt': ConflictEntry(key='b.txt'),
      },
    )
    data = idx.to_dict()
    idx2 = MergeIndex.from_dict(data)
    assert not idx2.is_resolved()
    assert len(idx2.conflicts) == 2

  def test_from_dict_with_null_conflicts(self) -> None:
    data = {'conflicts': None, 'resolved': None}
    idx = MergeIndex.from_dict(data)
    assert idx.conflicts == {}
    assert idx.resolved == {}

  def test_merge_analysis_result_round_trip(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=True,
      conflict_count=2,
      ancestor_epoch=1,
      classification=MergeClassification.conflict,
    )
    data = result.to_dict()
    result2 = MergeAnalysisResult.from_dict(data)
    assert result2.has_conflicts is True
    assert result2.conflict_count == 2


class TestStatusResultRoundTrip:
  def test_full_round_trip(self) -> None:
    entries = [
      StatusEntry(path='a.py', status='modified'),
      StatusEntry(path='b.py', status='added'),
      StatusEntry(path='c.py', status='deleted'),
      StatusEntry(path='d.py', status='unchanged'),
    ]
    sr = StatusResult(entries=entries)
    data = sr.to_dict()
    sr2 = StatusResult.from_dict(data)
    assert len(sr2.entries) == 4
    assert sr2.entries[0].path == 'a.py'
    assert sr2.entries[0].status == 'modified'
    assert sr2.entries[1].status == 'added'
    assert sr2.entries[2].status == 'deleted'
    assert sr2.entries[3].status == 'unchanged'

  def test_from_dict_with_null_entries(self) -> None:
    data = {'entries': None}
    sr = StatusResult.from_dict(data)
    assert sr.entries == []

  def test_from_dict_with_missing_entries(self) -> None:
    data = {}
    sr = StatusResult.from_dict(data)
    assert sr.entries == []


class TestAllStoreDataclassFromDictMissingFields:
  def test_file_entry_missing_optional_path(self) -> None:
    data = {'digest': 'abc', 'size': 10, 'mtime': 1.0}
    fe = FileEntry.from_dict(data)
    assert fe.original_path is None

  def test_file_entry_with_original_path(self) -> None:
    data = {'digest': 'abc', 'size': 10, 'mtime': 1.0, 'original_path': 'src/main.py'}
    fe = FileEntry.from_dict(data)
    assert fe.original_path == 'src/main.py'

  def test_diff_entry_missing_optional_fields(self) -> None:
    data = {'path': 'a.py', 'status': 'added'}
    de = DiffEntry.from_dict(data)
    assert de.old_hash is None
    assert de.new_hash is None
    assert de.text_diff is None

  def test_diff_entry_with_all_fields(self) -> None:
    data = {
      'path': 'a.py',
      'status': 'modified',
      'old_hash': 'h1',
      'new_hash': 'h2',
      'text_diff': 'diff text',
    }
    de = DiffEntry.from_dict(data)
    assert de.old_hash == 'h1'
    assert de.new_hash == 'h2'
    assert de.text_diff == 'diff text'

  def test_status_entry_round_trip(self) -> None:
    se = StatusEntry(path='f.py', status='modified')
    data = se.to_dict()
    se2 = StatusEntry.from_dict(data)
    assert se == se2

  def test_snapshot_manifest_from_dict_extra_keys_ignored(self) -> None:
    data = {
      'epoch': 0,
      'timestamp': 'ts',
      'entries': {},
      'extra_key': 'should be ignored',
    }
    sm = SnapshotManifest.from_dict(data)
    assert sm.epoch == 0
    assert not hasattr(sm, 'extra_key')

  def test_merge_analysis_from_dict_defaults(self) -> None:
    data = {'can_fast_forward': True, 'has_conflicts': False, 'conflict_count': 0}
    result = MergeAnalysisResult.from_dict(data)
    assert result.can_fast_forward is True
    assert result.classification == MergeClassification.up_to_date
    assert result.ancestor_epoch is None

  def test_snapshot_entry_round_trip(self) -> None:
    se = SnapshotEntry(epoch=5, timestamp='2025-01-01T00:00:00Z', file_count=42)
    data = se.to_dict()
    se2 = SnapshotEntry.from_dict(data)
    assert se == se2

  def test_snapshot_entry_from_dict(self) -> None:
    data = {'epoch': 0, 'timestamp': 'ts', 'file_count': 0}
    se = SnapshotEntry.from_dict(data)
    assert se.epoch == 0
    assert se.timestamp == 'ts'
    assert se.file_count == 0
