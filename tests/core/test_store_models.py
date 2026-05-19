"""Tests for Store ABC and supporting dataclasses."""

from autopilot.core.snapshot import FileEntry, SnapshotManifest
from autopilot.core.store.base import Store
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
from typing import Any, cast
import pytest


class TestStoreABC:
  def test_store_cannot_be_instantiated(self) -> None:
    with pytest.raises(NotImplementedError):
      Store(cast(Any, None))

  def test_store_is_a_class(self) -> None:
    assert isinstance(Store, type)

  def test_new_methods_raise_not_implemented(self) -> None:
    """All new methods on Store base raise NotImplementedError."""

    class MinimalStore(Store):
      def __init__(self):
        pass

    s = MinimalStore()
    with pytest.raises(NotImplementedError):
      s.snapshot('exp', 0)
    with pytest.raises(NotImplementedError):
      s.checkout('exp', 0)
    with pytest.raises(NotImplementedError):
      s.diff('exp', 0, 1)
    with pytest.raises(NotImplementedError):
      s.branch('exp')
    with pytest.raises(NotImplementedError):
      s.merge_analysis('exp', 'other')
    with pytest.raises(NotImplementedError):
      s.merge_preview('exp', 'other')
    with pytest.raises(NotImplementedError):
      s.merge_apply(MergeIndex())
    with pytest.raises(NotImplementedError):
      s.log('exp')
    with pytest.raises(NotImplementedError):
      s.status('exp')
    with pytest.raises(NotImplementedError):
      s.materialize('exp', 0)
    with pytest.raises(NotImplementedError):
      s.create_worktree('exp')
    with pytest.raises(NotImplementedError):
      s.remove_worktree('exp')
    with pytest.raises(NotImplementedError):
      s.list_worktrees()
    with pytest.raises(NotImplementedError):
      s.resolve_path('exp')
    with pytest.raises(NotImplementedError):
      s.save_state_dict({})
    with pytest.raises(NotImplementedError):
      s.load_state_dict()
    with pytest.raises(NotImplementedError):
      _ = s.config

  def test_method_signatures_accept_experiment_id(self) -> None:
    """Verify method signatures include experiment_id parameter."""

    class MinimalStore(Store):
      def __init__(self):
        pass

    s = MinimalStore()
    import inspect

    sig = inspect.signature(s.snapshot)
    params = list(sig.parameters.keys())
    assert 'experiment_id' in params
    assert 'epoch' in params

    sig = inspect.signature(s.checkout)
    params = list(sig.parameters.keys())
    assert 'experiment_id' in params
    assert 'epoch' in params

    sig = inspect.signature(s.resolve_path)
    params = list(sig.parameters.keys())
    assert 'experiment_id' in params
    assert 'epoch' in params


class TestFileEntry:
  def test_construction(self) -> None:
    e = FileEntry(digest='abc123', size=1024, mtime=1700000000.0)
    assert e.digest == 'abc123'
    assert e.size == 1024
    assert e.mtime == 1700000000.0

  def test_to_dict(self) -> None:
    e = FileEntry(digest='abc', size=10, mtime=1.0)
    d = e.to_dict()
    assert d == {'digest': 'abc', 'size': 10, 'mtime': 1.0, 'original_path': None}

  def test_from_dict(self) -> None:
    d = {'digest': 'abc', 'size': 10, 'mtime': 1.0}
    e = FileEntry.from_dict(d)
    assert e.digest == 'abc'
    assert e.size == 10

  def test_round_trip(self) -> None:
    e = FileEntry(digest='sha256hex', size=2048, mtime=1700000001.5)
    e2 = FileEntry.from_dict(e.to_dict())
    assert e == e2


class TestSnapshotManifest:
  def test_construction_empty(self) -> None:
    s = SnapshotManifest(epoch=0, timestamp='2025-01-01T00:00:00Z')
    assert s.epoch == 0
    assert s.entries == {}

  def test_construction_with_entries(self) -> None:
    entries = {
      'prompts::system.md': FileEntry(digest='aaa', size=100, mtime=1.0),
      'config::main.tf': FileEntry(digest='bbb', size=200, mtime=2.0),
    }
    s = SnapshotManifest(epoch=1, timestamp='2025-01-01T00:00:00Z', entries=entries)
    assert len(s.entries) == 2
    assert s.entries['prompts::system.md'].digest == 'aaa'

  def test_to_dict(self) -> None:
    s = SnapshotManifest(
      epoch=0,
      timestamp='ts',
      entries={'a::b.txt': FileEntry(digest='h', size=1, mtime=0.0)},
    )
    d = s.to_dict()
    assert d['epoch'] == 0
    assert d['entries']['a::b.txt'] == {
      'digest': 'h',
      'size': 1,
      'mtime': 0.0,
      'original_path': None,
    }

  def test_round_trip(self) -> None:
    entries = {
      'p1::file.py': FileEntry(digest='abc123', size=512, mtime=1700000000.0),
      'p2::data.json': FileEntry(digest='def456', size=1024, mtime=1700000001.0),
    }
    s = SnapshotManifest(epoch=3, timestamp='2025-06-15T12:00:00Z', entries=entries)
    s2 = SnapshotManifest.from_dict(s.to_dict())
    assert s2.epoch == s.epoch
    assert s2.timestamp == s.timestamp
    assert s2.entries == s.entries


class TestDiffEntry:
  def test_added(self) -> None:
    e = DiffEntry(path='a::new.py', status='added', new_hash='abc')
    assert e.status == 'added'
    assert e.old_hash is None

  def test_modified_with_diff(self) -> None:
    e = DiffEntry(
      path='p::f.txt', status='modified', old_hash='aaa', new_hash='bbb', text_diff='--- a\n+++ b'
    )
    assert e.text_diff is not None
    assert e.text_diff.startswith('---')

  def test_deleted(self) -> None:
    e = DiffEntry(path='p::old.txt', status='deleted', old_hash='aaa')
    assert e.new_hash is None

  def test_round_trip(self) -> None:
    e = DiffEntry(path='p::f.py', status='modified', old_hash='a', new_hash='b', text_diff='diff')
    e2 = DiffEntry.from_dict(e.to_dict())
    assert e == e2


class TestDiffResult:
  def test_empty(self) -> None:
    r = DiffResult()
    assert r.added() == []
    assert r.modified() == []
    assert r.deleted() == []

  def test_filtering(self) -> None:
    entries = [
      DiffEntry(path='a', status='added'),
      DiffEntry(path='b', status='modified', old_hash='x', new_hash='y'),
      DiffEntry(path='c', status='deleted', old_hash='z'),
      DiffEntry(path='d', status='added'),
    ]
    r = DiffResult(entries=entries)
    assert len(r.added()) == 2
    assert len(r.modified()) == 1
    assert len(r.deleted()) == 1

  def test_round_trip(self) -> None:
    entries = [
      DiffEntry(path='a', status='added', new_hash='h1'),
      DiffEntry(path='b', status='deleted', old_hash='h2'),
    ]
    r = DiffResult(entries=entries)
    r2 = DiffResult.from_dict(r.to_dict())
    assert len(r2.entries) == 2
    assert r2.entries[0].path == 'a'
    assert r2.entries[1].status == 'deleted'


class TestMergeStrategy:
  def test_members(self) -> None:
    assert MergeStrategy.normal.value == 'normal'
    assert MergeStrategy.ours.value == 'ours'
    assert MergeStrategy.theirs.value == 'theirs'
    assert MergeStrategy.union.value == 'union'


class TestConflictEntry:
  def test_construction(self) -> None:
    entry = ConflictEntry(
      key='prompts/system.txt',
      ours=FileEntry(digest='aaa', size=10, mtime=0.0),
    )
    assert entry.key == 'prompts/system.txt'
    assert entry.ancestor is None
    assert entry.theirs is None

  def test_round_trip(self) -> None:
    entry = ConflictEntry(
      key='k',
      ancestor=FileEntry(digest='a', size=1, mtime=0.0),
      ours=FileEntry(digest='b', size=2, mtime=0.0),
      theirs=FileEntry(digest='c', size=3, mtime=0.0),
    )
    data = entry.to_dict()
    entry2 = ConflictEntry.from_dict(data)
    assert entry2.key == 'k'
    assert entry2.ancestor is not None
    assert entry2.ancestor.digest == 'a'
    assert entry2.ours is not None
    assert entry2.ours.digest == 'b'
    assert entry2.theirs is not None
    assert entry2.theirs.digest == 'c'


class TestMergeAnalysisResult:
  def test_construction(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=True,
      has_conflicts=False,
      conflict_count=0,
      classification=MergeClassification.fast_forward,
    )
    assert result.can_fast_forward is True
    assert result.classification == 'fast_forward'

  def test_round_trip(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=True,
      conflict_count=3,
      ancestor_epoch=2,
      classification=MergeClassification.conflict,
    )
    data = result.to_dict()
    result2 = MergeAnalysisResult.from_dict(data)
    assert result2.has_conflicts is True
    assert result2.conflict_count == 3
    assert result2.ancestor_epoch == 2


class TestMergeIndex:
  def test_is_resolved_empty(self) -> None:
    idx = MergeIndex()
    assert idx.is_resolved() is True

  def test_is_resolved_with_conflicts(self) -> None:
    idx = MergeIndex(
      conflicts={'k': ConflictEntry(key='k')},
    )
    assert idx.is_resolved() is False


class TestStatusEntry:
  def test_modified(self) -> None:
    e = StatusEntry(path='p::f.txt', status='modified')
    assert e.status == 'modified'

  def test_round_trip(self) -> None:
    e = StatusEntry(path='p::f.py', status='deleted')
    e2 = StatusEntry.from_dict(e.to_dict())
    assert e == e2


class TestStatusResult:
  def test_empty(self) -> None:
    r = StatusResult()
    assert r.modified() == []
    assert r.added() == []
    assert r.deleted() == []
    assert r.unchanged() == []

  def test_grouping(self) -> None:
    entries = [
      StatusEntry(path='a', status='modified'),
      StatusEntry(path='b', status='unchanged'),
      StatusEntry(path='c', status='added'),
      StatusEntry(path='d', status='deleted'),
      StatusEntry(path='e', status='unchanged'),
    ]
    r = StatusResult(entries=entries)
    assert len(r.modified()) == 1
    assert len(r.unchanged()) == 2
    assert len(r.added()) == 1
    assert len(r.deleted()) == 1

  def test_round_trip(self) -> None:
    entries = [
      StatusEntry(path='a', status='modified'),
      StatusEntry(path='b', status='unchanged'),
    ]
    r = StatusResult(entries=entries)
    r2 = StatusResult.from_dict(r.to_dict())
    assert len(r2.entries) == 2
    assert r2.entries[0].status == 'modified'


class TestSnapshotEntry:
  def test_construction(self) -> None:
    e = SnapshotEntry(epoch=0, timestamp='2025-01-01T00:00:00Z', file_count=5)
    assert e.epoch == 0
    assert e.file_count == 5

  def test_round_trip(self) -> None:
    e = SnapshotEntry(epoch=3, timestamp='ts', file_count=12)
    e2 = SnapshotEntry.from_dict(e.to_dict())
    assert e == e2
