"""Tests for Store ABC and supporting dataclasses."""

from autopilot.core.store import (
  DiffEntry,
  DiffResult,
  FileEntry,
  MergeResult,
  SnapshotEntry,
  SnapshotManifest,
  StatusEntry,
  StatusResult,
  Store,
)
import pytest


class TestStoreABC:
  def test_store_cannot_be_instantiated(self) -> None:
    with pytest.raises(NotImplementedError):
      Store(path=None, slug='test', parameters=[])

  def test_store_is_a_class(self) -> None:
    assert isinstance(Store, type)


class TestFileEntry:
  def test_construction(self) -> None:
    e = FileEntry(hash='abc123', size=1024, mtime=1700000000.0)
    assert e.hash == 'abc123'
    assert e.size == 1024
    assert e.mtime == 1700000000.0

  def test_to_dict(self) -> None:
    e = FileEntry(hash='abc', size=10, mtime=1.0)
    d = e.to_dict()
    assert d == {'hash': 'abc', 'size': 10, 'mtime': 1.0}

  def test_from_dict(self) -> None:
    d = {'hash': 'abc', 'size': 10, 'mtime': 1.0}
    e = FileEntry.from_dict(d)
    assert e.hash == 'abc'
    assert e.size == 10

  def test_round_trip(self) -> None:
    e = FileEntry(hash='sha256hex', size=2048, mtime=1700000001.5)
    e2 = FileEntry.from_dict(e.to_dict())
    assert e == e2


class TestSnapshotManifest:
  def test_construction_empty(self) -> None:
    s = SnapshotManifest(epoch=0, timestamp='2025-01-01T00:00:00Z')
    assert s.epoch == 0
    assert s.entries == {}

  def test_construction_with_entries(self) -> None:
    entries = {
      'prompts::system.md': FileEntry(hash='aaa', size=100, mtime=1.0),
      'config::main.tf': FileEntry(hash='bbb', size=200, mtime=2.0),
    }
    s = SnapshotManifest(epoch=1, timestamp='2025-01-01T00:00:00Z', entries=entries)
    assert len(s.entries) == 2
    assert s.entries['prompts::system.md'].hash == 'aaa'

  def test_to_dict(self) -> None:
    s = SnapshotManifest(
      epoch=0,
      timestamp='ts',
      entries={'a::b.txt': FileEntry(hash='h', size=1, mtime=0.0)},
    )
    d = s.to_dict()
    assert d['epoch'] == 0
    assert d['entries']['a::b.txt'] == {'hash': 'h', 'size': 1, 'mtime': 0.0}

  def test_round_trip(self) -> None:
    entries = {
      'p1::file.py': FileEntry(hash='abc123', size=512, mtime=1700000000.0),
      'p2::data.json': FileEntry(hash='def456', size=1024, mtime=1700000001.0),
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


class TestMergeResult:
  def test_clean_merge(self) -> None:
    snap = SnapshotManifest(epoch=2, timestamp='ts', entries={})
    r = MergeResult(merged=True, merged_snapshot=snap)
    assert r.merged is True
    assert r.conflicts == []

  def test_conflicted_merge(self) -> None:
    r = MergeResult(merged=False, conflicts=['p::f.txt', 'p::g.txt'])
    assert r.merged is False
    assert len(r.conflicts) == 2

  def test_round_trip_with_snapshot(self) -> None:
    snap = SnapshotManifest(
      epoch=1,
      timestamp='ts',
      entries={'k': FileEntry(hash='h', size=1, mtime=0.0)},
    )
    r = MergeResult(merged=True, conflicts=[], merged_snapshot=snap)
    r2 = MergeResult.from_dict(r.to_dict())
    assert r2.merged is True
    assert r2.merged_snapshot is not None
    assert r2.merged_snapshot.entries['k'].hash == 'h'

  def test_round_trip_without_snapshot(self) -> None:
    r = MergeResult(merged=False, conflicts=['a'])
    r2 = MergeResult.from_dict(r.to_dict())
    assert r2.merged is False
    assert r2.merged_snapshot is None


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
