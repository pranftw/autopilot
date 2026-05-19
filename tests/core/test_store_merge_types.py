"""Tests for core merge types: MergeStrategy, ConflictEntry, MergeAnalysisResult, MergeIndex."""

from autopilot.core.errors import StoreError
from autopilot.core.snapshot import FileEntry, SnapshotManifest
from autopilot.core.store.types import (
  ConflictEntry,
  MergeAnalysisResult,
  MergeClassification,
  MergeIndex,
  MergeStrategy,
)
import pytest


class TestMergeStrategyRoundTrip:
  def test_each_member_round_trips(self) -> None:
    """MergeStrategy serializes via .value and restores via MergeStrategy(value)."""
    for member in MergeStrategy:
      serialized = member.value
      restored = MergeStrategy(serialized)
      assert restored == member

  def test_all_values(self) -> None:
    assert {m.value for m in MergeStrategy} == {'normal', 'ours', 'theirs', 'union'}


class TestConflictEntryRoundTripNoneSides:
  def test_all_none_sides(self) -> None:
    """ConflictEntry with all None sides survives to_dict/from_dict."""
    entry = ConflictEntry(key='prompts/system.txt')
    data = entry.to_dict()
    restored = ConflictEntry.from_dict(data)
    assert restored.key == 'prompts/system.txt'
    assert restored.ancestor is None
    assert restored.ours is None
    assert restored.theirs is None

  def test_partial_none_sides(self) -> None:
    """ConflictEntry with one non-None side round-trips correctly."""
    entry = ConflictEntry(
      key='config/rules.json',
      ancestor=FileEntry(digest='aaa', size=100, mtime=1.0),
      ours=None,
      theirs=FileEntry(digest='ccc', size=300, mtime=3.0),
    )
    data = entry.to_dict()
    restored = ConflictEntry.from_dict(data)
    assert restored.key == 'config/rules.json'
    assert restored.ancestor is not None
    assert restored.ancestor.digest == 'aaa'
    assert restored.ours is None
    assert restored.theirs is not None
    assert restored.theirs.digest == 'ccc'

  def test_all_sides_present(self) -> None:
    """ConflictEntry with all sides set round-trips with field equality."""
    entry = ConflictEntry(
      key='k',
      ancestor=FileEntry(digest='a', size=1, mtime=0.0),
      ours=FileEntry(digest='b', size=2, mtime=0.0),
      theirs=FileEntry(digest='c', size=3, mtime=0.0),
    )
    data = entry.to_dict()
    restored = ConflictEntry.from_dict(data)
    assert restored == entry


class TestMergeAnalysisResultClassifications:
  def test_fast_forward(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=True,
      has_conflicts=False,
      conflict_count=0,
      classification=MergeClassification.fast_forward,
    )
    data = result.to_dict()
    restored = MergeAnalysisResult.from_dict(data)
    assert restored.classification == MergeClassification.fast_forward
    assert restored.can_fast_forward is True

  def test_clean(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=False,
      conflict_count=0,
      classification=MergeClassification.clean,
    )
    data = result.to_dict()
    restored = MergeAnalysisResult.from_dict(data)
    assert restored.classification == MergeClassification.clean

  def test_conflict(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=True,
      conflict_count=5,
      ancestor_epoch=2,
      classification=MergeClassification.conflict,
    )
    data = result.to_dict()
    restored = MergeAnalysisResult.from_dict(data)
    assert restored.classification == MergeClassification.conflict
    assert restored.conflict_count == 5
    assert restored.ancestor_epoch == 2

  def test_up_to_date(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=False,
      conflict_count=0,
      classification=MergeClassification.up_to_date,
    )
    data = result.to_dict()
    restored = MergeAnalysisResult.from_dict(data)
    assert restored.classification == MergeClassification.up_to_date


class TestMergeIndexResolveExplicit:
  def test_resolve_moves_key_to_resolved(self) -> None:
    """resolve() moves key from conflicts to resolved."""
    entry = FileEntry(digest='resolved_hash', size=42, mtime=0.0)
    conflict = ConflictEntry(
      key='prompts/system.txt',
      ours=FileEntry(digest='ours_hash', size=10, mtime=0.0),
      theirs=FileEntry(digest='theirs_hash', size=20, mtime=0.0),
    )
    idx = MergeIndex(conflicts={'prompts/system.txt': conflict})
    idx.resolve('prompts/system.txt', entry)
    assert 'prompts/system.txt' not in idx.conflicts
    assert 'prompts/system.txt' in idx.resolved
    assert idx.resolved['prompts/system.txt'].digest == 'resolved_hash'

  def test_resolve_nonexistent_key_raises(self) -> None:
    idx = MergeIndex()
    with pytest.raises(StoreError, match='not in conflicts'):
      idx.resolve('missing', FileEntry(digest='x', size=1, mtime=0.0))


class TestMergeIndexResolveOurs:
  def test_resolves_to_ours_entry(self) -> None:
    """resolve_ours picks ours FileEntry from the ConflictEntry."""
    ours = FileEntry(digest='ours_hash', size=10, mtime=0.0)
    conflict = ConflictEntry(key='k', ours=ours, theirs=FileEntry(digest='t', size=5, mtime=0.0))
    idx = MergeIndex(conflicts={'k': conflict})
    idx.resolve_ours('k')
    assert idx.resolved['k'].digest == 'ours_hash'
    assert 'k' not in idx.conflicts


class TestMergeIndexResolveTheirs:
  def test_resolves_to_theirs_entry(self) -> None:
    """resolve_theirs picks theirs FileEntry from the ConflictEntry."""
    theirs = FileEntry(digest='theirs_hash', size=20, mtime=0.0)
    conflict = ConflictEntry(key='k', ours=FileEntry(digest='o', size=5, mtime=0.0), theirs=theirs)
    idx = MergeIndex(conflicts={'k': conflict})
    idx.resolve_theirs('k')
    assert idx.resolved['k'].digest == 'theirs_hash'
    assert 'k' not in idx.conflicts


class TestMergeIndexResolveOursMissingRaises:
  def test_missing_ours_raises_store_error(self) -> None:
    """resolve_ours with None ours side raises StoreError."""
    conflict = ConflictEntry(
      key='k',
      ours=None,
      theirs=FileEntry(digest='t', size=5, mtime=0.0),
    )
    idx = MergeIndex(conflicts={'k': conflict})
    with pytest.raises(StoreError, match='ours side is None'):
      idx.resolve_ours('k')


class TestMergeIndexIsResolvedFalse:
  def test_open_conflicts_false(self) -> None:
    idx = MergeIndex(
      conflicts={
        'a': ConflictEntry(key='a'),
        'b': ConflictEntry(key='b'),
      },
    )
    assert idx.is_resolved() is False


class TestMergeIndexIsResolvedTrue:
  def test_all_resolved_true(self) -> None:
    idx = MergeIndex(
      resolved={
        'a': FileEntry(digest='h1', size=1, mtime=0.0),
        'b': FileEntry(digest='h2', size=2, mtime=0.0),
      },
    )
    assert idx.is_resolved() is True


class TestMergeIndexToSnapshotManifest:
  def test_to_snapshot_yields_expected_keys(self) -> None:
    """to_snapshot() yields SnapshotManifest with only resolved keys."""
    idx = MergeIndex(
      resolved={
        'prompts/system.txt': FileEntry(digest='h1', size=10, mtime=0.0),
        'config/rules.json': FileEntry(digest='h2', size=20, mtime=0.0),
      },
    )
    snap = idx.to_snapshot()
    assert isinstance(snap, SnapshotManifest)
    assert set(snap.entries) == {'prompts/system.txt', 'config/rules.json'}
    assert snap.entries['prompts/system.txt'].digest == 'h1'

  def test_to_snapshot_with_conflicts_raises(self) -> None:
    """to_snapshot raises StoreError when conflicts remain."""
    idx = MergeIndex(
      conflicts={'k': ConflictEntry(key='k')},
      resolved={'ok': FileEntry(digest='h', size=1, mtime=0.0)},
    )
    with pytest.raises(StoreError, match='unresolved conflict'):
      idx.to_snapshot()
