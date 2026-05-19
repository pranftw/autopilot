"""Tests for Plan 09: store lock, merge agent, and merge analysis DRY consolidation.

Covers:
  - AutopilotFileLock-backed store locking and contention
  - exclusive_create wiring in store (worktree lock)
  - reraise_as_store_error error wrapping helper
  - _apply_single_resolution extracted MergeAgent helper
  - MergeClassification enum on MergeAnalysisResult (type, round-trip, JSON)
"""

from autopilot.ai.agents.agent import Agent
from autopilot.ai.merge_agent import MergeAgent
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store.store_io import reraise_as_store_error
from autopilot.ai.store_lock import StorageBackend, hash_bytes, hash_content
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError, TrackingError
from autopilot.core.snapshot import FileEntry
from autopilot.core.store.types import (
  ConflictEntry,
  MergeAnalysisResult,
  MergeClassification,
  MergeIndex,
)
from autopilot.tracking.file_lock import ConcurrentMutationError
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
import json
import pytest
import threading
import time


def _make_store(tmp_path: Path, files: dict[str, str]) -> tuple[FileStore, Path]:
  src = make_source_dir(tmp_path, files=files)
  config = make_store_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  return store, src


class StubAgent(Agent):
  """Agent stub returning a canned response."""

  def __init__(self, response: str) -> None:
    super().__init__()
    self._response = response

  def run(self, *args, **kwargs):
    return self._response


# 2.1: exclusive_create wiring


class TestExclusiveCreateStoreLock:
  """Verify StorageBackend.acquire_lock uses AutopilotFileLock."""

  def test_successful_acquire_creates_lock_file(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.store'
    backend = StorageBackend(config)
    backend.acquire_lock()
    assert backend._lock_file.exists()
    backend.release_lock()

  def test_contention_raises_store_error(self, tmp_path: Path) -> None:
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.store'
    backend = StorageBackend(config)
    barrier = threading.Barrier(2)
    err: list[BaseException | None] = [None]

    def holder() -> None:
      backend.acquire_lock()
      barrier.wait()
      time.sleep(0.15)
      backend.release_lock()

    def contender() -> None:
      barrier.wait()
      try:
        backend.acquire_lock()
      except ConcurrentMutationError as exc:
        err[0] = exc
      else:
        backend.release_lock()

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=contender)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert isinstance(err[0], ConcurrentMutationError)
    assert 'concurrent mutation' in str(err[0])

  def test_stale_lock_file_does_not_block(self, tmp_path: Path) -> None:
    """With filelock/flock, a leftover lock file does not prevent acquisition."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.store'
    backend = StorageBackend(config)
    backend._lock_file.parent.mkdir(parents=True, exist_ok=True)
    backend._lock_file.write_text('stale')
    backend.acquire_lock()
    backend.release_lock()


class TestExclusiveCreateWorktree:
  """Verify FileStore.create_worktree uses exclusive_create."""

  def test_worktree_lock_contention(self, tmp_path: Path) -> None:
    store, _src = _make_store(tmp_path, {'f.txt': 'hello'})
    store.snapshot('root', 0)
    worktrees_dir = store.config.worktrees_path
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    lock_path = worktrees_dir / 'root.lock'
    lock_path.write_text('held')
    with pytest.raises(StoreError, match='locked by another operation'):
      store.create_worktree('root')

  def test_worktree_creation_succeeds(self, tmp_path: Path) -> None:
    store, _src = _make_store(tmp_path, {'f.txt': 'hello'})
    store.snapshot('root', 0)
    wt_path = store.create_worktree('root')
    assert wt_path.exists()
    refs = store.load_refs()
    assert 'root' in refs.get('worktrees', {})


# 2.2: reraise_as_store_error helper


class TestReraiseAsStoreError:
  """Verify reraise_as_store_error wraps TrackingError into StoreError."""

  def test_wraps_with_message(self) -> None:
    original = TrackingError('disk full')
    with pytest.raises(StoreError, match='custom message') as exc_info:
      reraise_as_store_error(original, 'custom message')
    assert exc_info.value.__cause__ is original

  def test_preserves_exception_chain(self) -> None:
    original = TrackingError('io error')
    with pytest.raises(StoreError) as exc_info:
      reraise_as_store_error(original, str(original))
    assert isinstance(exc_info.value.__cause__, TrackingError)
    assert 'io error' in str(exc_info.value.__cause__)

  def test_store_load_refs_wraps_tracking_error(self, tmp_path: Path) -> None:
    """load_refs with corrupt JSON raises StoreError, not TrackingError."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.autopilot'
    store = FileStore(config)
    refs_file = config.refs_file
    refs_file.parent.mkdir(parents=True, exist_ok=True)
    refs_file.write_text('not valid json', encoding='utf-8')
    with pytest.raises(StoreError, match='failed to load refs'):
      store.load_refs()

  def test_store_load_snapshot_wraps_tracking_error(self, tmp_path: Path) -> None:
    """load_snapshot with corrupt JSON raises StoreError."""
    store, _src = _make_store(tmp_path, {'f.txt': 'base'})
    store.snapshot('root', 0)
    snap_path = config_snapshots_path(store) / 'root' / 'epoch_0.json'
    snap_path.write_text('broken json', encoding='utf-8')
    with pytest.raises(StoreError):
      store.load_snapshot('root', 0)


def config_snapshots_path(store: FileStore) -> Path:
  """Get snapshots directory from store internals."""
  return store._snapshots_dir


# 2.3: _apply_single_resolution helper


class TestApplySingleResolution:
  """Verify extracted resolution helper works in isolation."""

  def test_resolves_single_key(self, tmp_path: Path) -> None:
    store, _src = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)

    text_data = b'text content'
    text_digest = hash_bytes(text_data)
    store._store_object_bytes(text_digest, text_data)
    text_entry = FileEntry(digest=text_digest, size=len(text_data), mtime=0.0)

    conflict = ConflictEntry(
      key='source/main.py',
      ancestor=text_entry,
      ours=text_entry,
      theirs=text_entry,
    )
    index = MergeIndex(
      conflicts={'source/main.py': conflict},
      resolved={},
      preview_token='tok',
    )

    agent = MergeAgent(StubAgent(''), store)
    agent._apply_single_resolution(index, 'source/main.py', 'resolved text')

    assert 'source/main.py' not in index.conflicts
    assert 'source/main.py' in index.resolved
    expected_digest = hash_content('resolved text')
    assert index.resolved['source/main.py'].digest == expected_digest

  def test_multiple_resolutions_via_helper(self, tmp_path: Path) -> None:
    store, _src = _make_store(tmp_path, {'a.py': 'a', 'b.py': 'b'})
    store.snapshot('root', 0)

    def _entry(text: str) -> FileEntry:
      data = text.encode('utf-8')
      digest = hash_bytes(data)
      store._store_object_bytes(digest, data)
      return FileEntry(digest=digest, size=len(data), mtime=0.0)

    conflicts = {
      'source/a.py': ConflictEntry(
        key='source/a.py',
        ancestor=_entry('base a'),
        ours=_entry('ours a'),
        theirs=_entry('theirs a'),
      ),
      'source/b.py': ConflictEntry(
        key='source/b.py',
        ancestor=_entry('base b'),
        ours=_entry('ours b'),
        theirs=_entry('theirs b'),
      ),
    }
    index = MergeIndex(conflicts=conflicts, resolved={}, preview_token='tok')
    agent = MergeAgent(StubAgent(''), store)

    agent._apply_single_resolution(index, 'source/a.py', 'resolved a')
    agent._apply_single_resolution(index, 'source/b.py', 'resolved b')

    assert index.is_resolved()
    assert index.resolved['source/a.py'].digest == hash_content('resolved a')
    assert index.resolved['source/b.py'].digest == hash_content('resolved b')

  def test_parity_with_resolve_conflicts(self, tmp_path: Path) -> None:
    """resolve_conflicts and apply_resolution produce identical results."""
    store, src = _make_store(tmp_path, {'main.py': 'base\n'})
    store.snapshot('root', 0)
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'main.py').write_text('ours\n')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'main.py').write_text('theirs\n')
    store.snapshot('exp-b', 1)

    index = store.merge_preview('exp-a', 'exp-b')
    conflict_keys = sorted(index.conflicts)
    response_parts = [f'RESOLVED: {key}\n```\nmerged\n```' for key in conflict_keys]
    response = '\n\n'.join(response_parts)

    agent = MergeAgent(StubAgent(response), store)
    result = agent.resolve_conflicts(index)

    for key in conflict_keys:
      assert key in result.resolved
      assert result.resolved[key].digest == hash_content('merged')


# 2.4: MergeClassification in merge_analysis


class TestMergeAnalysisResultEnumType:
  """MergeAnalysisResult.classification is MergeClassification."""

  def test_field_type_is_enum(self) -> None:
    result = MergeAnalysisResult(
      can_fast_forward=True,
      has_conflicts=False,
      conflict_count=0,
      classification=MergeClassification.fast_forward,
    )
    assert isinstance(result.classification, MergeClassification)

  def test_merge_analysis_result_round_trip_with_enum(self) -> None:
    """Build with enum, to_dict stores string, from_dict reconstructs enum."""
    result = MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=True,
      conflict_count=2,
      ancestor_epoch=1,
      classification=MergeClassification.conflict,
    )
    data = result.to_dict()
    assert data['classification'] == 'conflict'
    assert isinstance(data['classification'], str)

    restored = MergeAnalysisResult.from_dict(data)
    assert isinstance(restored.classification, MergeClassification)
    assert restored.classification is MergeClassification.conflict

  def test_merge_classification_json_serialization(self) -> None:
    """json.dumps/loads round-trip preserves classification as wire string."""
    payload = {
      'classification': MergeClassification.clean.value,
      'can_fast_forward': False,
      'has_conflicts': False,
      'conflict_count': 0,
    }
    serialized = json.dumps(payload)
    deserialized = json.loads(serialized)
    assert deserialized['classification'] == 'clean'
    restored = MergeAnalysisResult.from_dict(deserialized)
    assert restored.classification is MergeClassification.clean

  def test_merge_analysis_returns_enum_members(self, tmp_path: Path) -> None:
    """FileStore.merge_analysis returns MergeClassification enum values."""
    store, src = _make_store(tmp_path, {'f.txt': 'base'})
    store.snapshot('root', 0)
    store.branch('feature')
    store.checkout('feature', 0)
    (src / 'f.txt').write_text('advanced')
    store.snapshot('feature', 1)
    result = store.merge_analysis('root', 'feature')
    assert isinstance(result.classification, MergeClassification)
    assert result.classification is MergeClassification.fast_forward

  def test_all_classifications_round_trip(self) -> None:
    """Every MergeClassification member round-trips through to_dict/from_dict."""
    for member in MergeClassification:
      result = MergeAnalysisResult(
        can_fast_forward=False,
        has_conflicts=False,
        conflict_count=0,
        classification=member,
      )
      data = result.to_dict()
      assert data['classification'] == member.value
      restored = MergeAnalysisResult.from_dict(data)
      assert restored.classification is member

  def test_legacy_unknown_string_raises_store_error(self) -> None:
    """Legacy 'unknown' classification in stored data raises StoreError."""
    data = {
      'can_fast_forward': False,
      'has_conflicts': False,
      'conflict_count': 0,
      'classification': 'unknown',
    }
    with pytest.raises(StoreError, match='unknown merge classification'):
      MergeAnalysisResult.from_dict(data)

  def test_missing_classification_defaults_to_up_to_date(self) -> None:
    """Missing classification key in dict defaults to up_to_date."""
    data = {
      'can_fast_forward': True,
      'has_conflicts': False,
      'conflict_count': 0,
    }
    restored = MergeAnalysisResult.from_dict(data)
    assert restored.classification is MergeClassification.up_to_date
