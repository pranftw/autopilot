"""Tests for MergeAgent: prompt building, conflict resolution, and apply."""

from autopilot.ai.agents.agent import Agent
from autopilot.ai.merge_agent import MergeAgent
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store_lock import hash_bytes, hash_content
from autopilot.core.snapshot import FileEntry
from autopilot.core.store.base import Store
from autopilot.core.store.types import ConflictEntry, MergeIndex, MergeStrategy
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
from unittest.mock import MagicMock
import pytest


class StubAgent(Agent):
  """Agent stub that returns a canned response."""

  def __init__(self, response: str) -> None:
    super().__init__()
    self._response = response

  def run(self, *args, **kwargs):
    return self._response


class FailAgent(Agent):
  """Agent stub that raises on run()."""

  def __init__(self, exc: Exception) -> None:
    super().__init__()
    self._exc = exc

  def run(self, *args, **kwargs):
    raise self._exc


def _make_store(tmp_path: Path, files: dict[str, str]) -> tuple[FileStore, Path]:
  src = make_source_dir(tmp_path, files=files)
  config = make_store_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  return store, src


def _make_conflict_index(
  store: FileStore,
  key: str = 'source/main.py',
  ancestor_text: str | None = 'base',
  ours_text: str | None = 'ours version',
  theirs_text: str | None = 'theirs version',
) -> MergeIndex:
  """Build a MergeIndex with one conflict using real store blobs."""

  def _store_text(text: str | None) -> FileEntry | None:
    if text is None:
      return None
    data = text.encode('utf-8')
    digest = hash_bytes(data)
    store._store_object_bytes(digest, data)
    return FileEntry(digest=digest, size=len(data), mtime=0.0)

  ancestor = _store_text(ancestor_text)
  ours = _store_text(ours_text)
  theirs = _store_text(theirs_text)

  conflict = ConflictEntry(key=key, ancestor=ancestor, ours=ours, theirs=theirs)
  return MergeIndex(
    conflicts={key: conflict},
    resolved={},
    experiment_id='exp-a',
    source_experiment_id='exp-b',
    strategy=MergeStrategy.normal,
    preview_token='test-token',
  )


class TestBuildResolutionPrompt:
  def test_formats_single_conflict(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)
    index = _make_conflict_index(store)
    agent = MergeAgent(StubAgent(''), store)
    prompt = agent.build_resolution_prompt(index)

    assert 'CONFLICT: source/main.py' in prompt
    assert '--- BASE ---' in prompt
    assert '--- OURS ---' in prompt
    assert '--- THEIRS ---' in prompt
    assert 'base' in prompt
    assert 'ours version' in prompt
    assert 'theirs version' in prompt

  def test_deleted_side_renders_deleted_token(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)
    index = _make_conflict_index(store, ours_text=None)
    agent = MergeAgent(StubAgent(''), store)
    prompt = agent.build_resolution_prompt(index)

    assert '(deleted)' in prompt
    assert 'theirs version' in prompt

  def test_multiple_conflicts_ordering(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)

    def _entry(text: str) -> FileEntry:
      data = text.encode('utf-8')
      digest = hash_bytes(data)
      store._store_object_bytes(digest, data)
      return FileEntry(digest=digest, size=len(data), mtime=0.0)

    conflicts = {
      'z_file': ConflictEntry(
        key='z_file',
        ancestor=_entry('a'),
        ours=_entry('b'),
        theirs=_entry('c'),
      ),
      'a_file': ConflictEntry(
        key='a_file',
        ancestor=_entry('x'),
        ours=_entry('y'),
        theirs=_entry('z'),
      ),
    }
    index = MergeIndex(conflicts=conflicts, resolved={}, preview_token='tok')
    agent = MergeAgent(StubAgent(''), store)
    prompt = agent.build_resolution_prompt(index)

    pos_a = prompt.index('CONFLICT: a_file')
    pos_z = prompt.index('CONFLICT: z_file')
    assert pos_a < pos_z

  def test_binary_side_non_utf8_sentinel(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)

    binary_data = b'\x80\x81\x82\xff'
    digest = hash_bytes(binary_data)
    store._store_object_bytes(digest, binary_data)
    binary_entry = FileEntry(digest=digest, size=len(binary_data), mtime=0.0)

    text_data = b'text content'
    text_digest = hash_bytes(text_data)
    store._store_object_bytes(text_digest, text_data)
    text_entry = FileEntry(digest=text_digest, size=len(text_data), mtime=0.0)

    conflict = ConflictEntry(
      key='binary_key',
      ancestor=binary_entry,
      ours=text_entry,
      theirs=binary_entry,
    )
    index = MergeIndex(conflicts={'binary_key': conflict}, resolved={}, preview_token='tok')
    agent = MergeAgent(StubAgent(''), store)
    prompt = agent.build_resolution_prompt(index)

    assert 'binary blob' in prompt
    assert 'cannot display' in prompt
    assert 'text content' in prompt

  def test_missing_object_side_renders_missing_sentinel(self, tmp_path: Path) -> None:
    """StoreError from read_object renders as missing, not binary."""
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)

    missing_entry = FileEntry(digest='nonexistent_hash', size=42, mtime=0.0)
    text_data = b'text content'
    text_digest = hash_bytes(text_data)
    store._store_object_bytes(text_digest, text_data)
    text_entry = FileEntry(digest=text_digest, size=len(text_data), mtime=0.0)

    conflict = ConflictEntry(
      key='missing_key',
      ancestor=missing_entry,
      ours=text_entry,
      theirs=text_entry,
    )
    index = MergeIndex(
      conflicts={'missing_key': conflict},
      resolved={},
      preview_token='tok',
    )
    agent = MergeAgent(StubAgent(''), store)
    prompt = agent.build_resolution_prompt(index)

    assert 'missing object' in prompt
    assert 'not in store' in prompt
    assert 'binary blob' not in prompt


class TestResolveConflicts:
  def test_mock_agent_parses_and_resolves(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)
    index = _make_conflict_index(store, key='source/main.py')

    response = 'RESOLVED: source/main.py\n```\nmerged content\n```\n'
    agent = MergeAgent(StubAgent(response), store)
    result = agent.resolve_conflicts(index)

    assert result.is_resolved()
    resolved_entry = result.resolved['source/main.py']
    expected_digest = hash_content('merged content')
    assert resolved_entry.digest == expected_digest

  def test_malformed_response_errors(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)
    index = _make_conflict_index(store)

    agent = MergeAgent(StubAgent('just some random text'), store)
    with pytest.raises(ValueError, match='no RESOLVED: headers'):
      agent.resolve_conflicts(index)

  def test_no_conflicts_noop(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)
    index = MergeIndex(conflicts={}, resolved={}, preview_token='tok')

    mock_agent = MagicMock(spec=Agent)
    agent = MergeAgent(mock_agent, store)
    result = agent.resolve_conflicts(index)

    assert result.is_resolved()
    mock_agent.run.assert_not_called()

  def test_agent_run_raises_network_error_propagates(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)
    index = _make_conflict_index(store)

    agent = MergeAgent(FailAgent(ConnectionError('network down')), store)
    with pytest.raises(ConnectionError, match='network down'):
      agent.resolve_conflicts(index)

  def test_missing_key_in_response_errors(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)
    index = _make_conflict_index(store, key='source/main.py')

    response = 'RESOLVED: wrong_key\n```\ncontent\n```\n'
    agent = MergeAgent(StubAgent(response), store)
    with pytest.raises(ValueError, match='missing keys'):
      agent.resolve_conflicts(index)

  def test_binary_conflict_rejected(self, tmp_path: Path) -> None:
    """resolve_conflicts rejects keys with binary/non-UTF-8 sides."""
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)

    binary_data = b'\x80\x81\x82\xff'
    digest = hash_bytes(binary_data)
    store._store_object_bytes(digest, binary_data)
    binary_entry = FileEntry(digest=digest, size=len(binary_data), mtime=0.0)

    text_data = b'text content'
    text_digest = hash_bytes(text_data)
    store._store_object_bytes(text_digest, text_data)
    text_entry = FileEntry(digest=text_digest, size=len(text_data), mtime=0.0)

    conflict = ConflictEntry(
      key='binary_file',
      ancestor=text_entry,
      ours=binary_entry,
      theirs=text_entry,
    )
    index = MergeIndex(
      conflicts={'binary_file': conflict},
      resolved={},
      preview_token='tok',
    )
    agent = MergeAgent(StubAgent(''), store)
    with pytest.raises(ValueError, match='cannot agent-resolve binary'):
      agent.resolve_conflicts(index)


class TestStoreBlobDelegation:
  def test_store_blob_delegates_to_store(self) -> None:
    """_store_blob calls store.store_blob directly."""
    mock_store = MagicMock(spec=Store)
    agent = MergeAgent(StubAgent(''), mock_store)
    agent._store_blob('abc123', b'data')
    mock_store.store_blob.assert_called_once_with('abc123', b'data')


class TestApplyResolution:
  def test_writes_blobs_and_calls_merge_apply(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path, {'main.py': 'base\n'})
    store.snapshot('root', 0)
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'main.py').write_text('ours change\n')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'main.py').write_text('theirs change\n')
    store.snapshot('exp-b', 1)

    merge_index = store.merge_preview('exp-a', 'exp-b')
    agent = MergeAgent(StubAgent(''), store)
    resolutions = dict.fromkeys(merge_index.conflicts, 'resolved content')
    manifest = agent.apply_resolution(merge_index, resolutions)

    assert manifest.epoch == 2
    assert len(manifest.entries) > 0

  def test_apply_resolution_calls_merge_apply_without_epoch_kwarg(self) -> None:
    """apply_resolution delegates to merge_apply without an epoch parameter."""
    import inspect

    sig = inspect.signature(MergeAgent.apply_resolution)
    params = list(sig.parameters)
    assert 'epoch' not in params
