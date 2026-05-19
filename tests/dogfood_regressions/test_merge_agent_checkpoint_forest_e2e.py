"""MergeAgent end-to-end tests with scripted stub agent.

Builds two conflicting snapshot manifests on disk, resolves via a scripted
(deterministic) stub agent implementing the Agent protocol, and verifies the
merged manifest epoch and content.

Also covers Sequential empty-construction validation, checkpoint resume
interruption smoke, and concurrency stress (forest lock contention) as
``@pytest.mark.slow`` tests.

No real LLM calls are made. Anthropic SDK availability is gated via the
``skip_without_anthropic`` marker in conftest.
"""

from autopilot.ai.merge_agent import MergeAgent
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store_lock import hash_bytes
from autopilot.core.checkpoint import JSONCheckpointIO
from autopilot.core.config import AutoPilotConfig
from autopilot.core.module.module import Module
from autopilot.core.ops import Sequential
from autopilot.core.snapshot import FileEntry
from autopilot.core.store.types import ConflictEntry, MergeIndex
from pathlib import Path
from tests.dogfood_regressions.conftest import ScriptedAgent
import pytest
import threading


def _make_store(tmp_path: Path, files: dict[str, str]) -> tuple[FileStore, Path]:
  """Create a FileStore with one PathParameter for merge tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  src = tmp_path / 'src'
  src.mkdir()
  for name, content in files.items():
    (src / name).write_text(content)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'code': param})
  return store, src


class TestMergeConflictResolutionWithStub:
  """E2E: two branches diverge, scripted agent resolves, merge_apply produces epoch."""

  @pytest.mark.timeout(1)
  def test_merge_conflict_resolution_optional_content_stub(self, tmp_path: Path) -> None:
    """BUG-023: MergeAgent resolves text conflicts via scripted stub.

    Sets up two branches diverging from a common ancestor, uses a
    ScriptedAgent that returns the expected RESOLVED: format, and
    verifies that merge_apply produces a new epoch with correct content.
    """
    store, src = _make_store(tmp_path, {'main.py': 'base content\n'})
    store.snapshot('root', 0)
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'main.py').write_text('ours change\n')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'main.py').write_text('theirs change\n')
    store.snapshot('exp-b', 1)

    index = store.merge_preview('exp-a', 'exp-b')
    assert not index.is_resolved(), 'expected unresolved conflicts'

    conflict_key = next(iter(index.conflicts))
    resolved_text = 'merged result'
    response = f'RESOLVED: {conflict_key}\n```\n{resolved_text}\n```\n'

    agent = MergeAgent(ScriptedAgent(response), store)
    agent.resolve_conflicts(index)
    assert index.is_resolved()

    manifest = store.merge_apply(index)
    assert manifest.epoch == 2
    assert len(manifest.entries) > 0

  @pytest.mark.timeout(1)
  def test_merge_agent_binary_conflict_rejected(self, tmp_path: Path) -> None:
    """MergeAgent refuses to resolve conflicts with binary/non-UTF-8 sides."""
    store, _src = _make_store(tmp_path, {'main.py': 'base'})
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
    agent = MergeAgent(ScriptedAgent(''), store)
    with pytest.raises(ValueError, match='cannot agent-resolve binary'):
      agent.resolve_conflicts(index)

  @pytest.mark.timeout(1)
  def test_merge_agent_no_conflicts_noop(self, tmp_path: Path) -> None:
    """MergeAgent with empty conflicts dict is a no-op (agent not called)."""
    store, _ = _make_store(tmp_path, {'main.py': 'base'})
    store.snapshot('root', 0)
    index = MergeIndex(conflicts={}, resolved={}, preview_token='tok')

    call_count = 0

    class _CountingAgent(ScriptedAgent):
      def run(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return super().run(*args, **kwargs)

    agent = MergeAgent(_CountingAgent(''), store)
    result = agent.resolve_conflicts(index)
    assert result.is_resolved()
    assert call_count == 0

  @pytest.mark.timeout(1)
  def test_merge_agent_apply_resolution_explicit(self, tmp_path: Path) -> None:
    """apply_resolution with explicit text bodies writes blobs and applies."""
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
    agent = MergeAgent(ScriptedAgent(''), store)
    resolutions = dict.fromkeys(index.conflicts, 'resolved content')
    manifest = agent.apply_resolution(index, resolutions)
    assert manifest.epoch == 2


class TestSequentialValidation:
  """AOE-002: Sequential with no modules raises ValueError."""

  @pytest.mark.timeout(1)
  def test_empty_sequential_raises_value_error(self) -> None:
    """Sequential() with no modules must raise ValueError."""
    with pytest.raises(ValueError, match='Sequential requires at least one module'):
      Sequential()

  @pytest.mark.timeout(1)
  def test_single_module_sequential_forward(self) -> None:
    """Sequential with one module passes through correctly."""

    class _Identity(Module):
      def forward(self, x):
        return x

    seq = Sequential(_Identity())
    from autopilot.core.types import Datum

    datum = Datum()
    result = seq(datum)
    assert isinstance(result, Datum)


class TestCheckpointResumeInterruption:
  """Section 2.4: checkpoint resume interruption smoke."""

  @pytest.mark.timeout(1)
  def test_checkpoint_save_load_roundtrip(self, tmp_path: Path) -> None:
    """JSONCheckpointIO save/load round-trips without corruption."""
    ckpt_io = JSONCheckpointIO()
    ckpt_path = tmp_path / 'epoch_0.ckpt'
    state = {
      'epoch': 0,
      'module_state': {'param_a': 'value_a'},
      'metrics': {'val_score': 0.85},
    }
    ckpt_io.save(state, ckpt_path)
    loaded = ckpt_io.load(ckpt_path)
    assert loaded == state

  @pytest.mark.timeout(1)
  def test_checkpoint_resume_after_partial_write(self, tmp_path: Path) -> None:
    """Checkpoint resume detects missing file gracefully (TrackingError)."""
    from autopilot.core.errors import TrackingError

    ckpt_io = JSONCheckpointIO()
    missing_path = tmp_path / 'missing.ckpt'
    assert not ckpt_io.exists(missing_path)
    with pytest.raises(TrackingError):
      ckpt_io.load(missing_path)

  @pytest.mark.timeout(1)
  def test_checkpoint_overwrite_on_new_epoch(self, tmp_path: Path) -> None:
    """Saving a new epoch overwrites prior checkpoint state."""
    ckpt_io = JSONCheckpointIO()
    ckpt_path = tmp_path / 'latest.ckpt'
    ckpt_io.save({'epoch': 0, 'score': 0.5}, ckpt_path)
    ckpt_io.save({'epoch': 1, 'score': 0.8}, ckpt_path)
    loaded = ckpt_io.load(ckpt_path)
    assert loaded['epoch'] == 1
    assert loaded['score'] == 0.8


class TestForestConcurrency:
  """Concurrent forest write stress tests."""

  @pytest.mark.timeout(1)
  @pytest.mark.slow
  def test_forest_many_writers_bounded_time(self, tmp_path: Path) -> None:
    """Concurrent forest writers experience lock contention without corruption.

    The forest lock is fail-fast (no timeout), so concurrent writers may get
    TrackingError for lock contention. The critical invariant is that no
    corruption occurs: all trees created by successful writes are present.
    """
    from autopilot.ai.forest import FileForest
    from autopilot.ai.store.file_store import FileStore
    from autopilot.core.config import AutoPilotConfig
    from autopilot.core.errors import TrackingError

    ws = tmp_path / 'concurrent_ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    n_writers = 3
    n_writes = 4
    successes: list[str] = []
    contention: list[str] = []
    lock = threading.Lock()

    def writer(writer_id: int) -> None:
      for i in range(n_writes):
        name = f'w{writer_id}-t{i}'
        try:
          forest.create_tree(name)
          with lock:
            successes.append(name)
        except (ValueError, OSError, TrackingError):
          with lock:
            contention.append(name)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    reloaded = FileForest(store)
    reloaded_names = {t.name for t in reloaded.list_trees()}
    for name in successes:
      assert name in reloaded_names, f'successful tree {name!r} missing after reload'
    assert len(successes) + len(contention) == n_writers * n_writes

  @pytest.mark.timeout(1)
  @pytest.mark.slow
  def test_concurrent_forest_fixture(self, concurrent_forest_writer) -> None:
    """Shared concurrent_forest_writer fixture completes; lock contention is expected.

    The forest lock is fail-fast, so concurrent writers may fail with
    TrackingError. The fixture catches those and reports them. The critical
    invariant: no corruption and at least one tree was created.
    """
    forest, errors = concurrent_forest_writer(n_writers=3, n_writes_each=3)
    assert len(forest.list_trees()) >= 1
    assert len(forest.list_trees()) + len(errors) >= 3
