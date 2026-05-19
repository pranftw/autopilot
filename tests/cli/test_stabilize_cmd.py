"""Tests for stabilize CLI command.

Covers:
  1. Stabilize with completed experiment in forest -- status found correctly
  2. Stabilize non-completed experiment -- error + exit 1
  3. Stabilize nonexistent experiment -- error + exit 1
  4. Stabilize with correct file paths -- files land at project-relative locations
  5. Stabilize with --json on error -- JSON envelope with error
  6. End-to-end: FileStore + snapshot + stabilize -> verify file content at correct path
  7. Invalid forest.json raises StoreError via FileStore.load_state_dict
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.stabilize import StabilizeCommand
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.parameter import Parameter
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
from unittest.mock import MagicMock
import json
import pytest


def _make_ctx(tmp_path: Path, use_json: bool = True) -> MagicMock:
  return make_mock_cli_context(tmp_path, use_json=use_json)


def _args(
  experiment_id: str = 'exp-001',
  parameter_prefix: str | None = None,
) -> MagicMock:
  args = MagicMock()
  args.experiment_id = experiment_id
  args.parameter_prefix = parameter_prefix
  return args


def _seed_forest_with_experiment(tmp_path: Path, experiment_id: str, status: str) -> None:
  """Seed forest.json with a properly structured forest containing one experiment."""
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id=experiment_id, hypothesis='test')
  if status in {'running', 'completed', 'failed', 'cancelled'}:
    exp.start()
  if status == 'completed':
    exp.complete(metrics={})
  elif status == 'failed':
    exp.fail(error='test failure')
  elif status == 'cancelled':
    exp.cancel()
  tree.add(Node(experiment=exp))
  forest.switch('main')
  forest.save()


class TestStabilizeCommand:
  def test_instantiates(self):
    cmd = StabilizeCommand()
    assert cmd.name == 'stabilize'

  def test_completed_experiment_found_in_forest(self, tmp_path: Path, capsys):
    """Completed experiment in forest is found via Forest.find_experiment."""
    _seed_forest_with_experiment(tmp_path, 'exp-001', 'completed')
    ctx = _make_ctx(tmp_path)

    cmd = StabilizeCommand()
    cmd.forward(ctx, _args('exp-001'))

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['copied'] == []

  def test_nonexistent_experiment_error_and_exit_1(self, tmp_path: Path):
    """Nonexistent experiment produces error and exits with code 1."""
    ctx = _make_ctx(tmp_path)
    ctx.output = MagicMock()
    cmd = StabilizeCommand()
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, _args('nonexistent'))
    assert exc_info.value.code == 1
    ctx.output.error.assert_called_once()
    assert 'not found' in ctx.output.error.call_args[0][0]

  def test_non_completed_experiment_error_and_exit_1(self, tmp_path: Path):
    """Non-completed experiment produces error and exits with code 1."""
    _seed_forest_with_experiment(tmp_path, 'exp-001', 'running')
    ctx = _make_ctx(tmp_path)
    ctx.output = MagicMock()
    cmd = StabilizeCommand()
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, _args('exp-001'))
    assert exc_info.value.code == 1
    ctx.output.error.assert_called_once()
    assert 'not completed' in ctx.output.error.call_args[0][0]

  def test_stabilize_copies_files_to_correct_paths(self, tmp_path: Path, capsys):
    """Stabilize uses original_path from manifest to place files at correct locations."""
    ctx = _make_ctx(tmp_path)
    config = ctx.config

    snapshots_dir = config.snapshots_path / 'exp-001'
    snapshots_dir.mkdir(parents=True)
    objects_dir = config.objects_path
    objects_dir.mkdir(parents=True)

    obj_hash = 'abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890'
    obj_dir = objects_dir / obj_hash[:2]
    obj_dir.mkdir()
    (obj_dir / obj_hash[2:]).write_text('param content')

    atomic_write_json(
      snapshots_dir / 'epoch_0.json',
      {
        'epoch': 0,
        'timestamp': '2024-01-01T00:00:00Z',
        'entries': {
          'param_0/prompt.txt': {
            'digest': obj_hash,
            'size': 13,
            'mtime': 0.0,
            'original_path': 'prompts/prompt.txt',
          },
        },
      },
    )

    cmd = StabilizeCommand()
    cmd.forward(ctx, _args('exp-001'))

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert len(envelope['result']['copied']) == 1
    assert (tmp_path / 'prompts' / 'prompt.txt').exists()
    assert (tmp_path / 'prompts' / 'prompt.txt').read_text() == 'param content'

  def test_stabilize_multiple_files_in_subdirs(self, tmp_path: Path, capsys):
    """Stabilize handles multiple files in nested directories."""
    ctx = _make_ctx(tmp_path)
    config = ctx.config

    snapshots_dir = config.snapshots_path / 'exp-001'
    snapshots_dir.mkdir(parents=True)
    objects_dir = config.objects_path
    objects_dir.mkdir(parents=True)

    hash_a = 'aa' + 'a1' * 31
    hash_b = 'bb' + 'b2' * 31
    (objects_dir / hash_a[:2]).mkdir()
    (objects_dir / hash_a[:2] / hash_a[2:]).write_text('file A')
    (objects_dir / hash_b[:2]).mkdir()
    (objects_dir / hash_b[:2] / hash_b[2:]).write_text('file B')

    atomic_write_json(
      snapshots_dir / 'epoch_0.json',
      {
        'epoch': 0,
        'timestamp': '2024-01-01T00:00:00Z',
        'entries': {
          'param_0/a.txt': {
            'digest': hash_a,
            'size': 6,
            'mtime': 0.0,
            'original_path': 'src/a.txt',
          },
          'param_0/sub/b.txt': {
            'digest': hash_b,
            'size': 6,
            'mtime': 0.0,
            'original_path': 'src/sub/b.txt',
          },
        },
      },
    )

    cmd = StabilizeCommand()
    cmd.forward(ctx, _args('exp-001'))

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert len(envelope['result']['copied']) == 2
    assert (tmp_path / 'src' / 'a.txt').read_text() == 'file A'
    assert (tmp_path / 'src' / 'sub' / 'b.txt').read_text() == 'file B'

  def test_stabilize_json_error_nonexistent(self, tmp_path: Path, capsys):
    """With --json and nonexistent experiment, JSON envelope with error is emitted."""
    ctx = _make_ctx(tmp_path, use_json=True)
    cmd = StabilizeCommand()
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, _args('no-such'))
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is False
    assert 'not found' in envelope['error']

  def test_stabilize_json_error_not_completed(self, tmp_path: Path, capsys):
    """With --json and non-completed experiment, JSON envelope with error is emitted."""
    _seed_forest_with_experiment(tmp_path, 'exp-001', 'failed')
    ctx = _make_ctx(tmp_path, use_json=True)
    cmd = StabilizeCommand()
    with pytest.raises(SystemExit) as exc_info:
      cmd.forward(ctx, _args('exp-001'))
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is False
    assert 'not completed' in envelope['error']

  def test_stabilize_missing_original_path_raises(self, tmp_path: Path):
    """Manifest entries without original_path raise KeyError."""
    config = AutoPilotConfig(workspace=tmp_path)
    snapshots_dir = config.snapshots_path / 'exp-001'
    snapshots_dir.mkdir(parents=True)
    config.objects_path.mkdir(parents=True)

    obj_hash = 'cc' + 'c3' * 31
    (config.objects_path / obj_hash[:2]).mkdir()
    (config.objects_path / obj_hash[:2] / obj_hash[2:]).write_text('content')

    atomic_write_json(
      snapshots_dir / 'epoch_0.json',
      {
        'epoch': 0,
        'timestamp': '2024-01-01T00:00:00Z',
        'entries': {
          'param_0/file.txt': {
            'digest': obj_hash,
            'size': 7,
            'mtime': 0.0,
          },
        },
      },
    )

    with pytest.raises(KeyError, match='original_path'):
      config.stabilize('exp-001')

  def test_empty_snapshot_copies_nothing(self, tmp_path: Path, capsys):
    ctx = _make_ctx(tmp_path)
    config = ctx.config

    snapshots_dir = config.snapshots_path / 'exp-001'
    snapshots_dir.mkdir(parents=True)
    atomic_write_json(
      snapshots_dir / 'epoch_0.json',
      {'epoch': 0, 'timestamp': '2024-01-01T00:00:00Z', 'entries': {}},
    )

    cmd = StabilizeCommand()
    cmd.forward(ctx, _args('exp-001'))

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['result']['copied'] == []

  def test_fallback_to_snapshots_dir(self, tmp_path: Path, capsys):
    """When forest JSON doesn't contain experiment, falls back to snapshots dir."""
    ctx = _make_ctx(tmp_path)
    config = ctx.config

    snapshots_dir = config.snapshots_path / 'exp-001'
    snapshots_dir.mkdir(parents=True)
    atomic_write_json(
      snapshots_dir / 'epoch_0.json',
      {'epoch': 0, 'timestamp': '2024-01-01T00:00:00Z', 'entries': {}},
    )

    cmd = StabilizeCommand()
    cmd.forward(ctx, _args('exp-001'))

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True

  def test_stabilize_uses_forest_find_experiment(self, tmp_path: Path, capsys):
    """Stabilize resolves experiment via Forest.find_experiment (plan 02)."""
    _seed_forest_with_experiment(tmp_path, 'exp-001', 'completed')
    ctx = _make_ctx(tmp_path)
    cmd = StabilizeCommand()
    cmd.forward(ctx, _args('exp-001'))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True

  def test_non_json_mode_output(self, tmp_path: Path, capsys):
    """Non-JSON mode prints human-readable output."""
    ctx = _make_ctx(tmp_path, use_json=False)
    config = ctx.config

    snapshots_dir = config.snapshots_path / 'exp-001'
    snapshots_dir.mkdir(parents=True)
    config.objects_path.mkdir(parents=True)

    obj_hash = 'dd' + 'd4' * 31
    (config.objects_path / obj_hash[:2]).mkdir()
    (config.objects_path / obj_hash[:2] / obj_hash[2:]).write_text('hello')

    atomic_write_json(
      snapshots_dir / 'epoch_0.json',
      {
        'epoch': 0,
        'timestamp': '2024-01-01T00:00:00Z',
        'entries': {
          'param_0/hello.txt': {
            'digest': obj_hash,
            'size': 5,
            'mtime': 0.0,
            'original_path': 'hello.txt',
          },
        },
      },
    )

    cmd = StabilizeCommand()
    cmd.forward(ctx, _args('exp-001'))

    captured = capsys.readouterr()
    assert 'Stabilized 1 file(s)' in captured.out
    assert (tmp_path / 'hello.txt').read_text() == 'hello'

  def test_all_status_values_reject_non_completed(self, tmp_path: Path):
    """All non-completed statuses are rejected."""
    for status in ['running', 'failed', 'cancelled']:
      ws = tmp_path / f'ws_{status}'
      ws.mkdir()
      _seed_forest_with_experiment(ws, 'exp-001', status)
      ctx = _make_ctx(ws)
      ctx.output = MagicMock()
      cmd = StabilizeCommand()
      with pytest.raises(SystemExit) as exc_info:
        cmd.forward(ctx, _args('exp-001'))
      assert exc_info.value.code == 1

  def test_resolve_experiment_invalid_forest_json_raises_store_error(self, tmp_path: Path):
    """Non-object JSON in forest.json raises StoreError via FileStore.load_state_dict."""
    ctx = _make_ctx(tmp_path)
    config = ctx.config
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text('[1, 2, 3]')

    cmd = StabilizeCommand()
    with pytest.raises(StoreError):
      cmd._resolve_experiment(ctx, 'exp-001')

  def test_stabilize_cross_tree_finds_experiment(self, tmp_path: Path, capsys):
    """Stabilize finds experiment on non-active tree via Forest.find_experiment."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    forest.create_tree('alpha')
    tree_beta = forest.create_tree('beta')
    exp = Experiment(experiment_id='exp-cross', hypothesis='cross-tree')
    exp.start()
    exp.complete(metrics={'f1': 0.9})
    tree_beta.add(Node(experiment=exp))
    forest.switch('alpha')
    forest.save()

    ctx = _make_ctx(tmp_path)
    cmd = StabilizeCommand()
    cmd.forward(ctx, _args('exp-cross'))
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True


class TestFileStoreSnapshotOriginalPath:
  """Test that FileStore._build_snapshot populates original_path."""

  def test_snapshot_includes_original_path(self, tmp_path: Path):
    """FileStore snapshot entries include original_path for PathParameter."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('hello world')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    manifest = store.snapshot('exp-001', 0)
    entry = manifest.entries['source/main.txt']
    assert entry.original_path == 'prompts/main.txt'

  def test_snapshot_nested_files_original_path(self, tmp_path: Path):
    """Nested files get correct original_path relative to workspace."""
    src_dir = tmp_path / 'src' / 'templates'
    src_dir.mkdir(parents=True)
    (src_dir / 'a.txt').write_text('alpha')
    (src_dir / 'b.txt').write_text('beta')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(src_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    manifest = store.snapshot('exp-001', 0)
    assert manifest.entries['source/a.txt'].original_path == 'src/templates/a.txt'
    assert manifest.entries['source/b.txt'].original_path == 'src/templates/b.txt'

  def test_snapshot_original_path_none_for_base_parameter(self, tmp_path: Path):
    """Base Parameter (no source) gets original_path=None."""
    config = AutoPilotConfig(workspace=tmp_path)
    param = Parameter()
    store = FileStore(config)
    store.register_parameters({'source': param})

    manifest = store.snapshot('exp-001', 0)
    assert len(manifest.entries) == 0


class TestEndToEndStabilize:
  """End-to-end: FileStore + snapshot + stabilize -> verify file content."""

  def test_full_round_trip(self, tmp_path: Path):
    """Create files, snapshot with FileStore, then stabilize to workspace."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'system.txt').write_text('You are a helpful assistant.')
    (prompts_dir / 'user.txt').write_text('Hello!')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    store.snapshot('exp-001', 0)

    (prompts_dir / 'system.txt').write_text('MODIFIED')
    (prompts_dir / 'user.txt').write_text('MODIFIED')

    copied = config.stabilize('exp-001')
    assert len(copied) == 2

    assert (tmp_path / 'prompts' / 'system.txt').read_text() == 'You are a helpful assistant.'
    assert (tmp_path / 'prompts' / 'user.txt').read_text() == 'Hello!'

  def test_stabilize_picks_latest_snapshot(self, tmp_path: Path):
    """Stabilize picks the latest epoch snapshot."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'file.txt').write_text('v1')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    store.snapshot('exp-001', 0)

    (prompts_dir / 'file.txt').write_text('v2')
    store.snapshot('exp-001', 1)

    (prompts_dir / 'file.txt').write_text('SHOULD NOT APPEAR')

    copied = config.stabilize('exp-001')
    assert len(copied) == 1
    assert (tmp_path / 'prompts' / 'file.txt').read_text() == 'v2'

  def test_stabilize_no_snapshots_returns_empty(self, tmp_path: Path):
    """Stabilize on experiment with no snapshots returns empty list."""
    config = AutoPilotConfig(workspace=tmp_path)
    result = config.stabilize('no-such-exp')
    assert result == []


class TestFileEntryOriginalPath:
  """Tests for FileEntry.original_path serialization round-trip."""

  def test_file_entry_with_original_path_to_dict(self):
    from autopilot.core.snapshot import FileEntry

    entry = FileEntry(digest='abc', size=10, mtime=1.0, original_path='prompts/file.txt')
    d = entry.to_dict()
    assert d['original_path'] == 'prompts/file.txt'

  def test_file_entry_without_original_path_to_dict(self):
    from autopilot.core.snapshot import FileEntry

    entry = FileEntry(digest='abc', size=10, mtime=1.0)
    d = entry.to_dict()
    assert d['original_path'] is None

  def test_file_entry_round_trip(self):
    from autopilot.core.snapshot import FileEntry

    original = FileEntry(digest='abc', size=10, mtime=1.0, original_path='src/file.py')
    restored = FileEntry.from_dict(original.to_dict())
    assert restored.original_path == 'src/file.py'
    assert restored.digest == 'abc'
    assert restored.size == 10

  def test_snapshot_manifest_round_trip_with_original_path(self):
    from autopilot.core.snapshot import FileEntry, SnapshotManifest

    manifest = SnapshotManifest(
      epoch=0,
      timestamp='2024-01-01T00:00:00Z',
      entries={
        'param_0/file.txt': FileEntry(
          digest='abc',
          size=10,
          mtime=1.0,
          original_path='dir/file.txt',
        ),
      },
    )
    d = manifest.to_dict()
    restored = SnapshotManifest.from_dict(d)
    assert restored.entries['param_0/file.txt'].original_path == 'dir/file.txt'
