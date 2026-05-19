"""Store UX improvement tests (sub-plan 03).

Tests for:
  - ``store log`` without ``--source`` (forest inference)
  - Union merge concatenation documentation in help text
  - Checkpoint filename ordering determinism
  - IsolatedEnvironment empty snapshot behavior
"""

from autopilot.ai.environment import IsolatedEnvironment
from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.checkpoint import JSONCheckpointIO
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from typing import Any
import argparse
import contextlib
import io
import json
import pytest


def _run_cli_expect_failure(workspace: Path, argv: list[str]) -> dict[str, Any]:
  """Run CLI with --json expecting a SystemExit, and return the JSON error envelope.

  Args:
    workspace: Workspace root directory.
    argv: CLI argument tokens.

  Returns:
    Parsed JSON envelope from captured stdout.
  """
  parser = build_parser()
  full_argv = [*list(argv), '--workspace', str(workspace), '--json']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)
  buf = io.StringIO()
  with pytest.raises(SystemExit), contextlib.redirect_stdout(buf):
    parsed.handler(ctx, parsed)
  output = buf.getvalue().strip()
  assert output, 'expected JSON error envelope on stdout'
  return json.loads(output)


def _build_workspace_with_store_snapshot(
  tmp_path: Path,
) -> tuple[Path, FileStore, FileForest]:
  """Build a workspace with a forest, tree, experiment, and one store snapshot.

  Returns:
    Tuple of (workspace_path, store, forest).
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  src_dir = ws / 'src'
  src_dir.mkdir()
  (src_dir / 'main.py').write_text('print("hello")', encoding='utf-8')

  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)

  param = PathParameter(source=str(src_dir), pattern='**/*.py')
  store = FileStore(config)
  store.register_parameters({'source': param})

  forest = FileForest(store)
  tree = forest.create_tree('main')
  exp = Experiment(experiment_id='exp-001', hypothesis='test')
  exp.start()
  exp.complete(metrics={'score': 0.9})
  tree.add(Node(experiment=exp))
  tree.head = 'exp-001'
  forest.switch('main')
  forest.save()

  store.snapshot('exp-001', 0)
  return ws, store, forest


class TestStoreLogWithoutSource:
  """store log inference when --source is omitted."""

  def test_store_log_without_source_uses_active_experiment(self, tmp_path: Path) -> None:
    """store log without --source succeeds when active experiment exists."""
    ws, _store, _forest = _build_workspace_with_store_snapshot(tmp_path)
    result = run_cli_no_context(
      ws,
      ['--experiment', 'exp-001', 'store', 'log'],
    )
    assert result['ok'] is True
    assert result['result']['count'] == 1

  def test_store_log_with_explicit_source(self, tmp_path: Path) -> None:
    """store log with explicit --source still works as before."""
    ws, _store, _forest = _build_workspace_with_store_snapshot(tmp_path)
    src_dir = ws / 'src'
    result = run_cli_no_context(
      ws,
      [
        '--experiment',
        'exp-001',
        'store',
        'log',
        '--source',
        str(src_dir),
      ],
    )
    assert result['ok'] is True
    assert result['result']['count'] == 1

  def test_store_log_without_source_no_active_fails(self, tmp_path: Path) -> None:
    """store log without --source fails with exact error substring."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.save()

    envelope = _run_cli_expect_failure(ws, ['--experiment', 'exp-missing', 'store', 'log'])
    assert envelope['ok'] is False
    assert 'No --source provided and no active experiment in forest' in envelope['error']


class TestUnionMergeDocumented:
  """Union merge concatenation semantics in merge-preview help text."""

  def test_union_merge_concatenation_documented(self) -> None:
    """merge-preview help contains the union merge canonical sentence."""
    parser = build_parser()
    store_parser = None
    for action in parser._actions:
      if isinstance(action, argparse._SubParsersAction):
        store_parser = action.choices.get('store')
        break
    assert store_parser is not None, 'store subcommand not found'

    store_help = ' '.join(store_parser.format_help().split())
    assert 'Union strategy concatenates' in store_help


class TestCheckpointOrdering:
  """Checkpoint filename zero-padding preserves lexicographic sort order."""

  def test_checkpoint_ordering_deterministic(self, tmp_path: Path) -> None:
    """12 epochs produce filenames that sort lexicographically in epoch order."""
    ckpt_dir = tmp_path / 'checkpoints'
    ckpt_dir.mkdir()
    checkpoint_io = JSONCheckpointIO()

    epoch_count = 12
    for epoch in range(epoch_count):
      path = ckpt_dir / f'epoch-{epoch:04d}.json'
      checkpoint_io.save({'epoch': epoch}, path)

    filenames = sorted(p.name for p in ckpt_dir.iterdir())
    expected = [f'epoch-{i:04d}.json' for i in range(epoch_count)]
    assert filenames == expected

    for idx, name in enumerate(filenames):
      epoch_str = name.replace('epoch-', '').replace('.json', '')
      assert int(epoch_str) == idx


class TestIsolatedEnvironmentNoSnapshot:
  """IsolatedEnvironment with no snapshot returns empty content."""

  def test_isolated_env_no_snapshot_returns_empty(self, tmp_path: Path) -> None:
    """Branch at latest_epoch=-1 without parent snapshot yields empty content."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    project_root = ws / 'project'
    project_root.mkdir()
    src_dir = project_root / 'src'
    src_dir.mkdir()
    (src_dir / 'main.py').write_text('original', encoding='utf-8')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.root = project_root

    store = FileStore(config)

    refs = {
      'HEAD': 'fresh-exp',
      'branches': {
        'fresh-exp': {
          'latest_epoch': -1,
        },
      },
    }
    store.save_refs(refs)

    env = IsolatedEnvironment(
      config=config,
      ignore_patterns=('.git', '__pycache__'),
      symlink_as_unit=(),
      core_files=(),
    )
    env._store = store

    content = env._get_snapshot_content('fresh-exp')
    assert content == {}
