"""Tests for the debug trend command.

Verifies metric trend analysis via the debug subcommand.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context


def _setup_workspace_with_experiments(
  tmp_path: Path,
  metrics_list: list[dict[str, float]],
  tree_name: str = 'main',
) -> Path:
  """Create workspace with completed experiments carrying given metrics."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree(tree_name)
  for i, metrics in enumerate(metrics_list):
    exp = Experiment(experiment_id=f'exp-{i:03d}', hypothesis=f'run {i}')
    exp.start()
    exp.complete(metrics=metrics)
    tree.add(Node(experiment=exp))
  forest.switch(tree_name)
  forest.save()
  return ws


def test_debug_trend_active_tree(tmp_path: Path):
  """debug trend on active tree returns trend result."""
  ws = _setup_workspace_with_experiments(
    tmp_path,
    [{'accuracy': 0.7}, {'accuracy': 0.8}, {'accuracy': 0.9}],
  )
  result = run_cli_no_context(ws, ['debug', 'trend', 'accuracy'])
  assert result['ok'] is True
  assert result['result']['trend'] is not None
  assert 'values' in result['result']['trend']


def test_debug_trend_all_trees(tmp_path: Path):
  """debug trend --all-trees returns trees dict keyed by tree name."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  for name in ['alpha', 'beta']:
    tree = forest.create_tree(name)
    for i in range(2):
      exp = Experiment(experiment_id=f'{name}-exp-{i}', hypothesis='h')
      exp.start()
      exp.complete(metrics={'score': 0.5 + i * 0.1})
      tree.add(Node(experiment=exp))

  forest.switch('alpha')
  forest.save()

  result = run_cli_no_context(ws, ['debug', 'trend', 'score', '--all-trees'])
  assert result['ok'] is True
  trees_result = result['result']['trees']
  assert 'alpha' in trees_result
  assert 'beta' in trees_result


def test_debug_trend_empty_tree(tmp_path: Path):
  """debug trend on empty tree returns trend: null."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('empty')
  forest.switch('empty')
  forest.save()

  result = run_cli_no_context(ws, ['debug', 'trend', 'accuracy'])
  assert result['ok'] is True
  assert result['result']['trend'] is None


def test_debug_trend_context_exempt(tmp_path: Path):
  """debug trend is context-exempt (read-only)."""
  ws = _setup_workspace_with_experiments(
    tmp_path,
    [{'acc': 0.5}],
  )
  result = run_cli_no_context(ws, ['debug', 'trend', 'acc'])
  assert result['ok'] is True
