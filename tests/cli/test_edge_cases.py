"""CLI edge case regression tests.

Covers: tree removal vs query --all-trees, NaN/Inf metric filter exclusion,
combined filter composition via CLI.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
import math


def _make_forest_with_trees(
  tmp_path: Path,
) -> tuple[FileForest, Path]:
  """Create a workspace with two trees, each containing completed experiments."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a = Experiment(experiment_id='exp-alpha', hypothesis='alpha')
  exp_a.start()
  exp_a.complete(metrics={'accuracy': 0.9})
  tree_a.add(Node(experiment=exp_a))

  tree_b = forest.create_tree('beta')
  exp_b = Experiment(experiment_id='exp-beta', hypothesis='beta')
  exp_b.start()
  exp_b.complete(metrics={'accuracy': 0.7})
  tree_b.add(Node(experiment=exp_b))

  forest.switch('alpha')
  forest.save()
  return forest, ws


def test_tree_remove_then_query_all_trees(tmp_path: Path) -> None:
  """After tree remove, query --all-trees omits removed tree's experiments."""
  _, ws = _make_forest_with_trees(tmp_path)

  result_before = run_cli_no_context(ws, ['query', '--all-trees'])
  assert result_before['ok']
  ids_before = {r['id'] for r in result_before['result']['experiments']}
  assert 'exp-alpha' in ids_before
  assert 'exp-beta' in ids_before

  run_cli(ws, ['tree', 'remove', 'beta'])

  result_after = run_cli_no_context(ws, ['query', '--all-trees'])
  assert result_after['ok']
  ids_after = {r['id'] for r in result_after['result']['experiments']}
  assert 'exp-alpha' in ids_after
  assert 'exp-beta' not in ids_after


def test_metric_filter_nan(tmp_path: Path) -> None:
  """NaN metric values are excluded from --metric-gt / --metric-lt filters."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_normal = Experiment(experiment_id='exp-normal', hypothesis='normal')
  exp_normal.start()
  exp_normal.complete(metrics={'score': 0.5})
  tree.add(Node(experiment=exp_normal))

  exp_nan = Experiment(experiment_id='exp-nan', hypothesis='nan')
  exp_nan.start()
  exp_nan.complete(metrics={'score': math.nan})
  tree.add(Node(experiment=exp_nan))

  forest.save()

  result = run_cli_no_context(ws, ['query', '--metric-gt', 'score:0.0'])
  assert result['ok']
  ids = {r['id'] for r in result['result']['experiments']}
  assert 'exp-normal' in ids
  assert 'exp-nan' not in ids

  result_lt = run_cli_no_context(ws, ['query', '--metric-lt', 'score:1.0'])
  assert result_lt['ok']
  ids_lt = {r['id'] for r in result_lt['result']['experiments']}
  assert 'exp-normal' in ids_lt
  assert 'exp-nan' not in ids_lt


def test_metric_filter_inf(tmp_path: Path) -> None:
  """Inf metric values are excluded from --metric-gt / --metric-lt filters."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_normal = Experiment(experiment_id='exp-ok', hypothesis='ok')
  exp_normal.start()
  exp_normal.complete(metrics={'score': 0.5})
  tree.add(Node(experiment=exp_normal))

  exp_inf = Experiment(experiment_id='exp-inf', hypothesis='inf')
  exp_inf.start()
  exp_inf.complete(metrics={'score': math.inf})
  tree.add(Node(experiment=exp_inf))

  exp_neginf = Experiment(experiment_id='exp-neginf', hypothesis='neginf')
  exp_neginf.start()
  exp_neginf.complete(metrics={'score': -math.inf})
  tree.add(Node(experiment=exp_neginf))

  forest.save()

  result = run_cli_no_context(ws, ['query', '--metric-gt', 'score:0.0'])
  assert result['ok']
  ids = {r['id'] for r in result['result']['experiments']}
  assert 'exp-ok' in ids
  assert 'exp-inf' not in ids
  assert 'exp-neginf' not in ids

  result_lt = run_cli_no_context(ws, ['query', '--metric-lt', 'score:1.0'])
  assert result_lt['ok']
  ids_lt = {r['id'] for r in result_lt['result']['experiments']}
  assert 'exp-ok' in ids_lt
  assert 'exp-inf' not in ids_lt
  assert 'exp-neginf' not in ids_lt


def test_query_combined_filters_cli(tmp_path: Path) -> None:
  """Combined --metric-gt + --metric-lt + --sort narrows and orders via CLI."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  for i in range(15):
    exp = Experiment(experiment_id=f'exp-{i:02d}', hypothesis=f'h{i}')
    exp.start()
    exp.complete(metrics={'accuracy': i * 0.06})
    tree.add(Node(experiment=exp))

  forest.save()

  result = run_cli_no_context(
    ws,
    ['query', '--metric-gt', 'accuracy:0.2', '--metric-lt', 'accuracy:0.7', '--sort', 'accuracy'],
  )
  assert result['ok']
  experiments = result['result']['experiments']
  assert len(experiments) > 0

  accuracies = [e['metrics']['accuracy'] for e in experiments]
  assert all(0.2 < a < 0.7 for a in accuracies)
  assert accuracies == sorted(accuracies, reverse=True)

  all_result = run_cli_no_context(ws, ['query'])
  all_count = len(all_result['result']['experiments'])
  assert len(experiments) < all_count
