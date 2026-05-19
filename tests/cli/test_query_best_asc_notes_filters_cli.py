"""Tests for query enhancements: --best with constraints, --asc, notes search."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
from tests.doubles import make_completed_experiment
import pytest


def _seed_metrics_workspace(ws: Path) -> None:
  """Seed a workspace with experiments carrying varied metrics for filter/sort tests."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  tree.add(
    Node(
      experiment=make_completed_experiment(
        'exp-cheap',
        'cheap run',
        {'accuracy': 0.85, 'precision': 0.75, 'loss': 0.3, 'cost_usd': 1.0, 'latency': 150.0},
      )
    )
  )
  tree.add(
    Node(
      experiment=make_completed_experiment(
        'exp-mid',
        'mid run',
        {'accuracy': 0.90, 'precision': 0.80, 'loss': 0.2, 'cost_usd': 5.0, 'latency': 100.0},
      )
    )
  )
  tree.add(
    Node(
      experiment=make_completed_experiment(
        'exp-expensive',
        'expensive run',
        {'accuracy': 0.95, 'precision': 0.90, 'loss': 0.1, 'cost_usd': 10.0, 'latency': 50.0},
      )
    )
  )
  tree.add(
    Node(
      experiment=make_completed_experiment(
        'exp-poor',
        'poor run',
        {'accuracy': 0.60, 'precision': 0.50, 'loss': 0.8, 'cost_usd': 0.3, 'latency': 300.0},
      )
    )
  )

  forest.save()


def _seed_notes_workspace(ws: Path) -> None:
  """Seed a workspace with experiments carrying notes and context log entries."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  tree = forest.create_tree('main')
  forest.switch('main')

  exp_a = Experiment(experiment_id='exp-notes', hypothesis='has notes')
  exp_a.start()
  exp_a.notes = 'This experiment uses the Acme dataset for evaluation'
  exp_a.complete(metrics={'accuracy': 0.9})
  tree.add(Node(experiment=exp_a))

  exp_b = Experiment(experiment_id='exp-ctx', hypothesis='has context log')
  exp_b.start()
  exp_b.add_context('initial run with Acme tools', source='user', epoch=0)
  exp_b.complete(metrics={'accuracy': 0.8})
  tree.add(Node(experiment=exp_b))

  exp_c = Experiment(experiment_id='exp-plain', hypothesis='no notes or acme')
  exp_c.start()
  exp_c.complete(metrics={'accuracy': 0.7})
  tree.add(Node(experiment=exp_c))

  exp_d = Experiment(experiment_id='exp-empty-notes', hypothesis='empty notes')
  exp_d.start()
  exp_d.notes = ''
  exp_d.complete(metrics={'accuracy': 0.6})
  tree.add(Node(experiment=exp_d))

  forest.save()


def _seed_cross_tree_workspace(ws: Path) -> None:
  """Seed workspace with two trees for cross-tree --best tests."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree_a = forest.create_tree('alpha')
  exp_a1 = Experiment(experiment_id='a-high', hypothesis='alpha high')
  exp_a1.start()
  exp_a1.complete(metrics={'val_accuracy': 0.95, 'loss': 0.1})
  tree_a.add(Node(experiment=exp_a1))

  exp_a2 = Experiment(experiment_id='a-low', hypothesis='alpha low')
  exp_a2.start()
  exp_a2.complete(metrics={'val_accuracy': 0.70, 'loss': 0.5})
  tree_a.add(Node(experiment=exp_a2))

  tree_b = forest.create_tree('beta')
  exp_b1 = Experiment(experiment_id='b-mid', hypothesis='beta mid')
  exp_b1.start()
  exp_b1.complete(metrics={'val_accuracy': 0.88, 'loss': 0.2})
  tree_b.add(Node(experiment=exp_b1))

  forest.switch('alpha')
  forest.save()


@pytest.fixture
def metrics_ws(tmp_path: Path) -> Path:
  """Workspace with metric-rich experiments."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_metrics_workspace(ws)
  return ws


@pytest.fixture
def notes_ws(tmp_path: Path) -> Path:
  """Workspace with notes-bearing experiments."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_notes_workspace(ws)
  return ws


@pytest.fixture
def cross_tree_ws(tmp_path: Path) -> Path:
  """Workspace with two trees for cross-tree tests."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  _seed_cross_tree_workspace(ws)
  return ws


# ---------------------------------------------------------------------------
# 4.1 Metric thresholds and sort direction
# ---------------------------------------------------------------------------


class TestMetricThresholdsAndSort:
  """Tests for --metric-gt, --metric-lt composition and --sort/--asc."""

  def test_query_multi_metric_gt(self, metrics_ws: Path) -> None:
    """Multiple --metric-gt predicates compose with AND."""
    result = run_cli_no_context(
      metrics_ws, ['query', '--metric-gt', 'accuracy:0.8', '--metric-gt', 'precision:0.7']
    )
    assert result['ok'] is True
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert 'exp-cheap' in ids
    assert 'exp-mid' in ids
    assert 'exp-expensive' in ids
    assert 'exp-poor' not in ids

  def test_query_multi_metric_lt(self, metrics_ws: Path) -> None:
    """Multiple --metric-lt predicates compose with AND."""
    result = run_cli_no_context(
      metrics_ws, ['query', '--metric-lt', 'loss:0.5', '--metric-lt', 'latency:200']
    )
    assert result['ok'] is True
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert 'exp-cheap' in ids
    assert 'exp-mid' in ids
    assert 'exp-expensive' in ids
    assert 'exp-poor' not in ids

  def test_query_sort_ascending(self, metrics_ws: Path) -> None:
    """--sort cost_usd --asc lists cheapest first."""
    result = run_cli_no_context(metrics_ws, ['query', '--sort', 'cost_usd', '--asc'])
    assert result['ok'] is True
    exps = result['result']['experiments']
    costs = [e['metrics']['cost_usd'] for e in exps]
    assert costs == sorted(costs)
    assert exps[0]['id'] == 'exp-poor'
    assert exps[-1]['id'] == 'exp-expensive'

  def test_query_sort_descending_default(self, metrics_ws: Path) -> None:
    """--sort accuracy keeps highest-first default."""
    result = run_cli_no_context(metrics_ws, ['query', '--sort', 'accuracy'])
    assert result['ok'] is True
    exps = result['result']['experiments']
    accuracies = [e['metrics']['accuracy'] for e in exps]
    assert accuracies == sorted(accuracies, reverse=True)
    assert exps[0]['id'] == 'exp-expensive'


# ---------------------------------------------------------------------------
# 4.2 Context and notes
# ---------------------------------------------------------------------------


class TestContextAndNotes:
  """Tests for --context-contains searching both notes and context log."""

  def test_query_context_contains_searches_notes(self, notes_ws: Path) -> None:
    """Notes body with 'Acme' matches --context-contains (case-insensitive)."""
    result = run_cli_no_context(notes_ws, ['query', '--context-contains', 'acme'])
    assert result['ok'] is True
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert 'exp-notes' in ids
    assert 'exp-ctx' in ids
    assert 'exp-plain' not in ids
    assert 'exp-empty-notes' not in ids

  def test_query_context_contains_searches_context_log(self, notes_ws: Path) -> None:
    """Match driven only by context_log entry reason; still found."""
    result = run_cli_no_context(notes_ws, ['query', '--context-contains', 'initial run'])
    assert result['ok'] is True
    exps = result['result']['experiments']
    ids = {e['id'] for e in exps}
    assert 'exp-ctx' in ids
    assert 'exp-notes' not in ids

  def test_query_context_contains_case_sensitive(self, notes_ws: Path) -> None:
    """--case-sensitive applies to both notes and context log."""
    result_exact = run_cli_no_context(
      notes_ws, ['query', '--context-contains', 'Acme', '--case-sensitive']
    )
    assert result_exact['ok'] is True
    exps_exact = result_exact['result']['experiments']
    ids_exact = {e['id'] for e in exps_exact}
    assert 'exp-notes' in ids_exact
    assert 'exp-ctx' in ids_exact

    result_lower = run_cli_no_context(
      notes_ws, ['query', '--context-contains', 'acme', '--case-sensitive']
    )
    assert result_lower['ok'] is True
    exps_lower = result_lower['result']['experiments']
    ids_lower = {e['id'] for e in exps_lower}
    assert 'exp-notes' not in ids_lower
    assert 'exp-ctx' not in ids_lower


# ---------------------------------------------------------------------------
# 4.3 --best composition and JSON
# ---------------------------------------------------------------------------


class TestBestCompositionAndJson:
  """Tests for --best after metric constraints and JSON shape."""

  def test_query_best_after_metric_constraints(self, metrics_ws: Path) -> None:
    """--best accuracy with --metric-gt cost_usd:0.5 picks best from filtered set."""
    result = run_cli_no_context(
      metrics_ws,
      ['query', '--best', 'accuracy', '--metric-gt', 'cost_usd:0.5'],
    )
    assert result['ok'] is True
    best = result['result']['best']
    assert best is not None
    assert best['id'] == 'exp-expensive'
    assert best['metrics']['cost_usd'] > 0.5

  def test_query_best_with_all_trees_and_constraints(self, cross_tree_ws: Path) -> None:
    """Cross-tree --best with --metric-lt filters correctly.

    Only a-high (val_accuracy=0.95, loss=0.1) and b-mid (val_accuracy=0.88,
    loss=0.2) satisfy loss < 0.3. a-high has the highest val_accuracy.
    """
    result = run_cli_no_context(
      cross_tree_ws,
      ['query', '--best', 'val_accuracy', '--metric-lt', 'loss:0.3', '--all-trees'],
    )
    assert result['ok'] is True
    best = result['result']['best']
    assert best is not None
    assert best['id'] == 'a-high'
    assert best['metrics']['loss'] < 0.3
    assert best['metrics']['val_accuracy'] == 0.95
    assert 'tree' in best
    assert best['tree'] == 'alpha'

  def test_query_constraints_exclude_all_experiments(self, metrics_ws: Path) -> None:
    """Overly strict constraints yield empty set, exit code 0."""
    result = run_cli_no_context(metrics_ws, ['query', '--metric-gt', 'accuracy:99.9'])
    assert result['ok'] is True
    assert result['result']['count'] == 0
    assert result['result']['experiments'] == []

  def test_query_best_json_shape(self, metrics_ws: Path) -> None:
    """--best --json includes expected keys on the best payload."""
    result = run_cli_no_context(metrics_ws, ['query', '--best', 'accuracy'])
    assert result['ok'] is True
    best = result['result']['best']
    assert best is not None
    expected_keys = {
      'id',
      'status',
      'hypothesis',
      'metrics',
      'spec_version',
      'created_at',
      'started_at',
      'metrics_trusted',
      'context_log',
    }
    assert expected_keys.issubset(set(best.keys()))

  def test_query_best_all_trees_json_shape(self, cross_tree_ws: Path) -> None:
    """--best --all-trees --json includes tree field on best object."""
    result = run_cli_no_context(cross_tree_ws, ['query', '--best', 'val_accuracy', '--all-trees'])
    assert result['ok'] is True
    best = result['result']['best']
    assert best is not None
    assert 'tree' in best
    assert best['tree'] in {'alpha', 'beta'}

  def test_query_sort_asc_json(self, metrics_ws: Path) -> None:
    """--sort --asc --json returns rows in ascending metric order."""
    result = run_cli_no_context(metrics_ws, ['query', '--sort', 'loss', '--asc'])
    assert result['ok'] is True
    exps = result['result']['experiments']
    losses = [e['metrics']['loss'] for e in exps]
    assert losses == sorted(losses)
    assert exps[0]['metrics']['loss'] <= exps[-1]['metrics']['loss']
