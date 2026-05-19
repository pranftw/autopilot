"""Tests for workspace-level report summary (FR#23, sub-plan 06)."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.report.command import gather_workspace_summary
from autopilot.cli.commands.report.compare import gather_summary
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context, seed_tree_with_experiments
import pytest


def _build_forest_with_experiments(
  tmp_path: Path,
  tree_experiments: dict[str, list[dict]],
  active_tree: str,
) -> tuple[FileForest, Path]:
  """Build a forest with multiple trees and experiments.

  Args:
    tmp_path: Temporary workspace root.
    tree_experiments: Mapping of tree name to list of experiment dicts
      (each with id, status, and optional metrics).
    active_tree: Name of the tree to set as active.

  Returns:
    Tuple of (forest, workspace_path).
  """
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  for tree_name, experiments in tree_experiments.items():
    seed_tree_with_experiments(forest, tree_name, experiments)

  forest.switch(active_tree)
  forest.save()
  return forest, ws


class TestReportSummaryWorkspaceLevel:
  """test_report_summary_workspace_level -- active tree aggregate."""

  def test_scope_is_tree_and_tree_name_present(self, tmp_path: Path) -> None:
    """Aggregate without --all-trees scopes to active tree."""
    forest, _ = _build_forest_with_experiments(
      tmp_path,
      {
        'main': [
          {'id': 'exp-1', 'status': 'completed', 'metrics': {'accuracy': 0.9}},
          {'id': 'exp-2', 'status': 'running'},
        ],
      },
      active_tree='main',
    )
    tree = forest.active
    result = gather_workspace_summary(forest, tree, all_trees=False)
    assert result['scope'] == 'tree'
    assert result['tree'] == 'main'
    assert result['experiments_count']['completed'] == 1
    assert result['experiments_count']['running'] == 1
    assert 'accuracy' in result['metric_summary']
    assert result['best_experiment'] is not None
    assert result['best_experiment']['id'] == 'exp-1'

  def test_only_completed_contribute_to_metrics(self, tmp_path: Path) -> None:
    """Running experiments do not contribute to metric_summary or best_experiment."""
    forest, _ = _build_forest_with_experiments(
      tmp_path,
      {
        'main': [
          {'id': 'exp-r', 'status': 'running', 'metrics': {'accuracy': 0.99}},
          {'id': 'exp-c', 'status': 'completed', 'metrics': {'accuracy': 0.8}},
        ],
      },
      active_tree='main',
    )
    tree = forest.active
    result = gather_workspace_summary(forest, tree, all_trees=False)
    assert result['metric_summary']['accuracy']['max'] == 0.8
    assert result['best_experiment']['id'] == 'exp-c'


class TestReportSummaryAllTrees:
  """test_report_summary_all_trees -- forest-wide aggregate."""

  def test_scope_workspace_no_tree_key(self, tmp_path: Path) -> None:
    """--all-trees sets scope=workspace and omits tree key."""
    forest, _ = _build_forest_with_experiments(
      tmp_path,
      {
        'alpha': [
          {'id': 'exp-a1', 'status': 'completed', 'metrics': {'loss': 0.3}},
        ],
        'beta': [
          {'id': 'exp-b1', 'status': 'completed', 'metrics': {'loss': 0.5}},
        ],
      },
      active_tree='alpha',
    )
    result = gather_workspace_summary(forest, None, all_trees=True)
    assert result['scope'] == 'workspace'
    assert 'tree' not in result
    assert result['experiments_count']['completed'] == 2

  def test_counts_include_all_trees(self, tmp_path: Path) -> None:
    """All experiments across all trees are counted."""
    forest, _ = _build_forest_with_experiments(
      tmp_path,
      {
        'alpha': [
          {'id': 'exp-a1', 'status': 'completed', 'metrics': {'accuracy': 0.9}},
          {'id': 'exp-a2', 'status': 'failed'},
        ],
        'beta': [
          {'id': 'exp-b1', 'status': 'running'},
        ],
      },
      active_tree='alpha',
    )
    result = gather_workspace_summary(forest, None, all_trees=True)
    assert result['experiments_count']['completed'] == 1
    assert result['experiments_count']['failed'] == 1
    assert result['experiments_count']['running'] == 1


class TestReportSummaryNoExperiments:
  """test_report_summary_no_experiments_empty -- empty tree."""

  def test_empty_tree_returns_empty_aggregates(self, tmp_path: Path) -> None:
    """Empty active tree yields zero counts, empty metrics, None best."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('empty')
    forest.switch('empty')
    forest.save()

    tree = forest.active
    result = gather_workspace_summary(forest, tree, all_trees=False)
    assert result['scope'] == 'tree'
    assert result['experiments_count'] == {}
    assert result['metric_summary'] == {}
    assert result['best_experiment'] is None


class TestReportSummaryJsonSchema:
  """test_report_summary_json_schema -- structural validation."""

  def test_result_has_required_keys(self, tmp_path: Path) -> None:
    """Aggregate result contains scope, experiments_count, metric_summary, best_experiment."""
    forest, _ = _build_forest_with_experiments(
      tmp_path,
      {
        'main': [
          {'id': 'exp-1', 'status': 'completed', 'metrics': {'f1': 0.75}},
        ],
      },
      active_tree='main',
    )
    tree = forest.active
    result = gather_workspace_summary(forest, tree, all_trees=False)
    assert 'scope' in result
    assert 'experiments_count' in result
    assert 'metric_summary' in result
    assert 'best_experiment' in result

  def test_cli_json_envelope(self, tmp_path: Path) -> None:
    """CLI report summary --json returns proper envelope with aggregate payload."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')

    exp = Experiment(experiment_id='exp-j', hypothesis='json test')
    exp.start()
    exp.complete(metrics={'precision': 0.85})
    tree.add(Node(experiment=exp))
    forest.save()

    envelope = run_cli_no_context(ws, ['report', 'summary'])
    assert envelope['ok'] is True
    payload = envelope['result']
    assert payload['scope'] == 'tree'
    assert 'experiments_count' in payload
    assert 'metric_summary' in payload


class TestReportSummaryMetricAggregation:
  """test_report_summary_metric_aggregation -- deterministic numeric checks."""

  def test_min_max_mean_computed_correctly(self, tmp_path: Path) -> None:
    """Verify min/max/mean against hand-calculated values."""
    forest, _ = _build_forest_with_experiments(
      tmp_path,
      {
        'main': [
          {'id': 'exp-1', 'status': 'completed', 'metrics': {'accuracy': 0.7, 'loss': 0.5}},
          {'id': 'exp-2', 'status': 'completed', 'metrics': {'accuracy': 0.9, 'loss': 0.3}},
          {'id': 'exp-3', 'status': 'completed', 'metrics': {'accuracy': 0.8}},
        ],
      },
      active_tree='main',
    )
    tree = forest.active
    result = gather_workspace_summary(forest, tree, all_trees=False)

    acc = result['metric_summary']['accuracy']
    assert acc['min'] == pytest.approx(0.7)
    assert acc['max'] == pytest.approx(0.9)
    assert acc['mean'] == pytest.approx(0.8)

    loss = result['metric_summary']['loss']
    assert loss['min'] == pytest.approx(0.3)
    assert loss['max'] == pytest.approx(0.5)
    assert loss['mean'] == pytest.approx(0.4)

  def test_best_experiment_selects_first_lexicographic_metric(self, tmp_path: Path) -> None:
    """best_experiment uses highest value on first metric key (lexicographic)."""
    forest, _ = _build_forest_with_experiments(
      tmp_path,
      {
        'main': [
          {'id': 'exp-lo', 'status': 'completed', 'metrics': {'accuracy': 0.7, 'f1': 0.99}},
          {'id': 'exp-hi', 'status': 'completed', 'metrics': {'accuracy': 0.95, 'f1': 0.5}},
        ],
      },
      active_tree='main',
    )
    tree = forest.active
    result = gather_workspace_summary(forest, tree, all_trees=False)
    assert result['best_experiment']['id'] == 'exp-hi'


class TestReportSummaryExitCode:
  """test_report_summary_exit_code -- successful invocation exits 0."""

  def test_aggregate_invocation_exits_zero(self, tmp_path: Path) -> None:
    """Successful aggregate report summary exits with code 0."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    forest.switch('main')

    exp = Experiment(experiment_id='exp-ok', hypothesis='exit test')
    exp.start()
    exp.complete(metrics={'accuracy': 0.5})
    tree.add(Node(experiment=exp))
    forest.save()

    envelope = run_cli_no_context(ws, ['report', 'summary'])
    assert envelope['ok'] is True
    assert envelope['result']['scope'] == 'tree'


def test_gather_summary_missing_experiment_raises(tmp_path: Path) -> None:
  """gather_summary raises ValueError for nonexistent experiment."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  with pytest.raises(ValueError, match='nonexistent'):
    gather_summary(forest, 'nonexistent')
