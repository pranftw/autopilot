"""CLI tests for ``recommend --metric-gt / --metric-lt`` filters (plan 06)."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import run_cli_no_context


def _seed_multi_experiment(tmp_path: Path) -> Path:
  """Create a workspace with 4 completed experiments at varying metrics."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')
  forest.switch('main')

  experiments = [
    ('exp-1', {'val_accuracy': 0.7, 'val_loss': 0.5}),
    ('exp-2', {'val_accuracy': 0.85, 'val_loss': 0.3}),
    ('exp-3', {'val_accuracy': 0.92, 'val_loss': 0.15}),
    ('exp-4', {'val_accuracy': 0.95, 'val_loss': 0.1}),
  ]
  for eid, metrics in experiments:
    exp = Experiment(experiment_id=eid)
    exp.start()
    exp.complete(metrics=metrics)
    tree.add(Node(experiment=exp))

  forest.save()
  return ws


def _seed_no_completed(tmp_path: Path) -> Path:
  """Create a workspace with experiments that are all non-terminal."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')
  forest.switch('main')

  for eid in ('exp-p1', 'exp-p2'):
    exp = Experiment(experiment_id=eid)
    exp.start()
    tree.add(Node(experiment=exp))

  forest.save()
  return ws


def _seed_single_experiment(tmp_path: Path) -> Path:
  """Create a workspace with exactly one completed experiment."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)

  tree = forest.create_tree('main')
  forest.switch('main')

  exp = Experiment(experiment_id='exp-only')
  exp.start()
  exp.complete(metrics={'val_accuracy': 0.88})
  tree.add(Node(experiment=exp))

  forest.save()
  return ws


class TestRecommendMetricGtFilter:
  """``--metric-gt`` filters candidates to those above threshold."""

  def test_recommend_metric_gt_filter(self, tmp_path: Path) -> None:
    """Only experiments with val_accuracy > 0.9 remain."""
    ws = _seed_multi_experiment(tmp_path)
    result = run_cli_no_context(
      ws,
      ['recommend', '--metric', 'val_accuracy', '--metric-gt', 'val_accuracy:0.9'],
    )
    assert result['ok'] is True
    rec = result['result']
    assert rec['experiment_id'] is not None
    evidence = rec.get('evidence', {})
    if 'candidates_count' in evidence:
      assert evidence['candidates_count'] == 2


class TestRecommendMetricLtFilter:
  """``--metric-lt`` filters candidates to those below threshold."""

  def test_recommend_metric_lt_filter(self, tmp_path: Path) -> None:
    """Only experiments with val_loss < 0.2 remain."""
    ws = _seed_multi_experiment(tmp_path)
    result = run_cli_no_context(
      ws,
      ['recommend', '--metric', 'val_accuracy', '--metric-lt', 'val_loss:0.2'],
    )
    assert result['ok'] is True
    rec = result['result']
    assert rec['experiment_id'] is not None
    evidence = rec.get('evidence', {})
    if 'candidates_count' in evidence:
      assert evidence['candidates_count'] == 2


class TestRecommendCombinedFilters:
  """Both GT and LT narrow list by intersection."""

  def test_recommend_combined_filters(self, tmp_path: Path) -> None:
    """Combining --metric-gt and --metric-lt restricts candidates."""
    ws = _seed_multi_experiment(tmp_path)
    result = run_cli_no_context(
      ws,
      [
        'recommend',
        '--metric',
        'val_accuracy',
        '--metric-gt',
        'val_accuracy:0.8',
        '--metric-lt',
        'val_loss:0.2',
      ],
    )
    assert result['ok'] is True
    rec = result['result']
    assert rec['experiment_id'] is not None
    evidence = rec.get('evidence', {})
    if 'candidates_count' in evidence:
      assert evidence['candidates_count'] == 2


class TestRecommendFilterExcludesAll:
  """Thresholds exclude every candidate -> investigate sentinel."""

  def test_recommend_filter_excludes_all(self, tmp_path: Path) -> None:
    """JSON ok True; result reflects sentinel Recommendation."""
    ws = _seed_multi_experiment(tmp_path)
    result = run_cli_no_context(
      ws,
      [
        'recommend',
        '--metric',
        'val_accuracy',
        '--metric-gt',
        'val_accuracy:0.99',
      ],
    )
    assert result['ok'] is True
    rec = result['result']
    assert rec['action'] == 'investigate'
    assert rec['experiment_id'] is None
    assert any('excluded' in r or 'filter' in r for r in rec['reasoning'])


class TestRecommendWithoutFiltersUnchanged:
  """Baseline ordering matches prior behavior with no filters."""

  def test_recommend_without_filters_unchanged(self, tmp_path: Path) -> None:
    """Without filters, same result as no-filter recommend."""
    ws = _seed_multi_experiment(tmp_path)
    result = run_cli_no_context(
      ws,
      ['recommend', '--metric', 'val_accuracy'],
    )
    assert result['ok'] is True
    rec = result['result']
    assert rec['experiment_id'] == 'exp-4'
    assert rec['confidence'] in {'high', 'medium', 'low'}


class TestRecommendNoCompletedExperiments:
  """All experiments non-terminal -> investigate sentinel."""

  def test_recommend_no_completed_experiments(self, tmp_path: Path) -> None:
    """No completed experiments -> investigate with reasoning."""
    ws = _seed_no_completed(tmp_path)
    result = run_cli_no_context(
      ws,
      ['recommend', '--metric', 'val_accuracy'],
    )
    assert result['ok'] is True
    rec = result['result']
    assert rec['action'] == 'investigate'
    assert rec['experiment_id'] is None
    assert rec['confidence'] == 'low'
    assert any('no completed' in r for r in rec['reasoning'])


class TestRecommendSingleExperiment:
  """Exactly one completed experiment -> low confidence."""

  def test_recommend_single_experiment(self, tmp_path: Path) -> None:
    """Single experiment recommendation has low confidence."""
    ws = _seed_single_experiment(tmp_path)
    result = run_cli_no_context(
      ws,
      ['recommend', '--metric', 'val_accuracy'],
    )
    assert result['ok'] is True
    rec = result['result']
    assert rec['experiment_id'] == 'exp-only'
    assert rec['confidence'] == 'low'
