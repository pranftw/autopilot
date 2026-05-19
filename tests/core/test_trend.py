"""Tests for TrendAnalyzer and TrendResult."""

from autopilot.core.node import Node
from autopilot.core.store.base import Store
from autopilot.core.tree import Tree
from autopilot.core.trend import (
  TrendAnalyzer,
  TrendResult,
)
from tests.core.conftest import completed_exp
from unittest.mock import MagicMock
import math
import pytest


@pytest.fixture
def mock_store():
  """Mock store for tree construction."""
  return MagicMock(spec=Store)


@pytest.fixture
def analyzer():
  """Fresh TrendAnalyzer instance."""
  return TrendAnalyzer()


def _make_tree(store, experiments):
  """Build a tree from a list of (id, metrics) tuples."""
  tree = Tree(name='trend-test', store=store)
  for eid, metrics in experiments:
    exp = completed_exp(eid, metrics=metrics)
    tree.add(Node(experiment=exp))
  return tree


class TestTrendResultRoundtrip:
  """TrendResult to_dict / from_dict identity."""

  def test_trend_result_roundtrip(self):
    """Crafted TrendResult survives to_dict -> from_dict."""
    original = TrendResult(
      metric='accuracy',
      values=[0.7, 0.8, 0.9],
      experiment_ids=['e1', 'e2', 'e3'],
      direction='improving',
      best_value=0.9,
      best_experiment_id='e3',
      latest_value=0.9,
      improvement_rate=0.1,
    )
    roundtripped = TrendResult.from_dict(original.to_dict())
    assert roundtripped.metric == original.metric
    assert roundtripped.values == original.values
    assert roundtripped.experiment_ids == original.experiment_ids
    assert roundtripped.direction == original.direction
    assert roundtripped.best_value == original.best_value
    assert roundtripped.best_experiment_id == original.best_experiment_id
    assert roundtripped.latest_value == original.latest_value
    assert roundtripped.improvement_rate == original.improvement_rate


class TestTrendResultValidation:
  """TrendResult constructor validation."""

  def test_trend_result_constructor_invalid_classification(self):
    """Bad direction raises ValueError."""
    with pytest.raises(ValueError, match='Invalid trend direction'):
      TrendResult(metric='x', direction='bogus')

  def test_trend_result_from_dict_missing_key(self):
    """from_dict with missing required key raises."""
    with pytest.raises(TypeError):
      TrendResult.from_dict({'direction': 'improving'})


class TestTrendImproving:
  """Monotonically improving sequences."""

  def test_trend_improving(self, mock_store, analyzer):
    """Strictly increasing values classify as improving."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.5}),
        ('e2', {'score': 0.6}),
        ('e3', {'score': 0.7}),
        ('e4', {'score': 0.8}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'improving'

  def test_trend_improving_with_ties(self, mock_store, analyzer):
    """Non-strict: ties do not break improving monotonicity."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.5}),
        ('e2', {'score': 0.5}),
        ('e3', {'score': 0.6}),
        ('e4', {'score': 0.6}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'improving'


class TestTrendDegrading:
  """Monotonically degrading sequences."""

  def test_trend_degrading(self, mock_store, analyzer):
    """Strictly decreasing values classify as degrading."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.9}),
        ('e2', {'score': 0.7}),
        ('e3', {'score': 0.5}),
        ('e4', {'score': 0.3}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'degrading'


class TestTrendPlateau:
  """Flat sequences within plateau threshold."""

  def test_trend_plateau(self, mock_store, analyzer):
    """Values within PLATEAU_THRESHOLD range -> plateau."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 1.000}),
        ('e2', {'score': 1.001}),
        ('e3', {'score': 0.999}),
        ('e4', {'score': 1.000}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'plateau'

  def test_trend_plateau_all_zero(self, mock_store, analyzer):
    """All-zero values -> plateau (max == 0 edge case)."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.0}),
        ('e2', {'score': 0.0}),
        ('e3', {'score': 0.0}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'plateau'


class TestTrendVolatile:
  """High coefficient of variation triggers volatile."""

  def test_trend_volatile(self, mock_store, analyzer):
    """Large swings with high CV -> volatile."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.2}),
        ('e2', {'score': 0.9}),
        ('e3', {'score': 0.1}),
        ('e4', {'score': 0.8}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'volatile'


class TestTrendBestExperiment:
  """Best value and experiment id selection."""

  def test_trend_best_experiment(self, mock_store, analyzer):
    """Best id matches best value under higher_is_better=True."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.5}),
        ('e2', {'score': 0.9}),
        ('e3', {'score': 0.7}),
      ],
    )
    result = analyzer.analyze(tree, 'score', higher_is_better=True)
    assert result.best_value == 0.9
    assert result.best_experiment_id == 'e2'

  def test_trend_best_experiment_lower(self, mock_store, analyzer):
    """Best id matches lowest value under higher_is_better=False."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.5}),
        ('e2', {'score': 0.9}),
        ('e3', {'score': 0.2}),
      ],
    )
    result = analyzer.analyze(tree, 'score', higher_is_better=False)
    assert result.best_value == 0.2
    assert result.best_experiment_id == 'e3'


class TestTrendImprovementRate:
  """Numeric improvement rate expectations."""

  def test_trend_improvement_rate(self, mock_store, analyzer):
    """Linear progression gives expected per-epoch rate."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 1.0}),
        ('e2', {'score': 2.0}),
        ('e3', {'score': 3.0}),
        ('e4', {'score': 4.0}),
      ],
    )
    result = analyzer.analyze(tree, 'score', higher_is_better=True)
    assert result.improvement_rate == pytest.approx(1.0)

  def test_trend_improvement_rate_lower(self, mock_store, analyzer):
    """Lower-is-better negates the raw rate."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 4.0}),
        ('e2', {'score': 3.0}),
        ('e3', {'score': 2.0}),
        ('e4', {'score': 1.0}),
      ],
    )
    result = analyzer.analyze(tree, 'score', higher_is_better=False)
    assert result.improvement_rate == pytest.approx(1.0)


class TestTrendWindow:
  """Window slices tail correctly."""

  def test_trend_window(self, mock_store, analyzer):
    """Window=3 on a 5-point series uses last 3 points."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.1}),
        ('e2', {'score': 0.2}),
        ('e3', {'score': 0.5}),
        ('e4', {'score': 0.6}),
        ('e5', {'score': 0.7}),
      ],
    )
    result = analyzer.analyze(tree, 'score', window=3)
    assert len(result.values) == 3
    assert result.values == [0.5, 0.6, 0.7]
    assert result.experiment_ids == ['e3', 'e4', 'e5']


class TestTrendEmptyTree:
  """Empty tree or no matching metrics."""

  def test_trend_empty_tree(self, mock_store, analyzer):
    """Empty tree -> insufficient_data with stable empty lists."""
    tree = Tree(name='empty', store=mock_store)
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'insufficient_data'
    assert result.values == []
    assert result.experiment_ids == []
    assert result.best_experiment_id is None
    assert result.improvement_rate is None

  def test_trend_no_matching_metric(self, mock_store, analyzer):
    """Experiments without the target metric -> insufficient_data."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'other': 0.5}),
        ('e2', {'other': 0.6}),
        ('e3', {'other': 0.7}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'insufficient_data'
    assert result.values == []


class TestTrendSingleExperiment:
  """One data point -> insufficient_data."""

  def test_trend_single_experiment(self, mock_store, analyzer):
    """Single experiment -> insufficient_data."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.5}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'insufficient_data'
    assert result.values == [0.5]
    assert result.best_value == 0.5
    assert result.best_experiment_id == 'e1'
    assert result.latest_value == 0.5

  def test_trend_two_experiments(self, mock_store, analyzer):
    """Two experiments (< MIN_DATA_POINTS=3) -> insufficient_data."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'score': 0.5}),
        ('e2', {'score': 0.6}),
      ],
    )
    result = analyzer.analyze(tree, 'score')
    assert result.direction == 'insufficient_data'
    assert len(result.values) == 2


class TestTrendLowerIsBetter:
  """Flip improving/degrading when higher_is_better=False."""

  def test_trend_lower_is_better(self, mock_store, analyzer):
    """Decreasing values improve when lower_is_better."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'loss': 1.0}),
        ('e2', {'loss': 0.8}),
        ('e3', {'loss': 0.6}),
        ('e4', {'loss': 0.4}),
      ],
    )
    result = analyzer.analyze(tree, 'loss', higher_is_better=False)
    assert result.direction == 'improving'

  def test_trend_lower_is_better_degrading(self, mock_store, analyzer):
    """Increasing values degrade when lower_is_better."""
    tree = _make_tree(
      mock_store,
      [
        ('e1', {'loss': 0.4}),
        ('e2', {'loss': 0.6}),
        ('e3', {'loss': 0.8}),
        ('e4', {'loss': 1.0}),
      ],
    )
    result = analyzer.analyze(tree, 'loss', higher_is_better=False)
    assert result.direction == 'degrading'


class TestTrendNanMetricValues:
  """NaN values are skipped in metric collection."""

  def test_trend_nan_metric_values(self, mock_store, analyzer):
    """NaN entries skipped; classification uses remaining points."""
    tree = Tree(name='nan-test', store=mock_store)

    exps = [
      ('e1', 0.5),
      ('e2', float('nan')),
      ('e3', 0.6),
      ('e4', 0.7),
      ('e5', float('nan')),
      ('e6', 0.8),
    ]
    for eid, val in exps:
      exp = completed_exp(eid, metrics={'score': val})
      tree.add(Node(experiment=exp))

    result = analyzer.analyze(tree, 'score')
    assert len(result.values) == 4
    assert all(not math.isnan(v) for v in result.values)
    assert result.direction == 'improving'
    assert 'e2' not in result.experiment_ids
    assert 'e5' not in result.experiment_ids
