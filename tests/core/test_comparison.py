"""Tests for metric comparison utilities and data models."""

from autopilot.core.comparison import EpochMetrics, MetricComparison, compare_metrics


class TestCompareMetrics:
  def test_clear_improvement(self):
    result = compare_metrics({'accuracy': 0.7}, {'accuracy': 0.9})
    assert result.improvement_detected is True
    assert result.regression_detected is False
    assert result.is_mixed is False

  def test_clear_regression(self):
    result = compare_metrics({'accuracy': 0.9}, {'accuracy': 0.7})
    assert result.regression_detected is True
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_mixed_results(self):
    result = compare_metrics(
      {'accuracy': 0.7, 'error_rate': 0.1},
      {'accuracy': 0.9, 'error_rate': 0.3},
      metric_metadata={'accuracy': True, 'error_rate': False},
    )
    assert result.is_mixed is True
    assert result.regression_detected is False
    assert result.improvement_detected is False

  def test_equal_metrics(self):
    result = compare_metrics({'accuracy': 0.8}, {'accuracy': 0.8})
    assert result.regression_detected is False
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_threshold_abs(self):
    result = compare_metrics(
      {'accuracy': 0.8},
      {'accuracy': 0.79},
      threshold_abs=0.05,
    )
    assert result.regression_detected is False
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_threshold_pct(self):
    result = compare_metrics(
      {'accuracy': 0.8},
      {'accuracy': 0.79},
      threshold_pct=0.05,
    )
    assert result.regression_detected is False
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_threshold_exceeded(self):
    result = compare_metrics(
      {'accuracy': 0.8},
      {'accuracy': 0.5},
      threshold_abs=0.05,
      threshold_pct=0.05,
    )
    assert result.regression_detected is True
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_per_metric_deltas(self):
    result = compare_metrics(
      {'accuracy': 0.7, 'recall': 0.6},
      {'accuracy': 0.9, 'recall': 0.5},
    )
    assert result.per_metric_deltas['accuracy'] > 0
    assert result.per_metric_deltas['recall'] < 0

  def test_empty_baseline(self):
    result = compare_metrics({}, {'accuracy': 0.8})
    assert result.regression_detected is False
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_empty_candidate(self):
    result = compare_metrics({'accuracy': 0.8}, {})
    assert result.regression_detected is False
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_single_metric(self):
    result = compare_metrics({'accuracy': 0.7}, {'accuracy': 0.8})
    assert result.improvement_detected is True
    assert result.regression_detected is False
    assert result.is_mixed is False

  def test_disjoint_keys(self):
    result = compare_metrics({'a': 1.0}, {'b': 2.0})
    assert result.regression_detected is False
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_subset_keys(self):
    result = compare_metrics(
      {'a': 0.8, 'b': 0.7, 'c': 0.6},
      {'a': 0.9},
    )
    assert 'a' in result.per_metric_deltas
    assert 'b' not in result.per_metric_deltas

  def test_metric_metadata_higher_is_better_true(self):
    result = compare_metrics(
      {'accuracy': 0.7},
      {'accuracy': 0.9},
      metric_metadata={'accuracy': True},
    )
    assert result.improvement_detected is True
    assert result.regression_detected is False
    assert result.is_mixed is False

  def test_metric_metadata_higher_is_better_false(self):
    result = compare_metrics(
      {'error_rate': 0.1},
      {'error_rate': 0.3},
      metric_metadata={'error_rate': False},
    )
    assert result.regression_detected is True
    assert result.improvement_detected is False
    assert result.is_mixed is False

  def test_metric_metadata_none_defaults_to_higher(self):
    result = compare_metrics(
      {'accuracy': 0.7},
      {'accuracy': 0.9},
    )
    assert result.improvement_detected is True
    assert result.regression_detected is False
    assert result.is_mixed is False

  def test_metric_metadata_mixed_directions(self):
    result = compare_metrics(
      {'accuracy': 0.7, 'loss': 0.5},
      {'accuracy': 0.9, 'loss': 0.3},
      metric_metadata={'accuracy': True, 'loss': False},
    )
    assert result.improvement_detected is True
    assert result.regression_detected is False
    assert result.is_mixed is False

  def test_no_name_heuristic(self):
    result = compare_metrics(
      {'error_rate': 0.1},
      {'error_rate': 0.3},
    )
    assert result.regression_detected is False
    assert result.improvement_detected is True
    assert result.is_mixed is False


class TestMetricComparisonRoundTrip:
  def test_round_trip(self):
    mc = MetricComparison(
      epoch=3,
      per_metric_deltas={'accuracy': -0.1, 'latency': 0.05},
      regressions=[{'metric': 'accuracy', 'delta': -0.1}],
      improvements=[{'metric': 'latency', 'delta': 0.05}],
    )
    d = mc.to_dict()
    assert d['epoch'] == 3
    assert d['per_metric_deltas'] == {'accuracy': -0.1, 'latency': 0.05}
    assert d['regressions'] == [{'metric': 'accuracy', 'delta': -0.1}]
    assert d['improvements'] == [{'metric': 'latency', 'delta': 0.05}]
    assert d['regression_detected'] is False
    assert d['improvement_detected'] is False
    assert d['is_mixed'] is True
    mc2 = MetricComparison.from_dict(d)
    assert mc2.epoch == 3
    assert mc2.per_metric_deltas == {'accuracy': -0.1, 'latency': 0.05}
    assert mc2.regressions == [{'metric': 'accuracy', 'delta': -0.1}]
    assert mc2.improvements == [{'metric': 'latency', 'delta': 0.05}]
    assert mc2.regression_detected is False
    assert mc2.improvement_detected is False
    assert mc2.is_mixed is True

  def test_computed_properties_only_regressions(self):
    mc = MetricComparison(
      epoch=1,
      regressions=[{'metric': 'a', 'delta': -0.1}],
      improvements=[],
    )
    assert mc.regression_detected is True
    assert mc.improvement_detected is False
    assert mc.is_mixed is False

  def test_computed_properties_only_improvements(self):
    mc = MetricComparison(
      epoch=1,
      regressions=[],
      improvements=[{'metric': 'a', 'delta': 0.1}],
    )
    assert mc.regression_detected is False
    assert mc.improvement_detected is True
    assert mc.is_mixed is False

  def test_from_dict_ignores_computed_keys(self):
    stored = {
      'epoch': 2,
      'per_metric_deltas': {'m': 0.2},
      'regressions': [],
      'improvements': [],
    }
    mc = MetricComparison.from_dict(
      {
        **stored,
        'regression_detected': True,
        'improvement_detected': True,
        'is_mixed': True,
      }
    )
    assert mc.epoch == 2
    assert mc.per_metric_deltas == {'m': 0.2}
    assert mc.regressions == []
    assert mc.improvements == []


class TestEpochMetricsRoundTrip:
  def test_round_trip(self):
    em = EpochMetrics(epoch=1, split='val', total=10, passed=8, accuracy=0.8)
    d = em.to_dict()
    em2 = EpochMetrics.from_dict(d)
    assert em2.epoch == 1
    assert em2.accuracy == 0.8
