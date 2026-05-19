"""Tests for metric comparison utilities and data models."""

from autopilot.core.comparison import (
  ComparatorMetric,
  Delta,
  MetricsComparator,
)
from autopilot.core.metric import Metric
from autopilot.core.types import Datum
import pytest

# --- New comparison API tests (Plan 1) ---


class _AccuracyMetric(Metric):
  higher_is_better = True

  def __init__(self) -> None:
    super().__init__()

  def name(self) -> str:
    return 'accuracy'

  def update(self, datum: Datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'accuracy': 0.0}


class _LossMetric(Metric):
  higher_is_better = False

  def __init__(self) -> None:
    super().__init__()

  def name(self) -> str:
    return 'loss'

  def update(self, datum: Datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'loss': 0.0}


class _UnknownDirectionMetric(Metric):
  higher_is_better = None

  def __init__(self) -> None:
    super().__init__()

  def name(self) -> str:
    return 'mystery'

  def update(self, datum: Datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'mystery': 0.0}


class TestDeltaRoundTrip:
  def test_basic_round_trip(self):
    d = Delta(
      metric='accuracy',
      baseline=0.7,
      candidate=0.9,
      delta=0.2,
      higher_is_better=True,
      significant=True,
    )
    data = d.to_dict()
    assert data['metric'] == 'accuracy'
    assert data['delta'] == 0.2
    assert data['higher_is_better'] is True
    assert data['significant'] is True
    d2 = Delta.from_dict(data)
    assert d2.metric == 'accuracy'
    assert d2.delta == 0.2

  def test_round_trip_higher_is_better_none(self):
    d = Delta(
      metric='mystery',
      baseline=1.0,
      candidate=2.0,
      delta=1.0,
      higher_is_better=None,
      significant=False,
    )
    data = d.to_dict()
    assert data['higher_is_better'] is None
    d2 = Delta.from_dict(data)
    assert d2.higher_is_better is None

  def test_round_trip_negative_delta(self):
    d = Delta(
      metric='loss',
      baseline=0.5,
      candidate=0.3,
      delta=-0.2,
      higher_is_better=False,
      significant=True,
    )
    data = d.to_dict()
    assert data['delta'] == -0.2
    d2 = Delta.from_dict(data)
    assert d2.delta == -0.2

  def test_round_trip_zero_delta(self):
    d = Delta(
      metric='accuracy',
      baseline=0.8,
      candidate=0.8,
      delta=0.0,
      higher_is_better=True,
      significant=False,
    )
    data = d.to_dict()
    assert data['delta'] == 0.0
    d2 = Delta.from_dict(data)
    assert d2.delta == 0.0
    assert d2.significant is False


class TestMetricsComparator:
  def test_compare_normal_case(self):
    comp = MetricsComparator([_AccuracyMetric(), _LossMetric()])
    deltas = comp.compare(
      {'accuracy': 0.7, 'loss': 0.5},
      {'accuracy': 0.9, 'loss': 0.3},
    )
    assert len(deltas) == 2
    acc = next(d for d in deltas if d.metric == 'accuracy')
    loss = next(d for d in deltas if d.metric == 'loss')
    assert acc.delta == pytest.approx(0.2)
    assert acc.higher_is_better is True
    assert loss.delta == pytest.approx(-0.2)
    assert loss.higher_is_better is False

  def test_compare_single_metric(self):
    comp = MetricsComparator([_AccuracyMetric()])
    deltas = comp.compare({'accuracy': 0.5}, {'accuracy': 0.8})
    assert len(deltas) == 1
    assert deltas[0].delta == pytest.approx(0.3)

  def test_compare_missing_metrics_skipped(self):
    comp = MetricsComparator([_AccuracyMetric(), _LossMetric()])
    deltas = comp.compare({'accuracy': 0.5}, {'loss': 0.3})
    assert len(deltas) == 0

  def test_compare_empty_dicts_return_empty(self):
    comp = MetricsComparator([_AccuracyMetric()])
    deltas = comp.compare({}, {})
    assert deltas == []

  def test_compare_partial_overlap(self):
    comp = MetricsComparator([_AccuracyMetric(), _LossMetric()])
    deltas = comp.compare(
      {'accuracy': 0.7, 'loss': 0.5},
      {'accuracy': 0.9},
    )
    assert len(deltas) == 1
    assert deltas[0].metric == 'accuracy'

  def test_duplicate_metric_raises(self):
    with pytest.raises(ValueError, match='duplicate metric'):
      MetricsComparator([_AccuracyMetric(), _AccuracyMetric()])


class TestMetricsComparatorIsImprovement:
  def test_higher_is_better_positive_delta(self):
    comp = MetricsComparator([_AccuracyMetric()])
    d = Delta(
      metric='accuracy',
      baseline=0.7,
      candidate=0.9,
      delta=0.2,
      higher_is_better=True,
      significant=True,
    )
    assert comp.is_improvement(d) is True

  def test_higher_is_better_negative_delta(self):
    comp = MetricsComparator([_AccuracyMetric()])
    d = Delta(
      metric='accuracy',
      baseline=0.9,
      candidate=0.7,
      delta=-0.2,
      higher_is_better=True,
      significant=True,
    )
    assert comp.is_improvement(d) is False

  def test_lower_is_better_negative_delta(self):
    comp = MetricsComparator([_LossMetric()])
    d = Delta(
      metric='loss',
      baseline=0.5,
      candidate=0.3,
      delta=-0.2,
      higher_is_better=False,
      significant=True,
    )
    assert comp.is_improvement(d) is True

  def test_lower_is_better_positive_delta(self):
    comp = MetricsComparator([_LossMetric()])
    d = Delta(
      metric='loss',
      baseline=0.3,
      candidate=0.5,
      delta=0.2,
      higher_is_better=False,
      significant=True,
    )
    assert comp.is_improvement(d) is False

  def test_higher_is_better_none_raises(self):
    comp = MetricsComparator([_UnknownDirectionMetric()])
    d = Delta(
      metric='mystery',
      baseline=1.0,
      candidate=2.0,
      delta=1.0,
      higher_is_better=None,
      significant=True,
    )
    with pytest.raises(ValueError, match='higher_is_better not set'):
      comp.is_improvement(d)


class TestMetricsComparatorBestIndex:
  def test_higher_is_better_returns_max(self):
    comp = MetricsComparator([_AccuracyMetric()])
    results = [{'accuracy': 0.7}, {'accuracy': 0.9}, {'accuracy': 0.8}]
    assert comp.best_index(results, 'accuracy') == 1

  def test_lower_is_better_returns_min(self):
    comp = MetricsComparator([_LossMetric()])
    results = [{'loss': 0.5}, {'loss': 0.3}, {'loss': 0.8}]
    assert comp.best_index(results, 'loss') == 1

  def test_ties_return_first_index(self):
    comp = MetricsComparator([_AccuracyMetric()])
    results = [{'accuracy': 0.9}, {'accuracy': 0.9}, {'accuracy': 0.7}]
    assert comp.best_index(results, 'accuracy') == 0

  def test_empty_results_raises(self):
    comp = MetricsComparator([_AccuracyMetric()])
    with pytest.raises(ValueError, match='results list is empty'):
      comp.best_index([], 'accuracy')

  def test_missing_metric_in_all_results_raises(self):
    comp = MetricsComparator([_AccuracyMetric()])
    results = [{'loss': 0.5}, {'loss': 0.3}]
    with pytest.raises(ValueError, match='no result dicts contain metric'):
      comp.best_index(results, 'accuracy')

  def test_missing_metric_in_some_results_skips(self):
    comp = MetricsComparator([_AccuracyMetric()])
    results = [{'loss': 0.5}, {'accuracy': 0.9}, {'accuracy': 0.7}]
    assert comp.best_index(results, 'accuracy') == 1

  def test_higher_is_better_none_raises(self):
    comp = MetricsComparator([_UnknownDirectionMetric()])
    results = [{'mystery': 1.0}]
    with pytest.raises(ValueError, match='higher_is_better not set'):
      comp.best_index(results, 'mystery')


class TestMetricsComparatorSignificance:
  def test_both_thresholds_zero_any_nonzero_significant(self):
    comp = MetricsComparator([_AccuracyMetric()])
    assert comp.is_significant(0.01, 1.0) is True
    assert comp.is_significant(-0.01, 1.0) is True
    assert comp.is_significant(0.0, 1.0) is False

  def test_threshold_abs_only(self):
    comp = MetricsComparator([_AccuracyMetric()], threshold_abs=0.05)
    assert comp.is_significant(0.06, 1.0) is True
    assert comp.is_significant(0.04, 1.0) is False
    assert comp.is_significant(0.05, 1.0) is False

  def test_threshold_pct_only(self):
    comp = MetricsComparator([_AccuracyMetric()], threshold_pct=0.1)
    assert comp.is_significant(0.2, 1.0) is True
    assert comp.is_significant(0.05, 1.0) is False

  def test_threshold_pct_baseline_zero_no_division_error(self):
    comp = MetricsComparator([_AccuracyMetric()], threshold_pct=0.1)
    assert comp.is_significant(0.5, 0.0) is False

  def test_both_thresholds_either_triggers(self):
    comp = MetricsComparator(
      [_AccuracyMetric()],
      threshold_abs=0.1,
      threshold_pct=0.5,
    )
    assert comp.is_significant(0.15, 1.0) is True
    assert comp.is_significant(0.05, 0.05) is True
    assert comp.is_significant(0.05, 1.0) is False

  def test_override_in_subclass(self):
    class CustomComparator(MetricsComparator):
      def is_significant(self, delta: float, baseline: float) -> bool:
        return abs(delta) > 1.0

    comp = CustomComparator([_AccuracyMetric()])
    assert comp.is_significant(1.5, 0.0) is True
    assert comp.is_significant(0.5, 0.0) is False

  def test_compare_uses_is_significant(self):
    comp = MetricsComparator(
      [_AccuracyMetric()],
      threshold_abs=0.5,
    )
    deltas = comp.compare({'accuracy': 0.7}, {'accuracy': 0.8})
    assert len(deltas) == 1
    assert deltas[0].significant is False

    deltas = comp.compare({'accuracy': 0.7}, {'accuracy': 1.3})
    assert len(deltas) == 1
    assert deltas[0].significant is True


class TestComparatorMetric:
  def test_stores_metric_name_and_higher_is_better(self):
    cm = ComparatorMetric('accuracy', higher_is_better=True)
    assert cm.metric_name == 'accuracy'
    assert cm.higher_is_better is True

  def test_higher_is_better_false(self):
    cm = ComparatorMetric('loss', higher_is_better=False)
    assert cm.metric_name == 'loss'
    assert cm.higher_is_better is False

  def test_higher_is_better_none(self):
    cm = ComparatorMetric('mystery', higher_is_better=None)
    assert cm.metric_name == 'mystery'
    assert cm.higher_is_better is None

  def test_name_method_returns_metric_name(self):
    cm = ComparatorMetric('f1_score', higher_is_better=True)
    assert cm.name() == 'f1_score'


class TestMetricsComparatorWithComparatorMetric:
  def test_compare_with_comparator_metric(self):
    cm = ComparatorMetric('accuracy', higher_is_better=True)
    comp = MetricsComparator([cm])
    deltas = comp.compare({'accuracy': 0.7}, {'accuracy': 0.9})
    assert len(deltas) == 1
    assert deltas[0].metric == 'accuracy'
    assert deltas[0].delta == pytest.approx(0.2)
    assert deltas[0].higher_is_better is True

  def test_compare_multiple_comparator_metrics(self):
    metrics = [
      ComparatorMetric('accuracy', higher_is_better=True),
      ComparatorMetric('loss', higher_is_better=False),
    ]
    comp = MetricsComparator(metrics)
    deltas = comp.compare(
      {'accuracy': 0.7, 'loss': 0.5},
      {'accuracy': 0.9, 'loss': 0.3},
    )
    assert len(deltas) == 2
    acc = next(d for d in deltas if d.metric == 'accuracy')
    loss = next(d for d in deltas if d.metric == 'loss')
    assert acc.delta == pytest.approx(0.2)
    assert loss.delta == pytest.approx(-0.2)

  def test_comparator_metric_higher_is_better_none(self):
    cm = ComparatorMetric('mystery', higher_is_better=None)
    comp = MetricsComparator([cm])
    deltas = comp.compare({'mystery': 1.0}, {'mystery': 2.0})
    assert len(deltas) == 1
    assert deltas[0].higher_is_better is None

  def test_best_index_with_comparator_metric(self):
    cm = ComparatorMetric('accuracy', higher_is_better=True)
    comp = MetricsComparator([cm])
    results = [{'accuracy': 0.7}, {'accuracy': 0.9}, {'accuracy': 0.8}]
    assert comp.best_index(results, 'accuracy') == 1

  def test_best_index_lower_is_better_comparator_metric(self):
    cm = ComparatorMetric('loss', higher_is_better=False)
    comp = MetricsComparator([cm])
    results = [{'loss': 0.5}, {'loss': 0.3}, {'loss': 0.8}]
    assert comp.best_index(results, 'loss') == 1

  def test_is_improvement_with_comparator_metric(self):
    cm = ComparatorMetric('accuracy', higher_is_better=True)
    comp = MetricsComparator([cm])
    delta = Delta(
      metric='accuracy',
      baseline=0.7,
      candidate=0.9,
      delta=0.2,
      higher_is_better=True,
      significant=True,
    )
    assert comp.is_improvement(delta) is True

  def test_mixed_metric_and_comparator_metric(self):
    metrics = [_AccuracyMetric(), ComparatorMetric('f1', higher_is_better=True)]
    comp = MetricsComparator(metrics)
    deltas = comp.compare(
      {'accuracy': 0.7, 'f1': 0.6},
      {'accuracy': 0.9, 'f1': 0.8},
    )
    assert len(deltas) == 2

  def test_duplicate_comparator_metric_raises(self):
    metrics = [
      ComparatorMetric('accuracy', higher_is_better=True),
      ComparatorMetric('accuracy', higher_is_better=False),
    ]
    with pytest.raises(ValueError, match='duplicate metric'):
      MetricsComparator(metrics)
