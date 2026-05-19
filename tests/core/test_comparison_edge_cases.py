"""Edge-case tests for MetricsComparator, Delta, and comparison utilities."""

from autopilot.core.comparison import Delta, MetricsComparator
from autopilot.core.metric import Metric
from autopilot.core.types import Datum
import inspect
import math
import pytest


class _HigherMetric(Metric):
  higher_is_better = True

  def __init__(self) -> None:
    super().__init__()

  def name(self) -> str:
    return 'accuracy'

  def update(self, datum: Datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'accuracy': 0.0}


class _LowerMetric(Metric):
  higher_is_better = False

  def __init__(self) -> None:
    super().__init__()

  def name(self) -> str:
    return 'loss'

  def update(self, datum: Datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'loss': 0.0}


class _UnknownMetric(Metric):
  higher_is_better = None

  def __init__(self) -> None:
    super().__init__()

  def name(self) -> str:
    return 'mystery'

  def update(self, datum: Datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'mystery': 0.0}


class TestNanHandling:
  def test_best_index_with_nan_values_returns_int(self) -> None:
    comp = MetricsComparator([_HigherMetric()])
    results = [
      {'accuracy': float('nan')},
      {'accuracy': 0.8},
      {'accuracy': 0.9},
    ]
    idx = comp.best_index(results, 'accuracy')
    assert isinstance(idx, int)
    assert 0 <= idx < len(results)

  def test_is_improvement_with_nan_delta(self) -> None:
    comp = MetricsComparator([_HigherMetric()])
    d = Delta(
      metric='accuracy',
      baseline=0.5,
      candidate=float('nan'),
      delta=float('nan'),
      higher_is_better=True,
      significant=False,
    )
    result = comp.is_improvement(d)
    assert isinstance(result, bool)

  def test_compare_with_nan_values(self) -> None:
    comp = MetricsComparator([_HigherMetric()])
    deltas = comp.compare(
      {'accuracy': 0.5},
      {'accuracy': float('nan')},
    )
    assert len(deltas) == 1
    assert math.isnan(deltas[0].candidate)


class TestAllEqualMetrics:
  def test_best_index_all_equal_returns_first(self) -> None:
    comp = MetricsComparator([_HigherMetric()])
    results = [{'accuracy': 0.8}, {'accuracy': 0.8}, {'accuracy': 0.8}]
    assert comp.best_index(results, 'accuracy') == 0

  def test_best_index_all_equal_lower_is_better(self) -> None:
    comp = MetricsComparator([_LowerMetric()])
    results = [{'loss': 0.3}, {'loss': 0.3}, {'loss': 0.3}]
    assert comp.best_index(results, 'loss') == 0


class TestIsSignificantOverride:
  def test_subclass_override_is_significant(self) -> None:
    class StrictComparator(MetricsComparator):
      def is_significant(self, delta: float, baseline: float) -> bool:
        return abs(delta) > 10.0

    comp = StrictComparator([_HigherMetric()])
    assert comp.is_significant(5.0, 1.0) is False
    assert comp.is_significant(15.0, 1.0) is True

  def test_override_used_in_compare(self) -> None:
    class AlwaysSignificant(MetricsComparator):
      def is_significant(self, delta: float, baseline: float) -> bool:
        return True

    comp = AlwaysSignificant([_HigherMetric()])
    deltas = comp.compare({'accuracy': 0.8}, {'accuracy': 0.8})
    assert len(deltas) == 1
    assert deltas[0].significant is True
    assert deltas[0].delta == 0.0


class TestEmptyMetricNames:
  def test_metric_with_empty_string_name(self) -> None:
    class EmptyNameMetric(Metric):
      higher_is_better = True

      def __init__(self) -> None:
        super().__init__()

      def name(self) -> str:
        return ''

      def update(self, datum: Datum) -> None:
        pass

      def compute(self) -> dict[str, float]:
        return {'': 0.0}

    comp = MetricsComparator([EmptyNameMetric()])
    deltas = comp.compare({'': 0.5}, {'': 0.9})
    assert len(deltas) == 1
    assert not deltas[0].metric
    assert deltas[0].delta == pytest.approx(0.4)


class TestDeltaSmallFloatDifferences:
  def test_very_small_positive_delta(self) -> None:
    d = Delta(
      metric='accuracy',
      baseline=0.9,
      candidate=0.9 + 1e-15,
      delta=1e-15,
      higher_is_better=True,
      significant=False,
    )
    assert d.delta > 0
    assert d.delta < 1e-14

  def test_very_small_negative_delta(self) -> None:
    d = Delta(
      metric='loss',
      baseline=0.1,
      candidate=0.1 - 1e-15,
      delta=-1e-15,
      higher_is_better=False,
      significant=False,
    )
    assert d.delta < 0
    assert abs(d.delta) < 1e-14

  def test_small_delta_round_trip(self) -> None:
    d = Delta(
      metric='m',
      baseline=1.0,
      candidate=1.0 + 1e-12,
      delta=1e-12,
      higher_is_better=True,
      significant=True,
    )
    data = d.to_dict()
    d2 = Delta.from_dict(data)
    assert d2.delta == pytest.approx(1e-12)


class TestBestIndexAllMissingKeys:
  def test_all_results_missing_metric_raises(self) -> None:
    comp = MetricsComparator([_HigherMetric()])
    results = [{'loss': 0.5}, {'loss': 0.3}, {'loss': 0.1}]
    with pytest.raises(ValueError, match='no result dicts contain metric'):
      comp.best_index(results, 'accuracy')

  def test_empty_results_raises(self) -> None:
    comp = MetricsComparator([_HigherMetric()])
    with pytest.raises(ValueError, match='results list is empty'):
      comp.best_index([], 'accuracy')

  def test_single_result_with_metric(self) -> None:
    comp = MetricsComparator([_HigherMetric()])
    results = [{'accuracy': 0.9}]
    assert comp.best_index(results, 'accuracy') == 0


class TestDeltaCandidateField:
  def test_delta_has_candidate_not_current(self) -> None:
    d = Delta(
      metric='m',
      baseline=0.5,
      candidate=0.8,
      delta=0.3,
      higher_is_better=True,
      significant=True,
    )
    assert hasattr(d, 'candidate')
    assert not hasattr(d, 'current')
    assert d.candidate == 0.8
    assert d.baseline == 0.5

  def test_delta_candidate_round_trip(self) -> None:
    d = Delta(
      metric='m',
      baseline=1.0,
      candidate=2.0,
      delta=1.0,
      higher_is_better=True,
      significant=True,
    )
    data = d.to_dict()
    assert 'candidate' in data
    assert 'current' not in data
    d2 = Delta.from_dict(data)
    assert d2.candidate == 2.0
    assert d2.baseline == 1.0


class _AccuracyMetric(Metric):
  higher_is_better = True

  def __init__(self) -> None:
    super().__init__()

  def name(self) -> str:
    return 'AccuracyMetric'

  def update(self, datum: Datum) -> None:
    pass

  def compute(self) -> dict[str, float]:
    return {'AccuracyMetric': 0.0}


class TestMetricsComparatorKeyNaming:
  def test_keys_must_match_metric_name(self) -> None:
    comp = MetricsComparator([_AccuracyMetric()])
    deltas = comp.compare(
      {'AccuracyMetric': 0.7},
      {'AccuracyMetric': 0.9},
    )
    assert len(deltas) == 1
    assert deltas[0].metric == 'AccuracyMetric'

  def test_mismatched_keys_produce_no_comparison(self) -> None:
    comp = MetricsComparator([_AccuracyMetric()])
    deltas = comp.compare(
      {'accuracy': 0.7},
      {'accuracy': 0.9},
    )
    assert len(deltas) == 0


def test_metrics_comparator_docstring_accurate():
  doc = inspect.getdoc(MetricsComparator)
  assert doc is not None
  assert 'named_children()' in doc
  assert 'metric.name()' in doc
  assert 'list[Metric]' in doc
  assert 'MetricCollection' in doc
