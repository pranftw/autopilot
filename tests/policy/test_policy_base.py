"""Tests for Policy and Metric base classes."""

from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.models import Result
from autopilot.core.types import Datum, GateResult
from autopilot.policy.gates import MinGate
from autopilot.policy.policy import Policy
from autopilot.policy.quality_first import QualityFirstMetric


class TestPolicyDefaults:
  def test_default_name(self) -> None:
    assert Policy().name() == 'Policy'

  def test_subclass_name(self) -> None:
    class Strict(Policy):
      pass

    assert Strict().name() == 'Strict'

  def test_default_evaluates_pass(self) -> None:
    r = Result(metrics={'accuracy': 0.9})
    assert Policy().forward(r) == GateResult.PASS

  def test_default_explain(self) -> None:
    r = Result(metrics={})
    explanation = Policy().explain(r)
    assert 'default pass' in explanation

  def test_subclass_overrides(self) -> None:
    class Strict(Policy):
      def forward(self, result: Result) -> GateResult:
        if result.metrics.get('accuracy', 0) < 0.9:
          return GateResult.FAIL
        return GateResult.PASS

    low = Result(metrics={'accuracy': 0.5})
    high = Result(metrics={'accuracy': 0.95})
    assert Strict().forward(low) == GateResult.FAIL
    assert Strict().forward(high) == GateResult.PASS


class TestMetricDefaults:
  def test_default_name(self) -> None:
    assert Metric().name() == 'Metric'

  def test_subclass_name(self) -> None:
    class Accuracy(Metric):
      def update(self, datum: Datum) -> None:
        pass

      def compute(self) -> dict[str, float]:
        return {}

    assert Accuracy().name() == 'Accuracy'

  def test_update_compute_cycle(self) -> None:
    class SumMetric(Metric):
      def __init__(self) -> None:
        super().__init__()
        self._total = 0.0

      def update(self, datum: Datum) -> None:
        self._total += datum.metrics.get('value', 0.0)

      def compute(self) -> dict[str, float]:
        return {'total': self._total}

      def reset(self) -> None:
        self._total = 0.0

    m = SumMetric()
    m.update(Datum(metrics={'value': 1.0}, metadata={'split': 'train'}))
    m.update(Datum(metrics={'value': 2.0}, metadata={'split': 'train'}))
    assert m.compute() == {'total': 3.0}
    m.reset()
    assert m.compute() == {'total': 0.0}


class TestMetricCollection:
  def test_compose_via_add(self) -> None:
    class A(Metric):
      def update(self, datum: Datum) -> None:
        pass

      def compute(self) -> dict[str, float]:
        return {'a': 1.0}

    class B(Metric):
      def update(self, datum: Datum) -> None:
        pass

      def compute(self) -> dict[str, float]:
        return {'b': 2.0}

    composite = A() + B()
    assert isinstance(composite, MetricCollection)
    assert composite.compute() == {'a': 1.0, 'b': 2.0}

  def test_collection_update_delegates(self) -> None:
    class Counter(Metric):
      def __init__(self) -> None:
        super().__init__()
        self._count = 0

      def update(self, datum: Datum) -> None:
        self._count += 1

      def compute(self) -> dict[str, float]:
        return {'count': float(self._count)}

    c1 = Counter()
    c2 = Counter()
    composite = MetricCollection({'c1': c1, 'c2': c2})
    composite.update(Datum(metadata={'split': 'train'}))
    assert c1._count == 1
    assert c2._count == 1

  def test_collection_reset(self) -> None:
    class CounterA(Metric):
      def __init__(self) -> None:
        super().__init__()
        self._count = 0

      def update(self, datum: Datum) -> None:
        self._count += 1

      def compute(self) -> dict[str, float]:
        return {'a_count': float(self._count)}

      def reset(self) -> None:
        self._count = 0

    class CounterB(Metric):
      def __init__(self) -> None:
        super().__init__()
        self._count = 0

      def update(self, datum: Datum) -> None:
        self._count += 1

      def compute(self) -> dict[str, float]:
        return {'b_count': float(self._count)}

      def reset(self) -> None:
        self._count = 0

    composite = MetricCollection({'x': CounterA(), 'y': CounterB()})
    composite.update(Datum(metadata={'split': 'train'}))
    composite.reset()
    assert composite.compute() == {'a_count': 0.0, 'b_count': 0.0}


class TestQualityFirstMetric:
  def test_update_and_compute(self) -> None:
    m = QualityFirstMetric()
    m.update(Datum(metrics={'accuracy': 0.8}, metadata={'split': 'train'}))
    m.update(Datum(metrics={'accuracy': 0.6}, metadata={'split': 'train'}))
    result = m.compute()
    assert result == {'accuracy': 0.7}

  def test_to_result_with_gates(self) -> None:
    m = QualityFirstMetric(gates=[MinGate('accuracy', 0.7)])
    m.update(Datum(metrics={'accuracy': 0.9}, metadata={'split': 'train'}))
    r = m.to_result()
    assert r.passed is True
    assert r.gates['accuracy'] == 'pass'

  def test_to_result_fails_gate(self) -> None:
    m = QualityFirstMetric(gates=[MinGate('accuracy', 0.9)])
    m.update(Datum(metrics={'accuracy': 0.5}, metadata={'split': 'train'}))
    r = m.to_result()
    assert r.passed is False
    assert r.gates['accuracy'] == 'fail'

  def test_reset_clears_state(self) -> None:
    m = QualityFirstMetric()
    m.update(Datum(metrics={'accuracy': 0.8}, metadata={'split': 'train'}))
    m.reset()
    assert m._accumulated == {}
