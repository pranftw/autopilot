"""Tests for sub-plan 08 sections 2.3 and 2.4: Gate prelude DRY and QualityFirstMetric fix.

Covers:
- _check_metric_available returns GateResult.FAIL for None, None for non-None.
- MinGate/MaxGate/RangeGate use _check_metric_available (no triplicated prelude).
- QualityFirstMetric.to_result evaluates each gate exactly once.
- Behavior unchanged for non-None metrics after refactor.
"""

from autopilot.core.models import Result
from autopilot.core.types import EvalDatum, GateResult
from autopilot.policy.gates import (
  CustomGate,
  MaxGate,
  MinGate,
  RangeGate,
  _check_metric_available,
)
from autopilot.policy.quality_first import QualityFirstMetric
from typing import Any, cast


class TestCheckMetricAvailable:
  """_check_metric_available module-level function."""

  def test_returns_fail_for_none(self) -> None:
    gate = MinGate('accuracy', 0.5)
    eval_result = Result(metrics={'other': 1.0})
    result = _check_metric_available(gate, None, eval_result)
    assert result == GateResult.FAIL
    assert gate.hint is not None

  def test_returns_none_for_float(self) -> None:
    gate = MinGate('accuracy', 0.5)
    eval_result = Result(metrics={'accuracy': 0.5})
    result = _check_metric_available(gate, 0.5, eval_result)
    assert result is None

  def test_returns_none_for_zero(self) -> None:
    gate = MinGate('loss', 0.5)
    eval_result = Result(metrics={'loss': 0.0})
    result = _check_metric_available(gate, 0.0, eval_result)
    assert result is None

  def test_returns_none_for_nan(self) -> None:
    gate = MinGate('score', 0.5)
    eval_result = Result(metrics={'score': float('nan')})
    result = _check_metric_available(gate, float('nan'), eval_result)
    assert result is None

  def test_returns_none_for_negative(self) -> None:
    gate = MinGate('delta', 0.5)
    eval_result = Result(metrics={'delta': -1.0})
    result = _check_metric_available(gate, -1.0, eval_result)
    assert result is None


class TestGatePreludeUsesHelper:
  """All gates produce FAIL for missing metrics via the shared helper."""

  def test_min_gate_none_metric(self) -> None:
    gate = MinGate('accuracy', 0.8)
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL

  def test_max_gate_none_metric(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL

  def test_range_gate_none_metric(self) -> None:
    gate = RangeGate('score', 0.0, 1.0)
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL

  def test_custom_gate_none_metric(self) -> None:
    gate = CustomGate('x', lambda v: True)
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL


class TestGateBehaviorUnchangedForNonNoneMetrics:
  """Refactor preserves gate behavior for valid numeric metrics."""

  def test_min_gate_pass(self) -> None:
    gate = MinGate('accuracy', 0.8)
    assert gate.forward(Result(metrics={'accuracy': 0.9})) == GateResult.PASSED

  def test_min_gate_fail(self) -> None:
    gate = MinGate('accuracy', 0.8)
    assert gate.forward(Result(metrics={'accuracy': 0.5})) == GateResult.FAIL

  def test_max_gate_pass(self) -> None:
    gate = MaxGate('loss', 1.0)
    assert gate.forward(Result(metrics={'loss': 0.5})) == GateResult.PASSED

  def test_max_gate_fail(self) -> None:
    gate = MaxGate('loss', 1.0)
    assert gate.forward(Result(metrics={'loss': 1.5})) == GateResult.FAIL

  def test_range_gate_pass(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    assert gate.forward(Result(metrics={'score': 0.5})) == GateResult.PASSED

  def test_range_gate_fail_below(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    assert gate.forward(Result(metrics={'score': 0.1})) == GateResult.FAIL

  def test_range_gate_fail_above(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    assert gate.forward(Result(metrics={'score': 0.9})) == GateResult.FAIL

  def test_custom_gate_pass(self) -> None:
    gate = CustomGate('x', lambda v: v > 0)
    assert gate.forward(Result(metrics={'x': 1.0})) == GateResult.PASSED

  def test_custom_gate_fail(self) -> None:
    gate = CustomGate('x', lambda v: v > 10)
    assert gate.forward(Result(metrics={'x': 1.0})) == GateResult.FAIL


class TestGateNaNBehaviorPreserved:
  """NaN metrics still produce FAIL (not intercepted by the helper)."""

  def test_min_gate_nan_fails(self) -> None:
    gate = MinGate('accuracy', 0.5)
    assert gate.forward(Result(metrics={'accuracy': float('nan')})) == GateResult.FAIL

  def test_max_gate_nan_fails(self) -> None:
    gate = MaxGate('loss', 1.0)
    assert gate.forward(Result(metrics={'loss': float('nan')})) == GateResult.FAIL

  def test_range_gate_nan_fails(self) -> None:
    gate = RangeGate('score', 0.0, 1.0)
    assert gate.forward(Result(metrics={'score': float('nan')})) == GateResult.FAIL


class TestQualityFirstMetricSingleGateEvaluation:
  """2.4: to_result evaluates each gate exactly once."""

  def test_gate_evaluated_once_in_to_result(self) -> None:
    call_count = 0

    class _CountingGate(MinGate):
      def forward(self, result: Result) -> GateResult:
        nonlocal call_count
        call_count += 1
        return super().forward(result)

    gate = _CountingGate('accuracy', 0.5)
    metric = QualityFirstMetric(gates=[gate])
    metric.update(EvalDatum(success=True, metrics={'accuracy': 0.9}))
    metric.to_result()
    assert call_count == 1

  def test_multiple_gates_each_evaluated_once(self) -> None:
    call_counts: dict[str, int] = {'accuracy': 0, 'loss': 0}

    class _CountingMinGate(MinGate):
      def forward(self, result: Result) -> GateResult:
        call_counts[self.metric] += 1
        return super().forward(result)

    class _CountingMaxGate(MaxGate):
      def forward(self, result: Result) -> GateResult:
        call_counts[self.metric] += 1
        return super().forward(result)

    gates = [_CountingMinGate('accuracy', 0.5), _CountingMaxGate('loss', 2.0)]
    metric = QualityFirstMetric(gates=cast(Any, gates))
    metric.update(EvalDatum(success=True, metrics={'accuracy': 0.9, 'loss': 0.5}))
    result = metric.to_result()

    assert call_counts == {'accuracy': 1, 'loss': 1}
    assert result.passed is True

  def test_to_result_failing_gate_single_eval(self) -> None:
    call_count = 0

    class _CountingGate(MinGate):
      def forward(self, result: Result) -> GateResult:
        nonlocal call_count
        call_count += 1
        return super().forward(result)

    gate = _CountingGate('accuracy', 0.95)
    metric = QualityFirstMetric(gates=[gate])
    metric.update(EvalDatum(success=True, metrics={'accuracy': 0.5}))
    result = metric.to_result()

    assert call_count == 1
    assert result.passed is False

  def test_to_result_with_explicit_metrics(self) -> None:
    """to_result(metrics=...) uses provided metrics instead of compute()."""
    gate = MinGate('accuracy', 0.8)
    metric = QualityFirstMetric(gates=[gate])
    result = metric.to_result(metrics={'accuracy': 0.9})
    assert result.passed is True
    assert len(result.gates) == 1
    assert result.gates[0].metric == 'accuracy'
    assert result.gates[0].passed is True

  def test_to_result_gates_list_populated(self) -> None:
    gate = MinGate('accuracy', 0.8)
    metric = QualityFirstMetric(gates=[gate])
    metric.update(EvalDatum(success=True, metrics={'accuracy': 0.9}))
    result = metric.to_result()
    assert len(result.gates) == 1
    assert result.gates[0].metric == 'accuracy'
    assert result.gates[0].passed is True


class TestQualityFirstMetricOutcomeParity:
  """2.4: outcome matches single-evaluation semantics."""

  def test_pass_outcome_with_mixed_gates(self) -> None:
    gates = [MinGate('accuracy', 0.5), MinGate('recall', 0.3, required=False)]
    metric = QualityFirstMetric(gates=cast(Any, gates))
    metric.update(EvalDatum(success=True, metrics={'accuracy': 0.9, 'recall': 0.1}))
    result = metric.to_result()
    assert result.passed is False
    assert result.gates[0].passed is True
    assert result.gates[1].passed is False

  def test_fail_outcome_required_gate_fails(self) -> None:
    gates = [MinGate('accuracy', 0.95)]
    metric = QualityFirstMetric(gates=cast(Any, gates))
    metric.update(EvalDatum(success=True, metrics={'accuracy': 0.5}))
    result = metric.to_result()
    assert result.passed is False
