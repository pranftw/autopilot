"""Tests for _gate_threshold_str formatting and QualityFirstPolicy.forward() gate population."""

from autopilot.core.constraint import ConstraintResult
from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.gates import (
  BudgetGate,
  Gate,
  MaxGate,
  MinGate,
  MonotonicGate,
  RangeGate,
)
from autopilot.policy.quality_first import QualityFirstPolicy, _gate_threshold_str
import pytest


class TestGateThresholdStr:
  """Unit tests for _gate_threshold_str formatting."""

  def test_gate_threshold_str_min_gate(self) -> None:
    assert _gate_threshold_str(MinGate('accuracy', 0.8)) == '>= 0.8'

  def test_gate_threshold_str_max_gate(self) -> None:
    assert _gate_threshold_str(MaxGate('loss', 1.0)) == '<= 1.0'

  def test_gate_threshold_str_range_gate(self) -> None:
    assert _gate_threshold_str(RangeGate('f1', min_value=0, max_value=1.0)) == '[0, 1.0]'

  def test_gate_threshold_str_monotonic_gate(self) -> None:
    gate = MonotonicGate('accuracy', direction='non_decreasing', epsilon=0.1)
    assert _gate_threshold_str(gate) == 'non_decreasing (epsilon=0.1)'

  def test_gate_threshold_str_budget_gate(self) -> None:
    assert _gate_threshold_str(BudgetGate(max_usd=50.0)) == '50.0 USD'

  def test_gate_threshold_str_unknown_raises_type_error(self) -> None:
    """Unrecognized gate type raises TypeError (CQ-001 regression guard)."""

    class UnknownGate(Gate):
      def forward(self, result: Result) -> GateResult:
        return GateResult.PASSED

    with pytest.raises(TypeError, match='unrecognized gate type'):
      _gate_threshold_str(UnknownGate('x'))


class TestForwardGatePopulation:
  """Tests for QualityFirstPolicy.forward() populating result.gates."""

  def test_forward_populates_gates_count(self) -> None:
    """Three gates produce exactly 3 ConstraintResult entries."""
    policy = QualityFirstPolicy(
      gates=[
        MinGate('accuracy', 0.5),
        RangeGate('f1', min_value=0, max_value=1.0),
        BudgetGate(max_usd=100.0),
      ]
    )
    result = Result(metrics={'accuracy': 0.9, 'f1': 0.8, 'cost_usd': 50.0})
    policy.forward(result)
    assert len(result.gates) == 3
    for cr in result.gates:
      assert isinstance(cr, ConstraintResult)

  def test_forward_empty_gates_list(self) -> None:
    """No gates configured yields empty gates list and PASSED."""
    policy = QualityFirstPolicy(gates=[])
    result = Result(metrics={'accuracy': 0.9})
    outcome = policy.forward(result)
    assert result.gates == []
    assert outcome == GateResult.PASSED

  def test_forward_gate_order_preserved(self) -> None:
    """Gate names in result.gates match insertion order."""
    policy = QualityFirstPolicy(
      gates=[
        MinGate('accuracy', 0.5),
        RangeGate('f1', min_value=0, max_value=1.0),
        BudgetGate(max_usd=100.0),
      ]
    )
    result = Result(metrics={'accuracy': 0.9, 'f1': 0.8, 'cost_usd': 50.0})
    policy.forward(result)
    names = [cr.name for cr in result.gates]
    assert names == ['MinGate', 'RangeGate', 'BudgetGate']

  def test_forward_pass_fail_correctness(self) -> None:
    """Failing MinGate has passed=False with message; passing has passed=True."""
    policy = QualityFirstPolicy(
      gates=[
        MinGate('accuracy', 0.8),
        MinGate('f1', 0.5),
      ]
    )
    result = Result(metrics={'accuracy': 0.5, 'f1': 0.9})
    policy.forward(result)

    failing = result.gates[0]
    assert failing.passed is False
    assert failing.message == 'MinGate failed'

    passing = result.gates[1]
    assert passing.passed is True
    assert passing.message is None

  def test_forward_threshold_strings_per_gate_type(self) -> None:
    """Composite policy threshold strings match expected formats."""
    policy = QualityFirstPolicy(
      gates=[
        MinGate('accuracy', 0.8),
        RangeGate('f1', min_value=0, max_value=1.0),
        BudgetGate(max_usd=100.0),
      ]
    )
    result = Result(metrics={'accuracy': 0.9, 'f1': 0.8, 'cost_usd': 50.0})
    policy.forward(result)
    thresholds = [cr.threshold for cr in result.gates]
    assert thresholds == ['>= 0.8', '[0, 1.0]', '100.0 USD']

  def test_forward_warn_optional_gate(self) -> None:
    """Optional failing gate yields WARN; its ConstraintResult.passed is False."""
    policy = QualityFirstPolicy(
      gates=[
        MinGate('accuracy', 0.5),
        MinGate('f1', 0.9, required=False),
      ]
    )
    result = Result(metrics={'accuracy': 0.8, 'f1': 0.5})
    outcome = policy.forward(result)
    assert outcome == GateResult.WARN

    optional_cr = result.gates[1]
    assert optional_cr.passed is False

  def test_forward_single_evaluation(self) -> None:
    """Each gate's forward is called exactly once per policy.forward()."""
    call_counts: dict[str, int] = {}

    class CountingMinGate(MinGate):
      def __init__(self, metric: str, label: str) -> None:
        super().__init__(metric, threshold=0.0)
        self._label = label
        call_counts[label] = 0

      def forward(self, result: Result) -> GateResult:
        self.hint = None
        call_counts[self._label] += 1
        return GateResult.PASSED

    policy = QualityFirstPolicy(
      gates=[
        CountingMinGate('a', 'gate_a'),
        CountingMinGate('b', 'gate_b'),
      ]
    )
    result = Result(metrics={'a': 1.0, 'b': 2.0})
    policy.forward(result)

    assert call_counts['gate_a'] == 1
    assert call_counts['gate_b'] == 1

  def test_forward_metric_value_captured(self) -> None:
    """ConstraintResult.value matches the actual metric value from result."""
    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.5)])
    result = Result(metrics={'accuracy': 0.75})
    policy.forward(result)
    assert result.gates[0].value == 0.75

  def test_forward_missing_metric_value_is_none(self) -> None:
    """ConstraintResult.value is None when metric is missing from result."""
    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.5)])
    result = Result(metrics={})
    policy.forward(result)
    assert result.gates[0].value is None
    assert result.gates[0].passed is False
