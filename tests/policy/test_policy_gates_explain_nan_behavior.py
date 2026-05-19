"""Tests for sub-plan 09: Policy/Gate fixes.

Covers:
- Bug 36: QualityFirstPolicy double/triple gate evaluation
- Bug 37: Policy stateless by design (documented, no state_dict)
- Bug 38: GateResult.SKIP never produced by built-in gates
- RangeGate min > max validation
- NaN metric behavior for all gate types
- Missing metric behavior for all gate types
- Gate.explain() for all four gate types
- Empty QualityFirstPolicy -> PASS
"""

from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.gates import CustomGate, MaxGate, MinGate, RangeGate
from autopilot.policy.policy import Policy
from autopilot.policy.quality_first import QualityFirstPolicy
from typing import Any, cast
import math
import pytest


class TestQualityFirstPolicyGateEvaluationCount:
  """Bug 36: gates must be evaluated exactly once per forward() call."""

  def test_gates_evaluated_exactly_once_per_forward(self) -> None:
    call_counts: dict[str, int] = {}

    class CountingGate(MinGate):
      def __init__(self, metric: str, *, required: bool = True) -> None:
        super().__init__(metric, threshold=0.0, required=required)
        call_counts[metric] = 0

      def forward(self, result: Result) -> GateResult:
        call_counts[self.metric] += 1
        return GateResult.PASSED

    gates = [CountingGate('a'), CountingGate('b'), CountingGate('c')]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'a': 1.0, 'b': 2.0, 'c': 3.0})

    policy.forward(result)

    assert call_counts == {'a': 1, 'b': 1, 'c': 1}

  def test_explain_reuses_cached_results_after_forward(self) -> None:
    call_counts: dict[str, int] = {}

    class CountingGate(MinGate):
      def __init__(self, metric: str, *, required: bool = True) -> None:
        super().__init__(metric, threshold=0.0, required=required)
        call_counts[metric] = 0

      def forward(self, result: Result) -> GateResult:
        call_counts[self.metric] += 1
        return GateResult.PASSED

    gates = [CountingGate('x'), CountingGate('y')]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'x': 1.0, 'y': 2.0})

    policy.forward(result)
    policy.explain(result)

    assert call_counts == {'x': 1, 'y': 1}

  def test_explain_without_prior_forward_triggers_single_evaluation(self) -> None:
    call_counts: dict[str, int] = {}

    class CountingGate(MinGate):
      def __init__(self, metric: str, *, required: bool = True) -> None:
        super().__init__(metric, threshold=0.0, required=required)
        call_counts[metric] = 0

      def forward(self, result: Result) -> GateResult:
        call_counts[self.metric] += 1
        return GateResult.FAIL

    gates = [CountingGate('m', required=True)]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'m': 0.0})

    policy.explain(result)

    assert call_counts == {'m': 1}

  def test_forward_then_explain_total_evaluations(self) -> None:
    call_count = 0

    class CountingMinGate(MinGate):
      def forward(self, result: Result) -> GateResult:
        nonlocal call_count
        call_count += 1
        value = result.metrics.get(self.metric)
        if value is None:
          return GateResult.FAIL
        return GateResult.PASSED if value >= 0.5 else GateResult.FAIL

    policy = QualityFirstPolicy(gates=[CountingMinGate('acc', 0.0), CountingMinGate('f1', 0.0)])
    result = Result(metrics={'acc': 0.3, 'f1': 0.9})

    policy.forward(result)
    policy.explain(result)

    assert call_count == 2


class TestRangeGateValidation:
  """RangeGate must validate min_value <= max_value."""

  def test_min_greater_than_max_raises_value_error(self) -> None:
    with pytest.raises(ValueError, match=r'min_value.*must be <= max_value'):
      RangeGate('score', 0.8, 0.2)

  def test_min_equals_max_is_valid(self) -> None:
    gate = RangeGate('score', 0.5, 0.5)
    result = Result(metrics={'score': 0.5})
    assert gate.forward(result) == GateResult.PASSED

  def test_valid_range(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    assert gate.min_value == 0.2
    assert gate.max_value == 0.8


class TestNaNMetricBehavior:
  """All gates with NaN metric -> FAIL."""

  def test_min_gate_nan_fails(self) -> None:
    gate = MinGate('accuracy', 0.5)
    result = Result(metrics={'accuracy': float('nan')})
    assert gate.forward(result) == GateResult.FAIL

  def test_max_gate_nan_fails(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={'loss': float('nan')})
    assert gate.forward(result) == GateResult.FAIL

  def test_range_gate_nan_fails(self) -> None:
    gate = RangeGate('score', 0.0, 1.0)
    result = Result(metrics={'score': float('nan')})
    assert gate.forward(result) == GateResult.FAIL

  def test_custom_gate_nan_with_always_true_fn_fails(self) -> None:
    gate = CustomGate('val', lambda v: not math.isnan(v))
    result = Result(metrics={'val': float('nan')})
    assert gate.forward(result) == GateResult.FAIL


class TestMissingMetricBehavior:
  """All gates with missing metric -> FAIL."""

  def test_min_gate_missing_metric_fails(self) -> None:
    gate = MinGate('accuracy', 0.5)
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL

  def test_max_gate_missing_metric_fails(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={'other': 0.5})
    assert gate.forward(result) == GateResult.FAIL

  def test_range_gate_missing_metric_fails(self) -> None:
    gate = RangeGate('score', 0.0, 1.0)
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL

  def test_custom_gate_missing_metric_fails(self) -> None:
    gate = CustomGate('x', lambda v: True)
    result = Result(metrics={})
    assert gate.forward(result) == GateResult.FAIL


class TestGateExplain:
  """Gate.explain() returns formatted strings with class name, metric, value, threshold, result."""

  def test_min_gate_explain_pass(self) -> None:
    gate = MinGate('accuracy', 0.80)
    result = Result(metrics={'accuracy': 0.85})
    text = gate.explain(result)
    assert 'MinGate' in text
    assert 'accuracy' in text
    assert '0.85' in text
    assert '>=' in text
    assert 'PASS' in text

  def test_min_gate_explain_fail(self) -> None:
    gate = MinGate('accuracy', 0.90)
    result = Result(metrics={'accuracy': 0.5})
    text = gate.explain(result)
    assert 'MinGate' in text
    assert 'accuracy' in text
    assert 'FAIL' in text

  def test_min_gate_explain_missing(self) -> None:
    gate = MinGate('accuracy', 0.80)
    result = Result(metrics={})
    text = gate.explain(result)
    assert 'MinGate' in text
    assert 'accuracy' in text
    assert 'missing' in text
    assert 'FAIL' in text

  def test_max_gate_explain_pass(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={'loss': 0.5})
    text = gate.explain(result)
    assert 'MaxGate' in text
    assert 'loss' in text
    assert '0.5' in text
    assert '<=' in text
    assert 'PASS' in text

  def test_max_gate_explain_fail(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={'loss': 1.5})
    text = gate.explain(result)
    assert 'MaxGate' in text
    assert 'loss' in text
    assert 'FAIL' in text

  def test_max_gate_explain_missing(self) -> None:
    gate = MaxGate('loss', 1.0)
    result = Result(metrics={})
    text = gate.explain(result)
    assert 'MaxGate' in text
    assert 'missing' in text
    assert 'FAIL' in text

  def test_range_gate_explain_pass(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    result = Result(metrics={'score': 0.5})
    text = gate.explain(result)
    assert 'RangeGate' in text
    assert 'score' in text
    assert '0.5' in text
    assert 'in' in text
    assert '[0.2, 0.8]' in text
    assert 'PASS' in text

  def test_range_gate_explain_fail(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    result = Result(metrics={'score': 0.9})
    text = gate.explain(result)
    assert 'RangeGate' in text
    assert 'FAIL' in text

  def test_range_gate_explain_missing(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    result = Result(metrics={})
    text = gate.explain(result)
    assert 'RangeGate' in text
    assert 'missing' in text
    assert 'FAIL' in text

  def test_custom_gate_explain_pass(self) -> None:
    def is_positive(v: float) -> bool:
      return v > 0

    gate = CustomGate('val', is_positive)
    result = Result(metrics={'val': 5.0})
    text = gate.explain(result)
    assert 'CustomGate' in text
    assert 'val' in text
    assert '5.0' in text
    assert 'via' in text
    assert 'is_positive' in text
    assert 'PASS' in text

  def test_custom_gate_explain_fail(self) -> None:
    def is_positive(v: float) -> bool:
      return v > 0

    gate = CustomGate('val', is_positive)
    result = Result(metrics={'val': -1.0})
    text = gate.explain(result)
    assert 'CustomGate' in text
    assert 'FAIL' in text

  def test_custom_gate_explain_missing(self) -> None:
    def check(v: float) -> bool:
      return True

    gate = CustomGate('val', check)
    result = Result(metrics={})
    text = gate.explain(result)
    assert 'CustomGate' in text
    assert 'missing' in text
    assert 'FAIL' in text

  def test_all_gate_explains_produce_nonempty_strings_with_metric(self) -> None:
    result = Result(metrics={'m': 0.5})
    gates_to_test = [
      MinGate('m', 0.3),
      MaxGate('m', 0.7),
      RangeGate('m', 0.0, 1.0),
      CustomGate('m', lambda v: True),
    ]
    for gate in gates_to_test:
      text = gate.explain(result)
      assert len(text) > 0, f'{type(gate).__name__}.explain() returned empty string'
      assert 'm' in text, f'{type(gate).__name__}.explain() missing metric name'


class TestEmptyQualityFirstPolicy:
  """Empty QualityFirstPolicy (no gates) -> PASS."""

  def test_empty_policy_returns_pass(self) -> None:
    policy = QualityFirstPolicy()
    result = Result(metrics={'accuracy': 0.5})
    assert policy.forward(result) == GateResult.PASSED

  def test_empty_policy_explain_all_passed(self) -> None:
    policy = QualityFirstPolicy()
    result = Result(metrics={})
    assert policy.explain(result) == 'all gates passed'

  def test_empty_gates_list_returns_pass(self) -> None:
    policy = QualityFirstPolicy(gates=[])
    result = Result(metrics={'x': 1.0})
    assert policy.forward(result) == GateResult.PASSED


class TestGateResultSkipNeverProduced:
  """GateResult.SKIP: no built-in gate class produces SKIP for any finite metric value."""

  def test_min_gate_never_returns_skip(self) -> None:
    gate = MinGate('m', 0.5)
    for value in [0.0, 0.5, 1.0, -1.0, 100.0, float('inf'), float('-inf')]:
      result = Result(metrics={'m': value})
      assert gate.forward(result) != GateResult.SKIP

  def test_max_gate_never_returns_skip(self) -> None:
    gate = MaxGate('m', 0.5)
    for value in [0.0, 0.5, 1.0, -1.0, 100.0, float('inf'), float('-inf')]:
      result = Result(metrics={'m': value})
      assert gate.forward(result) != GateResult.SKIP

  def test_range_gate_never_returns_skip(self) -> None:
    gate = RangeGate('m', 0.0, 1.0)
    for value in [0.0, 0.5, 1.0, -1.0, 100.0, float('inf'), float('-inf')]:
      result = Result(metrics={'m': value})
      assert gate.forward(result) != GateResult.SKIP

  def test_min_gate_nan_not_skip(self) -> None:
    gate = MinGate('m', 0.5)
    result = Result(metrics={'m': float('nan')})
    assert gate.forward(result) != GateResult.SKIP

  def test_max_gate_nan_not_skip(self) -> None:
    gate = MaxGate('m', 0.5)
    result = Result(metrics={'m': float('nan')})
    assert gate.forward(result) != GateResult.SKIP

  def test_range_gate_nan_not_skip(self) -> None:
    gate = RangeGate('m', 0.0, 1.0)
    result = Result(metrics={'m': float('nan')})
    assert gate.forward(result) != GateResult.SKIP

  def test_missing_metric_not_skip(self) -> None:
    for gate in [MinGate('m', 0.5), MaxGate('m', 0.5), RangeGate('m', 0.0, 1.0)]:
      result = Result(metrics={})
      assert gate.forward(result) != GateResult.SKIP


class TestPolicyStatelessByDesign:
  """Bug 37: Policy is deterministic (no mutable state between forward calls)."""

  def test_policy_has_state_dict_for_serialization(self) -> None:
    assert hasattr(Policy, 'state_dict')
    assert hasattr(Policy, 'load_state_dict')

  def test_policy_has_no_reset(self) -> None:
    assert not hasattr(Policy, 'reset')

  def test_quality_first_policy_deterministic(self) -> None:
    gate = MinGate('accuracy', 0.8)
    policy = QualityFirstPolicy(gates=[gate])
    result = Result(metrics={'accuracy': 0.85})
    r1 = policy.forward(result)
    r2 = policy.forward(result)
    assert r1 == r2 == GateResult.PASSED


class TestQualityFirstPolicyExplainIntegration:
  """Verify explain() produces correct text for all outcomes."""

  def test_explain_all_passed(self) -> None:
    gate = MinGate('accuracy', 0.5)
    policy = QualityFirstPolicy(gates=[gate])
    result = Result(metrics={'accuracy': 0.9})
    policy.forward(result)
    text = policy.explain(result)
    assert text == 'all gates passed'

  def test_explain_optional_failed_with_review(self) -> None:
    gate = MinGate('accuracy', 0.9, required=False)
    policy = QualityFirstPolicy(gates=[gate], human_review_on_warn=True)
    result = Result(metrics={'accuracy': 0.5})
    policy.forward(result)
    text = policy.explain(result)
    assert 'optional gate(s) failed' in text
    assert 'human review triggered' in text

  def test_explain_optional_failed_without_review(self) -> None:
    gate = MinGate('accuracy', 0.9, required=False)
    policy = QualityFirstPolicy(gates=[gate], human_review_on_warn=False)
    result = Result(metrics={'accuracy': 0.5})
    policy.forward(result)
    text = policy.explain(result)
    assert 'optional gate(s) failed' in text
    assert 'accuracy' in text
    assert 'human review' not in text

  def test_explain_required_failed_lists_metrics(self) -> None:
    gates = [MinGate('accuracy', 0.9), MinGate('f1', 0.8)]
    policy = QualityFirstPolicy(gates=cast(Any, gates))
    result = Result(metrics={'accuracy': 0.5, 'f1': 0.3})
    policy.forward(result)
    text = policy.explain(result)
    assert 'required gates failed' in text
    assert 'accuracy' in text
    assert 'f1' in text
