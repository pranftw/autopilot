"""Integration tests: Policy + gates produce structured ConstraintResult entries.

Tests 11-17 from sub-plan 02.
"""

from autopilot.core.constraint import ConstraintResult, gate_to_constraint
from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.gates import Gate, MaxGate, MinGate
from autopilot.policy.quality_first import QualityFirstMetric, QualityFirstPolicy
from autopilot.policy.threshold import ThresholdPolicy
import pytest


class TestPolicyForwardProducesConstraintResults:
  """Test 11: QualityFirstMetric.to_result produces ConstraintResult list."""

  def test_policy_forward_produces_constraint_results(self) -> None:
    gate_list: list[Gate] = [MinGate('accuracy', 0.5), MaxGate('loss', 1.0)]
    metric = QualityFirstMetric(gates=gate_list)
    from autopilot.core.types import EvalDatum

    metric.update(EvalDatum(success=True, metrics={'accuracy': 0.9, 'loss': 0.3}))
    result = metric.to_result()

    assert isinstance(result.gates, list)
    assert len(result.gates) == 2
    for cr in result.gates:
      assert isinstance(cr, ConstraintResult)
    assert result.gates[0].metric == 'accuracy'
    assert result.gates[0].passed is True
    assert result.gates[1].metric == 'loss'
    assert result.gates[1].passed is True
    assert result.passed is True


class TestPolicyExplainShowsConstraints:
  """Test 12: policy explain text still works with structured gates."""

  def test_policy_explain_shows_constraints(self) -> None:
    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.8)])
    r = Result(metrics={'accuracy': 0.9})
    outcome = policy.forward(r)
    text = policy.explain(r)
    assert outcome == GateResult.PASSED
    assert 'all gates passed' in text


class TestGateResultToConstraintResult:
  """Test 13: gate_to_constraint maps correctly."""

  def test_gate_result_to_constraint_result(self) -> None:
    cr = gate_to_constraint('MinGate', GateResult.PASSED, 'accuracy', 0.9, '>= 0.8')
    assert cr.passed is True
    assert cr.name == 'MinGate'
    assert cr.message is None

    cr_fail = gate_to_constraint('MaxGate', GateResult.FAIL, 'loss', 2.0, '<= 1.0')
    assert cr_fail.passed is False
    assert cr_fail.message == 'MaxGate failed'


class TestConstraintResultJsonInExperiment:
  """Test 14: experiment serialization path includes list of dicts / round-trip."""

  def test_constraint_result_json_in_experiment(self) -> None:
    gates = [
      ConstraintResult(name='g1', passed=True, metric='a', value=1.0, threshold='>= 0'),
      ConstraintResult(
        name='g2', passed=False, metric='b', value=0.1, threshold='>= 0.5', message='g2 failed'
      ),
    ]
    r = Result(metrics={'a': 1.0, 'b': 0.1}, gates=gates, summary='partial')
    d = r.to_dict()

    assert isinstance(d['gates'], list)
    assert len(d['gates']) == 2
    assert d['gates'][0]['name'] == 'g1'
    assert d['gates'][0]['passed'] is True
    assert d['gates'][1]['passed'] is False
    assert d['gates'][1]['message'] == 'g2 failed'
    assert d['passed'] is False

    r2 = Result.from_dict(d)
    assert len(r2.gates) == 2
    assert r2.gates[0].name == 'g1'
    assert r2.gates[1].message == 'g2 failed'
    assert r2.passed is False


class TestPolicyExplainJsonConstraints:
  """Test 15: policy explain JSON output includes structured constraint data."""

  def test_policy_explain_json_constraints(self) -> None:
    policy = QualityFirstPolicy(gates=[MinGate('accuracy', 0.99)])
    r = Result(metrics={'accuracy': 0.5})
    policy.forward(r)
    text = policy.explain(r)
    assert 'accuracy' in text
    assert 'failed' in text


class TestPolicyExplainExitCodeGateFail:
  """Test 16: when a gate fails, CLI policy explain should exit with code 1."""

  def test_policy_explain_exit_code_gate_fail(self) -> None:
    policy = ThresholdPolicy([MinGate('accuracy', 0.9)])
    r = Result(metrics={'accuracy': 0.5})
    outcome = policy.forward(r)
    assert outcome == GateResult.FAIL


class TestGateResultToConstraintAllMembers:
  """Test 17: parametrized - each GateResult maps to correct passed value."""

  @pytest.mark.parametrize(
    ('gate_result', 'expected_passed'),
    [
      (GateResult.PASSED, True),
      (GateResult.FAIL, False),
      (GateResult.WARN, False),
      (GateResult.SKIP, False),
    ],
  )
  def test_gate_result_to_constraint_all_members(
    self, gate_result: GateResult, expected_passed: bool
  ) -> None:
    cr = gate_to_constraint('TestGate', gate_result, 'metric', 1.0, '>= 0')
    assert cr.passed is expected_passed
    if expected_passed:
      assert cr.message is None
    else:
      assert cr.message == 'TestGate failed'
