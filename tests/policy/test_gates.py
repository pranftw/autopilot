"""Tests for the Gate class hierarchy."""

from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.gates import CustomGate, Gate, MaxGate, MinGate, RangeGate
import pytest


class TestMinGate:
  def test_forward_passes_when_metric_gte_threshold(self) -> None:
    gate = MinGate('accuracy', 0.8)
    sc = Result(metrics={'accuracy': 0.85})
    assert gate.forward(sc) == GateResult.PASS

  def test_forward_fails_when_metric_lt_threshold(self) -> None:
    gate = MinGate('accuracy', 0.8)
    sc = Result(metrics={'accuracy': 0.5})
    assert gate.forward(sc) == GateResult.FAIL

  def test_explain_returns_readable_string(self) -> None:
    gate = MinGate('accuracy', 0.8)
    sc = Result(metrics={'accuracy': 0.85})
    text = gate.explain(sc)
    assert 'accuracy' in text
    assert '0.850' in text or '0.85' in text
    assert '>=' in text
    assert 'pass' in text


class TestMaxGate:
  def test_forward_passes_when_metric_lte_threshold(self) -> None:
    gate = MaxGate('loss', 1.0)
    sc = Result(metrics={'loss': 0.5})
    assert gate.forward(sc) == GateResult.PASS

  def test_forward_fails_when_metric_gt_threshold(self) -> None:
    gate = MaxGate('loss', 1.0)
    sc = Result(metrics={'loss': 1.5})
    assert gate.forward(sc) == GateResult.FAIL


class TestRangeGate:
  def test_forward_passes_when_inside_range(self) -> None:
    gate = RangeGate('score', 0.0, 1.0)
    sc = Result(metrics={'score': 0.5})
    assert gate.forward(sc) == GateResult.PASS

  def test_forward_fails_when_outside_range(self) -> None:
    gate = RangeGate('score', 0.2, 0.8)
    below = Result(metrics={'score': 0.1})
    above = Result(metrics={'score': 0.9})
    assert gate.forward(below) == GateResult.FAIL
    assert gate.forward(above) == GateResult.FAIL


class TestCustomGate:
  def test_forward_with_lambda(self) -> None:
    gate = CustomGate('loss', lambda v: v < 0.5)
    sc = Result(metrics={'loss': 0.3})
    assert gate.forward(sc) == GateResult.PASS
    sc_fail = Result(metrics={'loss': 0.6})
    assert gate.forward(sc_fail) == GateResult.FAIL


class TestMissingMetric:
  def test_min_gate_fail_when_metric_missing(self) -> None:
    gate = MinGate('accuracy', 0.8)
    sc = Result(metrics={})
    assert gate.forward(sc) == GateResult.FAIL

  def test_max_gate_fail_when_metric_missing(self) -> None:
    gate = MaxGate('loss', 1.0)
    sc = Result(metrics={'other': 1.0})
    assert gate.forward(sc) == GateResult.FAIL

  def test_range_gate_fail_when_metric_missing(self) -> None:
    gate = RangeGate('score', 0.0, 1.0)
    sc = Result(metrics={})
    assert gate.forward(sc) == GateResult.FAIL

  def test_custom_gate_fail_when_metric_missing(self) -> None:
    gate = CustomGate('x', lambda v: True)
    sc = Result(metrics={})
    assert gate.forward(sc) == GateResult.FAIL


class TestGateCallAndExplain:
  def test_call_wraps_forward(self) -> None:
    gate = MinGate('accuracy', 0.8)
    sc = Result(metrics={'accuracy': 0.9})
    assert gate(sc) == gate.forward(sc)

  def test_gate_explain_returns_readable_string(self) -> None:
    class PassGate(Gate):
      def forward(self, result: Result) -> GateResult:
        return GateResult.PASS

    gate = PassGate('metric_a')
    sc = Result(metrics={})
    assert gate.explain(sc) == 'PassGate(metric_a): pass'


class TestGateReprAndRequired:
  def test_repr_includes_metric_name(self) -> None:
    gate = MinGate('f1', 0.75)
    rep = repr(gate)
    assert 'f1' in rep
    assert 'MinGate' in rep
    assert 'required=True' in rep

  def test_required_defaults_to_true(self) -> None:
    gate = MinGate('accuracy', 0.8)
    assert gate.required is True

  def test_required_false(self) -> None:
    gate = MinGate('accuracy', 0.8, required=False)
    assert gate.required is False
    assert 'required=False' in repr(gate)


class TestBaseGate:
  def test_forward_raises_not_implemented(self) -> None:
    gate = Gate('m')
    sc = Result(metrics={'m': 1.0})
    with pytest.raises(NotImplementedError):
      gate.forward(sc)
