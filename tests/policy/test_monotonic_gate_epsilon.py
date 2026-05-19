"""Tests for MonotonicGate epsilon (absolute tolerance) behavior."""

from autopilot.core.models import Result
from autopilot.core.types import GateResult
from autopilot.policy.gates import MonotonicGate
import pytest


def _make_result(metric: str, current: float, prev: float | None = None) -> Result:
  """Build a Result with current metric and optional _prev_ injection."""
  metrics: dict[str, float] = {metric: current}
  if prev is not None:
    metrics[f'_prev_{metric}'] = prev
  return Result(metrics=metrics)


class TestEpsilonZeroDefault:
  """Verify that default epsilon=0.0 preserves pre-existing strict behavior."""

  def test_epsilon_zero_default_unchanged_pass(self) -> None:
    gate = MonotonicGate('accuracy')
    result = _make_result('accuracy', current=0.9, prev=0.8)
    assert gate.forward(result) == GateResult.PASSED

  def test_epsilon_zero_default_unchanged_fail(self) -> None:
    gate = MonotonicGate('accuracy')
    result = _make_result('accuracy', current=0.79, prev=0.8)
    assert gate.forward(result) == GateResult.FAIL

  def test_epsilon_zero_non_increasing_pass(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing')
    result = _make_result('loss', current=0.3, prev=0.5)
    assert gate.forward(result) == GateResult.PASSED

  def test_epsilon_zero_non_increasing_fail(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing')
    result = _make_result('loss', current=0.6, prev=0.5)
    assert gate.forward(result) == GateResult.FAIL


class TestNonDecreasingWithEpsilon:
  """Non-decreasing direction: pass iff current >= prev - epsilon."""

  def test_non_decreasing_within_epsilon_passes(self) -> None:
    gate = MonotonicGate('accuracy', epsilon=0.1)
    result = _make_result('accuracy', current=0.75, prev=0.8)
    assert gate.forward(result) == GateResult.PASSED

  def test_non_decreasing_exceeds_epsilon_fails(self) -> None:
    gate = MonotonicGate('accuracy', epsilon=0.1)
    result = _make_result('accuracy', current=0.6, prev=0.8)
    assert gate.forward(result) == GateResult.FAIL

  def test_epsilon_exact_boundary(self) -> None:
    gate = MonotonicGate('accuracy', epsilon=0.25)
    result = _make_result('accuracy', current=0.75, prev=1.0)
    assert gate.forward(result) == GateResult.PASSED


class TestNonIncreasingWithEpsilon:
  """Non-increasing direction: pass iff current <= prev + epsilon."""

  def test_non_increasing_within_epsilon_passes(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing', epsilon=0.1)
    result = _make_result('loss', current=0.55, prev=0.5)
    assert gate.forward(result) == GateResult.PASSED

  def test_non_increasing_exceeds_epsilon_fails(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing', epsilon=0.1)
    result = _make_result('loss', current=0.7, prev=0.5)
    assert gate.forward(result) == GateResult.FAIL

  def test_non_increasing_exact_boundary(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing', epsilon=0.1)
    result = _make_result('loss', current=0.6, prev=0.5)
    assert gate.forward(result) == GateResult.PASSED


class TestEpsilonValidation:
  """Negative epsilon must raise ValueError."""

  def test_epsilon_negative_raises(self) -> None:
    with pytest.raises(ValueError, match='non-negative'):
      MonotonicGate('accuracy', epsilon=-0.01)

  def test_epsilon_zero_accepted(self) -> None:
    gate = MonotonicGate('accuracy', epsilon=0.0)
    assert gate._epsilon == 0.0


class TestFirstEpochWithEpsilon:
  """First epoch (no prior) should always pass regardless of epsilon."""

  def test_first_epoch_passes_with_epsilon(self) -> None:
    gate = MonotonicGate('accuracy', epsilon=0.5)
    result = Result(metrics={'accuracy': 0.1})
    assert gate.forward(result) == GateResult.PASSED

  def test_first_epoch_non_increasing_passes_with_epsilon(self) -> None:
    gate = MonotonicGate('loss', direction='non_increasing', epsilon=0.5)
    result = Result(metrics={'loss': 99.0})
    assert gate.forward(result) == GateResult.PASSED
