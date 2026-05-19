"""Tests for MonotonicGate.explain epsilon-accurate text and single forward call."""

from autopilot.core.models import Result
from autopilot.policy.gates import MonotonicGate
from unittest.mock import Mock


def _make_result(metric: str, current: float, prev: float | None = None) -> Result:
  """Build a Result with current metric and optional _prev_ injection."""
  metrics: dict[str, float] = {metric: current}
  if prev is not None:
    metrics[f'_prev_{metric}'] = prev
  return Result(metrics=metrics)


class TestExplainEpsilonBound:
  """Explain text must show epsilon-adjusted bound matching forward() logic."""

  def test_explain_non_decreasing_epsilon_bound_in_text(self) -> None:
    gate = MonotonicGate('m', epsilon=0.1)
    result = _make_result('m', 0.95, prev=1.0)
    explanation = gate.explain(result)
    assert '0.9' in explanation
    assert '>=' in explanation
    assert 'PASSED' in explanation

  def test_explain_non_increasing_epsilon_bound_in_text(self) -> None:
    gate = MonotonicGate('m', direction='non_increasing', epsilon=0.1)
    result = _make_result('m', 1.05, prev=1.0)
    explanation = gate.explain(result)
    assert '1.1' in explanation
    assert '<=' in explanation
    assert 'PASSED' in explanation

  def test_explain_zero_epsilon_uses_prev_as_bound(self) -> None:
    gate = MonotonicGate('m', epsilon=0.0)
    result = _make_result('m', 1.0, prev=0.8)
    explanation = gate.explain(result)
    assert '0.8' in explanation
    assert '>=' in explanation


class TestExplainEdgeCases:
  """Explain handles missing prior and missing current gracefully."""

  def test_explain_no_prior_still_reports_baseline(self) -> None:
    gate = MonotonicGate('m', epsilon=0.1)
    result = _make_result('m', 0.5)
    explanation = gate.explain(result)
    assert '(no prior)' in explanation

  def test_explain_missing_current_delegates_to_format_missing(self) -> None:
    gate = MonotonicGate('m', epsilon=0.1)
    result = Result(metrics={})
    explanation = gate.explain(result)
    expected = gate.format_missing_explanation()
    assert explanation == expected


class TestExplainForwardCallCount:
  """Explain must call forward() exactly once, not twice."""

  def test_monotonic_explain_calls_forward_once(self) -> None:
    gate = MonotonicGate('m', epsilon=0.1)
    result = _make_result('m', 0.95, prev=1.0)
    original_forward = gate.forward
    mock_forward = Mock(wraps=original_forward)
    gate.forward = mock_forward  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]
    gate.explain(result)
    assert mock_forward.call_count == 1
