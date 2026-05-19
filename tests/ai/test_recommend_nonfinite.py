"""Tests for non-finite metric value handling in _passes_thresholds."""

from autopilot.ai.recommend import _passes_thresholds


class TestPassesThresholdsNonFinite:
  """Non-finite values (NaN, Inf) must fail all threshold predicates."""

  def test_passes_thresholds_nan_metric_gt_fails(self) -> None:
    assert _passes_thresholds({'x': float('nan')}, [('x', 0.5)], None) is False

  def test_passes_thresholds_nan_metric_lt_fails(self) -> None:
    assert _passes_thresholds({'x': float('nan')}, None, [('x', 0.5)]) is False

  def test_passes_thresholds_positive_inf_metric_gt_fails(self) -> None:
    assert _passes_thresholds({'x': float('inf')}, [('x', 0.5)], None) is False

  def test_passes_thresholds_negative_inf_metric_lt_fails(self) -> None:
    assert _passes_thresholds({'x': float('-inf')}, None, [('x', 0.5)]) is False


class TestPassesThresholdsFinite:
  """Finite values still obey normal threshold semantics."""

  def test_passes_thresholds_finite_gt_satisfied(self) -> None:
    assert _passes_thresholds({'x': 1.0}, [('x', 0.5)], None) is True

  def test_passes_thresholds_finite_gt_violated(self) -> None:
    assert _passes_thresholds({'x': 0.3}, [('x', 0.5)], None) is False

  def test_passes_thresholds_finite_lt_satisfied(self) -> None:
    assert _passes_thresholds({'x': 0.3}, None, [('x', 0.5)]) is True

  def test_passes_thresholds_finite_lt_violated(self) -> None:
    assert _passes_thresholds({'x': 1.0}, None, [('x', 0.5)]) is False
