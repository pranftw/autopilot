"""Tests for regression comparison utilities."""

from autopilot.core.regression import (
  compare_metrics,
  is_regression,
  read_best_baseline,
  write_best_baseline,
)


class TestCompareMetrics:
  def test_clear_improvement(self):
    result = compare_metrics({'accuracy': 0.7}, {'accuracy': 0.9})
    assert result.overall_verdict == 'net_improvement'

  def test_clear_regression(self):
    result = compare_metrics({'accuracy': 0.9}, {'accuracy': 0.7})
    assert result.overall_verdict == 'net_regression'

  def test_mixed_results(self):
    result = compare_metrics(
      {'accuracy': 0.7, 'error_rate': 0.1},
      {'accuracy': 0.9, 'error_rate': 0.3},
    )
    assert result.overall_verdict == 'mixed'

  def test_equal_metrics(self):
    result = compare_metrics({'accuracy': 0.8}, {'accuracy': 0.8})
    assert result.overall_verdict == 'net_improvement'

  def test_threshold_abs(self):
    result = compare_metrics(
      {'accuracy': 0.8},
      {'accuracy': 0.79},
      threshold_abs=0.05,
    )
    assert result.overall_verdict == 'net_improvement'

  def test_threshold_pct(self):
    result = compare_metrics(
      {'accuracy': 0.8},
      {'accuracy': 0.79},
      threshold_pct=0.05,
    )
    assert result.overall_verdict == 'net_improvement'

  def test_threshold_exceeded(self):
    result = compare_metrics(
      {'accuracy': 0.8},
      {'accuracy': 0.5},
      threshold_abs=0.05,
      threshold_pct=0.05,
    )
    assert result.overall_verdict == 'net_regression'

  def test_per_category_deltas(self):
    result = compare_metrics(
      {'accuracy': 0.7, 'recall': 0.6},
      {'accuracy': 0.9, 'recall': 0.5},
    )
    assert result.per_category_deltas['accuracy'] > 0
    assert result.per_category_deltas['recall'] < 0

  def test_empty_baseline(self):
    result = compare_metrics({}, {'accuracy': 0.8})
    assert result.overall_verdict == 'net_improvement'

  def test_empty_candidate(self):
    result = compare_metrics({'accuracy': 0.8}, {})
    assert result.overall_verdict == 'net_improvement'

  def test_single_metric(self):
    result = compare_metrics({'accuracy': 0.7}, {'accuracy': 0.8})
    assert result.overall_verdict == 'net_improvement'

  def test_disjoint_keys(self):
    result = compare_metrics({'a': 1.0}, {'b': 2.0})
    assert result.overall_verdict == 'net_improvement'

  def test_subset_keys(self):
    result = compare_metrics(
      {'a': 0.8, 'b': 0.7, 'c': 0.6},
      {'a': 0.9},
    )
    assert 'a' in result.per_category_deltas
    assert 'b' not in result.per_category_deltas


class TestIsRegression:
  def test_true(self):
    result = compare_metrics({'accuracy': 0.9}, {'accuracy': 0.5})
    assert is_regression(result)

  def test_false(self):
    result = compare_metrics({'accuracy': 0.5}, {'accuracy': 0.9})
    assert not is_regression(result)


class TestBestBaseline:
  def test_read_no_file(self, tmp_path):
    assert read_best_baseline(tmp_path) is None

  def test_read_exists(self, tmp_path):
    write_best_baseline(tmp_path, epoch=3, metrics={'accuracy': 0.85})
    result = read_best_baseline(tmp_path)
    assert result == {'accuracy': 0.85}

  def test_write_creates_file(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'x': 0.5})
    assert (tmp_path / 'best_baseline.json').exists()

  def test_write_overwrites(self, tmp_path):
    write_best_baseline(tmp_path, epoch=1, metrics={'x': 0.5})
    write_best_baseline(tmp_path, epoch=2, metrics={'x': 0.9})
    result = read_best_baseline(tmp_path)
    assert result == {'x': 0.9}
