"""Tests for core/status.py experiment status gathering."""

from autopilot.core.artifacts.epoch import MetricComparisonArtifact
from autopilot.core.artifacts.experiment import SummaryArtifact
from autopilot.core.errors import TrackingError
from autopilot.core.memory import FileMemory
from autopilot.core.status import get_experiment_status
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
import pytest

_mc = MetricComparisonArtifact()
_summary = SummaryArtifact()


def _write_manifest(exp_dir: Path, slug: str = 'test-exp', epoch: int = 1) -> None:
  atomic_write_json(
    exp_dir / 'manifest.json',
    {
      'slug': slug,
      'title': 'Test',
      'current_epoch': epoch,
      'decision': None,
      'decision_reason': None,
    },
  )


class TestGetExperimentStatus:
  def test_basic_status(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir()
    _write_manifest(exp_dir, epoch=0)
    result = get_experiment_status(exp_dir)
    assert result['slug'] == 'test-exp'
    assert result['epoch'] == 0
    assert 'memory' in result

  def test_includes_metrics(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir()
    _write_manifest(exp_dir, epoch=2)
    (exp_dir / 'epoch_2').mkdir()
    _mc.write(
      {
        'per_metric_deltas': {'accuracy': 0.1},
        'regressions': [],
        'improvements': [{'metric': 'accuracy', 'delta': 0.1, 'baseline': 0.8, 'candidate': 0.9}],
      },
      exp_dir,
      epoch=2,
    )
    result = get_experiment_status(exp_dir)
    assert result['trained_epochs'] == 1
    assert 'last_metrics' in result
    assert result['last_metrics']['accuracy'] == 0.9

  def test_includes_stop_reason(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir()
    _write_manifest(exp_dir, epoch=3)
    _summary.write({'stop_reason': 'plateau', 'last_good_epoch': 2}, exp_dir)
    result = get_experiment_status(exp_dir)
    assert result['stop_reason'] == 'plateau'
    assert result['last_good_epoch'] == 2

  def test_crash_detection(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir()
    _write_manifest(exp_dir, epoch=5)
    atomic_write_json(
      exp_dir / 'run_state.json',
      {'epoch': 5, 'status': 'running'},
    )
    result = get_experiment_status(exp_dir)
    assert result['stop_reason'] == 'crash'

  def test_includes_baseline(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir()
    _write_manifest(exp_dir, epoch=2)
    atomic_write_json(exp_dir / 'best_baseline.json', {'accuracy': 0.9})
    result = get_experiment_status(exp_dir)
    assert result['best_baseline'] == {'accuracy': 0.9}

  def test_includes_regression(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir()
    _write_manifest(exp_dir, epoch=3)
    (exp_dir / 'epoch_3').mkdir()
    _mc.write(
      {
        'regression_detected': True,
        'regressions': [{'metric': 'accuracy', 'delta': -0.4}],
        'improvements': [],
      },
      exp_dir,
      epoch=3,
    )
    result = get_experiment_status(exp_dir)
    assert result['regression']['verdict'] == 'regression'
    assert 'accuracy' in result['regression']['regressed_metrics']

  def test_includes_memory_context(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir()
    _write_manifest(exp_dir, epoch=1)
    memory = FileMemory(exp_dir)
    memory.learn(epoch=1, outcome='worked', metrics={'accuracy': 0.8})
    result = get_experiment_status(exp_dir)
    assert result['memory']['total_records'] == 1

  def test_missing_manifest_raises(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'nonexistent'
    exp_dir.mkdir()
    with pytest.raises(TrackingError):
      get_experiment_status(exp_dir)

  def test_trained_epochs_count(self, tmp_path: Path) -> None:
    exp_dir = tmp_path / 'test-exp'
    exp_dir.mkdir()
    _write_manifest(exp_dir, epoch=1)
    for ep in range(1, 4):
      (exp_dir / f'epoch_{ep}').mkdir()
    result = get_experiment_status(exp_dir)
    assert result['trained_epochs'] == 3
