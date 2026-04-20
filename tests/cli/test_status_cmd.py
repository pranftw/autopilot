"""Tests for status CLI command."""

from autopilot.cli.commands.status_cmd import StatusCommand
from autopilot.cli.output import Output
from autopilot.core.stage_io import write_epoch_artifact, write_experiment_artifact
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
from unittest.mock import MagicMock
import json


def _make_ctx(tmp_path: Path, experiment: str = 'test-exp') -> MagicMock:
  ctx = MagicMock()
  ctx.experiment = experiment
  ctx.output = Output(use_json=True)
  exp_dir = tmp_path / experiment
  exp_dir.mkdir(parents=True, exist_ok=True)
  ctx.experiment_dir.return_value = exp_dir
  return ctx


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


class TestStatusCommand:
  def test_instantiates(self):
    cmd = StatusCommand()
    assert cmd.name == 'status'

  def test_no_experiment(self):
    cmd = StatusCommand()
    ctx = MagicMock()
    ctx.experiment = ''
    ctx.output = MagicMock()
    args = MagicMock()
    cmd.forward(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_happy_path(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _write_manifest(exp_dir, epoch=2)
    write_epoch_artifact(exp_dir, 2, 'epoch_metrics.json', {'accuracy': 0.9, 'total': 10})

    cmd = StatusCommand()
    args = MagicMock()
    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['slug'] == 'test-exp'
    assert envelope['result']['epoch'] == 2
    assert envelope['result']['trained_epochs'] == 1
    assert 'last_metrics' in envelope['result']
    assert envelope['result']['last_metrics']['accuracy'] == 0.9

  def test_includes_memory_context(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _write_manifest(exp_dir, epoch=0)

    cmd = StatusCommand()
    args = MagicMock()
    cmd.forward(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert 'memory' in envelope['result']
    assert 'total_records' in envelope['result']['memory']

  def test_includes_stop_reason_from_summary(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _write_manifest(exp_dir, epoch=3)
    write_experiment_artifact(
      exp_dir,
      'summary.json',
      {
        'stop_reason': 'plateau',
        'last_good_epoch': 2,
      },
    )

    cmd = StatusCommand()
    cmd.forward(ctx, MagicMock())
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['stop_reason'] == 'plateau'
    assert r['last_good_epoch'] == 2

  def test_includes_regression_from_epoch(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _write_manifest(exp_dir, epoch=3)
    write_epoch_artifact(exp_dir, 3, 'epoch_metrics.json', {'accuracy': 0.4})
    write_epoch_artifact(
      exp_dir,
      3,
      'regression_analysis.json',
      {
        'overall_verdict': 'net_regression',
        'regressions': [{'metric': 'accuracy', 'delta': -0.4}],
        'improvements': [],
      },
    )

    cmd = StatusCommand()
    cmd.forward(ctx, MagicMock())
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['regression']['verdict'] == 'net_regression'
    assert 'accuracy' in r['regression']['regressed_metrics']

  def test_includes_best_baseline(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _write_manifest(exp_dir, epoch=2)
    write_epoch_artifact(exp_dir, 2, 'epoch_metrics.json', {'accuracy': 0.8})
    atomic_write_json(exp_dir / 'best_baseline.json', {'accuracy': 0.9})

    cmd = StatusCommand()
    cmd.forward(ctx, MagicMock())
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['best_baseline'] == {'accuracy': 0.9}

  def test_crash_detection_from_run_state(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _write_manifest(exp_dir, epoch=5)
    atomic_write_json(
      exp_dir / 'run_state.json',
      {
        'epoch': 5,
        'status': 'running',
      },
    )

    cmd = StatusCommand()
    cmd.forward(ctx, MagicMock())
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['stop_reason'] == 'crash'

  def test_trained_epochs_count(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    _write_manifest(exp_dir, epoch=1)
    for ep in range(1, 4):
      (exp_dir / f'epoch_{ep}').mkdir()

    cmd = StatusCommand()
    cmd.forward(ctx, MagicMock())
    captured = capsys.readouterr()
    r = json.loads(captured.out)['result']
    assert r['trained_epochs'] == 3
