"""Status command -- comprehensive experiment health overview."""

from autopilot.cli.command import Command
from autopilot.core.memory import FileMemory
from autopilot.core.stage_io import read_epoch_artifact, read_experiment_artifact
from autopilot.tracking.manifest import load_manifest
from pathlib import Path
from typing import Any
import argparse


class StatusCommand(Command):
  """Show experiment status, regression state, and recent metrics."""

  name = 'status'
  help = 'show experiment status'

  def forward(self, ctx: Any, args: argparse.Namespace) -> None:
    experiment = ctx.experiment
    if not experiment:
      ctx.output.error('no experiment specified (use --experiment)')
      return

    exp_dir = ctx.experiment_dir()
    try:
      manifest = load_manifest(exp_dir)
    except Exception as e:
      ctx.output.error(f'cannot load experiment: {e}')
      return

    trained_epochs, latest_epoch = _scan_epoch_dirs(exp_dir)
    epoch = latest_epoch if latest_epoch > 0 else manifest.current_epoch

    result: dict[str, Any] = {
      'slug': manifest.slug,
      'epoch': manifest.current_epoch,
      'trained_epochs': trained_epochs,
      'decision': manifest.decision,
      'decision_reason': manifest.decision_reason,
    }

    summary = read_experiment_artifact(exp_dir, 'summary.json')
    if summary:
      result['stop_reason'] = summary.get('stop_reason')
      result['last_good_epoch'] = summary.get('last_good_epoch')

    run_state = read_experiment_artifact(exp_dir, 'run_state.json')
    if run_state:
      if run_state.get('status') == 'running':
        result['stop_reason'] = 'crash'
      elif not result.get('stop_reason'):
        result['stop_reason'] = run_state.get('stop_reason')
      if not result.get('last_good_epoch'):
        result['last_good_epoch'] = run_state.get('last_good_epoch')

    if epoch > 0:
      metrics = read_epoch_artifact(exp_dir, epoch, 'epoch_metrics.json')
      if metrics:
        result['last_metrics'] = metrics

    baseline = read_experiment_artifact(exp_dir, 'best_baseline.json')
    if baseline:
      result['best_baseline'] = baseline

    regression = _find_latest_regression(exp_dir, latest_epoch)
    if regression:
      result['regression'] = regression

    memory = FileMemory(exp_dir)
    mem_ctx = memory.context(epoch=epoch)
    result['memory'] = {
      'total_records': mem_ctx.total_records,
      'blocked_strategies': mem_ctx.blocked,
    }

    ctx.output.result(result)


def _scan_epoch_dirs(exp_dir: Any) -> tuple[int, int]:
  """Return (count, highest_epoch_number) from epoch dirs."""
  exp_path = Path(exp_dir)
  count = 0
  highest = 0
  if not exp_path.exists():
    return 0, 0
  for child in exp_path.iterdir():
    if child.is_dir() and child.name.startswith('epoch_'):
      try:
        num = int(child.name.split('_', 1)[1])
        count += 1
        highest = max(highest, num)
      except (ValueError, IndexError):
        pass
  return count, highest


def _find_latest_regression(exp_dir: Any, max_epoch: int) -> dict[str, Any] | None:
  for ep in range(max_epoch, 0, -1):
    data = read_epoch_artifact(exp_dir, ep, 'regression_analysis.json')
    if data and data.get('overall_verdict') in ('net_regression', 'mixed'):
      regressed = [r.get('metric', '') for r in data.get('regressions', [])]
      return {
        'epoch': ep,
        'verdict': data['overall_verdict'],
        'regressed_metrics': regressed,
      }
  return None
