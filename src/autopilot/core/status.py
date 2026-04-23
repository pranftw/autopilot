"""Experiment status gathering.

Library function for collecting experiment health: manifest, run state,
baseline, metrics, memory, and regression information.
"""

from autopilot.core.artifacts.experiment import BaselineArtifact, RunStateArtifact, SummaryArtifact
from autopilot.core.comparison import load_metric_comparison
from autopilot.core.memory import FileMemory
from autopilot.tracking.manifest import load_manifest
from pathlib import Path
from typing import Any


def get_experiment_status(experiment_dir: Path) -> dict[str, Any]:
  """Gather experiment status: manifest, run_state, baseline, metrics, memory."""
  manifest = load_manifest(experiment_dir)

  trained_epochs, latest_epoch = _scan_epoch_dirs(experiment_dir)
  epoch = latest_epoch if latest_epoch > 0 else manifest.current_epoch

  result: dict[str, Any] = {
    'slug': manifest.slug,
    'epoch': manifest.current_epoch,
    'trained_epochs': trained_epochs,
    'decision': manifest.decision,
    'decision_reason': manifest.decision_reason,
  }

  summary = SummaryArtifact().read_raw(experiment_dir)
  if summary:
    result['stop_reason'] = summary.get('stop_reason')
    result['last_good_epoch'] = summary.get('last_good_epoch')

  run_state = RunStateArtifact().read_raw(experiment_dir)
  if run_state:
    if run_state['status'] == 'running':
      result['stop_reason'] = 'crash'
    else:
      if 'stop_reason' not in result or result['stop_reason'] is None:
        result['stop_reason'] = run_state.get('stop_reason')
      if 'last_good_epoch' not in result or result['last_good_epoch'] is None:
        result['last_good_epoch'] = run_state.get('last_good_epoch')

  if epoch > 0:
    mc = load_metric_comparison(experiment_dir, epoch)
    if mc and mc.candidate_metrics:
      result['last_metrics'] = mc.candidate_metrics

  baseline_data = BaselineArtifact().read_raw(experiment_dir)
  if baseline_data:
    result['best_baseline'] = baseline_data

  regression = _find_latest_regression(experiment_dir, latest_epoch)
  if regression:
    result['regression'] = regression

  memory = FileMemory(experiment_dir)
  mem_ctx = memory.context(epoch=epoch)
  result['memory'] = {
    'total_records': mem_ctx.total_records,
    'blocked_strategies': mem_ctx.blocked,
  }

  return result


def _scan_epoch_dirs(exp_dir: Path) -> tuple[int, int]:
  """Return (count, highest_epoch_number) from epoch dirs."""
  count = 0
  highest = 0
  if not exp_dir.exists():
    return 0, 0
  for child in exp_dir.iterdir():
    if child.is_dir() and child.name.startswith('epoch_'):
      try:
        num = int(child.name.split('_', 1)[1])
        count += 1
        highest = max(highest, num)
      except (ValueError, IndexError):
        pass
  return count, highest


def _find_latest_regression(exp_dir: Path, max_epoch: int) -> dict[str, Any] | None:
  """Walk backwards from max_epoch to find the most recent regression."""
  for ep in range(max_epoch, 0, -1):
    mc = load_metric_comparison(exp_dir, ep)
    if mc is None:
      continue
    if mc.regression_detected or mc.is_mixed:
      regressed = [r['metric'] for r in mc.regressions]
      verdict = 'regression' if mc.regression_detected else 'mixed'
      return {
        'epoch': ep,
        'verdict': verdict,
        'regressed_metrics': regressed,
      }
  return None
