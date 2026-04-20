"""Regression comparison utilities.

Pure functions for metric comparison. No classes.
"""

from autopilot.core.stage_models import RegressionAnalysis
from autopilot.tracking.io import atomic_write_json, read_json
from pathlib import Path
import autopilot.core.paths as paths


def _higher_is_better(metric: str, per_category: dict[str, dict] | None) -> bool:
  if per_category and metric in per_category:
    return per_category[metric].get('higher_is_better', True)
  if metric == 'error_rate':
    return False
  return True


def _is_significant_change(
  delta: float,
  base_val: float,
  threshold_abs: float,
  threshold_pct: float,
) -> bool:
  if threshold_abs == 0.0 and threshold_pct == 0.0:
    return delta != 0.0
  sig_abs = threshold_abs > 0.0 and abs(delta) > threshold_abs
  sig_pct = threshold_pct > 0.0 and base_val != 0.0 and abs(delta) / abs(base_val) > threshold_pct
  return sig_abs or sig_pct


def compare_metrics(
  baseline: dict[str, float],
  candidate: dict[str, float],
  threshold_abs: float = 0.0,
  threshold_pct: float = 0.0,
  per_category: dict[str, dict] | None = None,
) -> RegressionAnalysis:
  """Compare two metric dicts. Returns verdict + per-key deltas."""
  per_category_deltas: dict[str, float] = {}
  regressions: list[dict] = []
  improvements: list[dict] = []

  common_keys = set(baseline.keys()) & set(candidate.keys())
  for key in sorted(common_keys):
    base_val = baseline[key]
    cand_val = candidate[key]
    delta = cand_val - base_val
    per_category_deltas[key] = delta

    higher = _higher_is_better(key, per_category)
    significant = _is_significant_change(delta, base_val, threshold_abs, threshold_pct)

    if higher:
      is_regression = delta < 0.0 and significant
      is_improvement = delta > 0.0 and significant
    else:
      is_regression = delta > 0.0 and significant
      is_improvement = delta < 0.0 and significant

    row = {'metric': key, 'delta': delta, 'baseline': base_val, 'candidate': cand_val}
    if is_regression:
      regressions.append(row)
    elif is_improvement:
      improvements.append(row)

  if regressions and not improvements:
    verdict = 'net_regression'
  elif regressions and improvements:
    verdict = 'mixed'
  else:
    verdict = 'net_improvement'

  return RegressionAnalysis(
    epoch=0,
    overall_verdict=verdict,
    per_category_deltas=per_category_deltas,
    regressions=regressions,
    improvements=improvements,
  )


def is_regression(analysis: RegressionAnalysis) -> bool:
  """True if analysis.overall_verdict == 'net_regression'."""
  return analysis.overall_verdict == 'net_regression'


def read_best_baseline(experiment_dir: Path) -> dict[str, float] | None:
  """Read best_baseline.json. None if no baseline yet."""
  data = read_json(paths.best_baseline_file(experiment_dir))
  if data is None:
    return None
  return data.get('metrics', {})


def write_best_baseline(experiment_dir: Path, epoch: int, metrics: dict[str, float]) -> None:
  """Atomically write best_baseline.json."""
  atomic_write_json(
    paths.best_baseline_file(experiment_dir),
    {'epoch': epoch, 'metrics': metrics},
  )
