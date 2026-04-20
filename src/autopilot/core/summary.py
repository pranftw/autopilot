"""Experiment summary functions.

Pure functions -- no class (subclassing is not expected).
"""

from autopilot.core.cost_tracker import CostTracker
from autopilot.core.stage_io import read_epoch_artifact
from autopilot.core.stage_models import ExperimentSummaryData, RegressionAnalysis
from autopilot.tracking.io import atomic_write_json, read_jsonl
from pathlib import Path
from typing import Any


def build_experiment_summary(
  experiment_dir: Path,
  loop_result: dict[str, Any],
  cost_tracker: CostTracker | None = None,
) -> ExperimentSummaryData:
  """Read artifacts from experiment_dir, assemble structured summary."""
  epochs = loop_result.get('epochs', [])
  total_epochs = loop_result.get('total_epochs', len(epochs))

  final_metrics: dict[str, float] = {}
  best_epoch = 0
  best_accuracy = -1.0

  for ep in epochs:
    metrics = ep.get('metrics', {})
    acc = metrics.get('accuracy', 0.0)
    if acc > best_accuracy:
      best_accuracy = acc
      best_epoch = ep.get('epoch', 0)
    final_metrics = metrics

  regressions: list[RegressionAnalysis] = []
  for ep in epochs:
    ep_num = ep.get('epoch', 0)
    ra_data = read_epoch_artifact(experiment_dir, ep_num, 'regression_analysis.json')
    if ra_data:
      regressions.append(RegressionAnalysis.from_dict(ra_data))

  kb_path = experiment_dir / 'knowledge_base.jsonl'
  kb_lines = read_jsonl(kb_path, strict=False) if kb_path.exists() else []
  memory_entries = len(kb_lines)

  cost_total = cost_tracker.total() if cost_tracker else None

  slug = experiment_dir.name

  return ExperimentSummaryData(
    slug=slug,
    total_epochs=total_epochs,
    final_metrics=final_metrics,
    best_epoch=best_epoch,
    stop_reason=loop_result.get('stop_reason'),
    last_good_epoch=loop_result.get('last_good_epoch', 0),
    promotions=[],
    regressions=regressions,
    cost_total=cost_total,
    memory_entries=memory_entries,
  )


def write_experiment_summary(
  experiment_dir: Path,
  summary: ExperimentSummaryData,
) -> Path:
  """Write summary.json. Returns path."""
  path = experiment_dir / 'summary.json'
  atomic_write_json(path, summary.to_dict())
  return path
