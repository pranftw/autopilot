"""Experiment summary functions.

Pure functions -- no class (subclassing is not expected).
Data model: ExperimentSummaryData.
"""

from autopilot.core.artifacts.experiment import SummaryArtifact
from autopilot.core.callbacks.cost import CostEntry, CostTrackerCallback
from autopilot.core.comparison import MetricComparison, load_metric_comparison
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ExperimentSummaryData(DictMixin):
  """Final experiment report."""

  slug: str | None = None
  total_epochs: int = 0
  final_metrics: dict[str, float] = field(default_factory=dict)
  best_epoch: int = 0
  stop_reason: str | None = None
  last_good_epoch: int = 0
  promotions: list[dict] = field(default_factory=list)
  comparisons: list[MetricComparison] = field(default_factory=list)
  cost_total: CostEntry | None = None
  memory_entries: int = 0

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ExperimentSummaryData':
    data = dict(data)
    data['comparisons'] = [MetricComparison.from_dict(c) for c in data.get('comparisons', [])]
    ct = data.get('cost_total')
    if ct is not None and isinstance(ct, dict):
      data['cost_total'] = CostEntry.from_dict(ct)
    else:
      data['cost_total'] = None
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def build_experiment_summary(
  experiment_dir: Path,
  loop_result: dict[str, Any],
  cost_tracker: CostTrackerCallback | None = None,
  monitor: str | None = None,
  monitor_higher_is_better: bool = True,
  memory_entries: int = 0,
) -> ExperimentSummaryData:
  """Read artifacts from experiment_dir, assemble structured summary."""
  epochs = loop_result.get('epochs', [])
  total_epochs = loop_result.get('total_epochs', len(epochs))

  final_metrics: dict[str, float] = {}
  best_epoch = 0
  best_value: float | None = None

  for ep in epochs:
    metrics = ep.get('metrics', {})
    if monitor and monitor in metrics:
      val = metrics[monitor]
      if best_value is None:
        is_better = True
      elif monitor_higher_is_better:
        is_better = val > best_value
      else:
        is_better = val < best_value
      if is_better:
        best_value = val
        best_epoch = ep.get('epoch', 0)
    elif not monitor and metrics:
      first_key = next(iter(metrics))
      val = metrics[first_key]
      if best_value is None or val > best_value:
        best_value = val
        best_epoch = ep.get('epoch', 0)
    final_metrics = metrics

  comparisons: list[MetricComparison] = []
  for ep in epochs:
    ep_num = ep.get('epoch', 0)
    mc = load_metric_comparison(experiment_dir, ep_num)
    if mc:
      comparisons.append(mc)

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
    comparisons=comparisons,
    cost_total=cost_total,
    memory_entries=memory_entries,
  )


def write_experiment_summary(
  experiment_dir: Path,
  summary: ExperimentSummaryData,
) -> Path:
  """Write summary.json. Returns path."""
  return SummaryArtifact().write(summary.to_dict(), experiment_dir)
