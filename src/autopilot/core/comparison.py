"""Metric comparison utilities.

Pure functions for metric comparison, plus the canonical typed reader.
Data models: MetricComparison, EpochMetrics.
"""

from autopilot.core.artifacts.epoch import MetricComparisonArtifact
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpochMetrics(DictMixin):
  """Structured forward pass result per epoch."""

  epoch: int = 0
  split: str | None = None
  total: int = 0
  passed: int = 0
  failed: int = 0
  accuracy: float = 0.0
  error_rate: float = 0.0
  latency_p95_ms: float = 0.0
  delta: dict[str, float] = field(default_factory=dict)
  gates: dict[str, str] = field(default_factory=dict)
  data_path: str | None = None


@dataclass
class MetricComparison(DictMixin):
  """Result of compare_metrics(). Booleans computed from lists, not magic strings."""

  epoch: int = 0
  per_metric_deltas: dict[str, float] = field(default_factory=dict)
  regressions: list[dict] = field(default_factory=list)
  improvements: list[dict] = field(default_factory=list)

  @property
  def regression_detected(self) -> bool:
    return bool(self.regressions) and not bool(self.improvements)

  @property
  def improvement_detected(self) -> bool:
    return bool(self.improvements) and not bool(self.regressions)

  @property
  def is_mixed(self) -> bool:
    return bool(self.regressions) and bool(self.improvements)

  @property
  def candidate_metrics(self) -> dict[str, float]:
    """Extract per-metric candidate values from comparison rows."""
    metrics: dict[str, float] = {}
    for row in [*self.regressions, *self.improvements]:
      if 'candidate' in row:
        metrics[row['metric']] = row['candidate']
    return metrics

  def to_dict(self) -> dict[str, Any]:
    d = super().to_dict()
    d['regression_detected'] = self.regression_detected
    d['improvement_detected'] = self.improvement_detected
    d['is_mixed'] = self.is_mixed
    return d


def _higher_is_better(metric: str, metric_metadata: dict[str, bool] | None) -> bool:
  if metric_metadata and metric in metric_metadata:
    return metric_metadata[metric]
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
  metric_metadata: dict[str, bool] | None = None,
) -> MetricComparison:
  """Compare two metric dicts. Returns MetricComparison with computed boolean properties."""
  per_metric_deltas: dict[str, float] = {}
  regressions: list[dict] = []
  improvements: list[dict] = []

  common_keys = set(baseline.keys()) & set(candidate.keys())
  for key in sorted(common_keys):
    base_val = baseline[key]
    cand_val = candidate[key]
    delta = cand_val - base_val
    per_metric_deltas[key] = delta

    higher = _higher_is_better(key, metric_metadata)
    significant = _is_significant_change(delta, base_val, threshold_abs, threshold_pct)

    if higher:
      is_reg = delta < 0.0 and significant
      is_imp = delta > 0.0 and significant
    else:
      is_reg = delta > 0.0 and significant
      is_imp = delta < 0.0 and significant

    row = {'metric': key, 'delta': delta, 'baseline': base_val, 'candidate': cand_val}
    if is_reg:
      regressions.append(row)
    elif is_imp:
      improvements.append(row)

  return MetricComparison(
    epoch=0,
    per_metric_deltas=per_metric_deltas,
    regressions=regressions,
    improvements=improvements,
  )


def load_metric_comparison(experiment_dir: Path, epoch: int) -> MetricComparison | None:
  """Read and deserialize metric_comparison.json for an epoch."""
  raw = MetricComparisonArtifact().read_raw(experiment_dir, epoch=epoch)
  if raw is None:
    return None
  return MetricComparison.from_dict(raw)
