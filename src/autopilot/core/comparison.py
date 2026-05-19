"""Metric comparison utilities.

Delta captures a per-metric comparison result. MetricsComparator orchestrates
comparisons across multiple metrics using Metric definitions for semantics.
Reads higher_is_better from each Metric to know direction. Significance is
checked via threshold_abs / threshold_pct -- no Comparison strategy class
hierarchy needed.
"""

from autopilot.core.metric import Metric, MetricCollection
from autopilot.core.serialization import DictMixin
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import Any, cast


@dataclass
class Delta(DictMixin):
  """Per-metric comparison result. Data only -- no judgment.

  Attributes:
    metric: Metric key compared (registered name on the comparator).
    baseline: Baseline (reference) metric value.
    candidate: Candidate metric value (field name is ``candidate``, not
      ``current``).
    delta: Candidate minus baseline.
    higher_is_better: Direction flag from the metric definition; None when
      unknown.
    significant: Whether the change exceeds configured significance thresholds.
  """

  metric: str
  baseline: float
  candidate: float
  delta: float
  higher_is_better: bool | None
  significant: bool


class ComparatorMetric:
  """Lightweight metric wrapper for MetricsComparator consumption.

  Unlike full Metric subclasses, this only carries ``metric_name`` and
  ``higher_is_better`` direction for comparison purposes.
  """

  def __init__(self, metric_name: str, higher_is_better: bool | None) -> None:
    """Create a comparison-only metric descriptor.

    Args:
      metric_name: Identifier used as the key in metric dicts.
      higher_is_better: Direction flag; ``None`` when direction is unknown.
    """
    self.metric_name = metric_name
    self.higher_is_better = higher_is_better

  def name(self) -> str:
    """Return the metric name for MetricsComparator keying.

    Returns:
      The ``metric_name`` this instance was constructed with.
    """
    return self.metric_name


class MetricsComparator:
  """Compare experiment metrics using Metric definitions for semantics.

  Baseline and candidate dict keys must match the names this comparator
  registers: for a list[Metric] argument, each key is metric.name()
  (default: class name, e.g. 'AccuracyMetric'). For a MetricCollection
  argument, keys are the collection's child attribute names (as returned
  by named_children()), which may differ from class names when metrics
  are stored under custom attribute keys. Mismatched keys silently
  produce no comparison.

  Reads higher_is_better from each Metric to know direction.
  Significance is checked via threshold_abs / threshold_pct.
  All comparison logic is here -- Metric stays clean (update/compute only).
  """

  def __init__(
    self,
    metrics: MetricCollection | Sequence[Metric | ComparatorMetric],
    threshold_abs: float = 0.0,
    threshold_pct: float = 0.0,
  ) -> None:
    """Create a comparator from metrics and significance thresholds.

    Args:
      metrics: Metric collection or sequence of objects with ``name()`` and
        ``higher_is_better``; keys derive from collection child names or
        each metric's ``name()``.
      threshold_abs: Absolute delta threshold for significance.
      threshold_pct: Relative delta threshold as a fraction of baseline magnitude.

    Raises:
      ValueError: When a sequence contains two metrics with the same name.
    """
    if isinstance(metrics, MetricCollection):
      self._metrics: dict[str, Metric | ComparatorMetric] = cast(
        Any, dict(metrics.named_children())
      )
    else:
      names_seen: dict[str, Metric | ComparatorMetric] = {}
      for m in metrics:
        name = m.name()
        if name in names_seen:
          msg = f'duplicate metric: {name}'
          raise ValueError(msg)
        names_seen[name] = m
      self._metrics = names_seen
    self._threshold_abs = threshold_abs
    self._threshold_pct = threshold_pct

  def compare(
    self,
    baseline: dict[str, float],
    candidate: dict[str, float],
  ) -> list[Delta]:
    """Compare two metric dicts.

    Args:
      baseline: Reference metrics keyed by registered metric names.
      candidate: Candidate metrics keyed by registered metric names.

    Returns:
      One Delta per metric present in both dicts and registered on this comparator.
    """
    deltas = []
    for name, metric in self._metrics.items():
      if name in baseline and name in candidate:
        base_val = baseline[name]
        cand_val = candidate[name]
        d = cand_val - base_val
        deltas.append(
          Delta(
            metric=name,
            baseline=base_val,
            candidate=cand_val,
            delta=d,
            higher_is_better=metric.higher_is_better,
            significant=self.is_significant(d, base_val),
          )
        )
    return deltas

  def is_improvement(self, delta: Delta) -> bool:
    """Whether the delta moves in the favorable direction for the metric.

    Args:
      delta: Comparison record including delta and higher_is_better.

    Returns:
      True if candidate improved vs baseline given directionality.

    Raises:
      ValueError: When higher_is_better is unset on the delta.
    """
    if delta.higher_is_better is None:
      msg = f'higher_is_better not set for {delta.metric}'
      raise ValueError(msg)
    if delta.higher_is_better:
      return delta.delta > 0
    return delta.delta < 0

  def best_index(
    self,
    results: list[dict[str, float]],
    metric: str,
  ) -> int:
    """Index of the best result for a given metric name.

    On ties, returns the first occurrence (lowest index).
    Skips result dicts missing the metric key.

    Args:
      results: List of per-run metric dicts.
      metric: Registered metric name to optimize.

    Returns:
      Index into results of the best value for metric.

    Raises:
      ValueError: When results is empty, higher_is_better is unset for the
        metric, or no dict in results contains the metric.
    """
    if not results:
      msg = f'best_index: results list is empty (metric={metric!r})'
      raise ValueError(
        msg,
      )
    metric_spec = self._metrics[metric]
    if metric_spec.higher_is_better is None:
      msg = (
        f'higher_is_better not set for metric {metric!r} '
        f'(mode must be set on Metric before best_index)'
      )
      raise ValueError(
        msg,
      )
    indices_and_values: list[tuple[int, float]] = []
    for i, r in enumerate(results):
      if metric not in r:
        continue
      indices_and_values.append((i, r[metric]))
    if not indices_and_values:
      all_keys: set[str] = set()
      for r in results:
        all_keys.update(r)
      union_keys = sorted(all_keys)
      msg = (
        f'best_index: no result dicts contain metric {metric!r}; '
        f'keys_present_across_results={union_keys!r}'
      )
      raise ValueError(
        msg,
      )
    if metric_spec.higher_is_better:
      return max(indices_and_values, key=itemgetter(1))[0]
    return min(indices_and_values, key=itemgetter(1))[0]

  def is_significant(self, delta: float, baseline: float) -> bool:
    """Check if a delta is significant given thresholds.

    Public method per CLAUDE.md: all customization hooks are public.
    Users may override for custom significance logic in subclasses.

    Args:
      delta: Candidate minus baseline for one metric.
      baseline: Baseline magnitude; used for relative threshold when non-zero.

    Returns:
      True if the change exceeds configured absolute or percentage thresholds
      (or any non-zero delta when both thresholds are zero).
    """
    if self._threshold_abs == 0.0 and self._threshold_pct == 0.0:
      return delta != 0.0
    abs_sig = self._threshold_abs > 0.0 and abs(delta) > self._threshold_abs
    pct_sig = (
      self._threshold_pct > 0.0
      and baseline != 0.0
      and abs(delta) / abs(baseline) > self._threshold_pct
    )
    return abs_sig or pct_sig
