"""Metric trajectory analysis over sequential experiments.

``TrendAnalyzer`` walks experiments in a ``Tree`` in node insertion order,
collects scalar float metric values, and classifies the trajectory as
improving / degrading / plateau / volatile / insufficient_data.

**Ordering**: experiments are iterated in ``Tree.nodes`` dict insertion
order — the order they were added to the tree.

**NaN policy**: ``math.isnan`` values are silently skipped and do not
count as data points.

**Classification order** (first match wins):

1. ``< MIN_DATA_POINTS`` (3) valid values -> ``insufficient_data``
2. Monotonically favorable (non-strict) -> ``improving``; all
   unfavorable -> ``degrading``
3. ``(max - min) / max < PLATEAU_THRESHOLD`` (or all equal) -> ``plateau``
4. ``CV = std_pop / mean > VOLATILITY_CV_THRESHOLD`` -> ``volatile``
5. Majority vote on successive deltas -> ``improving`` or ``degrading``;
   inconclusive -> ``volatile`` (conservative fallback)

**Window**: when ``window`` is not ``None``, only the last *window*
valid data points (post-NaN removal) are analyzed.

**Formulas**:

- ``improvement_rate = (last - first) / max(1, len(values) - 1)``
  Positive = improving for ``higher_is_better``; negate for lower.
- ``std_pop`` uses divisor ``N`` (population standard deviation).
- ``CV = std_pop(values) / mean(values)``
"""

from autopilot.core.serialization import DictMixin
from autopilot.core.tree import Tree
from dataclasses import dataclass, field
import math

VALID_TREND_DIRECTIONS = frozenset(
  {
    'improving',
    'degrading',
    'plateau',
    'volatile',
    'insufficient_data',
  }
)
PLATEAU_THRESHOLD = 0.01
VOLATILITY_CV_THRESHOLD = 0.3
MIN_DATA_POINTS = 3
NEAR_ZERO_EPSILON = 1e-15


@dataclass
class TrendResult(DictMixin):
  """Structured trend analysis output.

  Attributes:
    metric: Name of the analyzed metric.
    values: Collected scalar values in experiment order.
    experiment_ids: Parallel list of experiment ids for each value.
    direction: Classification string from ``VALID_TREND_DIRECTIONS``.
    best_value: Best value according to ``higher_is_better`` direction.
    best_experiment_id: Experiment id corresponding to ``best_value``.
    latest_value: Last valid data point in the analyzed sequence.
    improvement_rate: Average per-epoch improvement
      ``(last - first) / max(1, len(values) - 1)``.
  """

  metric: str
  values: list[float] = field(default_factory=list)
  experiment_ids: list[str] = field(default_factory=list)
  direction: str = 'insufficient_data'
  best_value: float = 0.0
  best_experiment_id: str | None = None
  latest_value: float = 0.0
  improvement_rate: float | None = None

  def __post_init__(self) -> None:
    """Validate direction is one of the allowed classifications.

    Raises:
      ValueError: When ``direction`` is not in ``VALID_TREND_DIRECTIONS``.
    """
    if self.direction not in VALID_TREND_DIRECTIONS:
      msg = (
        f'Invalid trend direction {self.direction!r}. '
        f'Must be one of: {", ".join(sorted(VALID_TREND_DIRECTIONS))}'
      )
      raise ValueError(msg)


class TrendAnalyzer:
  """Analyze metric trajectory over sequential experiments.

  Walks experiments in ``Tree.nodes`` insertion order, collects scalar
  float metric values (skipping NaN), and classifies the trajectory.
  """

  def analyze(
    self,
    tree: Tree,
    metric: str,
    *,
    higher_is_better: bool = True,
    window: int | None = None,
  ) -> TrendResult:
    """Classify the metric trend for experiments in *tree*.

    Args:
      tree: Experiment tree to analyze.
      metric: Metric key to read from ``experiment.metrics``.
      higher_is_better: When True, increasing values are ``improving``.
      window: When not None, restrict to the last *window* valid data
        points after NaN removal.

    Returns:
      ``TrendResult`` with classification, values, best, and rates.
    """
    values, experiment_ids = self._collect(tree, metric)

    if window is not None and len(values) > window:
      values = values[-window:]
      experiment_ids = experiment_ids[-window:]

    if len(values) < MIN_DATA_POINTS:
      return self._insufficient(metric, values, experiment_ids, higher_is_better)

    direction = self._classify(values, higher_is_better)
    return self._build_result(metric, values, experiment_ids, direction, higher_is_better)

  def _collect(
    self,
    tree: Tree,
    metric: str,
  ) -> tuple[list[float], list[str]]:
    """Walk tree nodes and collect valid metric values.

    Args:
      tree: Tree to iterate over.
      metric: Metric key to look up.

    Returns:
      Parallel lists of values and experiment ids.
    """
    values: list[float] = []
    experiment_ids: list[str] = []
    for node in tree.nodes.values():
      exp = node.experiment
      raw = exp.metrics.get(metric)
      if raw is None:
        continue
      if not isinstance(raw, (int, float)):
        continue
      val = float(raw)
      if math.isnan(val):
        continue
      values.append(val)
      experiment_ids.append(exp.id)
    return values, experiment_ids

  def _classify(self, values: list[float], higher_is_better: bool) -> str:
    """Run the classification algorithm on collected values.

    When all values are equal the sequence is classified as ``plateau``
    (step 3) directly — constant sequences bypass the monotonicity
    check because a flat line is more accurately described as a plateau
    than as "improving" or "degrading".

    Args:
      values: At least ``MIN_DATA_POINTS`` valid floats.
      higher_is_better: Direction for improvement.

    Returns:
      One of ``VALID_TREND_DIRECTIONS``.
    """
    all_equal = all(v == values[0] for v in values)
    if not all_equal:
      if self._is_monotonic_improving(values, higher_is_better):
        return 'improving'
      if self._is_monotonic_degrading(values, higher_is_better):
        return 'degrading'
    if self._is_plateau(values):
      return 'plateau'
    if self._is_volatile(values):
      return 'volatile'
    return self._majority_vote(values, higher_is_better)

  def _is_monotonic_improving(self, values: list[float], higher_is_better: bool) -> bool:
    """Check if all consecutive pairs move in the favorable direction.

    Non-strict: ties do not break monotonicity.

    Args:
      values: Sequence of metric values.
      higher_is_better: Direction for improvement.

    Returns:
      True when every consecutive pair is favorable (or equal).
    """
    for i in range(1, len(values)):
      if higher_is_better:
        if values[i] < values[i - 1]:
          return False
      elif values[i] > values[i - 1]:
        return False
    return True

  def _is_monotonic_degrading(self, values: list[float], higher_is_better: bool) -> bool:
    """Check if all consecutive pairs move in the unfavorable direction.

    Non-strict: ties do not break monotonicity.

    Args:
      values: Sequence of metric values.
      higher_is_better: Direction for improvement.

    Returns:
      True when every consecutive pair is unfavorable (or equal).
    """
    for i in range(1, len(values)):
      if higher_is_better:
        if values[i] > values[i - 1]:
          return False
      elif values[i] < values[i - 1]:
        return False
    return True

  def _is_plateau(self, values: list[float]) -> bool:
    """Check if the range relative to max is below the plateau threshold.

    When all values are equal (including all-zero), this is a plateau.
    Division-by-zero is avoided by checking ``max == min`` first.

    Args:
      values: Non-empty sequence of metric values.

    Returns:
      True when the values are within plateau range.
    """
    max_val = max(values)
    min_val = min(values)
    if max_val == min_val:
      return True
    if abs(max_val) < NEAR_ZERO_EPSILON:
      return False
    return (max_val - min_val) / abs(max_val) < PLATEAU_THRESHOLD

  def _is_volatile(self, values: list[float]) -> bool:
    """Check if the coefficient of variation exceeds the volatility threshold.

    Uses population standard deviation (divisor N).

    Args:
      values: Non-empty sequence of metric values.

    Returns:
      True when CV > ``VOLATILITY_CV_THRESHOLD``.
    """
    mean_val = sum(values) / len(values)
    if mean_val == 0:
      return False
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_pop = math.sqrt(variance)
    cv = std_pop / abs(mean_val)
    return cv > VOLATILITY_CV_THRESHOLD

  def _majority_vote(self, values: list[float], higher_is_better: bool) -> str:
    """Classify by majority sign of successive deltas.

    When inconclusive (tie or no clear majority), returns ``'volatile'``
    as a conservative fallback.

    Args:
      values: Sequence of metric values.
      higher_is_better: Direction for improvement.

    Returns:
      ``'improving'``, ``'degrading'``, or ``'volatile'``.
    """
    positive = 0
    negative = 0
    for i in range(1, len(values)):
      delta = values[i] - values[i - 1]
      if delta > 0:
        positive += 1
      elif delta < 0:
        negative += 1

    if positive > negative:
      return 'improving' if higher_is_better else 'degrading'
    if negative > positive:
      return 'degrading' if higher_is_better else 'improving'
    return 'volatile'

  def _insufficient(
    self,
    metric: str,
    values: list[float],
    experiment_ids: list[str],
    higher_is_better: bool,
  ) -> TrendResult:
    """Build a TrendResult for insufficient data.

    Args:
      metric: Metric name.
      values: Collected values (fewer than ``MIN_DATA_POINTS``).
      experiment_ids: Parallel experiment ids.
      higher_is_better: Direction for best-value selection.

    Returns:
      TrendResult with ``direction='insufficient_data'``.
    """
    if not values:
      return TrendResult(
        metric=metric,
        values=[],
        experiment_ids=[],
        direction='insufficient_data',
        best_value=0.0,
        best_experiment_id=None,
        latest_value=0.0,
        improvement_rate=None,
      )
    best_val, best_id = self._find_best(values, experiment_ids, higher_is_better)
    return TrendResult(
      metric=metric,
      values=values,
      experiment_ids=experiment_ids,
      direction='insufficient_data',
      best_value=best_val,
      best_experiment_id=best_id,
      latest_value=values[-1],
      improvement_rate=None,
    )

  def _build_result(
    self,
    metric: str,
    values: list[float],
    experiment_ids: list[str],
    direction: str,
    higher_is_better: bool,
  ) -> TrendResult:
    """Construct a full TrendResult with derived fields.

    Args:
      metric: Metric name.
      values: Collected values.
      experiment_ids: Parallel experiment ids.
      direction: Classification string.
      higher_is_better: Direction for best and improvement_rate.

    Returns:
      Fully populated ``TrendResult``.
    """
    best_val, best_id = self._find_best(values, experiment_ids, higher_is_better)
    raw_rate = (values[-1] - values[0]) / max(1, len(values) - 1)
    improvement_rate = raw_rate if higher_is_better else -raw_rate
    return TrendResult(
      metric=metric,
      values=values,
      experiment_ids=experiment_ids,
      direction=direction,
      best_value=best_val,
      best_experiment_id=best_id,
      latest_value=values[-1],
      improvement_rate=improvement_rate,
    )

  def _find_best(
    self,
    values: list[float],
    experiment_ids: list[str],
    higher_is_better: bool,
  ) -> tuple[float, str]:
    """Find the best value and its experiment id.

    Args:
      values: Non-empty list of metric values.
      experiment_ids: Parallel experiment ids.
      higher_is_better: When True, max is best; otherwise min.

    Returns:
      Tuple of (best_value, best_experiment_id).
    """
    if higher_is_better:
      idx = max(range(len(values)), key=lambda i: values[i])
    else:
      idx = min(range(len(values)), key=lambda i: values[i])
    return values[idx], experiment_ids[idx]
