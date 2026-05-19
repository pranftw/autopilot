"""Forest-aware recommendation engine.

ForestRecommender scans completed experiments in a Forest, ranks them
using MetricsComparator, and returns a structured Recommendation. Read-only:
never mutates forest or experiment state.
"""

from autopilot.core.comparison import MetricsComparator
from autopilot.core.enums import Status
from autopilot.core.forest import Forest
from autopilot.core.node import Node
from autopilot.core.recommend import Recommendation, Recommender
from autopilot.core.tree import Tree
from typing import Any
import math

REGRESSION_THRESHOLD = 0.1
BRANCH_MIN_EXPERIMENTS = 3
BRANCH_PLATEAU_RATIO = 0.01


def _completed_nodes_from_trees(trees: list[Tree]) -> list[Node]:
  """Collect nodes with completed experiments from a list of trees.

  Args:
    trees: Trees to scan.

  Returns:
    Nodes whose experiment status is completed.
  """
  nodes: list[Node] = []
  for tree in trees:
    nodes.extend(node for node in tree.query().all() if node.experiment.status == Status.completed)
  return nodes


def _resolve_trees(forest: Forest, scope: str | None) -> list[Tree]:
  """Resolve which trees to scan based on scope.

  Args:
    forest: Forest to scan.
    scope: None for all trees, 'tree:<name>' for a single tree.

  Returns:
    List of trees matching the scope.

  Raises:
    ValueError: When scope references an unknown tree name.
  """
  if scope is None:
    return forest.list_trees()
  if scope.startswith('tree:'):
    tree_name = scope[5:]
    tree = forest.get_tree(tree_name)
    if tree is None:
      msg = f'unknown tree {tree_name!r} in scope {scope!r}'
      raise ValueError(msg)
    return [tree]
  msg = f'invalid scope {scope!r}; expected None or "tree:<name>"'
  raise ValueError(msg)


def _apply_metric_filters(
  nodes: list[Node],
  metric_gt: list[tuple[str, float]] | None,
  metric_lt: list[tuple[str, float]] | None,
) -> list[Node]:
  """Filter nodes by metric threshold predicates.

  AND semantics: a node must pass all gt and lt checks. Missing metric
  keys exclude the candidate.

  Args:
    nodes: Candidate nodes to filter.
    metric_gt: (name, threshold) pairs requiring metrics[name] > threshold.
    metric_lt: (name, threshold) pairs requiring metrics[name] < threshold.

  Returns:
    Nodes satisfying all predicates.
  """
  if not metric_gt and not metric_lt:
    return nodes
  result: list[Node] = []
  for node in nodes:
    metrics = node.experiment.metrics
    if not _passes_thresholds(metrics, metric_gt, metric_lt):
      continue
    result.append(node)
  return result


def _passes_thresholds(
  metrics: dict[str, Any],
  metric_gt: list[tuple[str, float]] | None,
  metric_lt: list[tuple[str, float]] | None,
) -> bool:
  """Check if a single metrics dict passes all threshold predicates.

  Non-finite numeric values (NaN, positive/negative infinity) fail all
  predicates (fail closed). This matches the filtering semantics of
  ``QueryBuilder.metric_gt`` / ``metric_lt`` in ``core/query.py``.

  Args:
    metrics: Experiment metrics dict.
    metric_gt: Greater-than predicates.
    metric_lt: Less-than predicates.

  Returns:
    True when all predicates are satisfied.
  """
  if metric_gt:
    for name, threshold in metric_gt:
      val = metrics.get(name)
      if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        return False
      if float(val) <= threshold:
        return False
  if metric_lt:
    for name, threshold in metric_lt:
      val = metrics.get(name)
      if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
        return False
      if float(val) >= threshold:
        return False
  return True


class ForestRecommender(Recommender):
  """Forest-aware recommendation using MetricsComparator over completed experiments.

  Scans completed experiments, compares via MetricsComparator, and
  produces a recommendation with action, confidence, reasoning, and
  evidence. Read-only: does not call experiment.add_context() or
  mutate any forest/experiment state.

  Args:
    metric: Primary metric name to optimize.
    higher_is_better: Direction for the primary metric (default True).
  """

  def __init__(self, metric: str, *, higher_is_better: bool = True) -> None:
    """Create a forest recommender targeting a specific metric.

    Args:
      metric: Primary metric name to rank experiments by.
      higher_is_better: Whether higher values are better for the metric.
    """
    self.metric = metric
    self.higher_is_better = higher_is_better

  def recommend(
    self,
    forest: Forest,
    comparator: MetricsComparator,
    *,
    scope: str | None = None,
    metric_gt: list[tuple[str, float]] | None = None,
    metric_lt: list[tuple[str, float]] | None = None,
  ) -> Recommendation:
    """Produce a recommendation from forest data and metric comparisons.

    Rank experiments after applying threshold filters. A candidate must
    satisfy **all** supplied predicates (AND semantics across entries).
    ``metric_gt`` entries require ``metrics[name] > threshold``;
    ``metric_lt`` require ``metrics[name] < threshold``. Missing metric
    keys **exclude the candidate** (consistent with query filter behavior).

    When every candidate is filtered out, returns a sentinel
    ``Recommendation`` with ``action='investigate'`` and
    ``experiment_id=None`` rather than raising.

    Args:
      forest: Forest containing experiment trees to analyze.
      comparator: MetricsComparator for evidence-based ranking.
      scope: Optional 'tree:<name>' to restrict to a single tree.
        Unknown tree names propagate ``ValueError`` from ``_resolve_trees``.
      metric_gt: Repeatable (name, threshold) pairs; candidates must have
        ``metrics[name] > threshold`` for each entry.
      metric_lt: Repeatable (name, threshold) pairs; candidates must have
        ``metrics[name] < threshold`` for each entry.

    Returns:
      Structured Recommendation with action, confidence, and evidence.
    """
    trees = _resolve_trees(forest, scope)
    if not trees:
      return Recommendation(
        action='investigate',
        experiment_id=None,
        confidence='low',
        reasoning=['no trees in forest'],
      )

    completed = _completed_nodes_from_trees(trees)
    if not completed:
      return Recommendation(
        action='investigate',
        experiment_id=None,
        confidence='low',
        reasoning=['no completed experiments found'],
      )

    metric_nodes = [n for n in completed if self.metric in n.experiment.metrics]
    if not metric_nodes:
      return Recommendation(
        action='investigate',
        experiment_id=None,
        confidence='low',
        reasoning=['primary metric not found in any experiment'],
      )

    filtered = _apply_metric_filters(metric_nodes, metric_gt, metric_lt)
    if not filtered:
      return Recommendation(
        action='investigate',
        experiment_id=None,
        confidence='low',
        reasoning=['all candidates excluded by metric threshold filters'],
      )

    return self._rank_and_recommend(filtered, comparator)

  def _rank_and_recommend(
    self,
    nodes: list[Node],
    comparator: MetricsComparator,
  ) -> Recommendation:
    """Rank candidate nodes and produce the appropriate recommendation.

    Args:
      nodes: Completed nodes that have the primary metric.
      comparator: For computing deltas between experiments.

    Returns:
      Recommendation based on ranking and comparison analysis.
    """
    metric_dicts = [n.experiment.metrics for n in nodes]
    best_idx = comparator.best_index(metric_dicts, self.metric)
    best_node = nodes[best_idx]
    best_metrics = best_node.experiment.metrics
    best_value = best_metrics[self.metric]

    evidence: dict[str, Any] = {
      'best_experiment_id': best_node.experiment.id,
      'best_value': best_value,
      'metric': self.metric,
      'candidates_count': len(nodes),
      'metrics': best_metrics,
    }

    if len(nodes) == 1:
      return Recommendation(
        action='continue',
        experiment_id=best_node.experiment.id,
        confidence='low',
        reasoning=[
          f'only one completed experiment with metric {self.metric!r}',
          'more experiments needed for confident recommendation',
        ],
        evidence=evidence,
      )

    sorted_nodes = sorted(
      nodes,
      key=lambda n: n.experiment.metrics[self.metric],
      reverse=self.higher_is_better,
    )
    runner_up = sorted_nodes[1] if len(sorted_nodes) > 1 else None

    if runner_up is not None:
      deltas = comparator.compare(
        runner_up.experiment.metrics,
        best_metrics,
      )
      evidence['deltas'] = [d.to_dict() for d in deltas]
      evidence['runner_up_id'] = runner_up.experiment.id

    alternatives = [n.experiment.id for n in sorted_nodes[1:]]

    if self._should_branch(sorted_nodes):
      return Recommendation(
        action='branch',
        experiment_id=best_node.experiment.id,
        confidence='medium',
        reasoning=[
          f'plateau detected on {self.metric!r} across recent experiments',
          'branching may explore new optimization directions',
        ],
        alternatives=alternatives,
        evidence=evidence,
      )

    if self._detect_regression(nodes):
      return Recommendation(
        action='rollback',
        experiment_id=sorted_nodes[0].experiment.id,
        confidence='medium',
        reasoning=[
          f'latest experiments show regression on {self.metric!r}',
          f'best value: {best_value}',
        ],
        alternatives=alternatives,
        evidence=evidence,
      )

    if self._is_strong_candidate(sorted_nodes, comparator):
      return Recommendation(
        action='deploy',
        experiment_id=best_node.experiment.id,
        confidence='high',
        reasoning=[
          (
            f'experiment {best_node.experiment.id!r} is the clear leader '
            f'on {self.metric!r} with value {best_value}'
          ),
        ],
        alternatives=alternatives,
        evidence=evidence,
      )

    return Recommendation(
      action='continue',
      experiment_id=best_node.experiment.id,
      confidence='medium',
      reasoning=[
        (
          f'best experiment on {self.metric!r}: {best_node.experiment.id!r} with value {best_value}'
        ),
        'more experiments may improve confidence',
      ],
      alternatives=alternatives,
      evidence=evidence,
    )

  def _detect_regression(self, nodes: list[Node]) -> bool:
    """Check if recent experiments regressed compared to earlier ones.

    Uses completion timestamps to determine ordering. Regression is
    detected when the most recently completed experiment is significantly
    worse than the best overall.

    Args:
      nodes: Completed nodes (unsorted, original order).

    Returns:
      True when the latest completed experiment shows regression.
    """
    if len(nodes) < BRANCH_MIN_EXPERIMENTS:
      return False
    timestamped = [n for n in nodes if n.experiment.completed_at is not None]
    if len(timestamped) < BRANCH_MIN_EXPERIMENTS:
      return False
    timestamped.sort(key=lambda n: n.experiment.completed_at)
    latest = timestamped[-1]
    latest_val = latest.experiment.metrics[self.metric]
    best_val = (
      max(
        (n.experiment.metrics[self.metric] for n in nodes),
      )
      if self.higher_is_better
      else min(
        (n.experiment.metrics[self.metric] for n in nodes),
      )
    )
    if best_val == 0:
      return False
    gap = abs(best_val - latest_val) / abs(best_val)
    return gap > REGRESSION_THRESHOLD

  def _should_branch(self, sorted_nodes: list[Node]) -> bool:
    """Detect a plateau suggesting a new exploration direction.

    Args:
      sorted_nodes: Nodes sorted best-first by the primary metric.

    Returns:
      True when values are tightly clustered (plateau).
    """
    if len(sorted_nodes) < BRANCH_MIN_EXPERIMENTS:
      return False
    values = [n.experiment.metrics[self.metric] for n in sorted_nodes]
    val_range = max(values) - min(values)
    max_abs = max(abs(v) for v in values) if values else 1.0
    if max_abs == 0:
      return True
    return (val_range / max_abs) < BRANCH_PLATEAU_RATIO

  def _is_strong_candidate(
    self,
    sorted_nodes: list[Node],
    comparator: MetricsComparator,
  ) -> bool:
    """Determine if the best experiment is a strong deploy candidate.

    Requires at least 2 experiments and a significant delta vs runner-up.

    Args:
      sorted_nodes: Nodes sorted best-first by the primary metric.
      comparator: For significance testing.

    Returns:
      True when the best experiment is significantly better.
    """
    if len(sorted_nodes) < 2:
      return False
    best = sorted_nodes[0].experiment.metrics
    runner_up = sorted_nodes[1].experiment.metrics
    deltas = comparator.compare(runner_up, best)
    return any(d.metric == self.metric and d.significant for d in deltas)
