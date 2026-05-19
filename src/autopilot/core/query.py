"""Composable query engine for experiment tree nodes.

``QueryBuilder`` chains filter/structural/metric operations and executes
on terminal call. Everything is derived at execution time -- no cached
aggregates. Operates on ``Node`` objects with real ``Experiment`` references.

**Immutable chaining contract**: every chainable method (``filter``,
``where``, ``exclude``, ``metric_gt``, ``metric_lt``, ``metric_between``,
``metadata_contains``, ``order_by_metric``, etc.) returns a **new**
``QueryBuilder`` instance. The original builder is never mutated. Callers
must reassign the variable when composing filters::

    builder = builder.metric_gt('accuracy', 0.5)
    builder = builder.metric_lt('loss', 1.0)

Composition order matches the filter application order used by the
``query`` CLI command.
"""

from autopilot.core.comparison import ComparatorMetric, MetricsComparator
from autopilot.core.enums import Status
from autopilot.core.metadata import MetadataArtifact
from autopilot.core.node import Node
from collections.abc import Callable
from pathlib import Path
from typing import Any
import math


def _is_finite_number(value: Any) -> bool:
  """Check whether a value is a finite int or float (not NaN, not inf).

  Args:
    value: Metric value to check.

  Returns:
    True when value is numeric and finite.
  """
  if isinstance(value, bool):
    return False
  if isinstance(value, (int, float)):
    return math.isfinite(value)
  return False


class QueryBuilder:
  """Composable query over a list of Nodes.

  Every chainable method returns a **new** ``QueryBuilder`` instance;
  the original is never mutated. Callers must reassign the variable
  when composing (``b = b.metric_gt(...)``). Composition order matches
  the filter application order used by the ``query`` CLI command.

  Attributes:
    _nodes: Current candidate ``Node`` list for this chained query.
    _resolver: Callable mapping ``experiment_id`` to ``Node | None``.
    _steps: Human-readable audit trail of applied chain operations.

  Constructor takes the full node list and a resolver for experiment lookups.
  The resolver maps experiment_id -> Node.
  """

  def __init__(
    self,
    nodes: list[Node],
    resolver: Callable[[str], Node | None],
  ) -> None:
    """Capture the initial node list and id resolver.

    Args:
      nodes: Candidate ``Node`` instances for this query.
      resolver: Maps ``experiment_id`` to a ``Node`` when present.
    """
    self._nodes = list(nodes)
    self._resolver = resolver
    self._steps: list[str] = []

  def _copy(self, nodes: list[Node], step: str) -> 'QueryBuilder':
    qb = QueryBuilder(nodes, self._resolver)
    qb._steps = [*self._steps, step]
    return qb

  # chainable: filter

  def filter(self, **kwargs: Any) -> 'QueryBuilder':
    """Keep nodes whose ``experiment`` fields equal the given kwargs.

    Args:
      **kwargs: ``field_name=value`` predicates compared with equality on
        ``n.experiment``.

    Returns:
      New ``QueryBuilder`` filtering ``_nodes``; does not mutate this instance.
    """

    def matches(n: Node) -> bool:
      return all(getattr(n.experiment, key) == value for key, value in kwargs.items())

    filtered = [n for n in self._nodes if matches(n)]
    desc = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
    return self._copy(filtered, f'filter({desc})')

  def where(self, predicate: Callable[[Node], bool]) -> 'QueryBuilder':
    """Custom predicate with real object access.

    Returns:
      New builder retaining nodes for which ``predicate`` returns true.
    """
    filtered = [n for n in self._nodes if predicate(n)]
    return self._copy(filtered, 'where(<predicate>)')

  def exclude(self, **kwargs: Any) -> 'QueryBuilder':
    """Inverse of filter: exclude nodes matching all kwargs.

    Returns:
      New builder dropping nodes whose experiment fields equal all kwargs.
    """

    def matches(n: Node) -> bool:
      return all(getattr(n.experiment, key) == value for key, value in kwargs.items())

    filtered = [n for n in self._nodes if not matches(n)]
    desc = ', '.join(f'{k}={v!r}' for k, v in kwargs.items())
    return self._copy(filtered, f'exclude({desc})')

  # chainable: status convenience

  def completed(self) -> 'QueryBuilder':
    """Nodes with status == completed.

    Returns:
      New builder containing only completed experiments.
    """
    return self._copy(
      [n for n in self._nodes if n.experiment.status == Status.completed],
      'completed()',
    )

  def failed(self) -> 'QueryBuilder':
    """Nodes with status == failed.

    Returns:
      New builder containing only failed experiments.
    """
    return self._copy(
      [n for n in self._nodes if n.experiment.status == Status.failed],
      'failed()',
    )

  def running(self) -> 'QueryBuilder':
    """Nodes with status == running.

    Returns:
      New builder containing only running experiments.
    """
    return self._copy(
      [n for n in self._nodes if n.experiment.status == Status.running],
      'running()',
    )

  def pending(self) -> 'QueryBuilder':
    """Nodes with status == pending.

    Returns:
      New builder containing only pending experiments.
    """
    return self._copy(
      [n for n in self._nodes if n.experiment.status == Status.pending],
      'pending()',
    )

  def cancelled(self) -> 'QueryBuilder':
    """Nodes with status == cancelled.

    Returns:
      New builder containing only cancelled experiments.
    """
    return self._copy(
      [n for n in self._nodes if n.experiment.status == Status.cancelled],
      'cancelled()',
    )

  def terminal(self) -> 'QueryBuilder':
    """Nodes with terminal status (completed, failed, cancelled, invalidated).

    Returns:
      New builder containing experiments in a terminal state.
    """
    return self._copy(
      [n for n in self._nodes if n.experiment.is_terminal],
      'terminal()',
    )

  # chainable: metric predicates

  def metric_gt(self, name: str, value: float) -> 'QueryBuilder':
    """Nodes where experiment.metrics[name] > value.

    Non-finite metric values (NaN, Inf, -Inf) are excluded -- they do not
    satisfy any numeric comparison.

    Returns:
      New builder for nodes above the metric threshold.
    """
    filtered = [
      n
      for n in self._nodes
      if name in n.experiment.metrics
      and math.isfinite(n.experiment.metrics[name])
      and n.experiment.metrics[name] > value
    ]
    return self._copy(filtered, f'metric_gt({name!r}, {value})')

  def metric_lt(self, name: str, value: float) -> 'QueryBuilder':
    """Nodes where experiment.metrics[name] < value.

    Non-finite metric values (NaN, Inf, -Inf) are excluded -- they do not
    satisfy any numeric comparison.

    Returns:
      New builder for nodes below the metric threshold.
    """
    filtered = [
      n
      for n in self._nodes
      if name in n.experiment.metrics
      and math.isfinite(n.experiment.metrics[name])
      and n.experiment.metrics[name] < value
    ]
    return self._copy(filtered, f'metric_lt({name!r}, {value})')

  def metric_between(self, name: str, low: float, high: float) -> 'QueryBuilder':
    """Nodes where low <= experiment.metrics[name] <= high.

    Non-finite metric values (NaN, Inf, -Inf) are excluded.

    Returns:
      New builder for nodes within the inclusive metric band.
    """
    return self._copy(
      [
        n
        for n in self._nodes
        if name in n.experiment.metrics
        and math.isfinite(n.experiment.metrics[name])
        and low <= n.experiment.metrics[name] <= high
      ],
      f'metric_between({name!r}, {low}, {high})',
    )

  def has_metric(self, name: str) -> 'QueryBuilder':
    """Nodes where experiment.metrics contains the key.

    Returns:
      New builder retaining only nodes that recorded ``name``.
    """
    return self._copy(
      [n for n in self._nodes if name in n.experiment.metrics],
      f'has_metric({name!r})',
    )

  def metadata_contains(self, key: str, value: str, experiments_path: Path) -> 'QueryBuilder':
    """Nodes whose experiment metadata has ``key`` equal to ``value``.

    Loads ``MetadataArtifact`` per candidate experiment and compares
    using exact string equality: ``str(stored_value) == value``.
    Non-string metadata values are compared via ``str(stored_value)``.

    This method is immutable: returns a new builder without mutating
    the current instance.

    Args:
      key: Metadata key to match.
      value: Expected string value (compared after ``str()`` coercion).
      experiments_path: Base experiments directory for path resolution.

    Returns:
      New ``QueryBuilder`` retaining only matching experiments.
    """
    artifact = MetadataArtifact()

    def predicate(n: Node) -> bool:
      exp_dir = experiments_path / n.experiment.id
      stored = artifact.get(key, base_dir=exp_dir)
      if stored is None:
        return False
      return str(stored) == value

    filtered = [n for n in self._nodes if predicate(n)]
    return self._copy(filtered, f'metadata_contains({key!r}, {value!r})')

  # chainable: structural

  def ancestors_of(self, experiment_id: str) -> 'QueryBuilder':
    """Walk parent chain from node (closest ancestor first).

    Returns:
      New builder listing ancestors intersected with the current candidate set.
    """
    node = self._resolver(experiment_id)
    if node is None:
      return self._copy([], f'ancestors_of({experiment_id!r})')
    ancestors: list[Node] = []
    current = node.parent
    seen: set[str] = set()
    while current is not None and current.experiment.id not in seen:
      seen.add(current.experiment.id)
      ancestors.append(current)
      current = current.parent
    node_ids = {n.experiment.id for n in self._nodes}
    result = [a for a in ancestors if a.experiment.id in node_ids]
    return self._copy(result, f'ancestors_of({experiment_id!r})')

  def descendants_of(self, experiment_id: str) -> 'QueryBuilder':
    """All children recursively.

    Returns:
      New builder with every descendant of ``experiment_id`` in this tree slice.
    """
    children_map: dict[str, list[Node]] = {}
    for n in self._nodes:
      pid = n.parent.experiment.id if n.parent is not None else None
      if pid is not None:
        children_map.setdefault(pid, []).append(n)
    result: list[Node] = []
    stack = list(children_map.get(experiment_id, []))
    seen: set[str] = set()
    while stack:
      n = stack.pop(0)
      if n.experiment.id in seen:
        continue
      seen.add(n.experiment.id)
      result.append(n)
      stack.extend(children_map.get(n.experiment.id, []))
    return self._copy(result, f'descendants_of({experiment_id!r})')

  def children_of(self, experiment_id: str) -> 'QueryBuilder':
    """Direct children only.

    Returns:
      New builder listing immediate children under ``experiment_id``.
    """
    result = [
      n for n in self._nodes if n.parent is not None and n.parent.experiment.id == experiment_id
    ]
    return self._copy(result, f'children_of({experiment_id!r})')

  def siblings_of(self, experiment_id: str) -> 'QueryBuilder':
    """Nodes with same parent, excluding self.

    Returns:
      New builder of sibling nodes, or empty when ``experiment_id`` is unknown.
    """
    node = self._resolver(experiment_id)
    if node is None:
      return self._copy([], f'siblings_of({experiment_id!r})')
    parent_id = node.parent.experiment.id if node.parent is not None else None
    result = [
      n
      for n in self._nodes
      if n.experiment.id != experiment_id
      and (
        (n.parent is not None and n.parent.experiment.id == parent_id)
        if parent_id is not None
        else n.parent is None
      )
    ]
    return self._copy(result, f'siblings_of({experiment_id!r})')

  def roots(self) -> 'QueryBuilder':
    """Nodes with no parent.

    Returns:
      New builder containing only root experiments.
    """
    return self._copy(
      [n for n in self._nodes if n.parent is None],
      'roots()',
    )

  def leaves(self) -> 'QueryBuilder':
    """Nodes with no children.

    Returns:
      New builder containing experiments with no dependents in this slice.
    """
    parent_ids = {n.parent.experiment.id for n in self._nodes if n.parent is not None}
    return self._copy(
      [n for n in self._nodes if n.experiment.id not in parent_ids],
      'leaves()',
    )

  def depth(self, n: int) -> 'QueryBuilder':
    """Nodes at exact depth n from root.

    Returns:
      New builder whose nodes sit exactly ``n`` hops below a root.
    """
    result = [node for node in self._nodes if self._node_depth(node) == n]
    return self._copy(result, f'depth({n})')

  def depth_range(self, min_depth: int, max_depth: int) -> 'QueryBuilder':
    """Nodes within depth range [min_depth, max_depth].

    Returns:
      New builder filtered to nodes whose depth falls in the closed interval.
    """
    result = [node for node in self._nodes if min_depth <= self._node_depth(node) <= max_depth]
    return self._copy(result, f'depth_range({min_depth}, {max_depth})')

  # chainable: ordering

  def order_by(self, key: str, ascending: bool = True) -> 'QueryBuilder':
    """Sort by Experiment attribute.

    Returns:
      New builder with nodes sorted by ``getattr(experiment, key)``.
    """
    sorted_nodes = sorted(
      self._nodes,
      key=lambda n: getattr(n.experiment, key),
      reverse=not ascending,
    )
    direction = 'asc' if ascending else 'desc'
    return self._copy(sorted_nodes, f'order_by({key!r}, {direction})')

  def order_by_metric(self, name: str, ascending: bool = False) -> 'QueryBuilder':
    """Sort by metric value (descending by default).

    Returns:
      New builder with metric-bearing nodes first (sorted), then nodes lacking the metric.
    """
    with_metric = [n for n in self._nodes if name in n.experiment.metrics]
    without_metric = [n for n in self._nodes if name not in n.experiment.metrics]
    sorted_nodes = sorted(
      with_metric,
      key=lambda n: n.experiment.metrics[name],
      reverse=not ascending,
    )
    return self._copy(sorted_nodes + without_metric, f'order_by_metric({name!r})')

  # terminal methods

  def all(self) -> list[Node]:
    """Execute query and return all matching nodes.

    Returns:
      Copy of the current candidate list.
    """
    return list(self._nodes)

  def first(self) -> Node | None:
    """Execute query and return first match, or None.

    Returns:
      First ``Node`` in candidate order, or ``None`` when empty.
    """
    return self._nodes[0] if self._nodes else None

  def best(self, metric: str, higher_is_better: bool = True) -> Node | None:
    """Return the node with the best value for the given metric.

    Delegates to MetricsComparator.best_index. Nodes missing the metric
    are skipped.

    Returns:
      Winning ``Node``, or ``None`` when no node exposes ``metric``.
    """
    return self._best_worst(metric, higher_is_better)

  def worst(self, metric: str, higher_is_better: bool = True) -> Node | None:
    """Return the node with the worst value for the given metric.

    Uses same logic path as best() with inverted higher_is_better.
    Nodes missing the metric are skipped.

    Returns:
      Losing ``Node`` by the comparator, or ``None`` when metric is absent everywhere.
    """
    return self._best_worst(metric, not higher_is_better)

  def count(self) -> int:
    """Return number of matching nodes.

    Returns:
      Length of the current candidate list.
    """
    return len(self._nodes)

  def exists(self) -> bool:
    """Return True if any nodes match.

    Returns:
      False when the candidate list is empty; otherwise true.
    """
    return bool(self._nodes)

  # agent surface

  def explain(self) -> str:
    """Human-readable description of the query chain.

    Returns:
      Summary string enumerating chained operations.
    """
    if not self._steps:
      return 'query: all nodes'
    return 'query: ' + ' -> '.join(self._steps)

  def render(self) -> str:
    """Results as markdown table.

    Returns:
      Markdown string, or a placeholder when there are no rows.
    """
    nodes = self._nodes
    if not nodes:
      return '(no results)'
    lines = ['| id | status | hypothesis | metrics |', '|---|---|---|---|']
    for n in nodes:
      exp = n.experiment
      metrics_str = ', '.join(f'{k}={v}' for k, v in exp.metrics.items())
      hyp = '' if exp.hypothesis is None else exp.hypothesis
      lines.append(f'| {exp.id} | {exp.status.value} | {hyp} | {metrics_str} |')
    return '\n'.join(lines)

  # internal helpers

  def _node_depth(self, node: Node) -> int:
    depth = 0
    current = node.parent
    seen: set[str] = set()
    while current is not None and current.experiment.id not in seen:
      seen.add(current.experiment.id)
      depth += 1
      current = current.parent
    return depth

  def _best_worst(self, metric: str, higher_is_better: bool) -> Node | None:
    """Shared logic for best/worst using MetricsComparator.best_index.

    Skips nodes whose metric value is not a finite number (int or float,
    excluding NaN/inf). When no nodes have a valid numeric metric value,
    returns None.

    Returns:
      Selected ``Node`` or ``None`` when no nodes have a valid numeric metric.
    """
    nodes_with_metric = [
      n
      for n in self._nodes
      if metric in n.experiment.metrics and _is_finite_number(n.experiment.metrics[metric])
    ]
    if not nodes_with_metric:
      return None
    results = [n.experiment.metrics for n in nodes_with_metric]

    stub = ComparatorMetric(metric, higher_is_better)
    comparator = MetricsComparator([stub])
    idx = comparator.best_index(results, metric)
    return nodes_with_metric[idx]
