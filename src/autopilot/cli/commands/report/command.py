"""Reporting: experiment summaries, comparisons, and project narrative.

``autopilot report`` group: summaries, comparisons, narrative, and trend.
"""

from autopilot.ai.forest import FileForest
from autopilot.cli.command import Command
from autopilot.cli.commands.report.compare import ReportCompare, gather_summary
from autopilot.cli.commands.report.narrative import ReportNarrative
from autopilot.cli.commands.report.trend import ReportTrend
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest, require_active_tree
from autopilot.cli.primitives import argument, subcommand
from autopilot.core.experiment import Experiment
from autopilot.core.tree import Tree
from typing import Any
import argparse


class ReportCommand(Command):
  """``autopilot report`` group: summaries, comparisons, narrative, and trend."""

  name = 'report'
  help = 'Reports and comparisons'

  def __init__(self) -> None:
    """Wire report subcommands (``compare``, ``narrative``, ``trend``, and inline ``summary``)."""
    super().__init__()
    self.compare = ReportCompare()
    self.narrative = ReportNarrative()
    self.trend = ReportTrend()

  @argument('--all-trees', action='store_true', default=False, help='aggregate across all trees')
  @subcommand('summary', help_text='Summarize experiment outcomes')
  def summary(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Summarize a single experiment or aggregate across tree/workspace.

    When ``--experiment`` is set, delegates to the existing single-experiment
    ``gather_summary`` path. When omitted, aggregates the active tree
    (or all trees with ``--all-trees``).
    """
    slug = ctx.experiment
    forest = load_forest(ctx)
    if slug:
      ctx.output.info(f'Summarizing experiment {slug!r}...')
      try:
        result = gather_summary(forest, slug)
      except ValueError as exc:
        ctx.fail(str(exc))
    elif args.all_trees:
      result = gather_workspace_summary(forest, None, all_trees=True)
    else:
      tree = require_active_tree(ctx, forest)
      result = gather_workspace_summary(forest, tree, all_trees=False)
    ctx.output.result(result)


def gather_workspace_summary(
  forest: FileForest,
  tree: Tree | None,
  *,
  all_trees: bool = False,
) -> dict[str, Any]:
  """Aggregate metrics across experiments for workspace overview.

  Collects experiments from the active tree (default) or all trees
  (``all_trees=True``). Status counts include every experiment; metric
  aggregation and best-experiment selection use only ``completed``
  experiments.

  Args:
    forest: Loaded FileForest instance.
    tree: Active tree for single-tree scope (ignored when ``all_trees``).
    all_trees: When True, aggregate across all trees in the forest.

  Returns:
    Summary dict with ``scope``, ``experiments_count``, ``metric_summary``,
    and ``best_experiment`` keys. ``tree`` key present only when
    ``scope == 'tree'``.
  """
  trees = forest.list_trees() if all_trees else [tree] if tree is not None else []

  experiments: list[Experiment] = []
  for t in trees:
    experiments.extend(node.experiment for node in t.query().all())

  by_status: dict[str, int] = {}
  completed_experiments: list[Experiment] = []

  for exp in experiments:
    status_val = exp.status.value
    by_status[status_val] = by_status.get(status_val, 0) + 1
    if exp.status.value == 'completed':
      completed_experiments.append(exp)

  all_metrics: dict[str, list[float]] = {}
  for exp in completed_experiments:
    if exp.metrics:
      for key, val in exp.metrics.items():
        if isinstance(val, (int, float)):
          all_metrics.setdefault(key, []).append(float(val))

  metric_summary: dict[str, dict[str, float]] = {}
  for key, vals in sorted(all_metrics.items()):
    metric_summary[key] = {
      'min': min(vals),
      'max': max(vals),
      'mean': sum(vals) / len(vals),
    }

  best_experiment = _select_best_experiment(completed_experiments)

  result: dict[str, Any] = {
    'scope': 'workspace' if all_trees else 'tree',
    'experiments_count': by_status,
    'metric_summary': metric_summary,
    'best_experiment': best_experiment,
  }

  if not all_trees and tree is not None:
    result['tree'] = tree.name

  return result


def _select_best_experiment(
  completed_experiments: list[Experiment],
) -> dict[str, Any] | None:
  """Select the best experiment by highest value of the first lexicographic metric key.

  Args:
    completed_experiments: Experiments in ``completed`` status.

  Returns:
    Dict with ``id`` and ``metrics``, or None when no completed
    experiments have metrics.
  """
  best: dict[str, Any] | None = None
  best_value: float = float('-inf')
  best_key: str | None = None

  for exp in completed_experiments:
    if not exp.metrics:
      continue
    first_key = min(exp.metrics.keys())
    val = exp.metrics[first_key]
    if not isinstance(val, (int, float)):
      continue
    if best_key is None or first_key < best_key or (first_key == best_key and val > best_value):
      best_key = first_key
      best_value = float(val)
      best = {'id': exp.id, 'metrics': dict(exp.metrics)}

  return best
