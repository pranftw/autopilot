"""``report trend`` command: metric trajectory analysis.

Single-tree (default) or cross-tree (``--all-trees``) metric trajectory
analysis.  ``--all-trees`` runs ``TrendAnalyzer.analyze()`` once per tree
and returns a ``trees`` dict keyed by tree name.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest, require_active_tree
from autopilot.cli.primitives import Argument, Flag
from autopilot.core.trend import TrendAnalyzer, TrendResult
from typing import Any
import argparse


class ReportTrend(Command):
  """Analyze metric trajectory over sequential experiments.

  ``report trend <metric>`` classifies the trend as improving, degrading,
  plateau, volatile, or insufficient_data. Uses node insertion order as the
  experiment sequence.

  Flags:
    --lower           Lower metric values are better (default: higher).
    --window N        Restrict analysis to the last N valid data points.
    --all-trees       Aggregate trend analysis across all trees.
  """

  name = 'trend'
  help = 'Analyze metric trajectory over sequential experiments'

  metric_arg = Argument(
    'metric',
    metavar='METRIC',
    help='metric name to analyze',
  )
  lower = Flag(
    '--lower',
    help='lower metric values are better (default: higher)',
  )
  window = Argument(
    '--window',
    type=int,
    default=None,
    metavar='N',
    help='restrict to last N valid data points',
  )
  all_trees = Flag(
    '--all-trees',
    help='analyze trend across all trees (JSON: result.trees dict)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run trend analysis and emit result.

    When ``--all-trees`` is set, iterates every tree in the forest and
    returns a ``trees`` dict keyed by tree name. Otherwise analyzes only
    the active tree.

    Args:
      ctx: CLI context.
      args: Parsed namespace with ``metric``, ``lower``, ``window``,
        ``all_trees``.
    """
    forest = load_forest(ctx)
    higher_is_better = not args.lower
    analyzer = TrendAnalyzer()

    if args.all_trees:
      self._forward_all_trees(ctx, forest, analyzer, args, higher_is_better)
    else:
      self._forward_single_tree(ctx, forest, analyzer, args, higher_is_better)

  def _forward_single_tree(
    self,
    ctx: CLIContext,
    forest: Any,
    analyzer: TrendAnalyzer,
    args: argparse.Namespace,
    higher_is_better: bool,
  ) -> None:
    """Analyze the active tree (original behavior).

    Args:
      ctx: CLI context.
      forest: Loaded forest.
      analyzer: TrendAnalyzer instance.
      args: Parsed CLI args.
      higher_is_better: Metric direction.
    """
    tree = require_active_tree(ctx, forest)
    result = analyzer.analyze(
      tree,
      args.metric,
      higher_is_better=higher_is_better,
      window=args.window,
    )

    if not ctx.output.use_json:
      self._render_text(ctx, result)

    ctx.output.result(result.to_dict())

  def _forward_all_trees(
    self,
    ctx: CLIContext,
    forest: Any,
    analyzer: TrendAnalyzer,
    args: argparse.Namespace,
    higher_is_better: bool,
  ) -> None:
    """Analyze every tree in the forest.

    Args:
      ctx: CLI context.
      forest: Loaded forest.
      analyzer: TrendAnalyzer instance.
      args: Parsed CLI args.
      higher_is_better: Metric direction.
    """
    trees_result: dict[str, dict[str, Any] | None] = {}
    for tree in forest.list_trees():
      result = analyzer.analyze(
        tree,
        args.metric,
        higher_is_better=higher_is_better,
        window=args.window,
      )
      if not result.values:
        trees_result[tree.name] = None
      else:
        trees_result[tree.name] = result.to_dict()

    if not ctx.output.use_json:
      self._render_all_trees_text(ctx, trees_result)

    ctx.output.result({'trees': trees_result})

  def _render_text(self, ctx: CLIContext, result: TrendResult) -> None:
    """Render trend result as human-readable text.

    Args:
      ctx: CLI context for output.
      result: TrendResult instance.
    """
    ctx.output.info(f'Metric: {result.metric}')
    ctx.output.info(f'Direction: {result.direction}')
    ctx.output.info(f'Data points: {len(result.values)}')
    if result.best_experiment_id is not None:
      ctx.output.info(f'Best: {result.best_experiment_id} ({result.best_value})')
    if result.improvement_rate is not None:
      ctx.output.info(f'Improvement rate: {result.improvement_rate:.4f}')

  def _render_all_trees_text(
    self,
    ctx: CLIContext,
    trees_result: dict[str, dict[str, Any] | None],
  ) -> None:
    """Render combined all-trees trend as text with tree column.

    Args:
      ctx: CLI context for output.
      trees_result: Dict mapping tree name to TrendResult dict or None.
    """
    for tree_name, data in trees_result.items():
      if data is None:
        ctx.output.info(f'[{tree_name}] no data')
        continue
      direction = data.get('direction', 'unknown')
      values = data.get('values', [])
      best_id = data.get('best_experiment_id')
      best_val = data.get('best_value')
      ctx.output.info(f'[{tree_name}] {direction} ({len(values)} points)')
      if best_id is not None:
        ctx.output.info(f'  Best: {best_id} ({best_val})')
