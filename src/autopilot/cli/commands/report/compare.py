"""Report compare: baseline-centric experiment comparison and summary gathering.

``report compare`` accepts two or more experiment slugs as positional
arguments. First slug is baseline; each subsequent slug is a candidate.
Comparisons are **baseline-centric**: deltas and ``is_improvement``
semantics are always relative to the baseline slug position.

``report compare --all-trees`` compares the best experiment per tree
under a required ``--metric``.

Also provides ``gather_summary`` which collects experiment state and
event counts into a summary dict, used by both compare and summary
report handlers.
"""

from autopilot.ai.forest import FileForest
from autopilot.cli.command import Command
from autopilot.cli.commands.query import resolve_metric_name
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.primitives import Argument, Flag
from autopilot.core.comparison import ComparatorMetric, MetricsComparator
from autopilot.core.logger import load_events
from autopilot.core.metric_utils import infer_direction
from typing import Any
import argparse

REPORT_COMPARE_EPILOG = (
  'Baseline-centric comparison:\n'
  '  The first slug is the BASELINE; all subsequent slugs are CANDIDATES.\n'
  '  Deltas and verdicts are relative to the baseline (positive delta =\n'
  '  candidate value higher than baseline). Direction (higher-is-better\n'
  '  vs lower-is-better) is resolved per metric via infer_direction\n'
  '  heuristic, overridden by --lower-metric flags.\n'
  '\n'
  '  Example: autopilot report compare baseline-exp candidate-exp\n'
  '\n'
  '  For pairwise comparison across all pairs, see future --pairwise flag.\n'
)


def gather_summary(forest: FileForest, experiment_id: str) -> dict[str, Any]:
  """Gather experiment summary from Forest and events.jsonl under the experiment dir.

  Resolves ``exp_dir = forest.store.config.experiment_path(slug=experiment_id)``.
  Events loaded via ``load_events(exp_dir)``.

  Args:
    forest: Loaded FileForest instance.
    experiment_id: Experiment slug to look up.

  Returns:
    Summary dict with id, metrics, status, epoch, event_count.

  Raises:
    ValueError: When experiment_id is not found in the forest.
  """
  exp_dir = forest.store.config.experiment_path(slug=experiment_id)
  events = load_events(exp_dir)

  nodes = forest.query().filter(id=experiment_id).all()
  if not nodes:
    msg = f'experiment {experiment_id!r} not found in forest'
    raise ValueError(msg)

  node = nodes[0]
  state = node.experiment.state_dict()
  summary: dict[str, Any] = {'event_count': len(events)}
  summary['id'] = state['id']
  summary['hypothesis'] = state.get('hypothesis')
  summary['epoch'] = state.get('epoch', -1)
  summary['status'] = state.get('status')
  summary['metrics'] = state.get('metrics', {})
  return summary


class ReportCompare(Command):
  """Compare two or more experiments, or compare best-per-tree.

  Classic mode: ``report compare <baseline> <candidate> [<candidate2> ...]``.
  First slug is baseline; each subsequent slug is a candidate. Comparisons
  are pairwise baseline-vs-each-candidate with structured ``Delta`` records
  including significance flags.

  All-trees mode: ``report compare --all-trees --metric <name>``.
  Picks the best experiment per tree under the given metric, then selects
  an overall winner via ``MetricsComparator.best_index``.
  """

  name = 'compare'
  help = 'Compare experiments: first slug is baseline, rest are candidates (baseline-centric)'

  def register(self, subparsers: argparse._SubParsersAction) -> None:
    """Register compare subparser with baseline-centric epilog."""
    super().register(subparsers)
    parser = subparsers.choices[self.name]
    parser.epilog = REPORT_COMPARE_EPILOG
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

  slugs = Argument(
    'slugs',
    nargs='*',
    metavar='SLUG',
    help='experiment slugs: first is baseline, rest are candidates',
  )
  lower_metric = Argument(
    '--lower-metric',
    action='append',
    default=None,
    metavar='NAME',
    help='metric where lower is better (repeatable)',
  )
  union_metrics = Flag(
    '--union-metrics',
    help='include metrics present in only one experiment (missing side zero-filled)',
  )
  all_trees = Flag(
    '--all-trees',
    help='compare best experiment per tree (requires --metric)',
  )
  metric = Argument(
    '--metric',
    default=None,
    metavar='METRIC',
    help='metric to optimize (required with --all-trees)',
  )
  higher = Flag(
    '--higher',
    help='higher metric values are better for --all-trees (default)',
  )
  lower = Flag(
    '--lower',
    help='lower metric values are better for --all-trees',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run classic multi-way or all-trees comparison."""
    if args.all_trees:
      self._forward_all_trees(ctx, args)
    else:
      self._forward_classic(ctx, args)

  def _forward_classic(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run multi-way comparison with MetricsComparator deltas.

    Comparisons are baseline-centric: the first slug is the baseline, and
    each subsequent slug is compared pairwise against it. Deltas and
    ``is_improvement`` semantics are always relative to the baseline
    (positive delta = candidate value higher than baseline). Direction
    (higher-is-better vs lower-is-better) is resolved per metric via
    ``infer_direction`` heuristic, overridden by explicit
    ``--lower-metric`` flags.
    """
    slugs = args.slugs
    if len(slugs) < 2:
      ctx.fail('report compare requires at least 2 experiment slugs (baseline + candidate)')

    lower_metrics: set[str] = set(args.lower_metric) if args.lower_metric else set()
    use_union = args.union_metrics

    ctx.output.info('Comparing experiments...')
    forest = load_forest(ctx)

    try:
      summaries = [gather_summary(forest, slug) for slug in slugs]
    except ValueError as exc:
      ctx.fail(str(exc))
    baseline_metrics = summaries[0]['metrics']

    metric_comparisons: list[list[dict[str, Any]]] = []
    for candidate_summary in summaries[1:]:
      candidate_metrics = candidate_summary['metrics']

      if use_union:
        all_keys = set(baseline_metrics) | set(candidate_metrics)
      else:
        all_keys = set(baseline_metrics) & set(candidate_metrics)

      stubs = [
        ComparatorMetric(k, higher_is_better=False if k in lower_metrics else infer_direction(k))
        for k in sorted(all_keys)
      ]

      if not stubs:
        metric_comparisons.append([])
        continue

      comparator = MetricsComparator(stubs)

      baseline_filled = {k: baseline_metrics.get(k, 0.0) for k in all_keys}
      candidate_filled = {k: candidate_metrics.get(k, 0.0) for k in all_keys}
      deltas = comparator.compare(baseline_filled, candidate_filled)
      metric_comparisons.append([d.to_dict() for d in deltas])

    ctx.output.result(
      {
        'summaries': summaries,
        'metric_comparisons': metric_comparisons,
      }
    )

  def _forward_all_trees(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Compare best experiment per tree under a single metric."""
    if args.slugs:
      ctx.fail(
        'report compare --all-trees does not accept positional slugs;'
        ' it compares the best experiment per tree automatically'
      )
    if args.metric is None:
      ctx.fail(
        'report compare --all-trees requires --metric <name>;'
        ' specify the metric to optimize across trees'
      )

    higher_is_better = not args.lower
    forest = load_forest(ctx)

    all_nodes = []
    for tree in forest.list_trees():
      all_nodes.extend(tree.query().all())
    if not all_nodes:
      ctx.fail('no experiments found in any tree')

    resolved_metric = resolve_metric_name(all_nodes, args.metric)

    trees_payload: list[dict[str, Any]] = []
    winner_candidates: list[dict[str, float]] = []
    winner_indices: list[int] = []

    for idx, tree in enumerate(forest.list_trees()):
      best_node = tree.query().best(resolved_metric, higher_is_better=higher_is_better)
      if best_node is None:
        trees_payload.append({'tree': tree.name, 'best': None, 'metric_value': None})
        continue
      metric_value = best_node.experiment.metrics.get(resolved_metric)
      trees_payload.append(
        {
          'tree': tree.name,
          'best': best_node.experiment.id,
          'metric_value': metric_value,
        }
      )
      if metric_value is not None:
        winner_candidates.append({resolved_metric: metric_value})
        winner_indices.append(idx)

    winner: dict[str, Any] | None = None
    if winner_candidates:
      comparator = MetricsComparator(
        [ComparatorMetric(resolved_metric, higher_is_better=higher_is_better)]
      )
      best_idx = comparator.best_index(winner_candidates, resolved_metric)
      tree_idx = winner_indices[best_idx]
      winner_entry = trees_payload[tree_idx]
      winner = {
        'tree': winner_entry['tree'],
        'id': winner_entry['best'],
        'metric_value': winner_entry['metric_value'],
      }

    ctx.output.result(
      {
        'metric': resolved_metric,
        'higher_is_better': higher_is_better,
        'trees': trees_payload,
        'winner': winner,
      }
    )
