"""Recommendation command -- agent-facing evidence-based next-step advice.

Read-only, context-exempt, ``--json`` supported. Loads the forest, builds
a ForestRecommender, and prints a structured Recommendation or JSON
envelope.

``--metric-gt`` / ``--metric-lt`` (repeatable) pre-filter candidates
before ranking. Uses the same ``NAME:NUMBER`` validation as ``query``
(shared ``parse_metric_threshold_spec``). AND semantics: a candidate
must satisfy all supplied predicates.

JSON result schema::

  {
    'action': str,  # deploy | rollback | continue | branch | investigate
    'experiment_id': str | null,
    'confidence': str,  # high | medium | low
    'reasoning': list[str],
    'alternatives': list[str],
    'evidence': dict,
  }
"""

from autopilot.ai.recommend import ForestRecommender
from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest, parse_metric_threshold_spec
from autopilot.cli.primitives import Argument
from autopilot.core.comparison import ComparatorMetric, MetricsComparator
from autopilot.core.recommend import Recommendation
import argparse


class RecommendCommand(Command):
  """Produce an evidence-based recommendation from experiment data."""

  name = 'recommend'
  help = 'recommend next action based on experiment metrics'

  metric = Argument(
    '--metric',
    required=True,
    help='primary metric to optimize (required)',
  )
  scope = Argument(
    '--scope',
    default=None,
    help='scope filter: "tree:<name>" to restrict to one tree (default: all)',
  )
  lower = Argument(
    '--lower',
    action='store_true',
    default=False,
    help='lower metric values are better (default: higher is better)',
  )
  metric_gt_flag = Argument(
    '--metric-gt',
    action='append',
    default=None,
    dest='metric_gt',
    metavar='NAME:NUMBER',
    help='require metric > threshold (repeatable, AND semantics)',
  )
  metric_lt_flag = Argument(
    '--metric-lt',
    action='append',
    default=None,
    dest='metric_lt',
    metavar='NAME:NUMBER',
    help='require metric < threshold (repeatable, AND semantics)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Load forest, run recommender, and output the recommendation."""
    forest = load_forest(ctx)
    metric_name = args.metric
    higher_is_better = not args.lower

    gt_filters = _parse_filters(ctx, args.metric_gt, '--metric-gt')
    lt_filters = _parse_filters(ctx, args.metric_lt, '--metric-lt')

    comparator = MetricsComparator(
      [ComparatorMetric(metric_name, higher_is_better)],
    )
    recommender = ForestRecommender(
      metric_name,
      higher_is_better=higher_is_better,
    )

    rec = recommender.recommend(
      forest,
      comparator,
      scope=args.scope,
      metric_gt=gt_filters or None,
      metric_lt=lt_filters or None,
    )

    if ctx.output.use_json:
      ctx.output.result(rec.to_dict())
    else:
      _print_text(ctx, rec)


def _parse_filters(
  ctx: CLIContext,
  raw: list[str] | None,
  flag_label: str,
) -> list[tuple[str, float]]:
  """Parse repeatable metric threshold flags into validated tuples.

  Args:
    ctx: CLI context for error reporting.
    raw: Raw flag values from argparse (may be None).
    flag_label: Flag name for error messages.

  Returns:
    List of (metric_name, threshold) tuples.
  """
  if not raw:
    return []
  return [parse_metric_threshold_spec(ctx, spec, flag_label) for spec in raw]


def _print_text(ctx: CLIContext, rec: Recommendation) -> None:
  """Format a Recommendation as human-readable text output.

  Args:
    ctx: CLI context for output.
    rec: Recommendation to display.
  """
  ctx.output.info(f'Action: {rec.action}')
  ctx.output.info(f'Confidence: {rec.confidence}')
  if rec.experiment_id is not None:
    ctx.output.info(f'Experiment: {rec.experiment_id}')
  if rec.reasoning:
    ctx.output.info('Reasoning:')
    for line in rec.reasoning:
      ctx.output.info(f'  - {line}')
  if rec.alternatives:
    ctx.output.info(f'Alternatives: {", ".join(rec.alternatives)}')
