"""Reporting: experiment summaries and comparisons.

Delegates to tracking and normalization for data loading.
"""

from autopilot.cli.command import Argument, Command, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.config import resolve_experiment_dir
from autopilot.core.errors import TrackingError
from autopilot.core.normalization import load_split_summary
from autopilot.tracking.events import load_events
from autopilot.tracking.manifest import load_manifest
from typing import Any
import argparse


class ReportCompare(Command):
  name = 'compare'
  help = 'Compare experiments or runs'
  baseline = Argument('--baseline', default=None, metavar='SLUG', help='baseline experiment slug')
  candidate = Argument(
    '--candidate',
    default=None,
    metavar='SLUG',
    help='candidate experiment slug',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Compare two experiments side by side."""
    baseline_slug = args.baseline or ctx.experiment
    candidate_slug = args.candidate
    if not baseline_slug or not candidate_slug:
      ctx.output.error('compare requires --baseline and --candidate (or --experiment for baseline)')
      return
    ctx.output.info('Comparing experiments...')
    baseline_dir = resolve_experiment_dir(ctx.workspace, baseline_slug)
    candidate_dir = resolve_experiment_dir(ctx.workspace, candidate_slug)
    baseline = _gather_summary(baseline_dir)
    candidate = _gather_summary(candidate_dir)
    ctx.output.result(
      {
        'baseline': baseline,
        'candidate': candidate,
      }
    )


class ReportCommand(Command):
  name = 'report'
  help = 'Reports and comparisons'

  def __init__(self) -> None:
    super().__init__()
    self.compare = ReportCompare()

  @subcommand('summary', help='Summarize experiment outcomes')
  def summary(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Summarize an experiment's manifest, events, and split results."""
    slug = ctx.experiment
    if not slug:
      ctx.output.error('experiment slug required (--experiment)')
      return
    exp_dir = resolve_experiment_dir(ctx.workspace, slug)
    ctx.output.info(f'Summarizing experiment {slug!r}...')
    summary = _gather_summary(exp_dir)
    ctx.output.result(summary)


def _gather_summary(exp_dir: Any) -> dict[str, Any]:
  """Gather a full experiment summary from tracking files."""
  manifest = load_manifest(exp_dir)
  events = load_events(exp_dir)
  summaries: dict[str, Any] = {}
  for split in ['train', 'val', 'test']:
    try:
      summaries[split] = load_split_summary(exp_dir, split)
    except (TrackingError, OSError):
      summaries[split] = None

  return {
    'slug': manifest.slug,
    'current_epoch': manifest.current_epoch,
    'idea': manifest.idea,
    'decision': manifest.decision,
    'decision_reason': manifest.decision_reason,
    'event_count': len(events),
    'split_summaries': summaries,
    'hyperparams': manifest.hyperparams,
    'metadata': manifest.metadata,
  }
