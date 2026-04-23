"""Promotion workflow: plan changes, then execute when ready.

Uses Experiment class for lifecycle decisions.
"""

from autopilot.cli.command import Argument, Command, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.checkpoint import JSONCheckpoint
from autopilot.core.config import resolve_experiment_dir
from autopilot.core.experiment import PromotionExperiment
from autopilot.core.logger import JSONLogger
from autopilot.tracking.manifest import load_manifest
from pathlib import Path
from typing import Any
import argparse


class PromoteExecute(Command):
  name = 'execute'
  help = 'Execute promotion'
  reason = Argument('--reason', default=None, help='promotion reason')
  reviewer = Argument('--reviewer', default=None, help='reviewer name')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Execute the promotion, updating the experiment decision."""
    slug = ctx.experiment
    if not slug:
      ctx.output.error('experiment slug required (--experiment)')
      return
    exp_dir = resolve_experiment_dir(ctx.workspace, slug)

    if ctx.dry_run:
      check = _check_promotable(exp_dir)
      ctx.output.warn('dry-run: no promotion applied')
      ctx.output.result({**check, 'applied': False})
      return

    experiment = PromotionExperiment(
      exp_dir,
      slug=slug,
      logger=JSONLogger(exp_dir),
      checkpoint=JSONCheckpoint(),
    )
    if experiment.is_decided:
      ctx.output.result(
        {
          'slug': slug,
          'decision': experiment.decision,
          'error': 'experiment already decided',
        },
        ok=False,
      )
      return

    reason = args.reason or 'promoted via CLI'
    reviewer = args.reviewer or 'operator'
    experiment.promote(reason, reviewer=reviewer)

    ctx.output.info(f'Experiment {slug!r} promoted.')
    ctx.output.result(
      {
        'slug': slug,
        'decision': 'promoted',
        'reason': reason,
        'reviewer': reviewer,
      }
    )


class PromoteCommand(Command):
  name = 'promote'
  help = 'Promotion plan and execution'

  def __init__(self) -> None:
    super().__init__()
    self.execute = PromoteExecute()

  @subcommand('plan', help='Show what promotion would do')
  def plan(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show what promotion would do without applying it."""
    slug = ctx.experiment
    if not slug:
      ctx.output.error('experiment slug required (--experiment)')
      return
    exp_dir = resolve_experiment_dir(ctx.workspace, slug)
    ctx.output.info('Planning promotion...')
    check = _check_promotable(exp_dir)
    ctx.output.result(check, ok=not check['is_decided'])


def _check_promotable(exp_dir: Path) -> dict[str, Any]:
  """Check whether the experiment is in a promotable state."""
  manifest = load_manifest(exp_dir)
  result_path = exp_dir / 'result.json'
  has_result = result_path.is_file()
  return {
    'slug': manifest.slug,
    'is_decided': bool(manifest.decision),
    'has_result': has_result,
    'decision': manifest.decision,
  }
