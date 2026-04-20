"""Experiment lifecycle: create, list, inspect, and resume.

Uses Experiment class for creation, load_manifest for listing.
"""

from autopilot.cli.command import Argument, Command, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.checkpoint import JSONCheckpoint
from autopilot.core.checkpoints import load_checkpoint
from autopilot.core.config import resolve_experiment_dir
from autopilot.core.decisions import is_decided
from autopilot.core.errors import TrackingError
from autopilot.core.experiment import Experiment
from autopilot.core.logger import JSONLogger
from autopilot.tracking.manifest import load_manifest
import argparse


class ExperimentCreate(Command):
  name = 'create'
  help = 'Create a new experiment'
  slug = Argument('--slug', default='', help='experiment slug')
  idea = Argument('--idea', default='', help='short idea label')
  title = Argument('--title', default='', help='experiment title')
  hypothesis = Argument('--hypothesis', default='', help='hypothesis under test')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = args.slug or ctx.experiment
    if not slug:
      raise ValueError('experiment slug required (--slug or --experiment)')

    ctx.output.info(f'Creating experiment {slug!r}...')
    exp_dir = resolve_experiment_dir(ctx.workspace, slug, ctx.project)
    experiment = Experiment(
      experiment_dir=exp_dir,
      slug=slug,
      logger=JSONLogger(exp_dir),
      checkpoint=JSONCheckpoint(),
      title=args.title or slug,
      idea=args.idea,
      hypothesis=args.hypothesis,
    )
    ctx.output.result(
      {
        'slug': experiment.slug,
        'path': str(exp_dir),
      }
    )


class ExperimentCommand(Command):
  name = 'experiment'
  help = 'Experiment management'

  def __init__(self) -> None:
    super().__init__()
    self.create = ExperimentCreate()

  @subcommand('list', help='List experiments')
  def list(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    ctx.output.info('Listing experiments...')
    root = ctx.experiments_dir
    if not root.exists():
      ctx.output.result({'experiments': [], 'count': 0})
      return
    rows = []
    for exp_path in sorted(root.iterdir()):
      if not exp_path.is_dir():
        continue
      manifest_path = exp_path / 'manifest.json'
      decision = ''
      if manifest_path.is_file():
        try:
          m = load_manifest(exp_path)
          decision = m.decision
        except TrackingError:
          decision = 'error'
      rows.append({'slug': exp_path.name, 'decision': decision})
    ctx.output.table(rows, ['slug', 'decision'])
    ctx.output.result({'count': len(rows)})

  @subcommand('show', help='Show experiment details')
  def show(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = ctx.experiment
    if not slug:
      raise ValueError('experiment slug required (--experiment)')
    exp_dir = resolve_experiment_dir(ctx.workspace, slug, ctx.project)
    if not exp_dir.exists():
      ctx.output.result({'slug': slug, 'exists': False}, ok=False)
      return
    manifest = load_manifest(exp_dir)
    ctx.output.result(manifest.to_dict())

  @subcommand('status', help='Show experiment status')
  def status(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = ctx.experiment
    if not slug:
      raise ValueError('experiment slug required (--experiment)')
    exp_dir = resolve_experiment_dir(ctx.workspace, slug, ctx.project)
    checkpoint = load_checkpoint(exp_dir)
    if not checkpoint:
      ctx.output.result({'slug': slug, 'exists': False}, ok=False)
      return
    ctx.output.result(
      {
        'slug': checkpoint.slug,
        'current_epoch': checkpoint.current_epoch,
        'decision': checkpoint.decision,
      }
    )

  @subcommand('resume', help='Resume an experiment run')
  def resume(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = ctx.experiment
    if not slug:
      raise ValueError('experiment slug required (--experiment)')
    exp_dir = resolve_experiment_dir(ctx.workspace, slug, ctx.project)
    checkpoint = load_checkpoint(exp_dir)
    if not checkpoint:
      ctx.output.result({'slug': slug, 'error': 'no manifest found'}, ok=False)
      return
    if is_decided(checkpoint):
      ctx.output.result(
        {
          'slug': slug,
          'decision': checkpoint.decision,
          'resumable': False,
        },
        ok=False,
      )
      return
    ctx.output.info(f'Experiment {slug!r} is resumable (epoch {checkpoint.current_epoch})')
    ctx.output.result(
      {
        'slug': checkpoint.slug,
        'current_epoch': checkpoint.current_epoch,
        'resumable': True,
      }
    )
