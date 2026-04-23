"""Dataset registry and split operations: list, show, split, and seed.

Workspace commands for managing dataset directories and inspecting
split naming conventions used by the project.
"""

from autopilot.cli.command import Argument, Command, subcommand
from autopilot.cli.context import CLIContext
import argparse


class DatasetSplit(Command):
  """Inspect or validate a dataset split name.

  Currently validates that the split name resolves. Future: validate
  that the split directory contains expected files, check row counts,
  and verify content hashes against the dataset registry.
  """

  name = 'split'
  help = 'Inspect or validate split naming'
  split_name = Argument(
    'split_name',
    nargs='?',
    default=None,
    metavar='SPLIT',
    help='split name (defaults to --split)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Resolve and report the split name."""
    raw = args.split_name or ctx.split
    if not raw:
      ctx.output.info('No split specified.')
      ctx.output.result({'split': None, 'ok': True})
      return
    ctx.output.result({'split': raw, 'ok': True})


class DatasetCommand(Command):
  name = 'dataset'
  help = 'Dataset registry and splits'

  def __init__(self) -> None:
    super().__init__()
    self.split = DatasetSplit()

  @subcommand('list', help='List dataset split directories')
  def list(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List top-level dataset directories under the datasets root."""
    ctx.output.info('Listing dataset layout...')
    base = ctx.datasets_dir
    if not base.exists():
      ctx.output.result({'datasets': [], 'count': 0})
      return
    names = sorted(p.name for p in base.iterdir() if p.is_dir())
    ctx.output.result({'datasets': names, 'count': len(names)})

  @subcommand('show', help='Show dataset directory context')
  def show(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show context for the current or named dataset."""
    name = ctx.dataset or 'default'
    ctx.output.info(f'Showing dataset context for {name!r}...')
    ctx.output.result({'dataset': name, 'datasets_dir': str(ctx.datasets_dir)})

  @subcommand('seed', help='Seed dataset layout under autopilot/datasets')
  def seed(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create train/val/test split directories under the datasets root."""
    ctx.output.info('Seeding dataset directories...')
    base = ctx.datasets_dir
    for sub in ('train', 'val', 'test'):
      (base / sub).mkdir(parents=True, exist_ok=True)
    ctx.output.result({'datasets_dir': str(base), 'status': 'seeded'})
