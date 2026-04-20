"""Dataset registry and split operations: list, validate, materialize, and seed."""

from autopilot.cli.command import Argument, Command, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.datasets import create_dataset_snapshot, validate_dataset
from autopilot.core.errors import DatasetError
import argparse


def _dataset_profile_config(ctx: CLIContext) -> dict:
  """Profile-style dict with datasets.splits from the project Module."""
  if not ctx.module:
    raise DatasetError(
      'dataset validate/materialize require ctx.module (project CLI with module=)',
    )
  cfg = getattr(ctx.module, 'dataset_profile_config', None)
  if not isinstance(cfg, dict):
    raise DatasetError(
      'Module must set dataset_profile_config to a dict containing datasets.splits',
    )
  return cfg


class DatasetSplit(Command):
  name = 'split'
  help = 'Inspect or validate split naming'
  split_name = Argument(
    'split_name',
    nargs='?',
    default='',
    metavar='SPLIT',
    help='split name (defaults to --split)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
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
    ctx.output.info('Listing dataset layout...')
    base = ctx.datasets_dir
    if not base.exists():
      ctx.output.result({'datasets': [], 'count': 0})
      return
    names = sorted(p.name for p in base.iterdir() if p.is_dir())
    ctx.output.result({'datasets': names, 'count': len(names)})

  @subcommand('show', help='Show dataset directory context')
  def show(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    name = ctx.dataset or 'default'
    ctx.output.info(f'Showing dataset context for {name!r}...')
    ctx.output.result({'dataset': name, 'datasets_dir': str(ctx.datasets_dir)})

  @subcommand('validate', help='Validate dataset splits from module config')
  def validate(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    ctx.output.info('Validating datasets against module.dataset_profile_config...')
    cfg = _dataset_profile_config(ctx)
    entries = validate_dataset(ctx.workspace, cfg)
    rows = [{'name': e.name, 'split': e.split, 'rows': e.rows} for e in entries]
    ctx.output.table(rows, ['name', 'split', 'rows'])
    ctx.output.result({'validated': len(entries)})

  @subcommand('materialize', help='Materialize dataset snapshot metadata')
  def materialize(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    ctx.output.info('Materializing dataset snapshot (metadata only)...')
    cfg = _dataset_profile_config(ctx)
    entries = validate_dataset(ctx.workspace, cfg)
    snapshot = create_dataset_snapshot(entries)
    ctx.output.result({'entries': len(snapshot.entries), 'created_at': snapshot.created_at})

  @subcommand('seed', help='Seed dataset layout under autopilot/datasets')
  def seed(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    ctx.output.info('Seeding dataset directories...')
    base = ctx.datasets_dir
    for sub in ('train', 'val', 'test'):
      (base / sub).mkdir(parents=True, exist_ok=True)
    ctx.output.result({'datasets_dir': str(base), 'status': 'seeded'})
