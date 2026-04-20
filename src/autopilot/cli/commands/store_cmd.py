"""Store CLI: content-addressed versioning for experiment code."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store import FileStore
from autopilot.cli.command import Command, argument, subcommand
from autopilot.cli.context import CLIContext
from autopilot.core.parameter import Parameter
from pathlib import Path
import argparse
import autopilot.core.paths as paths


def _require_experiment(ctx: CLIContext) -> str:
  slug = ctx.experiment
  if not slug:
    raise ValueError('experiment slug required (--experiment)')
  return slug


def _store_root(ctx: CLIContext, args: argparse.Namespace) -> Path:
  if args.store:
    return Path(args.store).resolve()
  return paths.store(ctx.workspace, ctx.project)


def _parameters(args: argparse.Namespace) -> list[Parameter]:
  if not args.source:
    raise ValueError('source directory required (--source)')
  pattern = args.pattern
  resolved = Path(args.source).expanduser().resolve()
  return [PathParameter(source=str(resolved), pattern=pattern)]


def _open_file_store(ctx: CLIContext, args: argparse.Namespace) -> FileStore:
  root = _store_root(ctx, args)
  slug = _require_experiment(ctx)
  params = _parameters(args)
  return FileStore(root, slug, params)


class StoreCommand(Command):
  name = 'store'
  help = 'Content-addressed code store'

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @subcommand('create', help='Initialize a file store for an experiment')
  def create(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    slug = _require_experiment(ctx)
    root = _store_root(ctx, args)
    params = _parameters(args)
    store = FileStore(root, slug, params)
    ctx.output.result(
      {
        'slug': store.slug,
        'epoch': store.epoch,
        'path': str(root),
      }
    )

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @subcommand('snapshot', help='Record a new snapshot for the next sequential epoch')
  def snapshot(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    if ctx.dry_run:
      ctx.output.result({'dry_run': True, 'command': 'snapshot'})
      return
    store = _open_file_store(ctx, args)
    next_epoch = store.epoch + 1
    manifest = store.snapshot(next_epoch)
    ctx.output.result(
      {
        'epoch': manifest.epoch,
        'timestamp': manifest.timestamp,
        'file_count': len(manifest.entries),
      }
    )

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @subcommand('checkout', help='Restore tracked files to a snapshot')
  def checkout(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    if ctx.dry_run:
      ctx.output.result({'dry_run': True, 'command': 'checkout'})
      return
    store = _open_file_store(ctx, args)
    store.checkout(ctx.epoch)
    ctx.output.result({'slug': store.slug, 'epoch': ctx.epoch})

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @argument('--with-slug', required=True, metavar='SLUG', help='other experiment slug')
  @argument(
    '--epoch-a',
    type=int,
    default=None,
    help='epoch on current slug (default: latest)',
  )
  @argument('--epoch-b', type=int, default=0, help='epoch on the other slug')
  @subcommand('diff', help='Compare snapshots between slugs')
  def diff(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    store = _open_file_store(ctx, args)
    epoch_a = args.epoch_a if args.epoch_a is not None else store.epoch
    result = store.diff(epoch_a, args.with_slug, args.epoch_b)
    ctx.output.result(result.to_dict())

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @argument('--new-slug', required=True, metavar='SLUG', help='new branch slug')
  @argument('--from-epoch', type=int, default=0, help='fork point epoch')
  @subcommand('branch', help='Fork state into a new slug')
  def branch(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    if ctx.dry_run:
      ctx.output.result({'dry_run': True, 'command': 'branch'})
      return
    store = _open_file_store(ctx, args)
    store.branch(args.new_slug, args.from_epoch)
    ctx.output.result({'new_slug': args.new_slug, 'from_epoch': args.from_epoch})

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @argument('--from-slug', required=True, metavar='SLUG', help='slug to merge from')
  @argument(
    '--merge-epoch',
    type=int,
    default=None,
    metavar='N',
    help='epoch on from-slug (default: latest)',
  )
  @subcommand('merge', help='Three-way merge preview from another slug')
  def merge(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    store = _open_file_store(ctx, args)
    merge_result = store.merge(args.from_slug, args.merge_epoch)
    ctx.output.result(merge_result.to_dict())

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @subcommand('log', help='List snapshots for the experiment slug')
  def log(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    store = _open_file_store(ctx, args)
    entries = store.log()
    rows = [e.to_dict() for e in entries]
    ctx.output.table(rows, ['epoch', 'timestamp', 'file_count'])
    ctx.output.result({'count': len(entries)})

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @subcommand('status', help='Compare working tree to latest snapshot')
  def status(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    store = _open_file_store(ctx, args)
    st = store.status()
    ctx.output.result(st.to_dict())

  @argument('--source', required=True, help='source directory tracked by the store')
  @argument('--store', default='', help='store root (default: workspace .store)')
  @argument('--pattern', default='**/*', help='glob for tracked files under source')
  @subcommand('promote', help='Set baseline to a snapshot epoch')
  def promote(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    if ctx.dry_run:
      ctx.output.result({'dry_run': True, 'command': 'promote'})
      return
    store = _open_file_store(ctx, args)
    store.promote(ctx.epoch)
    ctx.output.result({'slug': store.slug, 'promoted_epoch': ctx.epoch})
