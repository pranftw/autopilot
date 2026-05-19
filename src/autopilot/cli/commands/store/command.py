"""Store CLI: content-addressed versioning for experiment code.

Subcommands:
  store create/snapshot/checkout/diff/branch/merge/log/status/promote
    -- existing VCS operations (require --source)
  store merge-analysis <experiment> <from> -- classify merge
  store merge-preview <experiment> <from> [--strategy] -- preview conflicts
  store merge-apply --token <token> -- apply a previewed merge
  store merge-resolve --token <token> <key> (--ours|--theirs|--content) -- resolve one conflict
  store worktree list [--json]            -- list active worktrees
  store worktree create <experiment-id> [--json] -- create empty worktree

Merge workflow ordering:
  merge-analysis -> merge-preview -> (merge-resolve)* -> merge-apply.
  The --token from merge-preview couples preview/resolve/apply invocations.

All subcommands support --json for agent-friendly structured output.

Note: there is no ``store list`` command. Use ``store log`` to list snapshots
for an experiment, or ``store status`` to compare the working tree against
the latest snapshot (GAP-006).
"""

from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import Command
from autopilot.cli.commands.store.helpers import (
  emit_doctor_text,
  open_file_store,
  open_forest_store,
  open_store_for_log,
  require_experiment,
)
from autopilot.cli.commands.store.merge import (
  MergeAnalysis,
  MergeApply,
  MergePreview,
  MergeResolve,
)
from autopilot.cli.commands.store.peripherals import (
  CopyEpoch,
  ReflogCommand,
  StashList,
  StashPop,
  StoreRecover,
  TagCommand,
  WorktreeCommand,
)
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import with_store_vcs_arguments
from autopilot.cli.primitives import argument, subcommand
from autopilot.core.errors import StoreError
import argparse


class StoreCommand(Command):
  """``autopilot store`` group: content-addressed versioning for experiment code."""

  name = 'store'
  help = 'Content-addressed code store'

  def __init__(self) -> None:
    """Wire store VCS subcommands, merge commands, worktree, tag, and stash groups."""
    super().__init__()
    self.merge_analysis_cmd = MergeAnalysis()
    self.merge_preview_cmd = MergePreview()
    self.merge_apply_cmd = MergeApply()
    self.merge_resolve_cmd = MergeResolve()
    self.worktree = WorktreeCommand()
    self.tag_cmd = TagCommand()
    self.stash_list_cmd = StashList()
    self.stash_pop_cmd = StashPop()
    self.copy_epoch_cmd = CopyEpoch()
    self.reflog_cmd = ReflogCommand()
    self.recover_cmd = StoreRecover()

  @with_store_vcs_arguments
  @subcommand('create', help_text='Initialize a file store for an experiment')
  def create(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Initialize a file store for an experiment."""
    slug = require_experiment(ctx)
    store = open_file_store(ctx, args)
    manifest = store.snapshot(slug, 0, context=ctx.context)
    ctx.output.result(
      {
        'slug': slug,
        'epoch': manifest.epoch,
        'path': str(store.config.store_path),
      }
    )

  @argument(
    '--force',
    action='store_true',
    default=False,
    help='create a new epoch even when file content is unchanged from the latest snapshot',
  )
  @with_store_vcs_arguments
  @subcommand('snapshot', help_text='Record a new snapshot for the next sequential epoch')
  def snapshot(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Record a new snapshot for the next sequential epoch.

    By default, skips the snapshot when file content is identical to the
    latest epoch (idempotent).  Use ``--force`` to record a new epoch
    even when files have not changed.
    """
    if ctx.dry_run:
      ctx.output.result({'dry_run': True, 'command': 'snapshot'})
      return
    slug = require_experiment(ctx)
    store = open_file_store(ctx, args)
    entries = store.log(slug)
    next_epoch = entries[-1].epoch + 1 if entries else 0
    manifest = store.snapshot(slug, next_epoch, context=ctx.context, force=args.force)
    skipped = manifest.epoch != next_epoch
    if skipped and not ctx.output.use_json:
      ctx.output.info('No changes detected; snapshot skipped.')
    ctx.output.result(
      {
        'epoch': manifest.epoch,
        'timestamp': manifest.timestamp,
        'file_count': len(manifest.entries),
        'skipped': skipped,
      }
    )

  @with_store_vcs_arguments
  @subcommand('checkout', help_text='Restore tracked files to a snapshot')
  def checkout(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Restore tracked files to a snapshot epoch.

    Dry-run mode performs all read-only validations (experiment existence,
    epoch validity, schema matching) before returning structured results,
    rather than returning immediately (BUG-017).
    """
    slug = require_experiment(ctx)
    store = open_file_store(ctx, args)
    if ctx.epoch is None:
      ctx.fail('--epoch is required for checkout')
    epoch = ctx.epoch
    if ctx.dry_run:
      try:
        info = store.validate_checkout(slug, epoch)
        ctx.output.result(
          {
            'dry_run': True,
            'command': 'checkout',
            'experiment': slug,
            'epoch': epoch,
            **info,
          }
        )
      except StoreError as exc:
        ctx.fail(str(exc))
      return
    store.checkout(slug, epoch, context=ctx.context)
    ctx.output.result({'slug': slug, 'epoch': epoch})

  @argument(
    '--epoch-a',
    type=int,
    default=None,
    help='first epoch to compare (default: 0)',
  )
  @argument('--epoch-b', type=int, default=None, help='second epoch to compare (default: latest)')
  @with_store_vcs_arguments
  @subcommand('diff', help_text='Compare snapshots within an experiment')
  def diff(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Compare snapshots within an experiment."""
    slug = require_experiment(ctx)
    store = open_file_store(ctx, args)
    epoch_a = args.epoch_a if args.epoch_a is not None else 0
    entries = store.log(slug)
    epoch_b = args.epoch_b if args.epoch_b is not None else entries[-1].epoch
    result = store.diff(slug, epoch_a, epoch_b)
    ctx.output.result(result.to_dict())

  @argument(
    '--restore',
    action='store_true',
    default=False,
    help='with --reset: also sync working-tree files (atomic reset_and_restore)',
  )
  @argument(
    '--reset',
    action='store_true',
    default=False,
    help='reset branch latest_epoch to -1 (re-run from clean state)',
  )
  @with_store_vcs_arguments
  @subcommand('branch', help_text='Fork state into a new experiment from HEAD')
  def branch(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Fork state into a new experiment or reset an existing branch.

    Default mode: create a new branch forking from HEAD.

    With ``--reset``: reset the branch's ``latest_epoch`` to -1 so the next
    ``snapshot()`` succeeds at epoch 0. Existing snapshots are retained.
    HEAD is unchanged. Use this to re-run an experiment from clean state
    (BUG-008).

    With ``--reset --restore``: atomically reset the branch tip and sync
    working-tree files via ``reset_and_restore``. When ``--epoch N`` is
    supplied, resets to that epoch and restores its files. Without
    ``--epoch``, resets to -1 and clears tracked files.
    """
    if args.restore and not args.reset:
      ctx.fail('--restore requires --reset')
    slug = require_experiment(ctx)
    if args.reset:
      store = open_forest_store(ctx, args, register_for_experiment=slug)
      if args.restore:
        if ctx.dry_run:
          ctx.output.result(
            {
              'dry_run': True,
              'command': 'branch',
              'reset': True,
              'restore': True,
              'epoch': ctx.epoch,
            }
          )
          return
        store.reset_and_restore(slug, ctx.epoch, context=ctx.context)
        ctx.output.result(
          {
            'experiment_id': slug,
            'reset': True,
            'restore': True,
            'epoch': ctx.epoch,
          }
        )
        return
      if ctx.dry_run:
        ctx.output.result({'dry_run': True, 'command': 'branch'})
        return
      store.reset_branch(slug)
      ctx.output.result({'experiment_id': slug, 'reset': True})
      return
    if ctx.dry_run:
      ctx.output.result({'dry_run': True, 'command': 'branch'})
      return
    store = open_file_store(ctx, args)
    store.branch(slug)
    ctx.output.result({'experiment_id': slug})

  @argument('--from-experiment', required=True, metavar='ID', help='experiment to merge from')
  @argument(
    '--merge-epoch',
    type=int,
    default=None,
    metavar='N',
    help='epoch on from-experiment (default: latest)',
  )
  @with_store_vcs_arguments
  @subcommand('merge', help_text='Three-way merge preview from another experiment')
  def merge(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Three-way merge preview from another experiment."""
    slug = require_experiment(ctx)
    store = open_file_store(ctx, args)
    merge_index = store.merge_preview(slug, args.from_experiment, args.merge_epoch)
    ctx.output.result(merge_index.to_dict())

  @argument(
    '--pattern',
    default='**/*',
    help='glob for tracked files under source',
  )
  @argument(
    '--store',
    default=None,
    help='store root (default: workspace .store)',
  )
  @argument(
    '--source',
    required=False,
    default=None,
    help=(
      'source directory tracked by the store '
      '(default: infer from active forest experiment when possible)'
    ),
  )
  @subcommand('log', help_text='List snapshots for the experiment')
  def log(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List all snapshots for the experiment.

    When ``--source`` is omitted, infers the store from the active forest
    experiment. Falls back to an explicit error when no active experiment
    is resolvable.
    """
    slug = require_experiment(ctx)
    store = open_store_for_log(ctx, args)
    entries = store.log(slug)
    rows = [e.to_dict() for e in entries]
    ctx.output.table(rows, ['epoch', 'timestamp', 'file_count'])
    ctx.output.result({'count': len(entries)})

  @with_store_vcs_arguments
  @subcommand('status', help_text='Compare working tree to latest snapshot')
  def status(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Compare working tree to the latest snapshot."""
    slug = require_experiment(ctx)
    store = open_file_store(ctx, args)
    st = store.status(slug)
    ctx.output.result(st.to_dict())

  @argument(
    '--repair',
    action='store_true',
    default=False,
    help='apply safe repairs for repairable issues (mutating; requires --context)',
  )
  @subcommand('doctor', help_text='Diagnose store health (read-only; --repair to fix)')
  def doctor(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run store health diagnostics, optionally applying repairs.

    Without ``--repair``, runs read-only diagnostics (context-exempt).
    With ``--repair``, applies safe repairs for repairable issues
    (requires ``--context``). ``--repair --dry-run`` previews repairs
    without applying them.
    """
    if args.repair and not ctx.dry_run and ctx.context is None:
      ctx.fail('--repair requires --context (mutating operation)')

    config = ctx.config
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    diagnostics = store.doctor()
    report = store.diagnostics_to_report(diagnostics)

    repaired: list[dict] = []
    if args.repair:
      try:
        repaired_entries = store.repair_diagnostics(
          diagnostics,
          dry_run=ctx.dry_run,
          context=ctx.context,
        )
        repaired = [e.to_dict() for e in repaired_entries]
      except StoreError as exc:
        ctx.fail(str(exc))

    if not ctx.output.use_json:
      emit_doctor_text(ctx, report, repair=args.repair, repaired=repaired)

    report['repaired'] = repaired
    report['dry_run'] = ctx.dry_run if args.repair else False
    ctx.output.result(report)

  @with_store_vcs_arguments
  @subcommand('promote', help_text='Set baseline to a snapshot epoch')
  def promote(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Set the baseline to a specific snapshot epoch."""
    if ctx.dry_run:
      ctx.output.result({'dry_run': True, 'command': 'promote'})
      return
    slug = require_experiment(ctx)
    store = open_file_store(ctx, args)
    if ctx.epoch is None:
      ctx.fail('--epoch is required for promote')
    store.materialize(slug, ctx.epoch)
    ctx.output.result({'slug': slug, 'promoted_epoch': ctx.epoch})

  @subcommand('stash', help_text='Capture current parameter state as a WIP stash')
  def stash(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Capture current registered parameter state as a numbered stash entry.

    Requires a prior snapshot for the active experiment so the store can
    infer parameter registration from the manifest schema.
    """
    slug = require_experiment(ctx)
    store = open_forest_store(ctx, register_for_experiment=slug)
    try:
      manifest = store.stash(context=ctx.context)
    except StoreError as exc:
      ctx.fail(str(exc))

    ctx.output.result(
      {
        'timestamp': manifest.timestamp,
        'entry_count': len(manifest.entries),
        'context': manifest.context,
      }
    )
