"""Store peripheral CLI commands: worktree, tag, stash, copy-epoch, reflog, recover.

These commands operate on the store's auxiliary features beyond the core
VCS operations (create/snapshot/checkout/diff/branch/log/status/doctor/promote).
"""

from autopilot.cli.command import Command
from autopilot.cli.commands.store.helpers import emit_reflog, open_forest_store, require_experiment
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.primitives import Argument
from autopilot.core.errors import StoreError
from autopilot.tracking.io import parse_timestamp, read_jsonl
from datetime import UTC, datetime, timedelta
import argparse
import re

OLDER_THAN_PATTERN = re.compile(r'^(\d+)d$')


class WorktreeList(Command):
  """List active worktrees."""

  name = 'list'
  help = 'List active worktrees'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List worktree ids from the forest store (table or JSON)."""
    forest = load_forest(ctx)
    store = forest.store
    worktrees = store.list_worktrees()

    if ctx.output.use_json:
      ctx.output.result({'worktrees': worktrees})
      return

    rows = [{'experiment_id': wt} for wt in worktrees]
    ctx.output.table(rows, ['experiment_id'])


class WorktreeCreate(Command):
  """Create an empty worktree directory for an experiment."""

  name = 'create'
  help = 'Create a worktree for an experiment'
  experiment_id = Argument('experiment_id', help='experiment ID')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create a worktree directory for the given experiment id."""
    forest = load_forest(ctx)
    wt_path = forest.store.create_worktree(args.experiment_id)
    ctx.output.result(
      {
        'ok': True,
        'experiment_id': args.experiment_id,
        'path': str(wt_path),
      }
    )


class WorktreeCommand(Command):
  """Manage store worktrees."""

  name = 'worktree'
  help = 'Manage worktrees'

  def __init__(self) -> None:
    """Wire worktree list and create commands."""
    super().__init__()
    self.list_cmd = WorktreeList()
    self.create = WorktreeCreate()


class TagCreate(Command):
  """Create an immutable tag on a store branch at a given epoch."""

  name = 'create'
  help = 'Create a tag for an experiment at a specific epoch'
  tag_name = Argument(
    'tag_name',
    help=(
      'tag name (max 128 chars; allowed: a-z, A-Z, 0-9, hyphen, underscore, dot; '
      'no slashes or path-like names)'
    ),
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create an immutable tag via the forest store."""
    store = open_forest_store(ctx)
    slug = require_experiment(ctx)

    if args.epoch is not None:
      epoch = args.epoch
    else:
      refs = store.load_refs()
      branches = refs.get('branches', {})
      branch = branches.get(slug)
      if branch is None:
        ctx.fail(f'experiment {slug!r} not found in store refs')
      epoch = branch['latest_epoch']
      if epoch < 0:
        ctx.fail(f'experiment {slug!r} has no snapshots to tag')

    try:
      store.tag(args.tag_name, slug, epoch, context=ctx.context)
    except StoreError as exc:
      ctx.fail(str(exc))

    payload = {
      'tag': args.tag_name,
      'experiment_id': slug,
      'epoch': epoch,
    }

    if not ctx.output.use_json:
      ctx.output.info(f'tagged {slug!r} epoch {epoch} as {args.tag_name!r}')

    ctx.output.result(payload)


class TagList(Command):
  """List all tags in the store."""

  name = 'list'
  help = 'List all tags'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List tags from the forest store."""
    store = open_forest_store(ctx)
    tags = store.list_tags()

    rows = [
      {
        'name': t.name,
        'experiment_id': t.experiment_id,
        'epoch': t.epoch,
        'context': t.context,
        'timestamp': t.timestamp,
        'manifest_digest': t.manifest_digest,
      }
      for t in tags
    ]

    if not ctx.output.use_json:
      if not rows:
        ctx.output.info('no tags found')
      else:
        ctx.output.table(rows, ['name', 'experiment_id', 'epoch', 'context', 'timestamp'])

    ctx.output.result({'tags': rows, 'count': len(rows)})


class TagVerify(Command):
  """Verify a tag's manifest digest for integrity checking."""

  name = 'verify'
  help = 'Verify tag manifest digest (read-only integrity check)'
  tag_name = Argument(
    'tag_name',
    help='tag name to verify',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Verify a tag's manifest digest and emit the result.

    Raises:
      SystemExit: With code 1 when verification fails (mismatch or no digest).
    """
    store = open_forest_store(ctx)

    try:
      result = store.verify_tag(args.tag_name)
    except StoreError as exc:
      ctx.fail(str(exc))

    if not ctx.output.use_json:
      verified = result.get('verified', False)
      if verified:
        ctx.output.info(f'tag {args.tag_name!r}: verified')
      else:
        reason = result.get('reason', 'unknown')
        ctx.output.info(f'tag {args.tag_name!r}: FAILED ({reason})')
        if reason == 'digest mismatch':
          ctx.output.info(f'  expected: {result["expected"]}')
          ctx.output.info(f'  actual:   {result["actual"]}')

    ctx.output.result(result)

    if not result.get('verified', False):
      raise SystemExit(1)


class TagCommand(Command):
  """Manage store tags (immutable named markers)."""

  name = 'tag'
  help = 'Manage tags'

  def __init__(self) -> None:
    """Wire tag create, list, and verify commands."""
    super().__init__()
    self.create = TagCreate()
    self.list_cmd = TagList()
    self.verify = TagVerify()


class StashList(Command):
  """List stash manifests (indices, timestamps, entry counts)."""

  name = 'stash-list'
  help = 'List stash entries'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List stash entries from the forest store."""
    store = open_forest_store(ctx)
    manifests = store.stash_list()
    rows = [
      {
        'index': idx,
        'timestamp': m.timestamp,
        'entry_count': len(m.entries),
        'context': m.context,
      }
      for idx, m in enumerate(manifests)
    ]

    if not ctx.output.use_json:
      if not rows:
        ctx.output.info('stash stack is empty')
      else:
        ctx.output.table(rows, ['index', 'timestamp', 'entry_count', 'context'])

    ctx.output.result({'stashes': rows, 'count': len(rows)})


class StashPop(Command):
  """Restore a stash entry to working parameters and remove it."""

  name = 'stash-pop'
  help = 'Pop a stash entry (restore and remove)'
  index = Argument(
    '--index',
    type=int,
    default=None,
    help='stash index to pop (default: newest / LIFO)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Pop a stash entry via the forest store.

    Registers parameters from the latest manifest schema so
    ``stash_pop`` can restore files to the correct locations.
    """
    slug = require_experiment(ctx)
    store = open_forest_store(ctx, register_for_experiment=slug)
    try:
      manifest = store.stash_pop(index=args.index, context=ctx.context)
    except StoreError as exc:
      ctx.fail(str(exc))

    ctx.output.result(
      {
        'popped_index': args.index,
        'timestamp': manifest.timestamp,
        'entry_count': len(manifest.entries),
      }
    )


class CopyEpoch(Command):
  """Copy a snapshot from one experiment branch to another."""

  name = 'copy-epoch'
  help = 'Copy a snapshot epoch from source to target branch (cherry-pick)'
  source_exp = Argument('source_exp', help='source experiment id')
  source_epoch = Argument('source_epoch', type=int, help='source epoch number')
  target_exp = Argument('target_exp', help='target experiment id')
  store = Argument('--store', default=None, help='store root path override')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Copy a snapshot manifest from source to target branch."""
    store = open_forest_store(ctx, args)
    try:
      manifest = store.copy_epoch(
        args.source_exp,
        args.source_epoch,
        args.target_exp,
        context=ctx.context,
      )
    except StoreError as exc:
      ctx.fail(str(exc))

    payload = {
      'epoch': manifest.epoch,
      'file_count': len(manifest.entries),
      'source_experiment_id': args.source_exp,
      'source_epoch': args.source_epoch,
      'target_experiment_id': args.target_exp,
    }

    if not ctx.output.use_json:
      ctx.output.info(
        f'copied epoch {args.source_epoch} from {args.source_exp!r} '
        f'to {args.target_exp!r} as epoch {manifest.epoch}'
      )
      ctx.output.info(f'files: {len(manifest.entries)}')

    ctx.output.result(payload)


class ReflogExpire(Command):
  """Expire old reflog entries by age."""

  name = 'expire'
  help = 'Remove reflog entries older than a duration'
  older_than = Argument(
    '--older-than',
    required=True,
    help='duration cutoff in days (e.g. 30d)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Expire reflog entries older than the specified duration.

    Parses ``--older-than`` as ``Nd`` (integer days with literal ``d`` suffix).
    With ``--dry-run``, reports the projected count without deleting.
    """
    raw = args.older_than
    match = OLDER_THAN_PATTERN.match(raw)
    if match is None:
      ctx.fail(f'invalid --older-than value {raw!r}: expected format Nd (e.g. 30d)')
    days = int(match.group(1))
    duration = timedelta(days=days)

    store = open_forest_store(ctx)

    if ctx.dry_run:
      cutoff = datetime.now(UTC) - duration
      count = 0
      for entry in store.iter_reflog():
        ts_str = entry.get('timestamp')
        if ts_str is None:
          continue
        try:
          ts = parse_timestamp(ts_str)
        except (ValueError, TypeError):
          continue
        if ts < cutoff:
          count += 1
      ctx.output.result({'ok': True, 'expired_count': count, 'dry_run': True})
      return

    count = store.expire_reflog(duration)

    if not ctx.output.use_json:
      ctx.output.info(f'expired {count} reflog entries older than {days}d')

    ctx.output.result({'ok': True, 'expired_count': count})


class StoreRecover(Command):
  """Recover branch tip from a reflog entry."""

  name = 'recover'
  help = 'Restore branch tip metadata from a reflog entry'
  reflog_entry = Argument(
    '--reflog-entry',
    type=int,
    required=True,
    help='0-based reflog entry index (same order as iter_reflog / debug store reflog)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Restore branch tip metadata from reflog entry at the given index.

    Updates refs so latest_epoch and HEAD match the entry. Does not run
    checkout or touch working-tree files.
    """
    if ctx.dry_run:
      ctx.output.result({'ok': True, 'dry_run': True, 'reflog_entry': args.reflog_entry})
      return

    store = open_forest_store(ctx)
    entry_index = args.reflog_entry

    entries = list(store.iter_reflog())
    if entry_index < 0 or entry_index >= len(entries):
      ctx.fail(
        f'reflog entry index {entry_index} out of range '
        f'(valid: 0..{len(entries) - 1 if entries else 0})'
      )

    entry = entries[entry_index]
    try:
      store.recover_from_reflog(entry_index)
    except StoreError as exc:
      ctx.fail(str(exc))

    experiment_id = entry.get('experiment_id')
    epoch = entry.get('new_epoch')

    if not ctx.output.use_json:
      ctx.output.info(f'recovered branch tip: experiment={experiment_id!r}, epoch={epoch}')

    ctx.output.result(
      {
        'ok': True,
        'experiment_id': experiment_id,
        'epoch': epoch,
      }
    )


class ReflogList(Command):
  """List reflog entries (read-only, same output as debug store reflog)."""

  name = 'list'
  help = 'List store reflog entries'
  limit = Argument(
    '-n',
    '--limit',
    type=int,
    default=None,
    help='max entries to show (most recent)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Display the append-only reflog via the shared emit_reflog helper.

    Reads ``reflog.jsonl`` from the store directory. Produces identical
    JSON output to ``debug store reflog`` for agent parser parity.
    """
    config = ctx.config
    reflog_path = config.store_path / 'reflog.jsonl'
    entries = read_jsonl(reflog_path, strict=False)
    emit_reflog(ctx, entries, limit=args.limit)


class ReflogCommand(Command):
  """Manage reflog lifecycle (list, expire old entries)."""

  name = 'reflog'
  help = 'Reflog lifecycle management'

  def __init__(self) -> None:
    """Wire reflog list and expire subcommands."""
    super().__init__()
    self.list_cmd = ReflogList()
    self.expire = ReflogExpire()
