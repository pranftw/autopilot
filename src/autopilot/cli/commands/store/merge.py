"""Store merge CLI: analysis, preview, resolve, and apply subcommands.

Three-step merge workflow:
  merge-analysis -> merge-preview -> (merge-resolve)* -> merge-apply.
  The --token from merge-preview couples preview/resolve/apply invocations.
"""

from autopilot.ai.store_lock import hash_bytes
from autopilot.cli.command import Command
from autopilot.cli.commands.store.helpers import open_forest_store
from autopilot.cli.context import CLIContext
from autopilot.cli.primitives import Argument
from autopilot.core.errors import StoreError
from autopilot.core.snapshot import FileEntry
from autopilot.core.store.base import Store
from autopilot.core.store.types import MergeIndex, MergeStrategy
from autopilot.tracking.io import atomic_write_json, read_json_dict
from pathlib import Path
import argparse

MERGE_CONFLICT_DISPLAY_LIMIT = 20


def _merge_preview_dir(ctx: CLIContext) -> Path:
  """Return the merge preview cache directory under the workspace .autopilot dir."""
  return ctx.autopilot_dir / 'merge_preview'


def _save_merge_index(ctx: CLIContext, merge_index: MergeIndex) -> None:
  """Persist a MergeIndex to the preview cache keyed by its preview_token."""
  cache_dir = _merge_preview_dir(ctx)
  cache_dir.mkdir(parents=True, exist_ok=True)
  path = cache_dir / f'{merge_index.preview_token}.json'
  atomic_write_json(path, merge_index.to_dict())


def _load_merge_index(ctx: CLIContext, token: str) -> MergeIndex:
  """Load a cached MergeIndex by preview token.

  Args:
    ctx: CLI context for path resolution.
    token: Preview token identifying the cached index.

  Returns:
    Deserialized MergeIndex.

  Raises:
    StoreError: If the cache file is missing or cannot be parsed.
  """
  path = _merge_preview_dir(ctx) / f'{token}.json'
  if not path.is_file():
    msg = (
      f'merge preview cache not found for token {token!r}; '
      f'run merge-preview first to generate a preview'
    )
    raise StoreError(msg)
  data = read_json_dict(path, f'merge preview {token}')
  return MergeIndex.from_dict(data)


def _conflict_entry_to_dict(entry: FileEntry | None) -> dict | None:
  """Project a FileEntry side to CLI JSON format (digest + size only).

  Returns:
    Dict with ``digest`` and ``size`` keys, or None when the entry is absent.
  """
  if entry is None:
    return None
  return {'digest': entry.digest, 'size': entry.size}


def _resolve_content_file(
  ctx: CLIContext,
  store: Store,
  merge_index: MergeIndex,
  key: str,
  content_arg: str,
) -> None:
  """Resolve a conflict key using content from a file path.

  Reads the file, validates UTF-8, stores the blob, and resolves in the index.

  Args:
    ctx: CLI context for error output.
    store: Store for blob storage (must support ``store_blob``).
    merge_index: MergeIndex being resolved.
    key: Conflict key to resolve.
    content_arg: Path string to the content file.
  """
  content_path = Path(content_arg).expanduser().resolve()
  if not content_path.is_file():
    ctx.fail(f'content file not found: {content_path}')
  raw_bytes = content_path.read_bytes()
  try:
    raw_bytes.decode('utf-8')
  except UnicodeDecodeError as exc:
    ctx.fail(f'content file is not valid UTF-8: {exc}')
  digest = hash_bytes(raw_bytes)
  store.store_blob(digest, raw_bytes)
  entry = FileEntry(digest=digest, size=len(raw_bytes), mtime=0.0)
  merge_index.resolve(key, entry)


class MergeAnalysis(Command):
  """Classify a potential merge between two experiments."""

  name = 'merge-analysis'
  help = 'Classify merge: fast-forward, clean, conflict, or up-to-date'
  experiment_id = Argument('experiment_id', help='target experiment (ours)')
  from_experiment_id = Argument('from_experiment_id', help='source experiment (theirs)')
  store = Argument('--store', default=None, help='store root path override')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run merge analysis and emit classification result."""
    store = open_forest_store(ctx, args)
    try:
      result = store.merge_analysis(args.experiment_id, args.from_experiment_id)
    except StoreError as exc:
      ctx.fail(str(exc))

    payload = {
      'classification': result.classification,
      'can_fast_forward': result.can_fast_forward,
      'has_conflicts': result.has_conflicts,
      'conflict_count': result.conflict_count,
      'ancestor_epoch': result.ancestor_epoch,
    }

    if not ctx.output.use_json:
      rows = [payload]
      columns = [
        'classification',
        'can_fast_forward',
        'has_conflicts',
        'conflict_count',
        'ancestor_epoch',
      ]
      ctx.output.table(rows, columns)

    ctx.output.result(payload)


class MergePreview(Command):
  """Compute a three-way merge preview and cache the MergeIndex.

  Union strategy concatenates text content line-by-line when ancestor
  exists, or raw concatenation otherwise.
  """

  name = 'merge-preview'
  help = (
    'Preview merge conflicts between two experiments. '
    'Union strategy concatenates text content line-by-line when ancestor '
    'exists, or raw concatenation otherwise.'
  )
  experiment_id = Argument('experiment_id', help='target experiment (ours)')
  from_experiment_id = Argument('from_experiment_id', help='source experiment (theirs)')
  strategy = Argument(
    '--strategy',
    default='normal',
    choices=['normal', 'ours', 'theirs', 'union'],
    help='merge strategy (default: normal)',
  )
  store = Argument('--store', default=None, help='store root path override')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Compute merge preview and persist MergeIndex to cache."""
    store = open_forest_store(ctx, args)
    try:
      strategy = MergeStrategy(args.strategy)
    except ValueError:
      ctx.fail(f'unknown merge strategy: {args.strategy!r}')

    try:
      merge_index = store.merge_preview(
        args.experiment_id,
        args.from_experiment_id,
        from_epoch=None,
        strategy=strategy,
      )
    except StoreError as exc:
      ctx.fail(str(exc))

    _save_merge_index(ctx, merge_index)

    conflicts_json: dict = {}
    for key, conflict in merge_index.conflicts.items():
      conflicts_json[key] = {
        'ancestor': _conflict_entry_to_dict(conflict.ancestor),
        'ours': _conflict_entry_to_dict(conflict.ours),
        'theirs': _conflict_entry_to_dict(conflict.theirs),
      }

    resolved_json: dict = {}
    for key, entry in merge_index.resolved.items():
      resolved_json[key] = {'digest': entry.digest, 'size': entry.size}

    payload = {
      'conflicts': conflicts_json,
      'resolved': resolved_json,
      'preview_token': merge_index.preview_token,
      'strategy': merge_index.strategy.value,
      'experiment_id': merge_index.experiment_id,
      'source_experiment_id': merge_index.source_experiment_id,
    }

    if not ctx.output.use_json:
      conflict_count = len(merge_index.conflicts)
      resolved_count = len(merge_index.resolved)
      ctx.output.info(f'strategy: {merge_index.strategy.value}')
      ctx.output.info(f'preview token: {merge_index.preview_token}')
      ctx.output.info(f'conflicts: {conflict_count}, resolved: {resolved_count}')
      if conflict_count > 0:
        for key in sorted(merge_index.conflicts)[:MERGE_CONFLICT_DISPLAY_LIMIT]:
          ctx.output.info(f'  conflict: {key}')
        if conflict_count > MERGE_CONFLICT_DISPLAY_LIMIT:
          ctx.output.info(f'  ... and {conflict_count - MERGE_CONFLICT_DISPLAY_LIMIT} more')

    ctx.output.result(payload)


class MergeApply(Command):
  """Apply a previewed merge from a cached MergeIndex."""

  name = 'merge-apply'
  help = 'Apply a previously previewed merge (requires --token from merge-preview)'
  token = Argument('--token', required=True, help='preview token from merge-preview')
  store = Argument('--store', default=None, help='store root path override')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Load cached MergeIndex and apply via store."""
    store = open_forest_store(ctx, args)

    try:
      merge_index = _load_merge_index(ctx, args.token)
    except StoreError as exc:
      ctx.fail(str(exc))

    try:
      manifest = store.merge_apply(merge_index)
    except StoreError as exc:
      ctx.fail(str(exc))

    cache_path = _merge_preview_dir(ctx) / f'{args.token}.json'
    cache_path.unlink(missing_ok=True)

    payload = {
      'epoch': manifest.epoch,
      'file_count': len(manifest.entries),
      'experiment_id': merge_index.experiment_id,
    }

    if not ctx.output.use_json:
      ctx.output.info(f'merge applied: epoch {manifest.epoch}')
      ctx.output.info(f'files: {len(manifest.entries)}')
      ctx.output.info(f'experiment: {merge_index.experiment_id}')

    ctx.output.result(payload)


class MergeResolve(Command):
  """Resolve a single conflict in a cached MergeIndex."""

  name = 'merge-resolve'
  help = 'Resolve one conflict key (--ours, --theirs, or --content <path>)'
  token = Argument('--token', required=True, help='preview token from merge-preview')
  key = Argument('key', help='conflict key to resolve')
  ours = Argument('--ours', action='store_true', default=False, help='resolve using ours side')
  theirs = Argument(
    '--theirs', action='store_true', default=False, help='resolve using theirs side'
  )
  content = Argument(
    '--content', default=None, metavar='PATH', help='resolve using content from file at PATH'
  )
  store = Argument('--store', default=None, help='store root path override')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Resolve one conflict key and re-save the cached MergeIndex."""
    store = open_forest_store(ctx, args)

    try:
      merge_index = _load_merge_index(ctx, args.token)
    except StoreError as exc:
      ctx.fail(str(exc))

    key = args.key
    if key not in merge_index.conflicts:
      ctx.fail(
        f'key {key!r} is not in conflicts; available conflict keys: {sorted(merge_index.conflicts)}'
      )

    mode_count = sum([args.ours, args.theirs, args.content is not None])
    if mode_count != 1:
      ctx.fail('exactly one of --ours, --theirs, or --content must be specified')

    resolution_label = self._apply_resolution(ctx, store, merge_index, key, args)

    _save_merge_index(ctx, merge_index)

    payload: dict = {'key': key, 'resolution': resolution_label}
    if resolution_label == 'content':
      payload['path'] = str(args.content)

    if not ctx.output.use_json:
      remaining = len(merge_index.conflicts)
      ctx.output.info(f'resolved {key!r} via {resolution_label}')
      if remaining > 0:
        ctx.output.info(f'{remaining} conflict(s) remaining')
      else:
        ctx.output.info('all conflicts resolved; ready for merge-apply')

    ctx.output.result(payload)

  def _apply_resolution(
    self,
    ctx: CLIContext,
    store: Store,
    merge_index: MergeIndex,
    key: str,
    args: argparse.Namespace,
  ) -> str:
    """Apply the chosen resolution mode and return the resolution label.

    Args:
      ctx: CLI context for error output.
      store: Store for blob storage.
      merge_index: MergeIndex being resolved.
      key: Conflict key to resolve.
      args: Parsed arguments with resolution mode flags.

    Returns:
      Resolution label string: ``'ours'``, ``'theirs'``, or ``'content'``.
    """
    if args.ours:
      try:
        merge_index.resolve_ours(key)
      except StoreError as exc:
        ctx.fail(str(exc))
      return 'ours'
    if args.theirs:
      try:
        merge_index.resolve_theirs(key)
      except StoreError as exc:
        ctx.fail(str(exc))
      return 'theirs'
    _resolve_content_file(ctx, store, merge_index, key, args.content)
    return 'content'
