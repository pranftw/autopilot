"""Shared CLI forest/tree/experiment loaders and argument descriptors.

Centralizes the store_path.mkdir -> FileStore -> FileForest bootstrap
sequence used by five CLI command modules. Guard helpers terminate via
ctx.fail for stable agent-facing exits.

Also provides ``CLIError`` for non-fatal user-input errors and
``resolve_epoch`` for consistent epoch argument resolution.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import CLIContext
from autopilot.cli.messages import (
  MSG_EPOCH_EMPTY_STORE,
  MSG_EPOCH_INVALID,
  MSG_EPOCH_NOT_FOUND,
  MSG_EPOCH_REQUIRED,
  MSG_FOREST_ONLY_STORE,
)
from autopilot.cli.primitives import Argument, argument
from autopilot.core.errors import AutoPilotError, StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.store.base import Store
from autopilot.core.tree import Tree
from autopilot.tracking.executions import resolve_command
import argparse
import math


def _wait_ms_to_timeout_s(wait_ms: int | None) -> float | None:
  """Convert CLI ``--wait`` milliseconds to lock timeout seconds.

  Args:
    wait_ms: ``None`` = fail-fast, ``0`` = block forever, ``N > 0`` = wait N ms.

  Returns:
    ``None`` (fail-fast), ``-1.0`` (block forever), or positive float seconds.

  Raises:
    ValueError: If wait_ms is negative.
  """
  if wait_ms is None:
    return None
  if wait_ms < 0:
    msg = f'--wait must be non-negative, got {wait_ms}'
    raise ValueError(msg)
  if wait_ms == 0:
    return -1.0
  return wait_ms / 1000.0


def load_forest(ctx: CLIContext) -> FileForest:
  """Bootstrap a FileForest from the CLI context's config.

  Creates the store directory if needed and hydrates the forest
  from persisted state. Threads the ``--wait`` timeout through to
  the forest and store lock layers.

  Args:
    ctx: CLI context whose ``config`` defines store layout.

  Returns:
    Loaded ``FileForest`` backed by a new ``FileStore``.
  """
  config = ctx.config
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  timeout_s = _wait_ms_to_timeout_s(ctx.wait_timeout_ms)
  store.lock_timeout_s = timeout_s
  forest = FileForest(store)
  forest.lock_timeout_s = timeout_s
  return forest


def require_active_tree(
  ctx: CLIContext,
  forest: FileForest,
  message: str | None = None,
) -> Tree:
  """Return the active tree or terminate via ctx.fail.

  Args:
    ctx: Context used for failure output.
    forest: Forest whose ``active`` tree is required.
    message: Optional override for the failure reason.

  Returns:
    The active ``Tree`` instance.
  """
  tree = forest.active
  if tree is None:
    ctx.fail(
      message or 'no active tree in forest. Run: autopilot tree create <name> --context "reason"'
    )
  return tree


EXPERIMENT_SLUG_PREFIX_MIN_LEN = 8


def resolve_active_tree_experiment_node(tree: Tree, token: str) -> Node | None:
  """Resolve an experiment by exact id or unique prefix on the active tree.

  Exact match is tried first. If that fails and the token is at least
  ``EXPERIMENT_SLUG_PREFIX_MIN_LEN`` characters long, a prefix search is
  attempted across all nodes in the tree.

  Args:
    tree: Active tree to search.
    token: Full experiment id or unique prefix (>= 8 chars for prefix).

  Returns:
    The matching ``Node``, or ``None`` when no match is found.

  Raises:
    CLIError: When the prefix matches multiple experiments (ambiguous).
  """
  exact = tree.get(token)
  if exact is not None:
    return exact

  if len(token) < EXPERIMENT_SLUG_PREFIX_MIN_LEN:
    return None

  matches = [n for n in tree.query().all() if n.experiment.id.startswith(token)]
  if len(matches) == 1:
    return matches[0]
  if len(matches) > 1:
    candidates = sorted(n.experiment.id for n in matches)
    msg = (
      f'ambiguous prefix {token!r} matches {len(matches)} experiments: '
      f'{", ".join(candidates)}; use a longer prefix or the full id'
    )
    raise CLIError(msg)
  return None


def require_experiment_node(
  ctx: CLIContext,
  tree: Tree,
  experiment_id: str,
) -> Node:
  """Return the node for experiment_id or terminate via ctx.fail.

  Tries exact match first, then prefix resolution via
  ``resolve_active_tree_experiment_node``.

  Args:
    ctx: Context used for failure output.
    tree: Tree to search.
    experiment_id: Experiment slug or unique prefix to resolve.

  Returns:
    Matching ``Node`` from the tree.
  """
  try:
    node = resolve_active_tree_experiment_node(tree, experiment_id)
  except CLIError as exc:
    ctx.fail(str(exc))
  if node is None:
    ctx.fail(
      f'Experiment {experiment_id!r} not found in tree {tree.name!r}. '
      'Run: autopilot query --json to list available experiments.'
    )
  return node


def journal_user_context(
  ctx: CLIContext,
  experiment: Experiment,
  args: argparse.Namespace,
) -> None:
  """Record user-provided ``--context`` on the experiment's context log.

  Called by handlers that resolve a live experiment for mutation. At most
  one call per CLI invocation (DRY-07). No-op when ``ctx.context`` is None
  (read-only exempt paths often omit ``--context``). DRY-04: dispatch owns
  ``ExecutionRecord.context``; handlers own ``experiment.add_context``.

  Args:
    ctx: CLI context carrying the ``--context`` value.
    experiment: Resolved experiment to journal on.
    args: Parsed namespace used to derive the command string.
  """
  if ctx.context is not None:
    command = resolve_command(args)
    experiment.add_context(ctx.context, source='user', command=command)


def store_vcs_arguments() -> list[Argument]:
  """Canonical --source/--store/--pattern descriptors for store VCS subcommands.

  Returns:
    Three ``Argument`` instances in declaration order.
  """
  return [
    Argument(
      '--source',
      required=True,
      help='source directory tracked by the store',
    ),
    Argument(
      '--store',
      default=None,
      help='store root (default: workspace .store)',
    ),
    Argument(
      '--pattern',
      default='**/*',
      help='glob for tracked files under source',
    ),
  ]


def with_store_vcs_arguments(handler):
  """Decorator that stacks --source/--store/--pattern on a @subcommand handler.

  Args:
    handler: Bound method or function already decorated with ``@subcommand``.

  Returns:
    Handler wrapped with additional ``@argument`` layers for VCS flags.
  """
  wrapped = handler
  for arg in store_vcs_arguments():
    wrapped = argument(*arg.flags, **arg.kwargs)(wrapped)
  return wrapped


def require_store_branch(ctx: CLIContext, store: Store, experiment_id: str) -> dict:
  """Validate that an experiment has a store branch, or fail with guidance.

  Checks that the experiment has a branch entry in the store refs. When
  absent, calls ``ctx.fail`` with the canonical forest-only guidance message
  pointing the user to ``store create``.

  Args:
    ctx: CLI context for failure output.
    store: Store whose refs to inspect.
    experiment_id: Experiment slug expected to have a branch.

  Returns:
    Branch metadata dict from refs (contains ``latest_epoch``, etc.).
  """
  refs = store.load_refs()
  branches = refs.get('branches', {})
  branch = branches.get(experiment_id)
  if branch is None:
    ctx.fail(MSG_FOREST_ONLY_STORE.format(experiment_id=experiment_id))
  return branch


def parse_metric_threshold_spec(
  ctx: CLIContext,
  spec: str,
  flag_label: str,
) -> tuple[str, float]:
  """Parse a ``NAME:NUMBER`` metric threshold specification.

  Validates that the spec contains exactly one colon separator with a
  non-empty metric name and a valid numeric threshold. Rejects NaN values.

  Shared between ``query`` and ``recommend`` for DRY validation.

  Args:
    ctx: CLI context for ``ctx.fail`` on validation errors.
    spec: Raw flag value (e.g. ``'accuracy:0.9'``).
    flag_label: Flag name for error messages (e.g. ``'--metric-gt'``).

  Returns:
    Tuple of (metric_name, threshold_value).
  """
  if ':' not in spec:
    ctx.fail(f'{flag_label} expects NAME:NUMBER, got {spec!r}')
  name, _, val_str = spec.partition(':')
  name = name.strip()
  val_str = val_str.strip()
  if not name:
    ctx.fail(f'{flag_label} expects NAME:NUMBER, got {spec!r}; metric name must not be empty')
  if not val_str:
    ctx.fail(f'{flag_label} expects NAME:NUMBER, got {spec!r}; threshold value must not be empty')
  try:
    value = float(val_str)
  except ValueError:
    ctx.fail(f'{flag_label} expects NAME:NUMBER, got {spec!r}; {val_str!r} is not a valid number')
  if math.isnan(value):
    ctx.fail(f'{flag_label} rejects NaN threshold in {spec!r}; NaN is not a useful threshold value')
  return name, value


class CLIError(AutoPilotError):
  """User-input or CLI state error that should map to ``ctx.fail``.

  Handlers catch ``CLIError`` and call ``ctx.fail(str(exc))`` so the
  user-visible message stays consistent with ``messages`` constants.
  """


def resolve_epoch(epoch_arg: str | int, store: Store, experiment_id: str) -> int:
  """Resolve an epoch argument to a concrete epoch number.

  Normalizes the CLI ``--epoch`` value: accepts an ``int``, numeric string
  (e.g. ``'3'``), or the literal ``'latest'`` (case-insensitive). When
  ``'latest'`` is given, the tip epoch is resolved via
  ``store.log(experiment_id)`` — the last entry's ``epoch`` attribute.

  Args:
    epoch_arg: Integer epoch, the string ``'latest'`` (case-normalized), or
      a numeric string (e.g. ``'3'``) parsed to ``int``.
    store: Store instance to query for snapshot history.
    experiment_id: Experiment slug scoping the epoch lookup.

  Returns:
    Resolved non-negative epoch number.

  Raises:
    CLIError: If *epoch_arg* is not ``'latest'``, int-convertible, or a
      valid epoch for the given experiment (negative, out of range, or
      store is empty when ``'latest'`` is requested).
  """
  if isinstance(epoch_arg, int):
    epoch = epoch_arg
  elif isinstance(epoch_arg, str):
    normalized = epoch_arg.strip()
    if normalized.lower() == 'latest':
      entries = _safe_store_log(store, experiment_id)
      if not entries:
        raise CLIError(MSG_EPOCH_EMPTY_STORE.format(experiment_id=experiment_id))
      return entries[-1].epoch
    try:
      epoch = int(normalized)
    except ValueError:
      raise CLIError(MSG_EPOCH_INVALID.format(value=epoch_arg)) from None
  else:
    raise CLIError(MSG_EPOCH_INVALID.format(value=epoch_arg))

  if epoch < 0:
    raise CLIError(MSG_EPOCH_INVALID.format(value=epoch_arg))

  entries = _safe_store_log(store, experiment_id)
  if not entries:
    raise CLIError(MSG_EPOCH_EMPTY_STORE.format(experiment_id=experiment_id))
  latest = entries[-1].epoch
  if epoch > latest:
    raise CLIError(
      MSG_EPOCH_NOT_FOUND.format(
        epoch=epoch,
        experiment_id=experiment_id,
        latest=latest,
      )
    )
  return epoch


def _safe_store_log(store: Store, experiment_id: str) -> list:
  """Call store.log, converting StoreError to CLIError.

  Returns:
    Snapshot entries from the store log.

  Raises:
    CLIError: If the store raises StoreError (e.g. experiment not found).
  """
  try:
    return store.log(experiment_id)
  except StoreError as exc:
    raise CLIError(str(exc)) from exc


def resolve_command_epoch(ctx: CLIContext, args: argparse.Namespace) -> int:
  """Return the epoch from args or ctx, using Store-backed resolution when available.

  Consolidates the identical ``_resolve_diagnose_epoch`` and
  ``_resolve_trace_epoch`` helpers. When an experiment is configured,
  delegates to ``resolve_epoch`` which supports ``'latest'`` and validates
  against the Store. Otherwise falls back to the raw epoch value from the
  CLI context.

  Args:
    ctx: CLI context with epoch and experiment.
    args: Parsed arguments containing the ``--epoch`` flag.

  Returns:
    Resolved epoch number.
  """
  epoch_arg = args.epoch if args.epoch is not None else ctx.epoch
  if epoch_arg is None:
    ctx.fail(MSG_EPOCH_REQUIRED)

  if isinstance(ctx.experiment, str):
    try:
      forest = load_forest(ctx)
      return resolve_epoch(epoch_arg, forest.store, ctx.experiment)
    except (CLIError, StoreError, OSError, TypeError, ValueError):
      pass

  return epoch_arg
