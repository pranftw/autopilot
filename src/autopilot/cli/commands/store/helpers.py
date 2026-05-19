"""Shared helpers for store CLI subcommands.

Provides common patterns for opening stores, requiring experiments,
constructing configs and parameters, emitting doctor reports, and
building reflog payloads (shared between ``debug store reflog`` and
``store reflog list``).
These helpers are used by ``command.py``, ``merge.py``, and
``peripherals.py`` to avoid circular imports between siblings.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.messages import MSG_EXPERIMENT_SLUG_REQUIRED
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.parameter import Parameter
from autopilot.core.store.base import Store
from pathlib import Path
from typing import Any
import argparse

ISO_DATETIME_DISPLAY_LEN = 19


def emit_doctor_text(
  ctx: CLIContext,
  report: dict[str, Any],
  *,
  repair: bool,
  repaired: list[dict],
) -> None:
  """Emit human-readable doctor report to the CLI output.

  Args:
    ctx: CLI context for output methods.
    report: Legacy doctor report dict from ``diagnostics_to_report``.
    repair: Whether ``--repair`` was requested.
    repaired: List of repaired entry dicts.
  """
  if report['healthy']:
    ctx.output.info('store is healthy')
  else:
    ctx.output.info('store has issues')
  if report['manifest_errors']:
    ctx.output.info(f'manifest errors: {len(report["manifest_errors"])}')
    for err in report['manifest_errors']:
      ctx.output.info(f'  {err}')
  if report['missing_blobs']:
    ctx.output.info(f'missing blobs: {len(report["missing_blobs"])}')
    for digest in report['missing_blobs']:
      ctx.output.info(f'  {digest}')
  if report['refs_issues']:
    ctx.output.info(f'refs issues: {len(report["refs_issues"])}')
    for issue in report['refs_issues']:
      ctx.output.info(f'  {issue}')
  ctx.output.info(f'orphan blobs: {report["orphan_count"]}')
  if repair and repaired:
    action_label = 'would repair' if ctx.dry_run else 'repaired'
    ctx.output.info(f'{action_label}: {len(repaired)} issue(s)')


def require_experiment(ctx: CLIContext) -> str:
  """Extract and validate the experiment slug from the CLI context.

  Args:
    ctx: CLI context with optional experiment field.

  Returns:
    Experiment slug string.

  Raises:
    ValueError: When no experiment is set on the context.
  """
  slug = ctx.experiment
  if not slug:
    raise ValueError(MSG_EXPERIMENT_SLUG_REQUIRED)
  return slug


def make_config(ctx: CLIContext, args: argparse.Namespace) -> AutoPilotConfig:
  """Build an AutoPilotConfig, optionally overriding the store path.

  Args:
    ctx: CLI context for workspace/project.
    args: Parsed arguments with optional ``--store`` override.

  Returns:
    AutoPilotConfig instance.
  """
  config = AutoPilotConfig(workspace=ctx.workspace, project=ctx.project)
  if args.store:
    config.store_path = Path(args.store).resolve()
  return config


def make_parameter(args: argparse.Namespace) -> PathParameter:
  """Construct a PathParameter from CLI arguments.

  Args:
    args: Parsed arguments with ``--source`` and ``--pattern``.

  Returns:
    PathParameter instance.

  Raises:
    ValueError: When ``--source`` is missing.
  """
  if not args.source:
    msg = 'source directory required (--source)'
    raise ValueError(msg)
  pattern = args.pattern
  resolved = Path(args.source).expanduser().resolve()
  return PathParameter(source=str(resolved), pattern=pattern)


def open_file_store(ctx: CLIContext, args: argparse.Namespace) -> FileStore:
  """Open a FileStore from explicit --source and --store arguments.

  Args:
    ctx: CLI context for workspace/project.
    args: Parsed arguments with ``--source``, ``--store``, ``--pattern``.

  Returns:
    Configured FileStore with a single ``source`` parameter registered.
  """
  config = make_config(ctx, args)
  param = make_parameter(args)
  store = FileStore(config)
  store.register_parameters({'source': param})
  return store


def open_forest_store(
  ctx: CLIContext,
  args: argparse.Namespace | None = None,
  *,
  register_for_experiment: str | None = None,
) -> Store:
  """Open the Store from the forest bootstrap path (no --source required).

  When ``args.store`` is provided, overrides the config store_path before
  opening the forest-backed store.

  When ``register_for_experiment`` is provided, rehydrates parameter
  registration from the latest snapshot manifest's ``ParameterSchema`` for
  that experiment. This is required for stash/stash-pop (which call
  ``_build_snapshot`` / ``_group_by_param`` and need live ``Parameter``
  objects).  Callers that do not need parameter access (merge, tag, etc.)
  should leave this as ``None``.

  Args:
    ctx: CLI context for workspace/project resolution.
    args: Parsed arguments; ``args.store`` overrides store path when set.
    register_for_experiment: When set, register parameters from the latest
      manifest schema for this experiment branch.

  Returns:
    Store backed by the workspace forest (concretely a FileStore).
  """
  if args is not None and args.store is not None:
    ctx.config.store_path = Path(args.store).resolve()
  forest = load_forest(ctx)
  store = forest.store
  if register_for_experiment is not None:
    register_parameters_from_latest_manifest(store, register_for_experiment)
  return store


def register_parameters_from_latest_manifest(store: Store, experiment_id: str) -> None:
  """Rehydrate parameter registration from the latest manifest schema.

  Loads the tip snapshot for ``experiment_id``, extracts the embedded
  ``ParameterSchema``, and constructs live ``PathParameter`` instances that
  are registered on the store.  This enables stash/stash-pop to operate
  on a forest-backed store that was opened without ``--source``.

  Only ``PathParameter`` is supported today; unknown ``type_name`` values
  raise ``StoreError`` with an actionable message.

  Args:
    store: Store to register parameters on (concretely a FileStore).
    experiment_id: Branch whose latest manifest provides the schema.

  Raises:
    StoreError: If no snapshots exist, the manifest lacks a schema, or an
      unsupported parameter type is encountered.
  """
  entries = store.log(experiment_id)
  if not entries:
    msg = (
      f'no snapshots exist for experiment {experiment_id!r}; '
      f'run "store snapshot" or "store create" first'
    )
    raise StoreError(msg)
  tip = entries[-1]
  manifest = store.load_snapshot(experiment_id, tip.epoch)
  if manifest.schema is None:
    msg = (
      f'snapshot at epoch {tip.epoch} for experiment {experiment_id!r} '
      f'has no parameter schema; cannot infer parameters for stash'
    )
    raise StoreError(msg)
  params: dict[str, Parameter] = {}
  for schema_entry in manifest.schema.parameters:
    if schema_entry.type_name == 'PathParameter':
      source = schema_entry.source
      if source is None:
        msg = (
          f'parameter {schema_entry.name!r} has type PathParameter but no source path '
          f'in the schema; cannot construct parameter'
        )
        raise StoreError(msg)
      resolved = Path(source).expanduser().resolve()
      pattern = schema_entry.pattern if schema_entry.pattern is not None else '**/*'
      params[schema_entry.name] = PathParameter(source=str(resolved), pattern=pattern)
    else:
      msg = (
        f'parameter {schema_entry.name!r} has unsupported type '
        f'{schema_entry.type_name!r} for schema-based registration; '
        f'only PathParameter is supported'
      )
      raise StoreError(msg)
  store.register_parameters(params)


def open_store_for_log(
  ctx: CLIContext,
  args: argparse.Namespace,
) -> Store:
  """Open a store for the ``store log`` command, inferring --source when absent.

  When ``args.source`` is provided, delegates to :func:`open_file_store`.
  Otherwise, opens the forest-backed store and uses the active experiment
  to look up registered parameters.

  Args:
    ctx: CLI context for workspace/project resolution.
    args: Parsed arguments with optional ``--source``, ``--store``, ``--pattern``.

  Returns:
    Store ready for ``store.log(experiment_id)``.
  """
  if args.source is not None:
    return open_file_store(ctx, args)
  forest = load_forest(ctx)
  tree = forest.active
  if tree is None or tree.head is None:
    ctx.fail('No --source provided and no active experiment in forest. Pass --source explicitly.')
  store = forest.store
  return store


def emit_reflog(
  ctx: CLIContext,
  entries: list[dict[str, Any]],
  *,
  limit: int | None = None,
) -> None:
  """Build and emit reflog payload to CLI output (shared by debug and store reflog list).

  Args:
    ctx: CLI context for output methods.
    entries: Raw reflog dicts from ``read_jsonl`` or equivalent.
    limit: When set, slice to the last N entries before output.
  """
  if limit is not None:
    entries = entries[-limit:]

  if not ctx.output.use_json:
    if not entries:
      ctx.output.info('reflog is empty')
    else:
      rows = [
        {
          'timestamp': (e.get('timestamp') or '')[:ISO_DATETIME_DISPLAY_LEN],
          'operation': e.get('operation') or '',
          'experiment_id': e.get('experiment_id') or '',
          'old_epoch': e.get('old_epoch'),
          'new_epoch': e.get('new_epoch'),
          'context': e.get('context'),
        }
        for e in entries
      ]
      ctx.output.table(
        rows,
        ['timestamp', 'operation', 'experiment_id', 'old_epoch', 'new_epoch', 'context'],
      )

  ctx.output.result({'entries': entries, 'count': len(entries)})
