"""Workspace management: initialize layout, health checks, directory tree, status, and journal.

The doctor command checks workspace structure only. Auth and provider
checks are project-specific and belong in preflight or project plugins.

``workspace init`` persists ``--context`` to ``.autopilot/workspace.json`` as
``description``. ``workspace status`` exposes ``description`` and exits ``1``
when composite health is unhealthy. JSON payload includes ``deployments``
(label/experiment/tree inventory across all trees, sorted by ``(tree, label)``)
and ``trees.detail`` (per-tree name, experiment_count, active flag, description)
alongside existing ``trees.count`` and ``trees.active`` keys.
``workspace doctor`` validates ``forest.json`` parse integrity.

``workspace journal`` aggregates context log entries across all trees and
experiments, annotates each entry with provenance (``experiment_id``,
``tree``), sorts chronologically using ``parse_timestamp()``, and supports
filters (``--source``, ``--limit``, ``--since``) plus ``--json`` envelope
parity.

Import terminal modules directly (e.g. ``from autopilot.core.module.module import Module``),
not package facade -- there is no ``__init__.py``.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.primitives import argument, subcommand
from autopilot.core.enums import Status
from autopilot.core.errors import StoreError, TrackingError
from autopilot.tracking.executions import load_executions
from autopilot.tracking.io import atomic_write_json, parse_timestamp, read_json_dict
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from typing import Any
import argparse

WORKSPACE_STATUS_RECENT_EXECUTIONS = 5
WORKSPACE_JSON_FILENAME = 'workspace.json'
JOURNAL_REASON_PREVIEW_LEN = 60


def _workspace_json_path(ctx: CLIContext) -> Path:
  """Resolve the path to ``.autopilot/workspace.json``.

  Args:
    ctx: CLI context with config layout.

  Returns:
    Path to the workspace metadata file.
  """
  return ctx.autopilot_dir / WORKSPACE_JSON_FILENAME


def _read_workspace_description(ctx: CLIContext) -> str | None:
  """Read the workspace description from ``.autopilot/workspace.json``.

  Returns ``None`` when the file is missing, corrupt, or lacks a string
  ``description`` key. Does not raise — this is a best-effort read for
  the status payload.

  Args:
    ctx: CLI context with workspace paths.

  Returns:
    The description string, or ``None`` when unavailable.
  """
  wj = _workspace_json_path(ctx)
  if not wj.is_file():
    return None
  try:
    data = read_json_dict(wj, WORKSPACE_JSON_FILENAME)
  except TrackingError:
    return None
  desc = data.get('description')
  if isinstance(desc, str):
    return desc
  return None


class WorkspaceCommand(Command):
  """``autopilot workspace`` group: init, doctor health checks, directory tree, and status.

  ``workspace init`` persists the ``--context`` reason as workspace
  ``description`` in ``.autopilot/workspace.json``.
  """

  name = 'workspace'
  help = 'Workspace management'

  @subcommand('init', help_text='Initialize workspace')
  def init(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create the standard autopilot directory layout.

    When ``--context`` is present (required for mutating commands under
    normal CLI dispatch), persists it as the workspace ``description``
    in ``.autopilot/workspace.json``.
    """
    ctx.output.info('Initializing workspace...')
    ctx.config.init_workspace()
    if ctx.context is not None:
      atomic_write_json(
        _workspace_json_path(ctx),
        {'description': ctx.context},
      )
    ctx.output.result({'workspace': str(ctx.workspace), 'status': 'initialized'})

  @argument(
    '--repair',
    action='store_true',
    default=False,
    help='repair missing workspace directories (mutating; requires --context)',
  )
  @subcommand('doctor', help_text='Check workspace health (--repair to fix)')
  def doctor(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Verify required directories exist and validate forest.json integrity.

    Without ``--repair``, runs read-only diagnostics (context-exempt).
    With ``--repair``, recreates missing workspace directories
    (requires ``--context``). ``--repair --dry-run`` previews repairs.
    """
    if args.repair and not ctx.dry_run and ctx.context is None:
      ctx.fail('--repair requires --context (mutating operation)')

    ctx.output.info('Checking workspace health...')

    checks, issues, forest_error = _run_workspace_checks(ctx)

    repaired: list[str] = []
    do_repair = args.repair
    if do_repair and issues:
      repaired = _repair_workspace_issues(ctx, issues, dry_run=ctx.dry_run)

    all_ok = not issues or (do_repair and len(repaired) == len(issues))
    if issues and not do_repair:
      for issue in issues:
        ctx.output.warn(f'missing: {issue}')

    result: dict[str, Any] = {
      'workspace': str(ctx.workspace),
      'healthy': all_ok,
      'checks': checks,
      'issues': issues,
    }
    if forest_error is not None:
      result['forest_error'] = forest_error
    if do_repair:
      result['repaired'] = repaired
      result['dry_run'] = ctx.dry_run

    ctx.output.result(result, ok=all_ok)

  @subcommand('tree', help_text='Show autopilot directory tree')
  def tree(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Display the autopilot directory tree structure."""
    ctx.output.info('Workspace tree (autopilot):')
    base = ctx.autopilot_dir
    if not base.exists():
      ctx.output.warn('autopilot directory does not exist; run workspace init')
      ctx.output.result({'tree': [], 'root': str(base)})
      return
    tree = [str(base), *tree_lines(base)]
    for line in tree:
      ctx.output.info(line)
    ctx.output.result({'root': str(base), 'lines': len(tree)})

  @subcommand('status', help_text='Show workspace status summary')
  def status(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Display a read-only workspace summary.

    JSON payload includes trees (count, active, per-tree detail with
    experiment_count / active flag / description), deployments inventory
    (label / experiment_id / tree, sorted by tree then label), experiments
    by status, store health, and recent executions.

    Exits non-zero when the composite health is unhealthy so agents and
    scripts can rely on the exit code.

    Raises:
      SystemExit: With code 1 when composite health is unhealthy.
    """
    payload = _build_status_payload(ctx)
    overall_ok = _workspace_status_overall_ok(payload)
    _print_status_text(ctx, payload)
    if not overall_ok and not ctx.output.use_json:
      ctx.output.warn('workspace unhealthy; run autopilot workspace init')
    ctx.output.result(payload, ok=overall_ok)
    if not overall_ok:
      raise SystemExit(1)

  @argument('--source', default=None, help='filter entries by source (exact match)')
  @argument('--limit', type=int, default=None, help='retain only the N most recent entries')
  @argument('--since', default=None, help='include entries at or after this ISO 8601 timestamp')
  @subcommand('journal', help_text='Aggregate context logs across all experiments')
  def journal(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Aggregate context log entries across all trees and experiments.

    Read-only command (no ``--context`` required). Iterates every tree and
    experiment node in the forest, collects context log entries, annotates
    each with ``experiment_id`` and ``tree`` provenance, sorts
    chronologically using ``parse_timestamp()``, and applies optional
    filters.

    Filters (applied in order):
      ``--source TEXT``: include entries where ``entry.source == TEXT``.
      ``--since ISO``: include entries at or after the given ISO 8601
        timestamp (datetime comparison via ``parse_timestamp``).
      ``--limit N``: retain only the N most recent entries after other
        filters. Result is re-sorted ascending for final emission.

    JSON output::

      {'ok': true, 'result': {'entries': [...]}, 'messages': []}

    Text mode prints concise lines: timestamp, tree, experiment, source,
    reason preview. Empty workspace yields ``entries: []``, exit 0.
    Malformed ``--since`` values are rejected via ``ctx.fail``.
    """
    since_dt = None
    if args.since is not None:
      try:
        since_dt = parse_timestamp(args.since)
      except ValueError:
        ctx.fail(
          f'invalid --since timestamp {args.since!r}; '
          'provide a valid ISO 8601 string (e.g. 2026-01-15T00:00:00Z)'
        )

    try:
      forest = load_forest(ctx)
    except (ValueError, OSError, StoreError):
      ctx.output.result({'entries': []})
      return

    entries = _collect_journal_entries(forest, source=args.source, since_dt=since_dt)

    if args.limit is not None and args.limit < len(entries):
      entries.sort(
        key=lambda e: (
          parse_timestamp(e['timestamp']),
          e['tree'],
          e['experiment_id'],
        ),
        reverse=True,
      )
      entries = entries[: args.limit]
      entries.sort(
        key=lambda e: (
          parse_timestamp(e['timestamp']),
          e['tree'],
          e['experiment_id'],
        ),
      )

    if not ctx.output.use_json:
      _print_journal_text(ctx, entries)

    ctx.output.result({'entries': entries})


WORKSPACE_DIR_CHECKS = {
  'experiments_dir': lambda ctx: ctx.experiments_dir,
  'records_dir': lambda ctx: ctx.records_dir,
  'datasets_dir': lambda ctx: ctx.datasets_dir,
  'projects_dir': lambda ctx: ctx.config.projects_path,
  'autopilot_exists': lambda ctx: ctx.autopilot_dir,
}


def _repair_workspace_issues(
  ctx: CLIContext,
  issues: list[str],
  *,
  dry_run: bool = False,
) -> list[str]:
  """Repair missing workspace directories.

  Creates directories for each issue that corresponds to a known workspace
  layout check. Returns the list of issues that were repaired.

  Args:
    ctx: CLI context with workspace paths.
    issues: List of check names that failed (from ``_run_workspace_checks``).
    dry_run: When True, do not create directories.

  Returns:
    List of issue names that were (or would be) repaired.
  """
  repaired: list[str] = []
  for issue in issues:
    resolver = WORKSPACE_DIR_CHECKS.get(issue)
    if resolver is not None:
      target = resolver(ctx)
      if not dry_run:
        target.mkdir(parents=True, exist_ok=True)
      repaired.append(issue)
  return repaired


def tree_lines(root: Path, prefix: str | None = None, max_depth: int = 6) -> list[str]:
  """Build indented tree-view lines for a directory hierarchy.

  Args:
    root: Root directory to walk.
    prefix: Indentation prefix for child entries.
    max_depth: Maximum recursion depth (0 stops immediately).

  Returns:
    List of formatted tree lines.
  """
  lines: list[str] = []
  if max_depth <= 0 or not root.exists():
    return lines
  pfx = '' if prefix is None else prefix
  entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
  for i, path in enumerate(entries):
    connector = '\\-- ' if i == len(entries) - 1 else '+-- '
    lines.append(f'{pfx}{connector}{path.name}')
    if path.is_dir():
      extension = '    ' if i == len(entries) - 1 else '|   '
      lines.extend(tree_lines(path, (pfx) + extension, max_depth - 1))
  return lines


def _run_workspace_checks(
  ctx: CLIContext,
) -> tuple[dict[str, bool], list[str], str | None]:
  """Run workspace layout and forest integrity checks.

  Shared by ``workspace doctor`` and ``_build_health_section`` so both
  commands agree on workspace health without duplicating logic.

  Args:
    ctx: CLI context with workspace paths and config.

  Returns:
    Tuple of (checks dict, issues list, forest_error string or None).
    ``issues`` collects names of failing checks. ``forest_error`` is set
    when ``forest.json`` exists but fails to parse as a JSON object.
  """
  checks: dict[str, bool] = {
    'workspace_exists': ctx.workspace.exists(),
    'autopilot_exists': ctx.autopilot_dir.exists(),
    'experiments_dir': ctx.experiments_dir.exists(),
    'records_dir': ctx.records_dir.exists(),
    'datasets_dir': ctx.datasets_dir.exists(),
    'projects_dir': ctx.config.projects_path.is_dir(),
  }

  forest_error: str | None = None
  forest_file = ctx.config.forest_file
  if forest_file.is_file():
    try:
      read_json_dict(forest_file, 'forest.json')
      checks['forest_json'] = True
    except TrackingError as exc:
      checks['forest_json'] = False
      forest_error = str(exc)

  issues = [k for k, v in checks.items() if not v]
  return checks, issues, forest_error


def _build_trees_section(forest: FileForest) -> dict[str, Any]:
  """Build the trees section of the workspace status payload.

  Includes ``detail`` list with per-tree metadata (name, experiment count,
  active flag, description) alongside the existing ``count`` and ``active``
  aggregate keys.

  Args:
    forest: Loaded forest to inspect.

  Returns:
    Dict with ``count``, ``active``, and ``detail`` keys.
  """
  trees = forest.list_trees()
  active = forest.active
  detail: list[dict[str, Any]] = [
    {
      'name': tree.name,
      'experiment_count': len(tree.nodes),
      'active': active is not None and tree.name == active.name,
      'description': tree.description,
    }
    for tree in trees
  ]
  return {
    'count': len(trees),
    'active': active.name if active is not None else None,
    'detail': detail,
  }


def _build_deployments_section(forest: FileForest) -> list[dict[str, str]]:
  """Build the deployments inventory from every tree in the forest.

  Scans all trees for nodes with a non-None ``deployed_as`` label and
  returns a deterministically sorted list of deployment records.

  Args:
    forest: Loaded forest whose trees contain experiment nodes.

  Returns:
    List of dicts with ``label``, ``experiment_id``, and ``tree`` keys,
    sorted by ``(tree.name, label)``.
  """
  deployments: list[dict[str, str]] = []
  for tree in forest.trees:
    for node in tree.nodes.values():
      label = node.deployed_as
      if label is None:
        continue
      deployments.append(
        {
          'label': label,
          'experiment_id': node.experiment.id,
          'tree': tree.name,
        }
      )
  deployments.sort(key=itemgetter('tree', 'label'))
  return deployments


def _build_experiments_section(forest: FileForest) -> dict[str, int]:
  """Build experiment counts grouped by Status value.

  Args:
    forest: Forest whose trees contain experiment nodes.

  Returns:
    Dict mapping each Status name to its count.
  """
  counts: dict[str, int] = {s.value: 0 for s in Status}
  for tree in forest.list_trees():
    for node in tree.query().all():
      status_value = node.experiment.status.value
      if status_value in counts:
        counts[status_value] += 1
  return counts


def _build_store_section(ctx: CLIContext, *, store_existed: bool = False) -> dict[str, Any]:
  """Build the store section with existence and path.

  Args:
    ctx: CLI context with config.
    store_existed: Whether the store directory existed before ``load_forest``
      potentially created it.

  Returns:
    Dict with ``exists`` and ``path`` keys.
  """
  return {
    'exists': store_existed,
    'path': str(ctx.config.store_path),
  }


def _build_health_section(
  ctx: CLIContext,
  *,
  store_existed: bool = False,
) -> dict[str, Any]:
  """Build health section with workspace doctor checks and optional store doctor.

  Delegates workspace layout and forest integrity checks to
  :func:`_run_workspace_checks` (shared with ``workspace doctor``) so both
  commands agree on ``workspace_doctor.healthy``. Then attempts to run
  ``FileStore.doctor()`` when the store directory pre-existed.

  Args:
    ctx: CLI context with config and workspace paths.
    store_existed: Whether the store directory existed before ``load_forest``
      potentially created it.

  Returns:
    Dict with ``workspace_doctor`` and optionally ``store_doctor`` keys.
  """
  workspace_checks, issues, forest_error = _run_workspace_checks(ctx)
  ws_doctor: dict[str, Any] = {
    'healthy': not issues,
    'checks': workspace_checks,
  }
  if forest_error is not None:
    ws_doctor['forest_error'] = forest_error

  health: dict[str, Any] = {'workspace_doctor': ws_doctor}

  if store_existed:
    try:
      store = FileStore(ctx.config)
      health['store_doctor'] = store.doctor_report()
    except StoreError as exc:
      health['store_doctor'] = {
        'healthy': False,
        'error': str(exc),
        'manifest_errors': [],
        'missing_blobs': [],
        'orphan_blobs': [],
        'orphan_count': 0,
        'refs_issues': [],
      }

  return health


def _build_executions_section(ctx: CLIContext) -> dict[str, Any]:
  """Build executions summary from the JSONL execution log.

  Reads all records, reports the total count, and includes a capped
  slice of the most recent entries.

  Args:
    ctx: CLI context with config for executions path.

  Returns:
    Dict with ``total`` and ``recent`` keys.
  """
  exec_path = ctx.config.executions_path
  if not exec_path.exists():
    return {'total': 0, 'recent': []}

  records = load_executions(exec_path)
  recent = records[-WORKSPACE_STATUS_RECENT_EXECUTIONS:]
  return {
    'total': len(records),
    'recent': [
      {
        'timestamp': rec.timestamp,
        'command': rec.command,
        'exit_code': rec.exit_code,
        'context': rec.context,
      }
      for rec in recent
    ],
  }


def _workspace_status_overall_ok(payload: dict[str, Any]) -> bool:
  """Determine top-level ``ok`` from composite health.

  False when workspace doctor unhealthy, store doctor present and unhealthy,
  or ``forest_error`` is present (forest failed to load).
  Absent ``store_doctor`` (store not pre-existing) does not fail.

  Args:
    payload: Full workspace status payload with ``health`` section.

  Returns:
    True only when all present health signals are healthy.
  """
  if payload.get('forest_error') is not None:
    return False
  health = payload.get('health', {})
  ws_doc = health.get('workspace_doctor', {})
  if not ws_doc.get('healthy', False):
    return False
  store_doc = health.get('store_doctor')
  return store_doc is None or store_doc.get('healthy', False)


def _build_status_payload(ctx: CLIContext) -> dict[str, Any]:
  """Assemble the full workspace status payload.

  Captures store_path existence before ``load_forest`` (which creates the
  directory) so that ``_build_health_section`` only runs store doctor when
  the store genuinely pre-existed.

  Args:
    ctx: CLI context with config, workspace paths, and output.

  Returns:
    Dict with ``trees``, ``experiments``, ``store``, ``executions``,
    ``health``, and ``forest_error`` sections. ``forest_error`` is ``None``
    when the forest loaded successfully, or a string error message on failure.
  """
  store_existed = ctx.config.store_path.exists()

  forest_error: str | None = None
  deployments: list[dict[str, str]] = []
  try:
    forest = load_forest(ctx)
    trees_section = _build_trees_section(forest)
    experiments_section = _build_experiments_section(forest)
    deployments = _build_deployments_section(forest)
  except (ValueError, OSError, StoreError) as exc:
    forest_error = str(exc)
    trees_section = {'count': 0, 'active': None, 'detail': []}
    experiments_section = {s.value: 0 for s in Status}

  return {
    'description': _read_workspace_description(ctx),
    'trees': trees_section,
    'deployments': deployments,
    'experiments': experiments_section,
    'store': _build_store_section(ctx, store_existed=store_existed),
    'executions': _build_executions_section(ctx),
    'health': _build_health_section(ctx, store_existed=store_existed),
    'forest_error': forest_error,
  }


def _print_status_text(ctx: CLIContext, payload: dict[str, Any]) -> None:
  """Print a human-readable workspace status summary.

  Includes workspace purpose, tree detail, deployments inventory, and health.

  Args:
    ctx: CLI context for output.
    payload: Assembled status payload.
  """
  description = payload.get('description')
  if description is not None:
    ctx.output.info(f'Purpose: {description}')

  trees = payload['trees']
  ctx.output.info(f'Trees: {trees["count"]} (active: {trees["active"] or "none"})')

  for detail in trees.get('detail', []):
    active_marker = ' *' if detail['active'] else ''
    desc = detail['description'] or '(none)'
    ctx.output.info(
      f'  {detail["name"]}{active_marker} -- {detail["experiment_count"]} experiments, {desc}'
    )

  experiments = payload['experiments']
  parts = [f'{status}: {count}' for status, count in experiments.items() if count > 0]
  ctx.output.info(f'Experiments: {", ".join(parts) if parts else "none"}')

  deployments = payload.get('deployments', [])
  if deployments:
    ctx.output.info('Deployments:')
    for dep in deployments:
      ctx.output.info(f'  {dep["label"]} -> {dep["experiment_id"]} (tree: {dep["tree"]})')

  store = payload['store']
  ctx.output.info(f'Store: {"exists" if store["exists"] else "not found"} ({store["path"]})')

  executions = payload['executions']
  ctx.output.info(f'Executions: {executions["total"]} total')
  for rec in executions['recent']:
    exit_marker = 'ok' if rec['exit_code'] == 0 else f'exit {rec["exit_code"]}'
    ctx.output.info(f'  {rec["timestamp"]} {rec["command"]} [{exit_marker}]')

  health = payload['health']
  ws_healthy = health['workspace_doctor']['healthy']
  ctx.output.info(f'Workspace health: {"healthy" if ws_healthy else "unhealthy"}')
  if 'store_doctor' in health:
    store_healthy = health['store_doctor'].get('healthy', False)
    ctx.output.info(f'Store health: {"healthy" if store_healthy else "unhealthy"}')


# ---------------------------------------------------------------------------
# workspace journal helpers
# ---------------------------------------------------------------------------


def _collect_journal_entries(
  forest: FileForest,
  *,
  source: str | None = None,
  since_dt: datetime | None = None,
) -> list[dict[str, Any]]:
  """Collect and merge context log entries across all trees and experiments.

  Each entry dict is the serialized ``ContextEntry`` augmented with
  ``experiment_id`` and ``tree`` provenance keys. Entries are sorted
  chronologically ascending with stable tie-breaking on
  ``(tree_name, experiment_id)``.

  Args:
    forest: Loaded forest to scan.
    source: If set, include only entries with this exact source value.
    since_dt: If set, include only entries at or after this datetime
      (compared via ``parse_timestamp``).

  Returns:
    Sorted list of annotated entry dicts.
  """
  merged: list[dict[str, Any]] = []
  for tree in forest.trees:
    for node in tree.nodes.values():
      exp = node.experiment
      for entry in exp.context_log.entries:
        if source is not None and entry.source != source:
          continue
        if since_dt is not None and parse_timestamp(entry.timestamp) < since_dt:
          continue
        record = {
          **entry.to_dict(),
          'experiment_id': exp.id,
          'tree': tree.name,
        }
        merged.append(record)

  merged.sort(
    key=lambda e: (
      parse_timestamp(e['timestamp']),
      e['tree'],
      e['experiment_id'],
    ),
  )
  return merged


def _print_journal_text(ctx: CLIContext, entries: list[dict[str, Any]]) -> None:
  """Print journal entries as concise text lines.

  Each line shows: timestamp, tree, experiment_id, source, and a truncated
  reason preview.

  Args:
    ctx: CLI context for output.
    entries: Sorted list of annotated entry dicts.
  """
  if not entries:
    ctx.output.info('no journal entries')
    return

  for entry in entries:
    source = entry.get('source') or '-'
    reason = entry['reason']
    if len(reason) > JOURNAL_REASON_PREVIEW_LEN:
      reason = reason[:JOURNAL_REASON_PREVIEW_LEN] + '...'
    ctx.output.info(
      f'{entry["timestamp"]}  [{entry["tree"]}] {entry["experiment_id"]}  {source}: {reason}'
    )
