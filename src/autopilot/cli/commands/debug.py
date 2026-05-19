"""Debug data collection, analysis, and execution inspection.

The collect subcommand runs the configured Module's forward pass in debug
mode and reports the result. Requires a configured module (via project CLI).

The executions child command inspects dispatch-written execution history
(``executions.jsonl``) via ``list``, ``show``, and ``tail`` subcommands.
JSON rows in ``list`` include all ``ExecutionRecord`` fields: ``timestamp``,
``command``, ``args`` (list), ``duration_ms``, ``exit_code``, ``stdout``,
``stderr``, ``experiment``, ``project``, ``extra`` (dict), and ``context``.

Additional introspection subcommands:
  - ``cost``: reads ``cost_summary.json`` from experiment directory.
  - ``parameters``: lists module parameters (name, type, requires_grad).
  - ``module-gradients``: summarizes current gradients on module parameters.
  - ``gradients``: extracts gradient summaries from the experiment context log.
  - ``params``: shows parameter state at a given store epoch.
  - ``optimizer-decisions``: surfaces ``.optimization/`` artifacts.
"""

from autopilot.ai.store.file_store import FileStore
from autopilot.cli.command import CLI, Command
from autopilot.cli.commands.ai import AICommand
from autopilot.cli.commands.checkout import CheckoutCommand
from autopilot.cli.commands.dataset import DatasetCommand
from autopilot.cli.commands.debug_commands import CommandsCatalog
from autopilot.cli.commands.diagnose import DiagnoseCommand
from autopilot.cli.commands.execute import ExecuteCommand
from autopilot.cli.commands.experiment.command import ExperimentCommand
from autopilot.cli.commands.optimize import OptimizeCommand
from autopilot.cli.commands.policy import PolicyCommand
from autopilot.cli.commands.project import ProjectCommand
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.cli.commands.query import QueryCommand
from autopilot.cli.commands.recommend import RecommendCommand
from autopilot.cli.commands.report.command import ReportCommand
from autopilot.cli.commands.stabilize import StabilizeCommand
from autopilot.cli.commands.status import StatusCommand
from autopilot.cli.commands.store.command import StoreCommand
from autopilot.cli.commands.store.helpers import emit_reflog
from autopilot.cli.commands.trace import TraceCommand
from autopilot.cli.commands.track import TrackCommand
from autopilot.cli.commands.tree import TreeCommand
from autopilot.cli.commands.undo import UndoGuideCommand
from autopilot.cli.commands.workspace import WorkspaceCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import CLIError, load_forest, require_active_tree, resolve_epoch
from autopilot.cli.messages import MSG_EXPERIMENT_SLUG_REQUIRED, MSG_NO_MODULE_CONFIGURED
from autopilot.cli.primitives import argument, subcommand
from autopilot.core.errors import StoreError
from autopilot.core.gradient import Gradient, NumericGradient
from autopilot.core.trend import TrendAnalyzer
from autopilot.tracking.executions import filter_executions, load_executions
from autopilot.tracking.io import read_json_dict, read_jsonl
from typing import Any
import argparse


def _build_catalog_cli() -> CLI:
  """Construct a CLI instance for command catalog introspection.

  Mirrors ``AutoPilotCLI`` registration without importing ``main.py``
  (which would create a circular import since ``main.py`` imports this module).
  Uses ``DebugCommand`` from this module directly.

  Returns:
    Fully populated CLI instance matching the live command surface.
  """
  cli = CLI()
  cli.ai = AICommand()
  cli.workspace = WorkspaceCommand()
  cli.project = ProjectCommand()
  cli.dataset = DatasetCommand()
  cli.experiment = ExperimentCommand()
  cli.optimize = OptimizeCommand()
  cli.debug = DebugCommand()
  cli.policy = PolicyCommand()
  cli.report = ReportCommand()
  cli.store = StoreCommand()
  cli.status = StatusCommand()
  cli.diagnose = DiagnoseCommand()
  cli.trace = TraceCommand()
  cli.propose = ProposeCommand()
  cli.tree = TreeCommand()
  cli.query = QueryCommand()
  cli.recommend = RecommendCommand()
  cli.checkout = CheckoutCommand()
  cli.stabilize = StabilizeCommand()
  cli.execute = ExecuteCommand()
  cli.track = TrackCommand()
  cli.undo_guide = UndoGuideCommand()
  return cli


EXEC_STDOUT_PREVIEW_LEN = 80
EXEC_ARGS_PREVIEW_LEN = 60
GRADIENT_PREVIEW_LEN = 256
ISO_DATETIME_DISPLAY_LEN = 19
EXEC_LIST_DEFAULT_LIMIT = 50
EXEC_TAIL_DEFAULT_LIMIT = 10
CONTEXT_PREVIEW_LEN = 40
PARAM_CONTENT_PREVIEW_LEN = 200


def _args_preview_text(args: list[str]) -> str:
  """Join args and truncate for human-readable table display.

  Args:
    args: Argument list from an execution record.

  Returns:
    Space-joined string capped at ``EXEC_ARGS_PREVIEW_LEN`` characters.
  """
  joined = ' '.join(args)
  if len(joined) <= EXEC_ARGS_PREVIEW_LEN:
    return joined
  return joined[:EXEC_ARGS_PREVIEW_LEN] + '...'


def _gradient_summary_preview(entry: dict[str, Any]) -> str:
  """Extract a truncated gradient summary string from a context log entry dict.

  Args:
    entry: Serialized context entry with metadata.

  Returns:
    Truncated string representation of gradient summaries.
  """
  metadata = entry.get('metadata') or {}
  summaries = metadata.get('gradient_summaries')
  if summaries is None:
    return ''
  return str(summaries)[:GRADIENT_PREVIEW_LEN]


def _context_preview(context: str | None) -> str:
  """Truncate context for summary display.

  Args:
    context: Raw context string, or None.

  Returns:
    Shortened preview capped at ``CONTEXT_PREVIEW_LEN`` characters.
  """
  if context is None:
    return ''
  if len(context) <= CONTEXT_PREVIEW_LEN:
    return context
  return context[:CONTEXT_PREVIEW_LEN] + '...'


class ExecutionsCommand(Command):
  """Inspect execution history. Subcommands: list, show, tail.

  Reads dispatch-level execution records from ``ctx.config.executions_path``
  (JSONL written by the dispatch tracking hook). All subcommands support
  ``--json`` via the shared ``Output`` class.
  """

  name = 'executions'
  help = 'Inspect execution history'

  @argument('-n', '--limit', type=int, default=EXEC_LIST_DEFAULT_LIMIT, help='max records')
  @argument('--command', dest='filter_command', default=None, help='filter by command name')
  @argument('--failures', action='store_true', default=False, help='only failures')
  @argument('--summary', action='store_true', default=False, help='compact one-line view')
  @argument(
    '--context-contains',
    dest='context_contains',
    default=None,
    help='filter records whose context contains this substring',
  )
  @subcommand('list', help_text='list recent executions')
  def list_executions(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List recent executions in tabular form.

    Supports ``--command``, ``--failures``, ``--context-contains`` filters
    and ``-n/--limit`` (default 50). ``--summary`` renders compact single-
    line rows in text mode. The ``idx`` column is the global positional
    index so ``show <idx>`` works directly from filtered output.

    JSON result schema: ``{executions: [...], count: int}``.
    """
    all_records = load_executions(ctx.config.executions_path)
    indexed = list(enumerate(all_records))
    if args.filter_command:
      indexed = [(i, r) for i, r in indexed if r.command == args.filter_command]
    if args.failures:
      indexed = [(i, r) for i, r in indexed if r.exit_code != 0]
    if args.context_contains is not None:
      needle = args.context_contains
      filtered = filter_executions(
        [r for _, r in indexed],
        predicate=lambda r: needle in (r.context or ''),
      )
      filtered_ids = {id(r) for r in filtered}
      indexed = [(i, r) for i, r in indexed if id(r) in filtered_ids]
    indexed = indexed[-args.limit :]
    rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for i, r in indexed:
      row: dict[str, Any] = {
        'idx': i,
        'timestamp': r.timestamp[:ISO_DATETIME_DISPLAY_LEN],
        'command': r.command,
        'exit_code': r.exit_code,
        'duration_ms': round(r.duration_ms, 1),
        'context': r.context,
        'experiment': r.experiment,
        'project': r.project,
        'args': list(r.args),
        'extra': dict(r.extra),
      }
      rows.append(row)
      table_rows.append({**row, 'args': _args_preview_text(list(r.args))})

    if args.summary and not ctx.output.use_json:
      summary_rows = [
        {
          'idx': r['idx'],
          'timestamp': r['timestamp'],
          'command': r['command'],
          'exit_code': r['exit_code'],
          'duration_ms': r['duration_ms'],
          'experiment': r['experiment'],
          'context': _context_preview(r['context']),
        }
        for r in table_rows
      ]
      ctx.output.table(
        summary_rows,
        ['idx', 'timestamp', 'command', 'exit_code', 'duration_ms', 'experiment', 'context'],
      )
    else:
      ctx.output.table(
        table_rows,
        [
          'idx',
          'timestamp',
          'command',
          'exit_code',
          'duration_ms',
          'context',
          'experiment',
          'args',
        ],
      )
    ctx.output.result({'executions': rows, 'count': len(rows)})

  @argument('index', type=int, help='record index from list output')
  @subcommand('show', help_text='show full execution record')
  def show_execution(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show the full execution record at a given global index.

    Positional ``index`` is validated against the loaded record list.
    Emits the complete record via ``to_dict()`` under ``result.execution``.
    Calls ``ctx.fail`` with an ``out of range`` message for invalid indices
    (negative or >= record count).

    JSON result schema: ``{execution: {...}}``.
    """
    records = load_executions(ctx.config.executions_path)
    idx = args.index
    if idx < 0 or idx >= len(records):
      ctx.fail(f'index {idx} out of range (0-{len(records) - 1})')
    record = records[idx]
    ctx.output.result({'execution': record.to_dict()})

  @argument('-n', '--limit', type=int, default=EXEC_TAIL_DEFAULT_LIMIT, help='number of records')
  @subcommand('tail', help_text='show last N executions')
  def tail_executions(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show the last N execution records (default 10).

    Table columns: ``idx``, ``timestamp``, ``command``, ``exit``,
    ``stdout_preview``. The ``stdout_preview`` column is truncated to
    80 characters with newlines replaced by spaces. ``idx`` values
    are the global positional indices into the full record list.

    JSON result schema: ``{tail: [...], count: int, total: int}``.
    """
    records = load_executions(ctx.config.executions_path)
    tail = records[-args.limit :]
    rows = [
      {
        'idx': len(records) - len(tail) + i,
        'timestamp': r.timestamp[:ISO_DATETIME_DISPLAY_LEN],
        'command': r.command,
        'exit': r.exit_code,
        'stdout_preview': (
          ('' if r.stdout is None else r.stdout)[:EXEC_STDOUT_PREVIEW_LEN].replace('\n', ' ')
        ),
      }
      for i, r in enumerate(tail)
    ]
    ctx.output.table(rows, ['idx', 'timestamp', 'command', 'exit', 'stdout_preview'])
    ctx.output.result({'tail': rows, 'count': len(rows), 'total': len(records)})


class DebugStoreCommand(Command):
  """Store maintenance subcommands under ``debug store``.

  Includes ``reflog`` (read-only audit trail) and ``prune-orphans``
  (mutating orphan cleanup).
  """

  name = 'store'
  help = 'Store maintenance'

  @argument('-n', '--limit', type=int, default=None, help='max entries to show')
  @subcommand('reflog', help_text='Show append-only store reflog')
  def reflog(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Display the append-only reflog of mutating store operations.

    Reads ``reflog.jsonl`` from the store directory. Corrupt tail lines
    are skipped with a warning (``strict=False``).

    JSON result schema: ``{entries: [...], count: int}``.
    """
    config = ctx.config
    reflog_path = config.store_path / 'reflog.jsonl'
    entries = read_jsonl(reflog_path, strict=False)
    emit_reflog(ctx, entries, limit=args.limit)

  @subcommand('prune-orphans', help_text='Remove orphaned object blobs')
  def prune_orphans(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Remove object blobs not reachable from any snapshot manifest.

    Walks all snapshot manifests, collects referenced digests, then
    removes any blobs in the objects directory that are unreachable.

    JSON result schema: ``{removed: [...], count: int}``.
    """
    config = ctx.config
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    removed = store.prune_orphans()

    if not ctx.output.use_json:
      if removed:
        ctx.output.info(f'Pruned {len(removed)} orphaned blob(s)')
      else:
        ctx.output.info('No orphaned blobs found')

    ctx.output.result({'removed': removed, 'count': len(removed)})


class DebugCommand(Command):
  """Debug data collection, analysis, and execution inspection.

  Child commands: ``collect`` (debug data from Module forward pass),
  ``commands`` (machine-readable catalog of all CLI commands),
  ``executions`` (inspect dispatch-written execution history via
  ``list``, ``show``, ``tail``), ``cost`` (read cost_summary.json with
  optional ``--detail``), ``parameters`` (list module parameters),
  ``module-gradients`` (summarize current gradients on module parameters),
  ``gradients`` (extract gradient summaries from experiment context log),
  ``params`` (parameter state at a store epoch),
  ``optimizer-decisions`` (surface ``.optimization/`` artifacts),
  ``store prune-orphans`` (remove unreachable blobs).
  """

  name = 'debug'
  help = 'Debug data collection and analysis'

  def __init__(self) -> None:
    """Wire debug subcommands including ``executions`` and ``store``."""
    super().__init__()
    self.executions = ExecutionsCommand()
    self.store = DebugStoreCommand()

  @subcommand('commands', help_text='Machine-readable catalog of all CLI commands')
  def commands(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Emit structured catalog of registered commands, arguments, and policies."""
    catalog = CommandsCatalog.build(_build_catalog_cli())
    ctx.output.result(catalog.to_dict())

  @subcommand('collect', help_text='Collect debug data')
  def collect(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Run the module in debug mode and report the observation."""
    slug = ctx.experiment
    if not slug:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)
    exp_dir = ctx.experiment_path(slug)

    if not ctx.module:
      ctx.fail('no module configured for debug')

    if ctx.dry_run:
      ctx.output.result({'command': 'debug', 'dry_run': True, 'success': True})
      return

    runtime_ctx: dict[str, Any] = {
      'workspace': str(ctx.workspace),
      'dry_run': ctx.dry_run,
      'experiment_path': str(exp_dir),
    }
    params: dict[str, Any] = {'command': 'debug'}
    observation = ctx.module(runtime_ctx, params)

    ctx.output.result(
      {
        'command': 'debug',
        'success': observation.success,
        'error': observation.error_message if not observation.success else None,
      },
      ok=observation.success,
    )

  @argument('--detail', action='store_true', default=False, help='show nested breakdown if present')
  @subcommand('cost', help_text='Show cost summary for an experiment')
  def cost(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Read ``cost_summary.json`` from the experiment directory.

    Resolves the experiment path via ``ctx.config.experiment_path`` and
    looks for the ``cost_summary.json`` artifact written by
    ``CostTrackerCallback.on_loop_end``.

    When ``--detail`` is set and the summary contains nested dict values,
    those are expanded into indented sub-rows in text mode. JSON mode
    includes all data regardless of the flag.

    JSON result schema: the aggregate ``CostEntry`` dict from the file.
    """
    slug = ctx.experiment
    if not slug:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)
    exp_dir = ctx.experiment_path(slug)
    cost_file = exp_dir / 'cost_summary.json'

    if not cost_file.exists():
      ctx.fail(
        f'cost_summary.json not found for experiment {slug!r}; '
        f'run training with CostTrackerCallback attached (optimize loop does this automatically)'
      )

    data = read_json_dict(cost_file, f'cost summary for {slug}')

    if not ctx.output.use_json:
      ctx.output.info(f'Cost summary for {slug}:')
      rows: list[dict[str, Any]] = []
      for k, v in data.items():
        if args.detail and isinstance(v, dict):
          rows.append({'metric': k, 'value': '{...}'})
          for sub_k, sub_v in v.items():
            rows.append({'metric': f'  {k}.{sub_k}', 'value': sub_v})
        else:
          rows.append({'metric': k, 'value': v})
      ctx.output.table(rows, ['metric', 'value'])

    ctx.output.result(data)

  @subcommand('parameters', help_text='List module parameters')
  def parameters(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List ``module.named_parameters()`` with type and requires_grad info.

    Requires a configured module on the CLI context. Cannot mutate
    parameters -- read-only introspection.

    JSON result schema: ``{parameters: {name: {type, requires_grad}}}``.
    """
    if not ctx.module:
      ctx.fail(MSG_NO_MODULE_CONFIGURED)

    param_info: dict[str, dict[str, Any]] = {}
    for name, param in ctx.module.named_parameters():
      param_info[name] = {
        'type': type(param).__name__,
        'requires_grad': param.requires_grad,
      }

    if not ctx.output.use_json:
      rows = [
        {'name': name, 'type': info['type'], 'requires_grad': info['requires_grad']}
        for name, info in param_info.items()
      ]
      ctx.output.table(rows, ['name', 'type', 'requires_grad'])

    ctx.output.result({'parameters': param_info})

  @subcommand('module-gradients', help_text='Show current gradients on module parameters')
  def module_gradients(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Summarize current gradients on module parameters.

    For each parameter with ``.grad`` set, shows gradient type and a
    preview capped at 256 characters. Parameters cleared by
    ``optimizer.zero_grad()`` show ``null``.

    JSON result schema: ``{gradients: {name: {type, preview} | null}}``.
    """
    if not ctx.module:
      ctx.fail(MSG_NO_MODULE_CONFIGURED)

    grad_info: dict[str, dict[str, Any] | None] = {}
    for name, param in ctx.module.named_parameters():
      grad = param.grad
      if grad is None:
        grad_info[name] = None
      elif isinstance(grad, NumericGradient):
        grad_info[name] = {
          'type': 'NumericGradient',
          'value': grad.value,
          'preview': grad.render()[:GRADIENT_PREVIEW_LEN],
        }
      elif isinstance(grad, Gradient):
        try:
          rendered = grad.render()[:GRADIENT_PREVIEW_LEN]
        except NotImplementedError:
          rendered = '<no render>'
        grad_info[name] = {
          'type': type(grad).__name__,
          'preview': rendered,
        }
      else:
        grad_info[name] = {
          'type': type(grad).__name__,
          'preview': str(grad)[:GRADIENT_PREVIEW_LEN],
        }

    if not ctx.output.use_json:
      rows = []
      for name, info in grad_info.items():
        if info is None:
          rows.append({'name': name, 'type': 'null', 'preview': ''})
        else:
          rows.append({'name': name, 'type': info['type'], 'preview': info['preview']})
      ctx.output.table(rows, ['name', 'type', 'preview'])

    ctx.output.result({'gradients': grad_info})

  @subcommand('gradients', help_text='Show gradient summaries from experiment context log')
  def gradients(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Extract gradient summary entries from the experiment context log.

    Walks the experiment's ``context_log`` entries and filters for those
    where ``source`` is ``'trainer'`` or ``'agent-optimizer'`` and
    ``metadata`` contains a ``'gradient_summaries'`` key. This data is
    emitted by ``Trainer._emit_gradient_journal`` and
    ``AgentOptimizer._agentic_step``.

    Requires ``--experiment`` slug. The experiment is loaded from the
    active tree in the forest.

    JSON result schema: ``{entries: [...], count: int}``.
    """
    slug = ctx.experiment
    if not slug:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)

    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)
    node = tree.get(slug)
    if node is None:
      ctx.fail(f'experiment {slug!r} not found in active tree')

    experiment = node.experiment
    gradient_sources = frozenset({'trainer', 'agent-optimizer'})
    entries: list[dict[str, Any]] = [
      entry.to_dict()
      for entry in experiment.context_log
      if entry.source in gradient_sources and 'gradient_summaries' in entry.metadata
    ]

    if not ctx.output.use_json:
      if not entries:
        ctx.output.info('no gradient summaries found in context log')
      else:
        rows = [
          {
            'epoch': e.get('epoch'),
            'source': e.get('source'),
            'summary': _gradient_summary_preview(e),
          }
          for e in entries
        ]
        ctx.output.table(rows, ['epoch', 'source', 'summary'])

    ctx.output.result({'entries': entries, 'count': len(entries)})

  @argument('epoch', type=str, help='epoch number or "latest"')
  @subcommand('params', help_text='Show parameter state at a store epoch')
  def params(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Read resolved parameter payloads at a given epoch without mutating working tree.

    Loads the snapshot manifest for the specified experiment and epoch from
    the store, then reads blob contents from the object store. Prints file
    paths and truncated content for text mode, structured dict for JSON.

    Requires a store-backed experiment. Experiments without a store get a
    failure message with guidance to run ``store create``.

    JSON result schema: ``{experiment: str, epoch: int, files: {key: {digest, size, content?}}}``.
    """
    slug = ctx.experiment
    if not slug:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)

    config = ctx.config
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    try:
      epoch = resolve_epoch(args.epoch, store, slug)
    except CLIError as exc:
      ctx.fail(
        f'{exc}; ensure the experiment has store snapshots '
        '(run `store create` then `store snapshot` to enable parameter inspection)'
      )

    try:
      snap = store.load_snapshot(slug, epoch)
    except StoreError as exc:
      ctx.fail(f'{exc}; run `store create` to enable parameter inspection')
    objects_dir = config.objects_path

    files: dict[str, dict[str, Any]] = {}
    for key, entry in snap.entries.items():
      info: dict[str, Any] = {
        'digest': entry.digest,
        'size': entry.size,
      }
      blob_path = objects_dir / entry.digest[:2] / entry.digest[2:]
      if blob_path.exists():
        try:
          content = blob_path.read_text(encoding='utf-8')
          info['content'] = content
        except UnicodeDecodeError:
          info['content'] = '<binary>'
      files[key] = info

    if not ctx.output.use_json:
      ctx.output.info(f'Parameters at epoch {epoch} for {slug}:')
      rows = [
        {
          'key': k,
          'digest': v['digest'][:12],
          'size': v['size'],
          'preview': (v.get('content') or '')[:PARAM_CONTENT_PREVIEW_LEN],
        }
        for k, v in files.items()
      ]
      ctx.output.table(rows, ['key', 'digest', 'size', 'preview'])

    ctx.output.result({'experiment': slug, 'epoch': epoch, 'files': files})

  @subcommand('optimizer-decisions', help_text='Show optimizer decision artifacts')
  def optimizer_decisions(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Surface ``.optimization/`` artifacts for an experiment.

    Lists files under ``<experiment_dir>/.optimization/`` sorted by name.
    These are epoch feedback files and todo lists written by
    ``AgentOptimizer`` during agentic optimization steps.

    Missing directory produces an empty result with info message.

    JSON result schema: ``{files: [{name, size, preview?}], count: int}``.
    """
    slug = ctx.experiment
    if not slug:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)

    exp_dir = ctx.experiment_path(slug)
    opt_dir = exp_dir / '.optimization'

    if not opt_dir.exists() or not opt_dir.is_dir():
      if not ctx.output.use_json:
        ctx.output.info(f'no .optimization/ directory for experiment {slug!r}')
      ctx.output.result({'files': [], 'count': 0})
      return

    file_entries: list[dict[str, Any]] = []
    for f in sorted(opt_dir.iterdir()):
      if not f.is_file():
        continue
      entry: dict[str, Any] = {
        'name': f.name,
        'size': f.stat().st_size,
      }
      try:
        preview = f.read_text(encoding='utf-8')[:PARAM_CONTENT_PREVIEW_LEN]
        entry['preview'] = preview
      except (UnicodeDecodeError, OSError):
        entry['preview'] = None
      file_entries.append(entry)

    if not ctx.output.use_json:
      if not file_entries:
        ctx.output.info(f'no files in .optimization/ for experiment {slug!r}')
      else:
        rows = [
          {'name': e['name'], 'size': e['size'], 'preview': (e.get('preview') or '')[:80]}
          for e in file_entries
        ]
        ctx.output.table(rows, ['name', 'size', 'preview'])

    ctx.output.result({'files': file_entries, 'count': len(file_entries)})

  @subcommand('profiler', help_text='Show profiler timing summary')
  def profiler(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Read ``profiler_summary.json`` from the experiment directory.

    Resolves the experiment path and reads the profiler summary artifact
    written by ``Trainer`` on fit completion when a profiler is configured.

    JSON result schema: the profiler ``describe()`` output dict keyed by
    action name, each with ``count``, ``total_ms``, ``mean_ms``.
    """
    slug = ctx.experiment
    if not slug:
      ctx.fail(MSG_EXPERIMENT_SLUG_REQUIRED)

    exp_dir = ctx.experiment_path(slug)
    profiler_file = exp_dir / 'profiler_summary.json'

    if not profiler_file.exists():
      ctx.fail(
        f'profiler_summary.json not found for experiment {slug!r}; '
        f'run training with profiler=SimpleProfiler() attached to the Trainer'
      )

    data = read_json_dict(profiler_file, f'profiler summary for {slug}')

    if not ctx.output.use_json:
      ctx.output.info(f'Profiler summary for {slug}:')
      rows: list[dict[str, Any]] = []
      for action, stats in data.items():
        if isinstance(stats, dict):
          rows.append(
            {
              'action': action,
              'count': stats['count'],
              'total_ms': stats['total_ms'],
              'mean_ms': stats['mean_ms'],
            }
          )
      ctx.output.table(rows, ['action', 'count', 'total_ms', 'mean_ms'])

    ctx.output.result(data)

  @argument('metric', help='Metric name to analyze (required).')
  @argument('--all-trees', action='store_true', default=False, help='Analyze all trees.')
  @subcommand('trend', help_text='Show metric trend analysis for the active tree')
  def trend(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show metric trend analysis for the active tree (or all trees).

    Requires a metric positional argument. Analyzes completed experiments
    in the tree ordered by creation time and reports trend direction.

    JSON result schema: ``{trend: TrendResult.to_dict()}`` or
    ``{trees: {name: TrendResult.to_dict() | null}}``.
    """
    if not args.metric:
      ctx.fail(
        'Metric name is required. Run: autopilot debug trend <metric> --json '
        '(e.g. autopilot debug trend val_accuracy --json)'
      )
    forest = load_forest(ctx)
    analyzer = TrendAnalyzer()
    if args.all_trees:
      results: dict[str, Any] = {}
      for tree in forest.list_trees():
        result = analyzer.analyze(tree, args.metric)
        results[tree.name] = result.to_dict() if result.values else None
      ctx.output.result({'trees': results})
    else:
      tree = require_active_tree(ctx, forest)
      result = analyzer.analyze(tree, args.metric)
      if not result.values:
        ctx.output.info('No analyzable experiments in active tree.')
        ctx.output.result({'trend': None})
        return
      ctx.output.result({'trend': result.to_dict()})
