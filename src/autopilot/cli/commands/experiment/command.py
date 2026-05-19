"""Experiment management: add, status, compare, remove, list, deploy, and notes.

Import terminal modules directly (e.g. ``from autopilot.core.module.module import Module``),
not package facade -- there is no ``__init__.py``.

Subcommands:
  autopilot experiment add --hypothesis "..." [--parent <id>] [--no-parent]
    [--baseline <id>] [--id <id>] [--spec-version VERSION] [--json]
                                              -- create experiment + node in active tree
  autopilot experiment status [id] [--json]  -- show experiment details
  autopilot experiment show [id] [--context-log] [--epoch-from N] [--epoch-to N]
    [--reason-contains STR] [--context-summary] [--json] -- show with context journal
  autopilot experiment compare <a> <b> [--json] -- side-by-side metrics comparison
  autopilot experiment remove <id> [--cascade] [--json] -- remove experiment from tree
  autopilot experiment list [--json]         -- list experiments (alias for query)
  autopilot experiment deploy <id> --as <name> [--replace] [--json] -- deploy under label
  autopilot experiment undeploy <label> [--json] -- remove deployment label
  autopilot experiment notes show <id> [--json] -- show experiment notes
  autopilot experiment notes write <id> --body '...' [--json] -- write experiment notes inline
  autopilot experiment notes write <id> --file <path> [--json] -- write notes from file

Compare enhancements (plan 11, plan 05):
  - Cross-tree lookup: when an experiment is not found in the active tree,
    all trees in the forest are searched before failing (BUG-019 fix).
  - Metric prefix normalization: mismatched prefixes (val_*/train_*/bare) are
    aligned by base name to produce meaningful deltas (TS-prefix-mismatch).
  - Verdict: JSON output includes a ``verdict`` field (improved/regressed/
    inconclusive) summarizing the overall direction of change (GAP-018).
  - Non-numeric metric robustness (plan 05): ``deltas`` is an ordered list of
    records (not a keyed dict). Each record has ``metric``, ``baseline``,
    ``candidate``, ``delta``, and ``type`` fields. ``type`` is ``'numeric'``
    (both values are finite numbers), ``'non_numeric'`` (at least one value is
    a string, bool, list, or NaN), or ``'missing'`` (one side has no value).
    Verdict is derived from numeric entries only.

Show enhancements (FR-010):
  - Cross-tree lookup for explicit experiment IDs (HEAD stays active-tree only).
  - JSON output always includes a ``tree`` field naming the owning tree.
  - Text mode emits an info line for cross-tree hits.

Show context filtering (FR-016):
  - ``--epoch-from`` / ``--epoch-to``: inclusive epoch range filter on entries.
    Entries where ``epoch is None`` are skipped when either bound is set.
  - ``--reason-contains``: case-sensitive substring match on reason field.
  - ``--context-summary``: emit ``{source: count}`` instead of full entries
    (requires ``--context-log``; fails with guidance if omitted).
  - All filters compose with AND: source + epoch range + reason substring.

All subcommands support --json for agent-friendly structured output.
Path resolution via ctx.config (no paths.* calls).
"""

from autopilot.cli.command import Command
from autopilot.cli.commands.experiment.compare import (
  ExperimentCompare,
  ExperimentImpact,
)
from autopilot.cli.commands.experiment.lifecycle import (
  ExperimentAdd,
  ExperimentCancel,
  ExperimentComplete,
  ExperimentFail,
  ExperimentInvalidate,
  ExperimentRemove,
)
from autopilot.cli.commands.experiment.lineage import ExperimentLineage
from autopilot.cli.commands.experiment.metadata import (
  ExperimentDeploy,
  ExperimentDeployLog,
  ExperimentMetadata,
  ExperimentNotes,
  ExperimentUndeploy,
)
from autopilot.cli.commands.experiment.timeline import ExperimentTimeline
from autopilot.cli.commands.query import QueryCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import (
  load_forest,
  require_active_tree,
  require_experiment_node,
)
from autopilot.cli.primitives import Argument, Flag
from autopilot.core.context import ContextEntry
from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from typing import Any
import argparse

EXPERIMENT_ID_HEX_LEN = 12


class ExperimentStatus(Command):
  """Show experiment details."""

  name = 'status'
  help = 'Show experiment status'
  experiment_id = Argument('id', nargs='?', default=None, help='experiment ID (default: HEAD)')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Print experiment details for HEAD or a given id."""
    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)

    eid = args.id
    if eid is None:
      eid = tree.head
      if eid is None:
        ctx.fail('no experiment specified and no HEAD set')

    node = require_experiment_node(ctx, tree, eid)

    exp = node.experiment
    ctx.output.result(
      {
        'id': exp.id,
        'hypothesis': exp.hypothesis,
        'status': exp.status.value,
        'metrics': exp.metrics,
        'dependencies': list(exp.dependencies),
        'epoch': exp.epoch,
        'error': exp.error,
        'notes': exp.notes,
        'spec_version': exp.spec_version,
        'created_at': exp.created_at,
        'started_at': exp.started_at,
        'completed_at': exp.completed_at,
        'failed_at': exp.failed_at,
        'cancelled_at': exp.cancelled_at,
        'invalidated_at': exp.invalidated_at,
        'deployed_as': node.deployed_as,
      }
    )


class ExperimentShow(Command):
  """Show experiment details with optional context journal.

  Like ``experiment status`` but supports ``--context-log`` to display the
  experiment's decision journal. Use ``--context-source`` to filter entries
  by source and ``--limit`` to show only the N most recent entries.

  JSON output always includes lineage and trust fields:
    ``parent`` (str | None): parent experiment id from node linkage.
    ``baseline`` (str | None): baseline experiment id from node linkage.
    ``dataset_fingerprint`` (Any | None): dataset fingerprint dict from
    ``experiment.dataset_meta``, matching the shape emitted by ``query``.
    ``metrics_trusted`` (bool): True only when status is ``completed``;
    False for all other statuses including ``invalidated``.

  Context log filtering (FR-016):
    ``--epoch-from`` / ``--epoch-to`` restrict entries to an inclusive epoch
    window. Entries with ``epoch is None`` are excluded when either bound is
    set.  ``--reason-contains`` performs a case-sensitive substring match on
    the reason field.  ``--context-summary`` replaces the full entry list
    with ``{source: count}`` aggregation (requires ``--context-log``).
  """

  name = 'show'
  help = 'Show experiment details with context journal'
  experiment_id = Argument('id', nargs='?', default=None, help='experiment ID (default: HEAD)')
  context_log_flag = Flag('--context-log', help='include context journal in output')
  context_source = Argument(
    '--context-source',
    default=None,
    help='filter journal entries by source',
  )
  limit = Argument('--limit', type=int, default=None, help='show N most recent entries')
  epoch_from = Argument(
    '--epoch-from',
    type=int,
    default=None,
    dest='epoch_from',
    help='inclusive lower bound on entry epoch',
  )
  epoch_to = Argument(
    '--epoch-to',
    type=int,
    default=None,
    dest='epoch_to',
    help='inclusive upper bound on entry epoch',
  )
  reason_contains = Argument(
    '--reason-contains',
    default=None,
    dest='reason_contains',
    help='case-sensitive substring match on reason',
  )
  context_summary_flag = Flag(
    '--context-summary',
    help='emit counts by source instead of full entries (requires --context-log)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Show experiment details, optionally with context journal entries.

    When ``args.id`` is explicit, searches all trees in the forest (cross-tree
    lookup). When falling back to HEAD, only the active tree is used.

    When ``--context-log`` is set, appends journal rows (timestamp, source,
    reason) to the text output. JSON mode includes a ``context_log`` array.
    ``--context-source`` filters entries via ``ContextLog.filter_by_source``.
    ``--limit`` slices the most recent N entries after filtering.
    ``--epoch-from`` / ``--epoch-to`` filter by epoch range (inclusive).
    ``--reason-contains`` applies case-sensitive substring match on reason.
    ``--context-summary`` replaces entries with ``{source: count}`` counts.
    """
    if args.context_summary and not args.context_log:
      ctx.fail('--context-summary requires --context-log; add --context-log to enable')

    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)

    eid = args.id
    if eid is None:
      eid = tree.head
      if eid is None:
        ctx.fail('no experiment specified and no HEAD set')
      node = require_experiment_node(ctx, tree, eid)
      owning_tree = tree
    else:
      result = forest.find_experiment(eid)
      if result is None:
        ctx.fail(
          f'experiment {eid!r} not found in any tree; '
          'verify the experiment id or check available experiments with query'
        )
      node, owning_tree = result

    exp = node.experiment
    tree_name = owning_tree.name

    if not ctx.output.use_json and tree_name != tree.name:
      ctx.output.info(f"resolved from tree '{tree_name}'")

    result: dict[str, Any] = {
      'id': exp.id,
      'tree': tree_name,
      'hypothesis': exp.hypothesis,
      'status': exp.status.value,
      'metrics': exp.metrics,
      'dependencies': list(exp.dependencies),
      'epoch': exp.epoch,
      'error': exp.error,
      'notes': exp.notes,
      'spec_version': exp.spec_version,
      'created_at': exp.created_at,
      'started_at': exp.started_at,
      'completed_at': exp.completed_at,
      'failed_at': exp.failed_at,
      'cancelled_at': exp.cancelled_at,
      'invalidated_at': exp.invalidated_at,
      'deployed_as': node.deployed_as,
      'parent': node.parent.experiment.id if node.parent is not None else None,
      'baseline': node.baseline.experiment.id if node.baseline is not None else None,
      'dataset_fingerprint': exp.dataset_meta.get('dataset_fingerprint'),
      'metrics_trusted': exp.status == Status.completed,
    }

    if args.context_log:
      entries = _filtered_context_entries(
        exp,
        args.context_source,
        args.limit,
        epoch_from=args.epoch_from,
        epoch_to=args.epoch_to,
        reason_substr=args.reason_contains,
      )

      if args.context_summary:
        summary = _build_context_summary(entries)
        result['context_summary'] = summary
        if not ctx.output.use_json:
          ctx.output.info(f'Context summary for {eid}:')
          rows = [{'source': src, 'count': str(cnt)} for src, cnt in summary.items()]
          ctx.output.table(rows, ['source', 'count'])
      else:
        result['context_log'] = [e.to_dict() for e in entries]
        if not ctx.output.use_json:
          ctx.output.info(f'Context journal for {eid}:')
          rows = [
            {
              'timestamp': e.timestamp[:19],
              'source': e.source or '',
              'reason': e.reason,
            }
            for e in entries
          ]
          ctx.output.table(rows, ['timestamp', 'source', 'reason'])

    ctx.output.result(result)


def _filtered_context_entries(
  exp: Experiment,
  source: str | None,
  limit: int | None,
  *,
  epoch_from: int | None = None,
  epoch_to: int | None = None,
  reason_substr: str | None = None,
) -> list[ContextEntry]:
  """Apply source, epoch, reason filters and limit to context log entries.

  Filters compose with AND: source + epoch range + reason substring all apply.
  Entries where ``entry.epoch is None`` are skipped when either epoch bound is set.

  Args:
    exp: Experiment whose context_log to query.
    source: If set, restrict to entries with this source value.
    limit: If set, take only the N most recent entries.
    epoch_from: Inclusive lower bound on entry epoch.
    epoch_to: Inclusive upper bound on entry epoch.
    reason_substr: Case-sensitive substring match on reason.

  Returns:
    Filtered and sliced list of ContextEntry objects.
  """
  if source is not None:
    entries = exp.context_log.filter_by_source(source)
  else:
    entries = exp.context_log.entries

  if epoch_from is not None or epoch_to is not None:
    entries = [e for e in entries if _epoch_in_range(e.epoch, epoch_from, epoch_to)]

  if reason_substr is not None:
    entries = [e for e in entries if reason_substr in e.reason]

  if limit is not None:
    entries = entries[-limit:]
  return entries


def _epoch_in_range(
  epoch: int | None,
  epoch_from: int | None,
  epoch_to: int | None,
) -> bool:
  """Check whether an entry's epoch falls within the requested range.

  Entries with ``epoch is None`` are excluded when any bound is set.

  Args:
    epoch: The entry's epoch value (may be None).
    epoch_from: Inclusive lower bound (or None for no lower bound).
    epoch_to: Inclusive upper bound (or None for no upper bound).

  Returns:
    True if the epoch satisfies the range constraint.
  """
  if epoch is None:
    return False
  if epoch_from is not None and epoch < epoch_from:
    return False
  return not (epoch_to is not None and epoch > epoch_to)


def _build_context_summary(entries: list[ContextEntry]) -> dict[str, int]:
  """Aggregate context entries by source into ``{source: count}`` counts.

  Sorted by descending count, then ascending key for stability.

  Args:
    entries: Filtered context entries to aggregate.

  Returns:
    Ordered dict of source to count.
  """
  counts: dict[str, int] = {}
  for entry in entries:
    key = entry.source or ''
    counts[key] = counts.get(key, 0) + 1
  return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


_QUERY_DEFAULTS = {
  'completed': False,
  'failed': False,
  'running': False,
  'pending': False,
  'terminal': False,
  'cancelled': False,
  'filter': None,
  'metric_gt': None,
  'metric_lt': None,
  'metric_between': None,
  'best': None,
  'higher': False,
  'lower': False,
  'sort': None,
  'all_trees': False,
  'context_contains': None,
  'context_source': None,
  'context_after': None,
  'created_after': None,
  'created_before': None,
  'case_sensitive': False,
  'compact': False,
  'include_invalidated': False,
  'deployed': False,
  'spec_version': None,
  'metadata_contains': None,
}


class ExperimentList(Command):
  """List experiments (alias for ``autopilot query``).

  Delegates to QueryCommand with default flags for convenience.
  Equivalent to running ``autopilot query`` in the same workspace.
  """

  name = 'list'
  help = 'List experiments (alias for query)'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Delegate to QueryCommand.forward with query-compatible defaults."""
    for attr, default in _QUERY_DEFAULTS.items():
      if not hasattr(args, attr):
        setattr(args, attr, default)
    QueryCommand().forward(ctx, args)


class ExperimentCommand(Command):
  """Manage experiments in the active tree.

  Subcommands: add, complete, fail, cancel, invalidate, deploy, undeploy,
  deploy-log, status, show, compare, remove, list, impact, lineage, timeline,
  notes (show/write), metadata (set/get/show).
  """

  name = 'experiment'
  help = 'Manage experiments'

  def __init__(self) -> None:
    """Wire experiment subcommands."""
    super().__init__()
    self.add = ExperimentAdd()
    self.complete_cmd = ExperimentComplete()
    self.fail_cmd = ExperimentFail()
    self.cancel = ExperimentCancel()
    self.invalidate = ExperimentInvalidate()
    self.deploy = ExperimentDeploy()
    self.undeploy = ExperimentUndeploy()
    self.deploy_log = ExperimentDeployLog()
    self.status_cmd = ExperimentStatus()
    self.show = ExperimentShow()
    self.compare = ExperimentCompare()
    self.remove = ExperimentRemove()
    self.list_cmd = ExperimentList()
    self.impact = ExperimentImpact()
    self.lineage = ExperimentLineage()
    self.timeline_cmd = ExperimentTimeline()
    self.notes = ExperimentNotes()
    self.metadata = ExperimentMetadata()
