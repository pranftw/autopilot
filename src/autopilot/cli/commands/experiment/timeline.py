"""Unified experiment timeline: chronological merge of context, execution, and reflog.

Read-only command that resolves an experiment across all trees via
``Forest.find_experiment`` and merges its context log, execution records,
and store reflog into a single chronological timeline.

JSON result schema::

  {
    'ok': True,
    'result': {
      'entries': [<TimelineEntry dict>, ...],
    },
    'messages': [...],
  }

Execution records are loaded via ``load_executions`` / ``filter_executions``
from ``tracking/executions.py`` (shared with ``debug executions`` and
``undo-guide``). Reflog entries are loaded via ``FileStore.iter_reflog()``.
"""

from autopilot.cli.command import Command
from autopilot.cli.commands.store.helpers import open_forest_store
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.primitives import Argument
from autopilot.core.errors import StoreError
from autopilot.core.timeline import build_timeline
from autopilot.tracking.executions import filter_executions, load_executions
from typing import Any
import argparse


class ExperimentTimeline(Command):
  """Show unified chronological timeline for an experiment.

  Merges context log entries, execution records, and store reflog
  operations into a single sorted timeline. Read-only; supports ``--json``.
  """

  name = 'timeline'
  help = 'Show unified experiment timeline'
  experiment_id = Argument('id', help='experiment ID')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Build and emit the unified timeline for the given experiment.

    Steps:
      1. Resolve experiment across all trees via Forest.find_experiment.
      2. Load context log from the experiment.
      3. Load execution records filtered to this experiment.
      4. Load reflog entries filtered to this experiment's branch.
      5. Merge via build_timeline and emit sorted entries.
    """
    forest = load_forest(ctx)
    eid = args.id

    found = forest.find_experiment(eid)
    if found is None:
      ctx.fail(
        f'experiment {eid!r} not found in any tree; '
        'verify the experiment id or use query to list available experiments'
      )

    node, owning_tree = found
    exp = node.experiment

    context_entries = exp.context_log.entries

    exec_path = ctx.config.executions_path
    if exec_path.exists():
      all_records = load_executions(exec_path)
      exp_records = filter_executions(all_records, experiment=eid)
    else:
      exp_records = []

    reflog_entries = _load_filtered_reflog(ctx, eid)

    timeline = build_timeline(
      experiment_id=eid,
      context_log=context_entries,
      execution_records=exp_records,
      reflog_entries=reflog_entries,
    )

    entries_dicts = [entry.to_dict() for entry in timeline]

    if not ctx.output.use_json:
      ctx.output.info(f'Timeline for {eid} (tree: {owning_tree.name})')
      if entries_dicts:
        rows = [
          {
            'timestamp': e['timestamp'][:19],
            'stream': e['stream'],
            'source': e.get('source') or '',
            'reason': e['reason'],
          }
          for e in entries_dicts
        ]
        ctx.output.table(rows, ['timestamp', 'stream', 'source', 'reason'])
      else:
        ctx.output.info('  (no timeline entries)')

    ctx.output.result({'entries': entries_dicts})


def _load_filtered_reflog(ctx: CLIContext, experiment_id: str) -> list[dict[str, Any]]:
  """Load reflog entries filtered to a specific experiment branch.

  Attempts to open the forest store and iterate reflog entries. If the
  store is not available or has no reflog, returns an empty list.

  Args:
    ctx: CLI context for workspace resolution.
    experiment_id: Experiment whose reflog entries to select.

  Returns:
    List of reflog dicts where experiment_id matches.
  """
  try:
    store = open_forest_store(ctx)
  except (OSError, KeyError, ValueError, StoreError):
    return []

  return [entry for entry in store.iter_reflog() if entry.get('experiment_id') == experiment_id]
