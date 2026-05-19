"""Trace command -- collect, inspect, and verify execution traces.

Includes ``trace verify`` subcommand for auditing context log + reflog
completeness against expected dimensions. ``--epochs`` inference and
JSON result shape are documented in the ``verify`` handler.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest, require_active_tree, resolve_command_epoch
from autopilot.cli.messages import MSG_EXPERIMENT_SLUG_REQUIRED
from autopilot.cli.primitives import argument, subcommand
from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.trace import is_policy_gate_entry, verify_trace_completeness
from typing import Any
import argparse


class TraceCommand(Command):
  """Collect and inspect execution traces."""

  name = 'trace'
  help = 'trace collection and inspection'

  @argument('--limit', type=int, default=0, help='max items')
  @subcommand('collect', help_text='collect trace data from epoch')
  def collect(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Collect trace data items from an epoch."""
    epoch = resolve_command_epoch(ctx, args)

    exp_dir = ctx.experiment_path()
    data = DataArtifact().read_raw(exp_dir, epoch=epoch)

    limit = args.limit
    if limit > 0:
      data = data[:limit]

    result: dict[str, Any] = {
      'epoch': epoch,
      'items': data,
      'count': len(data),
    }
    ctx.output.result(result)

  @argument('--node', default=None, help='node/id to inspect')
  @subcommand('inspect', help_text='inspect trace for a node')
  def inspect_trace(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Inspect trace data for a specific node."""
    node = args.node
    if not node:
      ctx.fail('--node is required')

    epoch = resolve_command_epoch(ctx, args)

    exp_dir = ctx.experiment_path()
    data = DataArtifact().read_raw(exp_dir, epoch=epoch)

    matched = []
    for idx, item in enumerate(data):
      datum_id = item.get('id')
      meta_node = item.get('metadata', {}).get('node')
      if node in {datum_id, meta_node} or node in str(datum_id) or node in str(meta_node):
        matched.append(
          {
            'batch_idx': idx,
            'id': datum_id,
            'success': item.get('success', True),
            'feedback': item.get('feedback'),
            'error_message': item.get('error_message'),
            'metadata': item.get('metadata', {}),
          }
        )

    result: dict[str, Any] = {
      'node': node,
      'epoch': epoch,
      'matches': matched,
      'count': len(matched),
    }
    ctx.output.result(result)

  @argument(
    '--epochs',
    type=int,
    default=None,
    help='epochs executed (default: infer from context log policy entries)',
  )
  @argument(
    '--check-cost',
    action='store_true',
    help='also require cost attribution entries per epoch',
  )
  @subcommand('verify', help_text='audit trace completeness for an experiment')
  def verify(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Audit context log + reflog completeness for the active experiment.

    Requires ``--experiment`` slug. Loads the experiment from the active tree,
    reads the context log and reflog, then runs ``verify_trace_completeness``
    across policy gate, gradient journal, store context, and optional cost
    dimensions.

    Epoch count is either explicit via ``--epochs N`` or inferred from the
    highest epoch index in policy gate context entries (max epoch + 1).

    JSON result: ``{experiment_id, epochs_run, report: TraceReport.to_dict()}``.
    Text mode: prints complete status, dimension details, and gaps.
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
    context_log = experiment.context_log

    reflog_entries: list[dict[str, Any]] = []
    store = forest.store
    if store is not None:
      reflog_entries = [
        entry for entry in store.iter_reflog() if entry.get('experiment_id') == slug
      ]

    epochs_run = args.epochs
    if epochs_run is not None:
      if epochs_run < 0:
        ctx.fail('--epochs must be non-negative')
    else:
      policy_epochs = [
        entry.epoch
        for entry in context_log
        if is_policy_gate_entry(entry) and entry.epoch is not None
      ]
      if policy_epochs:
        epochs_run = max(policy_epochs) + 1
      else:
        ctx.fail(
          'cannot infer epochs from context log (no policy gate entries found); '
          'pass --epochs N explicitly'
        )

    report = verify_trace_completeness(
      context_log,
      reflog_entries,
      epochs_run,
      check_cost=args.check_cost,
    )

    if not ctx.output.use_json:
      ctx.output.info(f'complete: {report.complete}')
      for dim in report.dimensions:
        for detail in dim.details:
          ctx.output.info(f'  {detail}')
      for gap in report.gaps:
        ctx.output.info(f'  gap: {gap}')

    result: dict[str, Any] = {
      'experiment_id': slug,
      'epochs_run': epochs_run,
      'report': report.to_dict(),
    }
    ctx.output.result(result)
