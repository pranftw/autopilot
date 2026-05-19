"""Read-only ``report narrative``: forest topology, counts, optional bests, tails."""

from autopilot.ai.forest import FileForest
from autopilot.cli.command import Command
from autopilot.cli.commands.query import resolve_metric_name
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.primitives import Argument, Flag
from autopilot.core.node import Node
from autopilot.core.tree import Tree
from autopilot.tracking.executions import ExecutionRecord, load_executions
from autopilot.tracking.io import parse_timestamp, read_jsonl
from typing import Any
import argparse

DEFAULT_CONTEXT_TAIL = 10
DEFAULT_EXECUTIONS_TAIL = 10
DEFAULT_REFLOG_TAIL = 5


class ReportNarrative(Command):
  """Aggregate tree stats, optional best metrics, recent context, executions, reflog."""

  name = 'narrative'
  help = 'Project handoff summary'
  metric = Argument(
    '--metric',
    default=None,
    metavar='METRIC',
    help='metric for best-per-tree selection (omit to skip best section)',
  )
  higher = Flag(
    '--higher',
    help='higher metric values are better (default)',
  )
  lower = Flag(
    '--lower',
    help='lower metric values are better',
  )
  context_tail = Argument(
    '--context-tail',
    type=int,
    default=DEFAULT_CONTEXT_TAIL,
    metavar='N',
    help=f'number of recent context entries (default: {DEFAULT_CONTEXT_TAIL})',
  )
  executions_tail = Argument(
    '--executions-tail',
    type=int,
    default=DEFAULT_EXECUTIONS_TAIL,
    metavar='N',
    help=f'number of recent CLI executions (default: {DEFAULT_EXECUTIONS_TAIL})',
  )
  reflog_tail = Argument(
    '--reflog-tail',
    type=int,
    default=DEFAULT_REFLOG_TAIL,
    metavar='N',
    help=f'number of recent reflog entries (default: {DEFAULT_REFLOG_TAIL})',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Build and emit a project narrative summary."""
    forest = load_forest(ctx)
    payload = _build_narrative(forest, ctx, args)

    if not ctx.output.use_json:
      _render_narrative_text(ctx, payload)

    ctx.output.result(payload)


def _build_narrative(
  forest: FileForest,
  ctx: CLIContext,
  args: argparse.Namespace,
) -> dict[str, Any]:
  """Build narrative payload (trees, recent context/executions, reflog tail)."""
  trees = forest.list_trees()
  higher_is_better = not args.lower

  tree_nodes: list[tuple[str, list[Node]]] = []
  all_nodes: list[Node] = []
  for tree in trees:
    nodes = tree.query().all()
    tree_nodes.append((tree.name, nodes))
    all_nodes.extend(nodes)

  resolved_metric: str | None = None
  if args.metric is not None and all_nodes:
    resolved_metric = resolve_metric_name(all_nodes, args.metric)

  trees_payload = [_tree_summary(tree, resolved_metric, higher_is_better) for tree in trees]

  recent_context = _collect_recent_context(tree_nodes, args.context_tail)
  recent_executions = _collect_recent_executions(ctx, args.executions_tail)
  reflog = _collect_reflog_tail(ctx, args.reflog_tail)

  return {
    'tree_count': len(trees),
    'trees': trees_payload,
    'recent_context': recent_context,
    'recent_executions': recent_executions,
    'reflog_tail': reflog,
  }


def _tree_summary(
  tree: Tree,
  resolved_metric: str | None,
  higher_is_better: bool,
) -> dict[str, Any]:
  """Per-tree histogram plus optional ``best`` metric selection."""
  nodes = tree.query().all()
  status_counts: dict[str, int] = {}
  for node in nodes:
    status_val = node.experiment.status.value
    status_counts[status_val] = status_counts.get(status_val, 0) + 1

  result: dict[str, Any] = {
    'name': tree.name,
    'experiment_count': len(nodes),
    'status_counts': status_counts,
  }

  if resolved_metric is not None:
    best_node = tree.query().best(resolved_metric, higher_is_better=higher_is_better)
    if best_node is not None:
      metric_value = best_node.experiment.metrics.get(resolved_metric)
      result['best'] = {
        'id': best_node.experiment.id,
        'metric': resolved_metric,
        'value': metric_value,
      }
    else:
      result['best'] = None

  return result


def _collect_recent_context(
  tree_nodes: list[tuple[str, list[Node]]],
  limit: int,
) -> list[dict[str, Any]]:
  """Newest-first context entries across trees, capped at ``limit``."""
  tuples: list[tuple[str, str, dict[str, Any]]] = []
  for tree_name, nodes in tree_nodes:
    for node in nodes:
      exp = node.experiment
      tuples.extend((exp.id, tree_name, entry.to_dict()) for entry in exp.context_log)

  tuples.sort(
    key=lambda t: parse_timestamp(t[2]['timestamp']),
    reverse=True,
  )
  tuples = tuples[:limit]

  return [
    {**entry_dict, 'experiment_id': exp_id, 'tree': tree_name}
    for exp_id, tree_name, entry_dict in tuples
  ]


def _collect_recent_executions(
  ctx: CLIContext,
  limit: int,
) -> list[dict[str, Any]]:
  """Tail of ``executions.jsonl`` as handoff-shaped dicts."""
  records = load_executions(ctx.config.executions_path)
  tail = records[-limit:] if limit < len(records) else records
  return [_execution_record_subset(rec) for rec in tail]


def _execution_record_subset(rec: ExecutionRecord) -> dict[str, Any]:
  """Pick timestamp, command, experiment, context, exit_code."""
  return {
    'timestamp': rec.timestamp,
    'command': rec.command,
    'experiment': rec.experiment,
    'context': rec.context,
    'exit_code': rec.exit_code,
  }


def _collect_reflog_tail(
  ctx: CLIContext,
  limit: int,
) -> list[dict[str, Any]]:
  """Last ``limit`` reflog rows (empty when missing); skips corrupt lines."""
  reflog_path = ctx.config.store_path / 'reflog.jsonl'
  entries = read_jsonl(reflog_path, strict=False)
  return entries[-limit:] if limit < len(entries) else list(entries)


def _render_narrative_text(
  ctx: CLIContext,
  payload: dict[str, Any],
) -> None:
  """Pretty-print narrative payload (non-JSON mode)."""
  tree_count = payload['tree_count']
  if tree_count == 0:
    ctx.output.info('No trees in forest. Run `tree create` to start.')
    return

  ctx.output.info(f'Project narrative: {tree_count} tree(s)')
  ctx.output.info('')

  for tree_info in payload['trees']:
    _render_tree_block(ctx, tree_info)

  _render_context_block(ctx, payload['recent_context'])
  _render_executions_block(ctx, payload['recent_executions'])
  _render_reflog_block(ctx, payload['reflog_tail'])


def _render_tree_block(ctx: CLIContext, tree_info: dict[str, Any]) -> None:
  """Print one tree header, status histogram, optional best."""
  name = tree_info['name']
  exp_count = tree_info['experiment_count']
  status_counts = tree_info['status_counts']
  status_str = ', '.join(f'{k}: {v}' for k, v in sorted(status_counts.items()))
  ctx.output.info(f'  Tree {name!r}: {exp_count} experiment(s)')
  if status_str:
    ctx.output.info(f'    Status: {status_str}')
  best = tree_info.get('best')
  if best is not None:
    ctx.output.info(f'    Best: {best["id"]} ({best["metric"]}={best["value"]})')


def _render_context_block(ctx: CLIContext, recent_context: list[dict[str, Any]]) -> None:
  """Print recent decisions list."""
  if not recent_context:
    return
  ctx.output.info('')
  ctx.output.info('Recent decisions:')
  for entry in recent_context:
    exp_id = entry['experiment_id']
    reason = entry['reason']
    source = entry['source']
    ctx.output.info(f'  [{exp_id}] ({source}) {reason}')


def _render_executions_block(ctx: CLIContext, recent_executions: list[dict[str, Any]]) -> None:
  """Print recent executions list."""
  if not recent_executions:
    return
  ctx.output.info('')
  ctx.output.info('Recent CLI executions:')
  for rec in recent_executions:
    cmd = rec['command']
    ts = rec['timestamp'][:19]
    exit_code = rec['exit_code']
    ctx.output.info(f'  {ts}  {cmd}  (exit={exit_code})')


def _render_reflog_block(ctx: CLIContext, reflog: list[dict[str, Any]]) -> None:
  """Print reflog tail or a missing-message."""
  if reflog:
    ctx.output.info('')
    ctx.output.info('Recent store reflog:')
    for entry in reflog:
      op = entry.get('operation') or ''
      exp_id = entry.get('experiment_id') or ''
      ts = (entry.get('timestamp') or '')[:19]
      ctx.output.info(f'  {ts}  {op}  {exp_id}')
  else:
    ctx.output.info('')
    ctx.output.info('No reflog entries.')
