"""Experiment lineage: walk ancestor chain via Node.parent pointers.

Read-only command that resolves an experiment across all trees via
``Forest.find_experiment`` and walks the parent chain to produce a
deterministic ancestor list ordered from immediate parent to root.

JSON result schema::

  {
    'experiment_id': str,
    'tree': str,
    'depth': int,
    'ancestors': [{'id': str, 'status': str, 'metrics': dict}, ...],
  }

``depth == len(ancestors)``: root experiment has depth 0; each hop
toward root increments depth by one. ``ancestors[0]`` is the immediate
parent; increasing index walks toward the ultimate root.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.primitives import Argument
from autopilot.core.node import Node
from typing import Any
import argparse


def collect_ancestors(node: Node) -> list[dict[str, Any]]:
  """Walk parent pointers; immediate parent first.

  Args:
    node: Starting node whose lineage to traverse.

  Returns:
    List of ancestor dicts ordered from immediate parent to root.
    Each dict has keys: ``id``, ``status``, ``metrics``.
  """
  out: list[dict[str, Any]] = []
  current = node.parent
  while current is not None:
    exp = current.experiment
    out.append(
      {
        'id': exp.id,
        'status': exp.status.value,
        'metrics': dict(exp.metrics),
      }
    )
    current = current.parent
  return out


class ExperimentLineage(Command):
  """Show ancestor chain for an experiment.

  Resolves the experiment across all trees via ``Forest.find_experiment``,
  then walks ``Node.parent`` pointers from immediate parent to root.
  """

  name = 'lineage'
  help = 'Show experiment ancestor chain'
  experiment_id = Argument('id', help='experiment ID to trace lineage for')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Traverse parent chain and emit lineage result.

    Text mode prints a human-readable indented list:
      depth 0: <immediate parent id> (status) metrics={...}
      depth 1: <grandparent id> (status) metrics={...}
      ...

    JSON mode returns the standard envelope with result containing
    ``experiment_id``, ``tree``, ``depth``, and ``ancestors``.
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
    ancestors = collect_ancestors(node)
    depth = len(ancestors)

    if not ctx.output.use_json:
      ctx.output.info(f'Lineage for {eid} (tree: {owning_tree.name}, depth: {depth})')
      if ancestors:
        rows = [
          {
            'hop': str(i),
            'id': a['id'],
            'status': a['status'],
            'metrics': str(a['metrics']),
          }
          for i, a in enumerate(ancestors)
        ]
        ctx.output.table(rows, ['hop', 'id', 'status', 'metrics'])
      else:
        ctx.output.info('  (root experiment, no ancestors)')

    ctx.output.result(
      {
        'experiment_id': eid,
        'tree': owning_tree.name,
        'depth': depth,
        'ancestors': ancestors,
      }
    )
