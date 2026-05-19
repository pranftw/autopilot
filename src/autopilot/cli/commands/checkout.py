"""Checkout command: navigate to an experiment by setting tree HEAD and restoring Store state.

autopilot checkout <experiment-id> [--json]

Sets the active tree's HEAD to the given experiment and restores the
Store's parameter state for that experiment. This is the high-level
navigation command. Distinct from 'autopilot store checkout' which is
a low-level Store VCS operation.

Path resolution via ctx.config (no paths.* calls).
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import (
  journal_user_context,
  load_forest,
  require_active_tree,
  require_experiment_node,
)
from autopilot.cli.primitives import Argument
from autopilot.core.errors import StoreError
from typing import Any
import argparse


def _find_dependents(forest: Any, target_id: str) -> list[str]:
  """Find experiments that declare *target_id* as a dependency.

  Scans all trees in the forest for experiments whose ``dependencies``
  list contains *target_id*.

  Args:
    forest: Forest instance to scan.
    target_id: Experiment id to look for in dependency lists.

  Returns:
    Sorted list of experiment ids that depend on *target_id*.
  """
  dependents: list[str] = []
  for t in forest.list_trees():
    dependents.extend(
      node.experiment.id for node in t.query().all() if target_id in node.experiment.dependencies
    )
  return sorted(set(dependents))


class CheckoutCommand(Command):
  """Set tree HEAD and restore Store state for an experiment."""

  name = 'checkout'
  help = 'Navigate to an experiment (set HEAD + restore state)'
  experiment_id = Argument('experiment_id', help='experiment ID to checkout')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Set tree HEAD to the experiment and persist the forest."""
    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)

    eid = args.experiment_id
    node = require_experiment_node(ctx, tree, eid)
    journal_user_context(ctx, node.experiment, args)

    dependents = _find_dependents(forest, eid)
    if dependents:
      dep_list = ', '.join(dependents)
      ctx.output.warn(
        f'experiments declaring a dependency on {eid!r}: {dep_list}. '
        f'Run `autopilot experiment impact {eid}` to see full impact.'
      )

    try:
      tree.checkout(eid, context=ctx.context)
    except (RuntimeError, ValueError, StoreError) as exc:
      ctx.fail(
        f'store checkout failed ({type(exc).__name__}): {exc};'
        ' verify the experiment and epoch exist with store log'
      )
    forest.save()

    exp = node.experiment
    ctx.output.result(
      {
        'ok': True,
        'experiment_id': eid,
        'status': exp.status.value,
        'hypothesis': exp.hypothesis,
      }
    )
