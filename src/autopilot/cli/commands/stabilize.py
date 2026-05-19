"""CLI command for stabilizing an experiment.

Resolves experiments cross-tree via ``Forest.find_experiment`` for primary
lookup. Falls back to snapshot directory presence for orphan experiments
outside the forest graph.

Stabilize selects the latest epoch by numeric extraction from
``epoch_<n>.json`` filenames (not lexicographic sort). Only files matching
``^epoch_(digit+).json$`` participate in ordering; other files in the
snapshots directory are silently ignored (BUG-034).

Multi-project overwrite hazard (BUG-045): when multiple projects share a
workspace, ``stabilize`` writes files into the project root based on
``original_path`` from the snapshot manifest. If two projects produce files
with the same ``original_path``, a second stabilize overwrites the first's
artifacts. Use distinct ``original_path`` layouts per project, or scope
with ``--parameter-prefix``.
"""

from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest
from autopilot.cli.primitives import Argument
from autopilot.core.enums import Status
import argparse


class StabilizeCommand(Command):
  """Copy experiment parameter files back to the project root.

  Selects the latest epoch by numeric extraction from ``epoch_<n>.json``
  filenames. Cross-project note: when multiple projects share a workspace,
  use ``--parameter-prefix`` to scope which parameters are copied and avoid
  overwriting artifacts from other projects.

  Copies **all** parameter files including optimizer-mutated artifacts
  (e.g. ``.optimization/`` feedback files). Use ``--parameter-prefix``
  to filter when only a subset should transfer.

  Resolves experiments cross-tree via ``Forest.find_experiment``. Falls back
  to snapshot directory presence for orphan experiments outside the forest.
  """

  name = 'stabilize'
  help = 'Stabilize experiment results into project root'
  experiment_id = Argument('experiment_id', help='experiment ID to stabilize')
  parameter_prefix = Argument(
    '--parameter-prefix',
    default=None,
    metavar='PREFIX',
    help='only copy entries whose manifest key starts with PREFIX/',
  )

  def _resolve_experiment(self, ctx: CLIContext, experiment_id: str) -> str | None:
    """Locate experiment via Forest.find_experiment or snapshot fallback.

    Uses ``Forest.find_experiment`` for primary lookup. Falls back to checking
    the snapshots directory for orphan experiments that may not appear in the
    forest graph.

    Returns:
      Status string when found.
    """
    config = ctx.config
    config.store_path.mkdir(parents=True, exist_ok=True)

    forest = load_forest(ctx)
    result = forest.find_experiment(experiment_id)

    if result is not None:
      node, _ = result
      return node.experiment.status.value

    snapshots_dir = config.snapshots_path / experiment_id
    if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
      return Status.completed.value

    ctx.fail(
      f'experiment {experiment_id!r} not found in any tree; '
      'verify the experiment id or check available experiments with query'
    )
    return None

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Execute stabilization: validate experiment, then copy files."""
    experiment_id = args.experiment_id
    experiment_status = self._resolve_experiment(ctx, experiment_id)

    if experiment_status and experiment_status != Status.completed.value:
      msg = (
        f'experiment {experiment_id!r} is not completed '
        f'(status={experiment_status!r}); only completed experiments can be stabilized'
      )
      ctx.fail(msg)

    copied = ctx.config.stabilize(experiment_id, parameter_prefix=args.parameter_prefix)

    result = {'copied': [str(p) for p in copied]}

    if ctx.output.use_json:
      ctx.output.result(result)
    else:
      if copied:
        ctx.output.info(f'Stabilized {len(copied)} file(s) from {experiment_id!r}:')
        for p in copied:
          ctx.output.info(f'  {p}')
      elif args.parameter_prefix is not None:
        ctx.output.info(
          f'no parameters matched prefix {args.parameter_prefix!r} for experiment {experiment_id!r}'
        )
      else:
        ctx.output.info(f'No files to stabilize for {experiment_id!r}')
      ctx.output.result(result)
