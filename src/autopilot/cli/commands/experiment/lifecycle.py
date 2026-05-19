"""Experiment lifecycle commands: add, complete, fail, cancel, invalidate, remove.

Mutating operations that transition experiment status and persist the forest.
"""

from autopilot.ai.fingerprint import compute_fingerprint
from autopilot.cli.command import Command
from autopilot.cli.commands.experiment.compare import collect_all_dependencies
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import (
  journal_user_context,
  load_forest,
  require_active_tree,
  require_experiment_node,
)
from autopilot.cli.primitives import Argument, Flag
from autopilot.core.errors import ExperimentError
from autopilot.core.experiment import Experiment, validate_dependency_ids
from autopilot.core.node import Node
from pathlib import Path
from typing import Any
import argparse
import json
import uuid

EXPERIMENT_ID_HEX_LEN = 12


def _apply_dataset_fingerprint(ctx: CLIContext, exp: Experiment, dataset_path_str: str) -> None:
  """Compute and attach dataset fingerprint to experiment metadata.

  Args:
    ctx: CLI context for failure output.
    exp: Experiment to attach the fingerprint to.
    dataset_path_str: Raw --dataset-path argument string.
  """
  dataset_path = Path(dataset_path_str).expanduser().resolve()
  if not dataset_path.exists():
    ctx.fail(
      f'{type(dataset_path).__name__} {str(dataset_path)!r} does not exist;'
      ' create the file/directory or fix the --dataset-path'
    )
  fp = compute_fingerprint([dataset_path])
  exp.dataset_meta['dataset_fingerprint'] = fp.to_dict()


def _check_parent_flag_conflict(ctx: Any, args: argparse.Namespace) -> None:
  """Fail if ``--parent`` and ``--no-parent`` are both provided.

  Args:
    ctx: CLI context (used for ``ctx.fail``).
    args: Parsed argparse namespace.
  """
  if args.no_parent and args.parent is not None:
    ctx.fail(
      '--parent and --no-parent are mutually exclusive. '
      'Use --no-parent for an orphan experiment, '
      'or --parent <id> for an explicit parent, '
      'or omit both to auto-parent to HEAD.'
    )


def _resolve_parent(
  ctx: Any,
  tree: Any,
  args: argparse.Namespace,
) -> tuple[str | None, Any]:
  """Resolve parent experiment id and node from CLI args.

  When ``--no-parent`` is set, no parent is resolved even when HEAD exists.
  When ``--parent`` is explicit, that id is used. Otherwise HEAD is the default.

  Args:
    ctx: CLI context (used for ``ctx.fail`` on lookup errors).
    tree: Active tree to look up the parent node.
    args: Parsed argparse namespace with ``parent`` and ``no_parent`` fields.

  Returns:
    Tuple of (parent_id, parent_node) where either or both may be None.
  """
  parent_id = args.parent
  if parent_id is None and not args.no_parent and tree.head is not None:
    parent_id = tree.head
  parent_node = None
  if parent_id is not None:
    parent_node = require_experiment_node(ctx, tree, parent_id)
  return parent_id, parent_node


class ExperimentAdd(Command):
  """Add a new experiment to the active tree.

  Auto-parents to HEAD if no ``--parent`` is given and ``--no-parent`` is
  absent. Parent experiment must be in a terminal state (completed, failed,
  cancelled). If HEAD is non-terminal, add fails with an error -- complete
  or cancel the parent first.

  Use ``--no-parent`` to create an orphan experiment (no parent, even when
  HEAD exists). ``--parent <id>`` and ``--no-parent`` are mutually exclusive.

  After a successful add, tree HEAD is immediately set to the new experiment
  id (FRICTION-003). This ensures follow-up commands that default to HEAD
  operate on the most recently added experiment without requiring a manual
  checkout or switch.
  """

  name = 'add'
  help = 'Add an experiment to the active tree'
  hypothesis = Argument(
    '--hypothesis', required=False, default=None, help='experiment hypothesis (optional)'
  )
  parent = Argument('--parent', default=None, help='parent experiment ID')
  no_parent = Argument(
    '--no-parent',
    action='store_true',
    default=False,
    dest='no_parent',
    help='create without a parent (skip HEAD auto-parenting)',
  )
  baseline = Argument('--baseline', default=None, help='baseline experiment ID')
  experiment_id = Argument('--id', default=None, dest='experiment_id', help='experiment ID')
  depends_on = Argument(
    '--depends-on',
    action='append',
    default=None,
    dest='depends_on',
    help='experiment id this experiment depends on (repeatable)',
  )
  dataset_path = Argument(
    '--dataset-path',
    default=None,
    dest='dataset_path',
    help='dataset file or directory path to fingerprint',
  )
  spec_version = Argument(
    '--spec-version',
    default=None,
    dest='spec_version',
    help='spec/schema version string for this experiment',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create a new experiment node on the active tree and save the forest.

    When ``--dataset-path`` is provided, computes a SHA-256 fingerprint of
    the given file or directory and stores it in ``experiment.dataset_meta``.
    """
    _check_parent_flag_conflict(ctx, args)

    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)

    eid = args.experiment_id or uuid.uuid4().hex[:EXPERIMENT_ID_HEX_LEN]
    exp = Experiment(experiment_id=eid, hypothesis=args.hypothesis)

    if args.spec_version is not None:
      if not args.spec_version.strip():
        ctx.fail(
          "--spec-version must be non-empty; provide a version string (e.g. 'v1', '2024-01')"
        )
      exp.spec_version = args.spec_version

    if args.dataset_path is not None:
      _apply_dataset_fingerprint(ctx, exp, args.dataset_path)

    parent_id, parent_node = _resolve_parent(ctx, tree, args)

    baseline_node = None
    baseline_id = args.baseline
    if baseline_id is None and parent_id is not None:
      baseline_id = parent_id
    if baseline_id is not None:
      baseline_node = require_experiment_node(ctx, tree, baseline_id)

    raw_deps = args.depends_on or []
    if raw_deps:

      def _resolve_dep(dep_id: str) -> Node | None:
        pair = forest.find_experiment(dep_id)
        if pair is None:
          return None
        return pair[0]

      try:
        validated = validate_dependency_ids(
          raw_deps,
          self_id=eid,
          resolve=_resolve_dep,
          all_dependencies=lambda: collect_all_dependencies(forest),
        )
      except ExperimentError as exc:
        ctx.fail(str(exc))
      exp.dependencies = validated

    journal_user_context(ctx, exp, args)
    node = Node(experiment=exp, parent=parent_node, baseline=baseline_node)
    tree.add(node)
    tree.head = eid
    forest.save()

    result: dict[str, Any] = {
      'ok': True,
      'experiment_id': eid,
      'hypothesis': exp.hypothesis,
      'parent': parent_id,
      'baseline': baseline_id,
      'spec_version': exp.spec_version,
    }
    if exp.dependencies:
      result['dependencies'] = exp.dependencies
    if exp.dataset_meta.get('dataset_fingerprint') is not None:
      result['dataset_fingerprint'] = exp.dataset_meta['dataset_fingerprint']
    ctx.output.result(result)


class ExperimentComplete(Command):
  """Complete an experiment (pending or running -> completed).

  Accepts both ``pending`` and ``running`` status so CLI-only workflows
  that never call ``Trainer.start()`` can finalize directly. Resolves
  the experiment in the active tree, calls ``experiment.complete()``
  with optional parsed ``--metrics`` JSON, journals user context, and
  persists the forest.
  """

  name = 'complete'
  help = 'Complete an experiment'
  experiment_id = Argument('id', help='experiment ID to complete')
  metrics = Argument('--metrics', default=None, help='JSON metrics object (e.g. \'{"acc": 0.9}\')')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Complete the experiment and persist the forest."""
    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)
    node = require_experiment_node(ctx, tree, args.id)

    metrics_parsed: dict[str, float] | None = None
    if args.metrics is not None:
      try:
        metrics_parsed = json.loads(args.metrics)
      except (json.JSONDecodeError, TypeError) as exc:
        ctx.fail(f'invalid --metrics JSON: {exc}; expected a JSON object like \'{{"key": 1.0}}\'')

    journal_user_context(ctx, node.experiment, args)

    try:
      node.experiment.complete(metrics=metrics_parsed)
    except ExperimentError as exc:
      ctx.fail(str(exc))

    forest.save()
    ctx.output.result(
      {
        'ok': True,
        'experiment_id': args.id,
        'status': node.experiment.status.value,
        'metrics': node.experiment.metrics,
      }
    )


def _fail_error_for_json(experiment: Experiment, ctx: CLIContext) -> str | None:
  """Resolve the error string to surface in experiment fail JSON results.

  Precedence:
    1. ``experiment.error`` (set by ``fail(error=...)`` with ``--error``).
    2. ``ctx.context`` (the mandatory ``--context`` CLI flag value).
    3. First matching context log entry with a trainer failure reason or
       error metadata.

  Args:
    experiment: The experiment that was just failed.
    ctx: CLI context carrying the ``--context`` flag value.

  Returns:
    Best-effort error string, or None when nothing is available.
  """
  if experiment.error is not None:
    return experiment.error
  if ctx.context is not None:
    return ctx.context
  for entry in reversed(experiment.context_log.entries):
    if entry.source == 'trainer' and 'failed' in entry.reason.lower():
      return entry.reason
    if entry.metadata is not None:
      err = entry.metadata.get('error')
      if err is not None and err:
        return str(err)
  return None


class ExperimentFail(Command):
  """Fail an experiment (pending/running -> failed).

  Marks the experiment as failed with an optional ``--error`` string and
  optional ``--metrics`` JSON object for recording diagnostic metrics on
  failure. Experiments in ``pending`` or ``running`` status can be failed.

  JSON output includes an ``error`` field resolved via a fallback chain:
  ``--error`` value > ``--context`` flag > trainer failure context log entry.
  The fallback is display-only and does not mutate ``experiment.error``.
  """

  name = 'fail'
  help = 'Fail an experiment'
  experiment_id = Argument('id', help='experiment ID to fail')
  error = Argument('--error', default=None, help='failure error message')
  metrics = Argument('--metrics', default=None, help='JSON metrics object (e.g. \'{"acc": 0.1}\')')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Fail the experiment and persist the forest."""
    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)
    node = require_experiment_node(ctx, tree, args.id)

    metrics_parsed: dict[str, Any] | None = None
    if args.metrics is not None:
      try:
        metrics_parsed = json.loads(args.metrics)
      except (json.JSONDecodeError, TypeError) as exc:
        ctx.fail(f'invalid --metrics JSON: {exc}; expected a JSON object like \'{{"key": 1.0}}\'')
      if not isinstance(metrics_parsed, dict):
        ctx.fail(
          f'--metrics must be a JSON object, got {type(metrics_parsed).__name__}; '
          f'expected a JSON object like \'{{"key": 1.0}}\''
        )

    journal_user_context(ctx, node.experiment, args)

    try:
      node.experiment.fail(error=args.error, metrics=metrics_parsed)
    except ExperimentError as exc:
      ctx.fail(str(exc))

    forest.save()
    display_error = _fail_error_for_json(node.experiment, ctx)
    ctx.output.result(
      {
        'ok': True,
        'experiment_id': args.id,
        'status': node.experiment.status.value,
        'error': display_error,
        'metrics': node.experiment.metrics,
      }
    )


class ExperimentCancel(Command):
  """Cancel an experiment (pending/running -> cancelled).

  Marks the experiment as cancelled. Only non-terminal experiments
  can be cancelled.
  """

  name = 'cancel'
  help = 'Cancel an experiment'
  experiment_id = Argument('id', help='experiment ID to cancel')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Cancel the experiment and persist the forest."""
    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)
    node = require_experiment_node(ctx, tree, args.id)

    journal_user_context(ctx, node.experiment, args)

    try:
      node.experiment.cancel()
    except ExperimentError as exc:
      ctx.fail(str(exc))

    forest.save()
    ctx.output.result(
      {
        'ok': True,
        'experiment_id': args.id,
        'status': node.experiment.status.value,
      }
    )


class ExperimentInvalidate(Command):
  """Invalidate a completed experiment (completed -> invalidated).

  Marks a historically bad experiment without deleting it from the tree.
  Only experiments in ``completed`` status can be invalidated. Requires
  ``--reason`` to document why the experiment is being invalidated.
  """

  name = 'invalidate'
  help = 'Invalidate a completed experiment'
  experiment_id = Argument('id', help='experiment ID to invalidate')
  reason = Argument('--reason', required=True, help='reason for invalidation (required)')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Invalidate the experiment and persist the forest."""
    if not args.reason or not args.reason.strip():
      ctx.fail('--reason must be non-empty; provide a reason for invalidation')

    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)
    node = require_experiment_node(ctx, tree, args.id)

    journal_user_context(ctx, node.experiment, args)

    try:
      node.experiment.invalidate(reason=args.reason)
    except ExperimentError as exc:
      ctx.fail(str(exc))

    forest.save()
    ctx.output.result(
      {
        'ok': True,
        'experiment_id': args.id,
        'status': node.experiment.status.value,
        'invalidated_at': node.experiment.invalidated_at,
      }
    )


class ExperimentRemove(Command):
  """Remove an experiment from the active tree.

  Wraps ``Tree.remove(experiment_id, cascade=bool)``. Clears HEAD if the
  removed node was HEAD. Use ``--cascade`` to remove descendants.
  """

  name = 'remove'
  help = 'Remove an experiment from the tree'
  experiment_id = Argument('id', help='experiment ID to remove')
  cascade = Flag('--cascade', help='also remove all descendant experiments')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Remove experiment node and persist the forest."""
    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)

    eid = args.id
    node = tree.get(eid)
    if node is None:
      ctx.fail(f'experiment {eid!r} not found in active tree')

    journal_user_context(ctx, node.experiment, args)

    try:
      tree.remove(eid, cascade=args.cascade)
    except ValueError as exc:
      ctx.fail(str(exc))
    forest.save()

    ctx.output.result({'removed': eid, 'cascade': args.cascade})
