"""Experiment metadata, notes, and deployment commands.

Subcommands for metadata key-value pairs, experiment notes (show/write),
and deployment lifecycle (deploy/undeploy/deploy-log).

Metadata commands resolve experiments cross-tree via ``Forest.find_experiment``
so agents need not switch trees to read or set metadata on any experiment.
"""

from autopilot.ai.deployment import deployment_log_for_workspace, emit_deployment_event
from autopilot.cli.command import Command
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import (
  journal_user_context,
  load_forest,
  require_active_tree,
  require_experiment_node,
)
from autopilot.cli.primitives import Argument, Flag
from autopilot.core.decision import DecisionEntry
from autopilot.core.metadata import MetadataArtifact
from autopilot.core.node import Node
from autopilot.tracking.io import BINARY_SNIFF_BYTES
from pathlib import Path
from typing import Any
import argparse


def _resolve_experiment_for_metadata(ctx: CLIContext, forest: Any, experiment_id: str) -> Node:
  """Resolve an experiment cross-tree for metadata operations.

  Args:
    ctx: CLI context for failure output.
    forest: Forest instance to search.
    experiment_id: Experiment id to look up.

  Returns:
    The matching Node.
  """
  result = forest.find_experiment(experiment_id)
  if result is None:
    ctx.fail(
      f'Experiment {experiment_id!r} not found in any tree. '
      'Run: autopilot query --all-trees --json to list available experiments.'
    )
  node, _ = result
  return node


class ExperimentMetadataSet(Command):
  """Set a metadata key-value pair on an experiment.

  Mutating command: requires ``--context``. Persists metadata to
  ``{experiment_path}/metadata.json`` via ``MetadataArtifact``.
  Resolves experiments cross-tree via ``Forest.find_experiment``.
  """

  name = 'set'
  help = 'Set a metadata key-value pair'
  experiment_id = Argument('id', help='experiment ID')
  key_arg = Argument('key', help='metadata key')
  value_arg = Argument('value', help='metadata value')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Write one key-value pair to experiment metadata."""
    forest = load_forest(ctx)
    require_active_tree(ctx, forest)
    node = _resolve_experiment_for_metadata(ctx, forest, args.id)
    journal_user_context(ctx, node.experiment, args)
    exp_dir = ctx.experiment_path(node.experiment.id)
    artifact = MetadataArtifact()
    artifact.set(args.key, args.value, base_dir=exp_dir)
    if not ctx.output.use_json:
      ctx.output.info(f'{args.key}={args.value}')
    ctx.output.result(
      {
        'experiment_id': node.experiment.id,
        'key': args.key,
        'value': args.value,
      }
    )


class ExperimentMetadataGet(Command):
  """Get a single metadata value by key.

  Read-only and context-exempt. Returns the value or null/empty when
  the key is not set. Resolves experiments cross-tree.
  """

  name = 'get'
  help = 'Get a metadata value by key'
  experiment_id = Argument('id', help='experiment ID')
  key_arg = Argument('key', help='metadata key')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Print a single metadata value."""
    forest = load_forest(ctx)
    require_active_tree(ctx, forest)
    node = _resolve_experiment_for_metadata(ctx, forest, args.id)
    exp_dir = ctx.experiment_path(node.experiment.id)
    artifact = MetadataArtifact()
    value = artifact.get(args.key, base_dir=exp_dir)
    if not ctx.output.use_json:
      ctx.output.info(str(value) if value is not None else '(not set)')
    ctx.output.result(
      {
        'experiment_id': node.experiment.id,
        'key': args.key,
        'value': value,
      }
    )


class ExperimentMetadataShow(Command):
  """Show all metadata key-value pairs for an experiment.

  Read-only and context-exempt. Returns ``{}`` when no metadata is set.
  Resolves experiments cross-tree.
  """

  name = 'show'
  help = 'Show all metadata key-value pairs'
  experiment_id = Argument('id', help='experiment ID')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Dump all metadata as text or JSON."""
    forest = load_forest(ctx)
    require_active_tree(ctx, forest)
    node = _resolve_experiment_for_metadata(ctx, forest, args.id)
    exp_dir = ctx.experiment_path(node.experiment.id)
    artifact = MetadataArtifact()
    data = artifact.show(base_dir=exp_dir)
    if not ctx.output.use_json:
      if not data:
        ctx.output.info('(no metadata)')
      else:
        for key, value in data.items():
          ctx.output.info(f'{key}={value}')
    ctx.output.result(
      {
        'experiment_id': node.experiment.id,
        'metadata': data,
      }
    )


class ExperimentMetadata(Command):
  """Manage experiment metadata (set, get, show)."""

  name = 'metadata'
  help = 'Manage experiment metadata'

  def __init__(self) -> None:
    """Wire metadata subcommands (set, get, show)."""
    super().__init__()
    self.set_cmd = ExperimentMetadataSet()
    self.get_cmd = ExperimentMetadataGet()
    self.show_cmd = ExperimentMetadataShow()


class ExperimentNotesShow(Command):
  """Show the notes for an experiment.

  Unlike ``experiment show`` which defaults to HEAD when no id is given,
  ``notes show`` requires a positional ``id`` argument. There is no HEAD
  default for the notes subcommand -- callers must always specify the
  experiment id explicitly.
  """

  name = 'show'
  help = 'Show experiment notes'
  experiment_id = Argument('id', help='experiment ID')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Print current experiment notes."""
    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)
    node = require_experiment_node(ctx, tree, args.id)
    notes = node.experiment.notes
    if not ctx.output.use_json:
      ctx.output.info(notes if notes is not None else '(no notes)')
    ctx.output.result({'experiment_id': args.id, 'notes': notes})


class ExperimentNotesWrite(Command):
  """Write or overwrite notes for an experiment.

  Notes text is supplied via --body (inline) or --file (read from path).
  The two flags are mutually exclusive; one must be provided.
  Migration: positional text argument is removed. Use --body '...' instead.
  """

  name = 'write'
  help = 'Write experiment notes (use --body or --file)'
  experiment_id = Argument('id', help='experiment ID')
  body = Argument('--body', default=None, help='notes text to write inline')
  file = Argument('--file', default=None, help='read notes from UTF-8 file path')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Set experiment notes from inline body or file and persist the forest."""
    if args.body is not None and args.file is not None:
      ctx.fail('--body and --file are mutually exclusive')
    if args.body is None and args.file is None:
      ctx.fail('Provide --body or --file')

    content = self._read_file(ctx, Path(args.file)) if args.file is not None else args.body

    forest = load_forest(ctx)
    tree = require_active_tree(ctx, forest)
    node = require_experiment_node(ctx, tree, args.id)
    journal_user_context(ctx, node.experiment, args)
    node.experiment.notes = content
    forest.save()
    if ctx.output.use_json:
      ctx.output.result({'bytes_written': len(content.encode('utf-8'))})
      return
    ctx.output.info(f'Notes written for experiment {args.id}')

  def _read_file(self, ctx: CLIContext, path: Path) -> str:
    """Read and validate a UTF-8 text file for notes content.

    Args:
      ctx: CLI context for error reporting.
      path: File path to read.

    Returns:
      Decoded UTF-8 string content of the file.
    """
    if not path.exists():
      ctx.fail(f'File not found: {path}')
    raw = path.read_bytes()
    if b'\x00' in raw[:BINARY_SNIFF_BYTES]:
      ctx.fail('Binary files not supported for notes. Use a text file.')
    try:
      return raw.decode('utf-8')
    except UnicodeDecodeError:
      ctx.fail(f'File is not valid UTF-8: {path}')


class ExperimentNotes(Command):
  """Manage experiment notes (show, write)."""

  name = 'notes'
  help = 'Manage experiment notes'

  def __init__(self) -> None:
    """Wire notes subcommands (show, write)."""
    super().__init__()
    self.show = ExperimentNotesShow()
    self.write = ExperimentNotesWrite()


class ExperimentDeploy(Command):
  """Deploy an experiment under a named deployment label.

  Usage: ``experiment deploy <id> --as <deployment-name> [--replace]``

  Persists ``deployed_as`` on the Node. Deployment names must be
  forest-wide unique: attempting to deploy a second experiment with
  the same name while another node already holds it fails unless
  ``--replace`` is passed. Deploying the same experiment again with
  the same name is idempotent (no-op). Deploying an already-deployed
  experiment under a different name fails (explicit undeploy required).

  When ``--replace`` is used and another experiment holds the label,
  that experiment's label is cleared and a deployment context entry is
  added to both the newly deployed and the previously deployed experiment.
  Cross-tree replacement is supported (the previous holder may live in
  any tree in the forest).

  Resolves experiments cross-tree via ``Forest.find_experiment`` so agents
  need not switch trees before deploying.
  """

  name = 'deploy'
  help = 'Deploy an experiment under a named label'
  experiment_id = Argument('id', help='experiment ID to deploy')
  deploy_name = Argument('--as', required=True, dest='deploy_name', help='deployment name')
  replace_flag = Flag('--replace', help='take label from another experiment if held')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Set deployment label on the node and persist the forest."""
    forest = load_forest(ctx)

    result_pair = forest.find_experiment(args.id)
    if result_pair is None:
      ctx.fail(
        f'Experiment {args.id!r} not found in any tree. '
        'Run: autopilot query --all-trees --json to list available experiments.'
      )
    node, _ = result_pair

    deploy_name = args.deploy_name

    if node.deployed_as == deploy_name:
      ctx.output.result(
        {
          'ok': True,
          'experiment_id': args.id,
          'deployed_as': deploy_name,
          'idempotent': True,
        }
      )
      return

    if node.deployed_as is not None and node.deployed_as != deploy_name:
      ctx.fail(
        f'experiment {args.id!r} is already deployed as {node.deployed_as!r}; '
        f'cannot redeploy under a different name without undeploying first'
      )

    try:
      previous = forest.deploy(node, deploy_name, replace=args.replace)
    except ValueError as exc:
      ctx.fail(str(exc))
      return

    journal_user_context(ctx, node.experiment, args)

    node.experiment.add_context(
      f'deployed as {deploy_name}',
      source='deployment',
      metadata=DecisionEntry.deployment(
        label=deploy_name,
        experiment_id=args.id,
      ),
    )

    result: dict[str, Any] = {
      'ok': True,
      'experiment_id': args.id,
      'deployed_as': deploy_name,
    }

    if previous is not None:
      previous.experiment.add_context(
        f'deployment label {deploy_name!r} transferred to {args.id!r}',
        source='deployment',
        metadata=DecisionEntry.deployment(
          label=deploy_name,
          experiment_id=args.id,
          previous_id=previous.experiment.id,
        ),
      )
      result['replaced_experiment_id'] = previous.experiment.id

    forest.save()

    prev_id = previous.experiment.id if previous is not None else None
    action = 'replace' if previous is not None else 'deploy'
    log = deployment_log_for_workspace(ctx.config.workspace)
    emit_deployment_event(
      log,
      label=deploy_name,
      experiment_id=args.id,
      action=action,
      previous_experiment_id=prev_id,
      context=ctx.context,
    )

    ctx.output.result(result)


class ExperimentUndeploy(Command):
  """Remove a deployment label from whichever experiment holds it.

  Usage: ``experiment undeploy <label>``

  Scans all trees in the forest for a node with ``deployed_as == label``
  and clears it. Fails with non-zero exit when no node holds the label.
  Journals a deployment context entry on the affected experiment and
  persists the forest. Events are written only after successful persistence.
  """

  name = 'undeploy'
  help = 'Remove a deployment label'
  label = Argument('label', help='deployment label to clear')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Clear the deployment label and persist the forest."""
    forest = load_forest(ctx)

    try:
      cleared = forest.undeploy(args.label)
    except ValueError as exc:
      ctx.fail(str(exc))
      return
    if cleared is None:
      ctx.fail(
        f'no experiment is deployed as {args.label!r}; '
        f'check the label name or list deployments with query --deployed'
      )
      return

    journal_user_context(ctx, cleared.experiment, args)

    cleared.experiment.add_context(
      f'undeployed label {args.label}',
      source='deployment',
      metadata=DecisionEntry.deployment(
        label=args.label,
        experiment_id=cleared.experiment.id,
      ),
    )

    forest.save()

    log = deployment_log_for_workspace(ctx.config.workspace)
    emit_deployment_event(
      log,
      label=args.label,
      experiment_id=cleared.experiment.id,
      action='undeploy',
      context=ctx.context,
    )

    ctx.output.result(
      {
        'ok': True,
        'label': args.label,
        'experiment_id': cleared.experiment.id,
      }
    )


class ExperimentDeployLog(Command):
  """Show deployment event history from the workspace JSONL log.

  Lists deployment events (deploy, undeploy, replace) as a text table or
  ``--json`` envelope. Read-only and context-exempt.

  Use ``--label`` to filter events by deployment label. When omitted,
  all events are returned in chronological order (oldest first).
  """

  name = 'deploy-log'
  help = 'Show deployment event history'
  label = Argument('--label', default=None, help='filter events by deployment label')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Query and display deployment events."""
    autopilot_dir = ctx.autopilot_dir
    if not autopilot_dir.is_dir():
      ctx.fail(
        f'no .autopilot directory at {ctx.config.workspace}; run "autopilot workspace init" first'
      )

    log = deployment_log_for_workspace(ctx.config.workspace)
    events = log.query(label=args.label)

    if not ctx.output.use_json:
      if not events:
        ctx.output.info('No deployment events found.')
      else:
        rows = [
          {
            'timestamp': e.timestamp[:19],
            'label': e.label,
            'experiment_id': e.experiment_id,
            'action': e.action,
            'previous': e.previous_experiment_id or '',
          }
          for e in events
        ]
        ctx.output.table(rows, ['timestamp', 'label', 'experiment_id', 'action', 'previous'])

    ctx.output.result(
      {
        'ok': True,
        'events': [e.to_dict() for e in events],
      }
    )
