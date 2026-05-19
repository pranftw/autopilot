"""Tree management: list, create, show, describe, and switch exploration trees.

Subcommands:
  autopilot tree list [--json]         -- list all trees in the forest
  autopilot tree create <name> [--description ...] [--json] -- create a new tree
  autopilot tree show [name] [--json]  -- show tree structure (render or to_dict)
  autopilot tree describe [name] [--json] -- show tree metadata (lightweight)
  autopilot tree switch <name> [--no-checkout] [--bind] [--json] -- switch active tree

All subcommands support --json for agent-friendly structured output.
Path resolution via ctx.config (no paths.* calls).

Import terminal modules directly (e.g. ``from autopilot.core.module.module import Module``),
not package facade -- there is no ``__init__.py``.
"""

from autopilot.ai.environment import bind_path_parameters
from autopilot.cli.command import CLI, Command
from autopilot.cli.commands.store.helpers import register_parameters_from_latest_manifest
from autopilot.cli.context import CLIContext
from autopilot.cli.helpers import load_forest, require_active_tree, require_store_branch
from autopilot.cli.primitives import Argument, Flag
from autopilot.core.errors import ConfigError, StoreError
from autopilot.core.forest import validate_tree_name
from autopilot.core.module.module import Module
from pathlib import Path
import argparse


class TreeList(Command):
  """List all exploration trees in the forest."""

  name = 'list'
  help = 'List exploration trees'

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """List trees in the forest with optional active marker."""
    forest = load_forest(ctx)
    trees = forest.list_trees()
    active = forest.active
    active_name = active.name if active else None

    if ctx.output.use_json:
      rows = []
      for t in trees:
        node_count = t.query().count()
        rows.append(
          {
            'name': t.name,
            'description': t.description,
            'nodes': node_count,
            'active': t.name == active_name,
          }
        )
      ctx.output.result({'trees': rows})
      return

    rows = []
    for t in trees:
      node_count = t.query().count()
      marker = '*' if t.name == active_name else ''
      rows.append(
        {
          'name': t.name,
          'description': '' if t.description is None else t.description,
          'nodes': str(node_count),
          'active': marker,
        }
      )
    ctx.output.table(rows, ['name', 'description', 'nodes', 'active'])


class TreeCreate(Command):
  """Create a new exploration tree.

  ``--description`` is optional; when omitted, the global ``--context``
  value is used as the tree description so agents that always supply
  ``--context`` get a meaningful tree purpose without an extra flag.
  """

  name = 'create'
  help = 'Create a new tree'
  tree_name = Argument('name', help='tree name')
  description = Argument('--description', default=None, help='tree description')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Create a tree, switch to it, and report success.

    Validates the tree name via ``validate_tree_name`` before reaching
    ``forest.create_tree`` so validation failures produce explicit
    ``ctx.fail()`` messages instead of generic dispatch errors.

    Description resolution: explicit ``--description`` wins; otherwise
    falls back to global ``--context``.
    """
    forest = load_forest(ctx)
    name = args.name
    try:
      validate_tree_name(name)
    except ValueError as exc:
      ctx.fail(
        f'{exc}; provide a non-empty tree name (ASCII letters, digits, hyphen, underscore, dot)'
      )
    desc = args.description
    if desc is None and ctx.context is not None:
      desc = ctx.context
    try:
      forest.create_tree(name, description=desc)
    except ValueError as exc:
      ctx.fail(str(exc))
    forest.switch(name)
    ctx.output.result({'ok': True, 'tree': name})


class TreeShow(Command):
  """Show tree structure. Omit name to show active tree."""

  name = 'show'
  help = 'Show tree structure'
  tree_name = Argument('name', nargs='?', default=None, help='tree name (default: active)')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Render or JSON-serialize the named tree or the active tree."""
    forest = load_forest(ctx)
    name = args.name
    if name:
      tree = forest.get_tree(name)
      if tree is None:
        ctx.fail(f'tree {name!r} not found')
    else:
      tree = require_active_tree(ctx, forest)

    if ctx.output.use_json:
      ctx.output.result(tree.to_dict())
      return

    ctx.output.info(tree.render())


DISK_STATE_ADVISORY = (
  'note: switching the active tree does not sync working tree files '
  'or PathParameter binds; run checkout or activate the environment '
  'to align disk state with the new tree'
)


class TreeSwitch(Command):
  """Switch the active exploration tree.

  By default, switching also runs ``store checkout`` for the HEAD
  experiment's tip snapshot so that working tree files are synced to
  the new active tree.

  Pass ``--no-checkout`` to opt out of the automatic checkout; in that
  case a disk-state advisory is emitted recommending a manual
  ``store checkout`` or environment activation.

  With ``--bind``, after checkout, ``PathParameter`` instances on the
  project module are rebound to worktree-relative paths using
  :func:`autopilot.ai.environment.bind_path_parameters`. Requires
  ``-p <project>`` global flag so the module can be loaded.
  ``--bind`` combined with ``--no-checkout`` is an error.
  """

  name = 'switch'
  help = 'Switch active tree (auto-checkouts by default; --no-checkout to skip)'
  tree_name = Argument('name', help='tree name to switch to')
  no_checkout = Flag(
    '--no-checkout',
    help='skip automatic checkout after switching (emits disk-state advisory)',
  )
  bind = Flag(
    '--bind',
    help='rebind PathParameter roots after checkout (requires -p <project>)',
  )

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Switch the forest's active tree, checking out HEAD tip by default.

    Default: runs ``store checkout`` for the HEAD experiment's latest
    snapshot after switching. Pass ``--no-checkout`` to skip.
    """
    skip_checkout = args.no_checkout is True
    run_checkout_after_switch = not skip_checkout

    if args.bind and skip_checkout:
      ctx.fail(
        '--bind requires checkout; remove --no-checkout or omit --bind '
        '(e.g. tree switch <name> --bind)'
      )

    forest = load_forest(ctx)
    forest.switch(args.name)

    if not run_checkout_after_switch:
      ctx.output.info(DISK_STATE_ADVISORY)
      ctx.output.result({'ok': True, 'active': args.name})
      return

    tree = forest.active
    if tree is None:
      ctx.fail(f'cannot checkout: tree {args.name!r} has no active state after switch')

    if tree.head is None:
      ctx.fail('cannot checkout: tree has no HEAD experiment; add an experiment first')

    store = forest.store
    experiment_id = tree.head
    branch = require_store_branch(ctx, store, experiment_id)

    latest_epoch = branch['latest_epoch']
    if latest_epoch < 0:
      advisory = f'skipped checkout: no snapshots on branch {experiment_id!r}'
      ctx.output.info(advisory)
      ctx.output.result({'ok': True, 'active': args.name})
      return

    try:
      register_parameters_from_latest_manifest(store, experiment_id)
    except (StoreError, OSError):
      advisory = (
        f'skipped checkout: could not rehydrate parameter schema for '
        f'{experiment_id!r}; run store checkout with -p <project> '
        f'to materialize files'
      )
      ctx.output.info(advisory)
      ctx.output.result({'ok': True, 'active': args.name})
      return

    try:
      store.checkout(experiment_id, latest_epoch, context=ctx.context)
    except StoreError as exc:
      ctx.fail(f'checkout failed for {experiment_id!r} at epoch {latest_epoch}: {exc}')

    bound_count = 0
    if args.bind:
      bound_count = self._bind_parameters(ctx, store)

    ctx.output.info(f'checked out experiment {experiment_id!r} at epoch {latest_epoch}')
    result_payload: dict = {
      'ok': True,
      'active': args.name,
      'checkout': True,
      'experiment_id': experiment_id,
      'epoch': latest_epoch,
    }
    if args.bind:
      result_payload['bind'] = True
      result_payload['bound_parameters'] = bound_count
    ctx.output.result(result_payload)

  def _bind_parameters(self, ctx: CLIContext, store: object) -> int:
    """Load the project module and bind its PathParameters.

    Args:
      ctx: CLI context for project/module resolution.
      store: Store instance (unused directly; module may reference it).

    Returns:
      Count of PathParameters that were bound.
    """
    if ctx.project is None:
      ctx.fail(
        '--bind requires -p <project> to load the module; '
        'pass -p <project> --workspace <path> along with --bind'
      )

    module = self._load_project_module(ctx)
    if module is None:
      ctx.fail(
        'could not load module for project; ensure the project CLI defines a module via run()'
      )

    cfg_root = Path(ctx.config.root)
    try:
      rebound = bind_path_parameters(module, cfg_root, cfg_root)
    except ConfigError as exc:
      ctx.fail(f'bind failed: {exc}')
    return len(rebound)

  def _load_project_module(self, ctx: CLIContext) -> Module | None:
    """Attempt to load the project module via the CLI project registry.

    Args:
      ctx: CLI context carrying project slug.

    Returns:
      Module instance, or None if the project CLI has no module configured.
    """
    project_cls = CLI.lookup_project(ctx.project)
    if project_cls is None:
      return None
    project_cli = project_cls()
    return project_cli.module


class TreeDescribe(Command):
  """Show lightweight metadata for one exploration tree."""

  name = 'describe'
  help = 'Show tree metadata (name, description, head, node count)'
  tree_name = Argument('name', nargs='?', default=None, help='tree name (default: active)')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Emit tree metadata without full node serialization."""
    forest = load_forest(ctx)
    name = args.name
    if name:
      tree = forest.get_tree(name)
      if tree is None:
        ctx.fail(f'tree {name!r} not found')
    else:
      tree = require_active_tree(ctx, forest)

    payload = {
      'name': tree.name,
      'description': tree.description,
      'head': tree.head,
      'node_count': tree.query().count(),
      'created_at': None,
    }

    if ctx.output.use_json:
      ctx.output.result(payload)
      return

    desc = '' if tree.description is None else tree.description
    ctx.output.info(f'name: {tree.name}')
    ctx.output.info(f'description: {desc}')
    ctx.output.info(f'head: {tree.head}')
    ctx.output.info(f'node_count: {payload["node_count"]}')


class TreeRemove(Command):
  """Remove a tree from the forest (irreversible).

  Calls ``Forest.remove_tree`` which persists under forest lock.
  If the removed tree was active, the active selection is cleared.
  """

  name = 'remove'
  help = 'Remove a tree from the forest (irreversible)'
  tree_name = Argument('name', help='tree name to remove')

  def forward(self, ctx: CLIContext, args: argparse.Namespace) -> None:
    """Remove the named tree and persist the forest."""
    forest = load_forest(ctx)
    name = args.name
    try:
      forest.remove_tree(name)
    except ValueError as exc:
      ctx.fail(f'{exc}; use tree list to see available trees')
    ctx.output.result({'ok': True, 'removed': name})


class TreeCommand(Command):
  """Manage exploration trees."""

  name = 'tree'
  help = 'Manage exploration trees'

  def __init__(self) -> None:
    """Wire tree subcommands (list, create, show, describe, switch, remove)."""
    super().__init__()
    self.list_cmd = TreeList()
    self.create = TreeCreate()
    self.show = TreeShow()
    self.describe = TreeDescribe()
    self.switch = TreeSwitch()
    self.remove = TreeRemove()
