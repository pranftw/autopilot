"""Collection of exploration Trees.

Forest manages multiple Trees within a project. Each Tree represents
an independent exploration direction. Forest coordinates with Store
for persistence via save_state_dict/load_state_dict.
"""

from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from autopilot.core.store.base import Store
from autopilot.core.store.types import TAG_NAME_ALLOWED_RE, TAG_NAME_MAX_LEN
from autopilot.core.tree import Tree
from collections.abc import Callable
from typing import Any


def validate_tree_name(name: str) -> None:
  """Validate a tree name using the same rules as store tag names.

  Args:
    name: Proposed tree name.

  Raises:
    ValueError: If ``name`` is empty, whitespace-only, too long, or contains
      invalid characters (including path-like or dotted-edge cases matching
      ``validate_tag_name``).
  """
  stripped = name.strip()
  if not stripped:
    msg = 'tree name must not be empty or whitespace-only'
    raise ValueError(msg)
  if stripped != name:
    msg = 'tree name must not have leading or trailing whitespace'
    raise ValueError(msg)
  if len(stripped) > TAG_NAME_MAX_LEN:
    msg = f'tree name exceeds {TAG_NAME_MAX_LEN} characters: {stripped!r}'
    raise ValueError(msg)
  if not TAG_NAME_ALLOWED_RE.match(stripped):
    msg = (
      f'tree name {stripped!r} contains invalid characters. '
      'Allowed: ASCII letters, digits, hyphen (-), underscore (_), and dot (.).'
    )
    raise ValueError(msg)
  if stripped.startswith('.') or stripped.endswith('.'):
    msg = f'tree name {stripped!r} must not start or end with "."'
    raise ValueError(msg)
  if '..' in stripped:
    msg = f'tree name {stripped!r} must not contain ".."'
    raise ValueError(msg)


class Forest:
  """Collection of exploration Trees with cross-tree queries.

  Not thread-safe. Concurrent mutations (add/remove/switch) may corrupt
  state. query() during mutation may raise RuntimeError. Single-writer model.

  Manages multiple Trees within a project. Each Tree represents
  an independent exploration direction. Coordinates with Store
  for persistence. Use the ``trees`` property for ergonomic enumeration
  (alias of ``list_trees()``).

  Args:
    store: shared persistence layer
  """

  def __init__(self, store: Store) -> None:
    """Create an empty forest using the given store for persistence.

    Args:
      store: Shared persistence layer for save/load.
    """
    self.store = store
    self._trees: dict[str, Tree] = {}
    self._active_name: str | None = None

  def create_tree(self, name: str, description: str | None = None) -> Tree:
    """Create a new tree registered under ``name``.

    Args:
      name: Unique tree name within this forest.
      description: Optional human-readable description.

    Returns:
      The newly created ``Tree``.

    Raises:
      ValueError: When ``name`` contains invalid characters, exceeds length
        limits, or already exists. Tree names follow the same charset/length
        rules as store tag names.
    """
    validate_tree_name(name)
    if name in self._trees:
      msg = f'tree {name!r} already exists'
      raise ValueError(msg)
    tree = Tree(name=name, store=self.store, description=description)
    self._trees[name] = tree
    return tree

  def get_tree(self, name: str) -> Tree | None:
    """Look up a tree by name.

    Returns:
      The ``Tree`` instance, or ``None`` when missing.
    """
    return self._trees.get(name)

  def list_trees(self) -> list[Tree]:
    """Return every tree in arbitrary dict iteration order.

    Returns:
      List of ``Tree`` instances.
    """
    return list(self._trees.values())

  @property
  def trees(self) -> list[Tree]:
    """All trees (alias of list_trees() for discoverability).

    Returns:
      List of ``Tree`` instances from ``list_trees()``.
    """
    return self.list_trees()

  def remove_tree(self, name: str) -> None:
    """Remove a tree and clear active selection when it matches.

    Args:
      name: Tree name to drop.

    Raises:
      ValueError: When no tree is registered under ``name``.
    """
    if name not in self._trees:
      msg = f'tree {name!r} not found'
      raise ValueError(msg)
    del self._trees[name]
    if self._active_name == name:
      self._active_name = None

  @property
  def active(self) -> Tree | None:
    """Currently active tree, or None."""
    if self._active_name is None:
      return None
    return self._trees.get(self._active_name)

  def switch(self, name: str) -> Tree:
    """Set the active tree to ``name``.

    Args:
      name: Name of an existing tree.

    Returns:
      The tree that became active.

    Raises:
      ValueError: When ``name`` is not registered.
    """
    if name not in self._trees:
      msg = f'tree {name!r} not found'
      raise ValueError(msg)
    self._active_name = name
    return self._trees[name]

  def query(self) -> QueryBuilder:
    """Cross-tree query over nodes from ALL trees.

    Deduplicates by ``experiment_id``: when the same id appears in
    multiple trees, the first occurrence (in dict iteration order of
    ``_trees``) is kept and subsequent duplicates are silently dropped.
    This matches the deterministic ordering guaranteed by Python 3.7+
    dict insertion order.

    Within a single tree, duplicate ids are already rejected by
    ``Tree.add``.

    Returns:
      ``QueryBuilder`` spanning deduplicated nodes from every tree.
    """
    all_nodes: list[Node] = []
    all_nodes_map: dict[str, Node] = {}
    for tree in self._trees.values():
      for node in tree.query().all():
        eid = node.experiment.id
        if eid in all_nodes_map:
          continue
        all_nodes.append(node)
        all_nodes_map[eid] = node
    return QueryBuilder(all_nodes, all_nodes_map.get)

  def state_dict(self) -> dict[str, Any]:
    """Serialization for persistence.

    Returns:
      Dict with active tree name and per-tree serialized state.
    """
    return {
      'active': self._active_name,
      'trees': [tree.state_dict() for tree in self._trees.values()],
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    """Restore forest from state dict. Resolves experiment IDs to objects."""
    # active may be None when no tree is selected
    self._active_name = state.get('active')
    self._trees = {}

    tree_states = state['trees']
    for tree_state in tree_states:
      name = tree_state['name']
      tree = Tree(
        name=name,
        store=self.store,
        description=tree_state['description'],
      )
      experiments: dict[str, Experiment] = {}
      for nd in tree_state['nodes']:
        exp_id = nd['experiment']
        if exp_id not in experiments:
          exp = Experiment(experiment_id=exp_id)
          experiments[exp_id] = exp

      def make_resolver(
        exps: dict[str, Experiment],
        bound_tree: Tree = tree,
      ) -> Callable[[str], Experiment | Node]:
        def resolver(id_str: str) -> Experiment | Node:
          node = bound_tree.get(id_str)
          if node is not None:
            return node
          if id_str in exps:
            return exps[id_str]
          msg = f'cannot resolve id: {id_str!r}'
          raise ValueError(msg)

        return resolver

      tree.load_state_dict(tree_state, make_resolver(experiments))
      self._trees[name] = tree

  def save(self) -> None:
    """Convenience: persist forest via store."""
    self.store.save_state_dict(self.state_dict())

  def load(self) -> None:
    """Convenience: hydrate forest from store."""
    state = self.store.load_state_dict()
    if state is not None:
      self.load_state_dict(state)

  def to_dict(self) -> dict[str, Any]:
    """Structured dict for JSON serialization.

    Returns:
      Dict with active name and ``tree.to_dict()`` payloads.
    """
    return {
      'active': self._active_name,
      'trees': [tree.to_dict() for tree in self._trees.values()],
    }

  def undeploy(self, label: str) -> Node | None:
    """Clear ``deployed_as`` for the node that holds ``label``, if any.

    Scans all trees in the forest. At most one node should hold a given
    label (forest-wide uniqueness invariant). The first match is cleared.

    Args:
      label: Deployment label to clear (forest-wide).

    Returns:
      The node that was cleared, or ``None`` when no node held ``label``.

    Raises:
      ValueError: When ``label`` is empty or whitespace-only.
    """
    if not label or not label.strip():
      msg = 'deployment label must be non-empty'
      raise ValueError(msg)
    for tree in self.list_trees():
      for node in tree.query().all():
        if node.deployed_as == label:
          node.deployed_as = None
          return node
    return None

  def deploy(self, node: Node, label: str, *, replace: bool = False) -> Node | None:
    """Assign ``label`` to ``node``, enforcing forest-wide uniqueness.

    When ``replace`` is True, clears any existing holder of ``label`` across the
    forest before assigning. Callers must ``save()`` to persist.

    When another node already holds ``label``:
      - ``replace=False``: raises ``ValueError`` with an actionable message.
      - ``replace=True``: clears ``deployed_as`` on all current holders, then
        assigns ``label`` to ``node``. Returns the last cleared previous holder.

    If ``node`` already holds ``label``, the assignment is still applied
    (idempotent for the label map).

    Args:
      node: Node whose experiment receives the deployment label.
      label: Non-empty deployment name.
      replace: When ``True``, clear any existing holder of ``label`` first.

    Returns:
      The previous holder when ``replace=True`` cleared one; otherwise ``None``.

    Raises:
      ValueError: When ``label`` is empty/whitespace-only, or when ``label``
        is already held and ``replace`` is ``False``.
    """
    if not label or not label.strip():
      msg = 'deployment label must be non-empty'
      raise ValueError(msg)
    previous: Node | None = None
    for tree in self.list_trees():
      for n in tree.query().all():
        if n.deployed_as != label or n is node:
          continue
        if not replace:
          msg = (
            f'deployment label {label!r} already used by experiment '
            f'{n.experiment.id!r}. Use --replace to swap.'
          )
          raise ValueError(msg)
        n.deployed_as = None
        previous = n
    node.deployed_as = label
    return previous

  def find_experiment(self, experiment_id: str) -> tuple[Node, Tree] | None:
    """Locate a node by experiment id across all trees.

    Checks ``self.active`` first when set, then every other tree in
    ``list_trees()`` order (active tree is not scanned twice).

    Args:
      experiment_id: Experiment identifier to search for.

    Returns:
      ``(node, tree)`` when found; ``None`` when absent. When the same
      experiment id exists in multiple trees, the first match in search
      order wins (active tree first, then remaining trees in dict/list order).
    """
    active = self.active
    if active is not None:
      node = active.get(experiment_id)
      if node is not None:
        return node, active
    for tree in self._trees.values():
      if tree is active:
        continue
      node = tree.get(experiment_id)
      if node is not None:
        return node, tree
    return None

  def __repr__(self) -> str:
    """Return tree count and active name for debugging."""
    return f'Forest(trees={len(self._trees)}, active={self._active_name!r})'
