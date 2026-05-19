"""Experiment exploration DAG (not a git directory tree object).

Nodes are experiments; edges encode fork/parent relationships (see Node).

Tree manages a DAG of Nodes. Each node holds a real Experiment object.
Queries derive everything at execution time via QueryBuilder.

Forest-level HEAD is Tree._head. refs.json HEAD (FileStore) updates inside
store.checkout. Tree.checkout sets Tree._head then calls store.checkout so
both stay in sync.

Branching precondition on add(): if parent is provided, the parent
experiment must be terminal (completed, failed, or cancelled). This
enforces the immutability invariant -- a parent's files won't change
after a child branches from it.
"""

from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from autopilot.core.store.base import Store
from collections.abc import Callable
from typing import Any
import logging

logger = logging.getLogger(__name__)


class Tree:
  """Single experiment DAG for exploration tracking.

  The name **Tree** refers to this experiment exploration graph only; it is
  not a Git repository tree or worktree. Autopilot's file worktrees live
  under ``Store`` / ``FileStore``, not this type.

  Manages a set of Nodes forming a DAG. Each node holds a real ``Experiment``
  object (not a bare string handle). Queries derive everything at execution
  time via ``QueryBuilder`` -- no cached aggregates.

  Attributes:
    name: Stable tree identifier within a ``Forest`` or project overlay.
    store: ``Store`` used for checkpoint/checkout coordination.
    description: Optional human/agent-facing annotation.
    on_change: Optional callback invoked after mutating ops (``add``, ``remove``,
      ``update``).
    nodes: Read-only property returning a shallow copy of the ``experiment_id``
      -> ``Node`` mapping. Mutating the returned dict does not affect the tree.
    _head: Current ``experiment_id`` for checkout-oriented workflows, if set.

  Args:
    name: Tree identity string.
    store: ``Store`` instance for checkout state restoration.
    description: Optional agent-friendly context.
    on_change: Optional callback invoked after state-mutating operations
      (add, remove, update). ``FileForest`` sets this to ``self.save``.
  """

  def __init__(
    self,
    name: str,
    store: Store,
    description: str | None = None,
    on_change: Callable[[], None] | None = None,
  ) -> None:
    """Create an empty DAG with optional mutation callback.

    Args:
      name: Stable identifier for this tree within a forest.
      store: Store coordinating snapshots and checkout.
      description: Optional annotation for agents or UIs.
      on_change: Optional zero-arg hook after ``add``/``remove``/``update``.
    """
    self.name = name
    self.store = store
    self.description = description
    self.on_change = on_change
    self._nodes: dict[str, Node] = {}
    self._head: str | None = None

  def add(self, node: Node) -> None:
    """Register node in tree and optionally auto-branch in the store.

    Accepts a Node wrapping an Experiment, not a bare Experiment.
    Use Node(experiment=exp, parent_id=parent) to construct.

    When a ``store`` is attached and the node has a parent,
    ``store.branch(node.experiment.id)`` is attempted before committing
    the node into ``_nodes``. If the branch call fails (e.g. no HEAD
    set, branch already exists, store not initialized), the error is
    logged at DEBUG and the node is still added (BUG-035). Store
    branches are also auto-created by ``snapshot(exp_id, 0)`` in
    normal Trainer flows, so auto-branch failure is non-fatal.

    The ``remove()`` method does **not** prune the corresponding store
    branch or its blobs (BUG-041). Use ``store.prune_orphans()`` to
    clean up orphaned blobs.

    Validates:
    - no duplicate experiment.id
    - parent exists in tree if specified on node
    - parent experiment must be terminal if parent is provided

    Raises:
      ValueError: On duplicate id, missing parent, or non-terminal parent.
    """
    eid = node.experiment.id
    if eid in self._nodes:
      msg = f'duplicate experiment id: {eid!r}'
      raise ValueError(msg)
    if node.parent is not None:
      pid = node.parent.experiment.id
      if pid not in self._nodes:
        msg = f'parent {pid!r} not found in tree'
        raise ValueError(msg)
      if not node.parent.experiment.is_terminal:
        msg = (
          f'parent {pid!r} is not terminal '
          f'(status={node.parent.experiment.status.value}); '
          f'cannot branch from non-terminal experiment'
        )
        raise ValueError(msg)
      if self.store is not None:
        try:
          self.store.branch(eid)
        except (StoreError, NotImplementedError) as exc:
          logger.debug('auto-branch skipped for %r: %s', eid, exc)
    self._nodes[eid] = node
    if self.on_change is not None:
      self.on_change()

  def update(self, experiment_id: str, **kwargs: Any) -> None:
    """Update experiment fields. Only metrics and error are accepted.

    Status changes must go through Experiment lifecycle methods
    (start/complete/fail/cancel), not through Tree.update().

    Raises:
      TypeError: When disallowed keyword arguments are supplied.
      ValueError: When ``experiment_id`` is not registered.
    """
    allowed = {'metrics', 'error'}
    bad_keys = set(kwargs) - allowed
    if bad_keys:
      msg = f'unknown keyword argument(s): {", ".join(sorted(bad_keys))}'
      raise TypeError(msg)
    node = self._nodes.get(experiment_id)
    if node is None:
      msg = f'experiment {experiment_id!r} not found in tree'
      raise ValueError(msg)
    if 'metrics' in kwargs:
      node.experiment.metrics = kwargs['metrics']
    if 'error' in kwargs:
      node.experiment.error = kwargs['error']
    if self.on_change is not None:
      self.on_change()

  def remove(self, experiment_id: str, *, cascade: bool = False) -> None:
    """Remove an experiment from the tree.

    If cascade=False and node has children, raises ValueError. If
    cascade=True, removes node and all descendants. If the removed node
    was HEAD, HEAD is cleared to None.

    Note: removing a tree node does **not** delete store branches or blobs.
    Orphan pruning (``store.prune_orphans()``) is out of scope for this
    operation (BUG-041).

    Raises:
      ValueError: When the id is unknown, or when children exist without ``cascade``.
    """
    if experiment_id not in self._nodes:
      msg = f'experiment {experiment_id!r} not found in tree'
      raise ValueError(msg)
    children = [
      n
      for n in self._nodes.values()
      if n.parent is not None and n.parent.experiment.id == experiment_id
    ]
    if children and not cascade:
      msg = 'cannot remove node with children; use cascade=True'
      raise ValueError(msg)
    if cascade:
      removed_ids = self._collect_descendants(experiment_id)
      removed_ids.add(experiment_id)
      for eid in removed_ids:
        del self._nodes[eid]
    else:
      removed_ids = {experiment_id}
      del self._nodes[experiment_id]
    if self._head in removed_ids:
      self._head = None
    if self.on_change is not None:
      self.on_change()

  def get(self, experiment_id: str) -> Node | None:
    """Lookup node by experiment.id.

    Returns:
      The ``Node`` when present, else ``None``.
    """
    return self._nodes.get(experiment_id)

  def roots(self) -> list[Node]:
    """Nodes with no parent.

    Returns:
      List of root ``Node`` instances.
    """
    return [n for n in self._nodes.values() if n.parent is None]

  @property
  def head(self) -> str | None:
    """Current active experiment_id, or None."""
    return self._head

  @head.setter
  def head(self, value: str | None) -> None:
    """Set the active experiment_id without triggering store checkout.

    Use ``checkout()`` when filesystem alignment via ``Store.checkout``
    is needed. This setter only updates the in-memory HEAD pointer
    (e.g. after ``experiment add`` to track the latest node).
    """
    self._head = value

  @property
  def nodes(self) -> dict[str, Node]:
    """Shallow copy of experiment_id -> Node mapping.

    Mutating the returned dict does not mutate the tree's internal
    registry. Use ``add()`` / ``remove()`` for state changes.

    Returns:
      New dict with the same Node references as internal storage.
    """
    return dict(self._nodes)

  def checkout(self, experiment_id: str, context: str | None = None) -> None:
    """Set Tree._head and call Store.checkout (refs HEAD + filesystem restore).

    Keeps forest head and refs.json HEAD aligned; delegate here instead of
    mutating Tree._head without store.checkout.

    Args:
      experiment_id: Experiment to checkout.
      context: Optional reason/provenance string for audit traceability.

    Raises:
      ValueError: When ``experiment_id`` is not registered in this tree.
    """
    node = self._nodes.get(experiment_id)
    if node is None:
      msg = f'experiment {experiment_id!r} not found in tree'
      raise ValueError(msg)
    self._head = experiment_id
    self.store.checkout(experiment_id, node.experiment.epoch, context=context)

  def query(self) -> QueryBuilder:
    """Create QueryBuilder scoped to this tree's nodes.

    Returns:
      Builder bound to this tree's ``Node`` list and id resolver.
    """
    return QueryBuilder(
      list(self._nodes.values()),
      self._nodes.get,
    )

  def render(self) -> str:
    """Markdown tree visualization (recursive, indented).

    Returns:
      Multi-line string, or a placeholder when the tree has no roots.
    """
    root_nodes = self.roots()
    if not root_nodes:
      return f'{self.name}\n(empty tree)'
    lines = [self.name]
    for i, root in enumerate(root_nodes):
      is_last = i == len(root_nodes) - 1
      self._render_node(root, lines, '', is_last)
    return '\n'.join(lines)

  def to_dict(self) -> dict[str, Any]:
    """Structured JSON with all nodes serialized via Node.to_dict.

    Returns:
      Dict including name, description, head, and serialized nodes.
    """
    return {
      'name': self.name,
      'description': self.description,
      'head': self._head,
      'nodes': [n.to_dict() for n in self._nodes.values()],
    }

  def state_dict(self) -> dict[str, Any]:
    """Serialization for persistence.

    Returns:
      Same payload as ``to_dict`` for checkpointing.
    """
    return self.to_dict()

  def load_state_dict(
    self,
    state: dict[str, Any],
    resolver: Callable[[str], Experiment | Node],
  ) -> None:
    """Restore tree from state dict. Resolver hydrates IDs to objects."""
    self.name = state['name']
    self.description = state['description']
    # head may be None when no experiment is checked out
    self._head = state.get('head')
    self._nodes = {}
    node_dicts = state['nodes']
    self._load_nodes_topological(node_dicts, resolver)

  def _load_nodes_topological(
    self,
    node_dicts: list[dict[str, Any]],
    resolver: Callable[[str], Experiment | Node],
  ) -> None:
    """Load nodes in topological order (parents before children).

    Raises:
      ValueError: On cycles or permanently unresolved parent references.
    """
    remaining = list(node_dicts)
    resolved_nodes: dict[str, Node] = {}

    def combined_resolver(id_str: str) -> Experiment | Node:
      if id_str in resolved_nodes:
        return resolved_nodes[id_str]
      return resolver(id_str)

    max_iterations = len(remaining) + 1
    for _ in range(max_iterations):
      if not remaining:
        break
      unresolved: list[dict[str, Any]] = []
      for nd in remaining:
        # parent is None for root nodes
        parent_id = nd.get('parent')
        if parent_id is not None and parent_id not in resolved_nodes:
          unresolved.append(nd)
          continue
        node = Node.from_dict(nd, combined_resolver)
        resolved_nodes[node.experiment.id] = node
        self._nodes[node.experiment.id] = node
      if len(unresolved) == len(remaining):
        msg = 'cycle detected or unresolvable parent references in tree nodes'
        raise ValueError(msg)
      remaining = unresolved

  # internal helpers

  def _collect_descendants(self, experiment_id: str) -> set[str]:
    """Collect all descendant experiment_ids recursively.

    Returns:
      Set of child and deeper experiment ids (excludes ``experiment_id``).
    """
    result: set[str] = set()
    stack = [experiment_id]
    while stack:
      eid = stack.pop()
      for n in self._nodes.values():
        if n.parent is not None and n.parent.experiment.id == eid and n.experiment.id not in result:
          result.add(n.experiment.id)
          stack.append(n.experiment.id)
    return result

  def _render_node(
    self,
    node: Node,
    lines: list[str],
    prefix: str,
    is_last: bool,
  ) -> None:
    connector = '+-- '
    exp = node.experiment
    metric_label = ''
    if exp.metrics:
      metric_label = ' ' + ' '.join(f'{k}={v}' for k, v in exp.metrics.items())
    head_marker = ' [HEAD]' if self._head == exp.id else ''
    lines.append(f'{prefix}{connector}{exp.id} ({exp.status.value}){metric_label}{head_marker}')
    children = [
      n for n in self._nodes.values() if n.parent is not None and n.parent.experiment.id == exp.id
    ]
    child_prefix = prefix + ('    ' if is_last else '|   ')
    for i, child in enumerate(children):
      child_is_last = i == len(children) - 1
      self._render_node(child, lines, child_prefix, child_is_last)

  def __repr__(self) -> str:
    """Return name and node count for debugging."""
    return f'Tree(name={self.name!r}, nodes={len(self._nodes)})'
