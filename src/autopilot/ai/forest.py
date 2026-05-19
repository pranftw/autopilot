"""File-backed Forest implementation.

FileForest persists tree structure to .autopilot/forest.json via Store.
Loads/saves automatically. Creates Trees with Store reference.

Auto-saves forest.json after every state-mutating operation
(create_tree, switch, and any Tree.add/remove/update called on a
tree owned by FileForest). Implementation: Tree gets on_change
callback set to self.save on all trees managed by FileForest.

``FileForest.load`` accepts an injectable ``experiment_factory`` callable
to reconstruct ``Experiment`` subclasses (e.g. ``AutoPilotExperiment``)
rather than flattening to the base ``Experiment`` (BUG-040). The default
factory creates base ``Experiment`` instances and restores state from
``experiment_state``. Callers at application boundaries (e.g. CLI) can
pass a factory that produces subclass instances, avoiding string-key
registries in library code.
"""

from autopilot.ai.store.file_store import FileStore
from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.node import Node
from autopilot.core.tree import Tree
from autopilot.tracking.file_lock import AutopilotFileLock
from collections.abc import Callable
from typing import Any


class FileForest(Forest):
  """File-backed Forest that persists to forest.json via FileStore.

  Constructor takes a FileStore instance (not a Config). The store
  provides the persistence path. Example: FileForest(store=FileStore(config)).

  Constructor hydrates from forest.json if it exists. All state-mutating
  operations auto-save. Trees created by FileForest get an on_change
  callback wired to self.save for automatic persistence.

  An optional ``experiment_factory`` callable can be passed to reconstruct
  ``Experiment`` subclasses (e.g. ``AutoPilotExperiment``) rather than
  flattening to base ``Experiment`` (BUG-040). The factory receives the
  serialized ``experiment_state`` dict and must return an ``Experiment``
  instance. The default factory creates a base ``Experiment`` and restores
  state via ``load_state_dict``.

  Args:
    store: FileStore instance for persistence
    experiment_factory: Optional callable ``(node_data: dict) -> Experiment``
      for subclass hydration.
  """

  def __init__(
    self,
    store: FileStore,
    experiment_factory: Callable[[dict[str, Any]], Experiment] | None = None,
  ) -> None:
    """Hydrate forest state from disk and attach auto-save callbacks.

    Args:
      store: File store used to load and persist ``forest.json``.
      experiment_factory: Optional callable to produce ``Experiment`` subclass
        instances from serialized state. Receives the ``experiment_state`` dict.
    """
    self._experiment_factory = experiment_factory
    self.lock_timeout_s: float | None = None
    super().__init__(store)
    self.load()

  def create_tree(self, name: str, description: str | None = None) -> Tree:
    """Create a new tree with auto-save callback.

    Returns:
      New :class:`Tree` instance wired to persist on change.
    """
    tree = super().create_tree(name, description)
    tree.on_change = self.save
    self.save()
    return tree

  def switch(self, name: str) -> Tree:
    """Switch active tree and auto-save.

    Returns:
      The tree that is now active.
    """
    tree = super().switch(name)
    self.save()
    return tree

  def remove_tree(self, name: str) -> None:
    """Remove tree and auto-save."""
    super().remove_tree(name)
    self.save()

  def save(self) -> None:
    """Serialize all trees and nodes, persist via store.

    Acquires an exclusive file lock at ``store_path/forest.lock`` to
    prevent concurrent writers from corrupting the serialized file.
    """
    lock_path = self.store.config.store_path / 'forest.lock'
    with AutopilotFileLock(lock_path, self.lock_timeout_s, operation='forest_save'):
      self.store.save_state_dict(self.state_dict())

  def load(self) -> None:
    """Load from store, deserialize trees and nodes, resolve references."""
    state = self.store.load_state_dict()
    if state is None:
      return
    self._load_with_experiments(state)

  def _load_with_experiments(self, state: dict[str, Any]) -> None:
    """Deserialize state, creating real Experiment objects and resolving references.

    Uses ``_experiment_factory`` when provided to reconstruct Experiment
    subclasses (BUG-040). Falls back to base ``Experiment`` with
    ``load_state_dict`` when no factory is configured.
    """
    self._active_name = state.get('active')
    self._trees = {}

    tree_states = state['trees']
    for tree_state in tree_states:
      name = tree_state['name']
      tree = Tree(
        name=name,
        store=self.store,
        description=tree_state['description'],
        on_change=self.save,
      )

      experiments: dict[str, Experiment] = {}
      for nd in tree_state['nodes']:
        exp_id = nd['experiment']
        if exp_id not in experiments:
          exp_state = nd.get('experiment_state', {})
          experiments[exp_id] = self._hydrate_experiment(exp_id, exp_state)

      def make_resolver(
        exps: dict[str, Experiment],
        t: Tree,
      ) -> Callable[[str], Experiment | Node]:
        def resolver(id_str: str) -> Experiment | Node:
          existing = t.get(id_str)
          if existing is not None:
            return existing
          if id_str in exps:
            return exps[id_str]
          msg = f'cannot resolve id: {id_str!r}'
          raise ValueError(msg)

        return resolver

      tree.load_state_dict(tree_state, make_resolver(experiments, tree))
      tree.on_change = self.save
      self._trees[name] = tree

  def _hydrate_experiment(
    self,
    experiment_id: str,
    exp_state: dict[str, Any],
  ) -> Experiment:
    """Create an Experiment from serialized state, using factory if available.

    Args:
      experiment_id: Experiment identifier.
      exp_state: Serialized experiment state dict.

    Returns:
      Experiment instance (possibly a subclass when factory is configured).
    """
    if self._experiment_factory is not None:
      return self._experiment_factory(exp_state)
    exp = Experiment(experiment_id=experiment_id)
    if exp_state:
      exp.load_state_dict(exp_state)
    return exp

  def state_dict(self) -> dict[str, Any]:
    """Extended serialization that includes experiment state.

    Returns:
      Dict with ``active`` tree name and serialized trees including node payloads.
    """
    result = {
      'active': self._active_name,
      'trees': [],
    }
    for tree in self._trees.values():
      tree_dict = tree.state_dict()
      enriched_nodes = []
      for nd in tree_dict['nodes']:
        node = tree.get(nd['experiment'])
        if node is not None:
          nd['experiment_state'] = node.experiment.state_dict()
        enriched_nodes.append(nd)
      tree_dict['nodes'] = enriched_nodes
      result['trees'].append(tree_dict)
    return result

  def __repr__(self) -> str:
    """Return a short debug representation with tree count and active name."""
    return f'FileForest(trees={len(self._trees)}, active={self._active_name!r})'
