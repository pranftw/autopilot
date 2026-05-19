"""Experiment branching tests: root -> child -> grandchild lineage.

Verifies:
1. Create root experiment, complete it
2. Branch: create child experiment from root
3. Complete child with better metrics
4. Branch again: create grandchild
5. Query: ancestors_of(grandchild) -> [child, root]
6. Query: best across all -> grandchild
7. Stabilize grandchild
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.enums import Status
from autopilot.core.node import Node
from autopilot.core.trainer.trainer import Trainer
from tests.integration.doubles import MinimalPathModule, TwoBatchTrainDatamodule


def _train_experiment(config, path_param, store, tree, exp_id, accuracy, parent_node=None):
  """Create, add to tree, and train an experiment."""
  exp = AutoPilotExperiment(experiment_id=exp_id)
  node = Node(experiment=exp, parent=parent_node)
  tree.add(node)

  module = MinimalPathModule(path_param, accuracy=accuracy)
  cb = StoreCheckpointCallback()
  trainer = Trainer(
    callbacks=[cb],
    experiment=exp,
    store=store,
    config=config,
    tree=tree,
  )
  trainer.fit(module, datamodule=TwoBatchTrainDatamodule(), max_epochs=1)
  return exp, node


def test_root_child_grandchild_branching(integration_workspace_with_store) -> None:
  """Full branching lineage: root -> child -> grandchild with queries."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('branching-tree')

  # 1. Create and complete root experiment
  root_exp, root_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'root',
    0.3,
  )
  assert root_exp.status == Status.completed

  # 2. Branch child from root (tree-level branching; store auto-creates branch on snapshot)
  child_exp, child_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'child',
    0.6,
    parent_node=root_node,
  )
  assert child_exp.status == Status.completed

  # 4. Branch grandchild from child
  gc_exp, _gc_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'grandchild',
    0.95,
    parent_node=child_node,
  )
  assert gc_exp.status == Status.completed

  # 5. Query: ancestors_of(grandchild) -> [child, root] (closest ancestor first)
  ancestors = tree.query().ancestors_of('grandchild').all()
  ancestor_ids = [a.experiment.id for a in ancestors]
  assert ancestor_ids == ['child', 'root'], (
    f'Expected ancestors in order [child, root], got {ancestor_ids}'
  )

  # 6. Query: best across all -> grandchild
  best = tree.query().completed().best('AccuracyMetric', higher_is_better=True)
  assert best is not None
  assert best.experiment.id == 'grandchild'

  # 7. Stabilize grandchild
  copied = config.stabilize(best.experiment.id)
  assert len(copied) > 0
  for p in copied:
    assert p.exists()


def test_descendants_of_root(integration_workspace_with_store) -> None:
  """descendants_of(root) returns child and grandchild."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('branching-tree')

  _root_exp, root_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'desc-root',
    0.3,
  )
  _child_exp, child_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'desc-child',
    0.6,
    parent_node=root_node,
  )
  _gc_exp, _gc_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'desc-grandchild',
    0.9,
    parent_node=child_node,
  )

  descendants = tree.query().descendants_of('desc-root').all()
  desc_ids = {d.experiment.id for d in descendants}
  assert desc_ids == {'desc-child', 'desc-grandchild'}


def test_children_of_root(integration_workspace_with_store) -> None:
  """children_of(root) returns direct children only."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('branching-tree')

  _root_exp, root_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'ch-root',
    0.3,
  )

  _train_experiment(
    config,
    path_param,
    store,
    tree,
    'ch-child-1',
    0.5,
    parent_node=root_node,
  )
  _train_experiment(
    config,
    path_param,
    store,
    tree,
    'ch-child-2',
    0.7,
    parent_node=root_node,
  )

  children = tree.query().children_of('ch-root').all()
  child_ids = {c.experiment.id for c in children}
  assert child_ids == {'ch-child-1', 'ch-child-2'}


def test_siblings_query(integration_workspace_with_store) -> None:
  """siblings_of returns nodes with same parent, excluding self."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('branching-tree')

  _root_exp, root_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'sib-root',
    0.3,
  )

  _exp_a, _node_a = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'sib-a',
    0.5,
    parent_node=root_node,
  )
  _exp_b, _node_b = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'sib-b',
    0.7,
    parent_node=root_node,
  )

  siblings = tree.query().siblings_of('sib-a').all()
  sibling_ids = {s.experiment.id for s in siblings}
  assert sibling_ids == {'sib-b'}


def test_roots_and_leaves_query(integration_workspace_with_store) -> None:
  """roots() returns root nodes, leaves() returns leaf nodes."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('branching-tree')

  _root_exp, root_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'rl-root',
    0.3,
  )
  _train_experiment(
    config,
    path_param,
    store,
    tree,
    'rl-leaf',
    0.8,
    parent_node=root_node,
  )

  roots = tree.query().roots().all()
  assert len(roots) == 1
  assert roots[0].experiment.id == 'rl-root'

  leaves = tree.query().leaves().all()
  assert len(leaves) == 1
  assert leaves[0].experiment.id == 'rl-leaf'


def test_depth_query(integration_workspace_with_store) -> None:
  """depth(n) filters nodes at specific depth."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('branching-tree')

  _root_exp, root_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'depth-root',
    0.2,
  )
  _child_exp, child_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'depth-child',
    0.5,
    parent_node=root_node,
  )
  _train_experiment(
    config,
    path_param,
    store,
    tree,
    'depth-gc',
    0.8,
    parent_node=child_node,
  )

  at_depth_0 = tree.query().depth(0).all()
  assert len(at_depth_0) == 1
  assert at_depth_0[0].experiment.id == 'depth-root'

  at_depth_1 = tree.query().depth(1).all()
  assert len(at_depth_1) == 1
  assert at_depth_1[0].experiment.id == 'depth-child'

  at_depth_2 = tree.query().depth(2).all()
  assert len(at_depth_2) == 1
  assert at_depth_2[0].experiment.id == 'depth-gc'


def test_branching_preserves_store_history(integration_workspace_with_store) -> None:
  """Each branch in the store has independent snapshot history."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('branching-tree')

  _root_exp, root_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'hist-root',
    0.3,
  )

  root_log = store.log('hist-root')
  assert len(root_log) == 1

  _child_exp, _child_node = _train_experiment(
    config,
    path_param,
    store,
    tree,
    'hist-child',
    0.7,
    parent_node=root_node,
  )

  child_log = store.log('hist-child')
  assert len(child_log) >= 1

  # Root log is unchanged
  root_log_after = store.log('hist-root')
  assert len(root_log_after) == len(root_log)
