"""Multi-tree tests: forest with multiple trees.

Verifies:
1. Create forest with 3 trees
2. Add experiments to each
3. Switch between trees
4. Forest-level query across all trees
5. Compare experiments from different trees
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.ai.forest import FileForest
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.node import Node
from autopilot.core.trainer.trainer import Trainer
from tests.integration.doubles import MinimalPathModule, TwoBatchTrainDatamodule


def _run_experiment(config, path_param, store, tree, exp_id, accuracy, parent_node=None):
  """Create and run an experiment through Trainer.fit."""
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


def test_three_trees_with_experiments(integration_workspace_with_store) -> None:
  """Create forest with 3 trees, add experiments, verify isolation."""
  config, path_param, store, forest = integration_workspace_with_store

  tree_a = forest.create_tree('alpha', description='feature A')
  tree_b = forest.create_tree('beta', description='feature B')
  tree_c = forest.create_tree('gamma', description='feature C')

  assert len(forest.list_trees()) == 3

  _exp_a, _node_a = _run_experiment(config, path_param, store, tree_a, 'exp-a', 0.7)
  _exp_b, _node_b = _run_experiment(config, path_param, store, tree_b, 'exp-b', 0.4)
  _exp_c, _node_c = _run_experiment(config, path_param, store, tree_c, 'exp-c', 0.9)

  assert len(tree_a.query().all()) == 1
  assert len(tree_b.query().all()) == 1
  assert len(tree_c.query().all()) == 1

  assert tree_a.query().all()[0].experiment.id == 'exp-a'
  assert tree_b.query().all()[0].experiment.id == 'exp-b'
  assert tree_c.query().all()[0].experiment.id == 'exp-c'


def test_switch_between_trees(integration_workspace_with_store) -> None:
  """Switching active tree changes forest.active."""
  _, _, _, forest = integration_workspace_with_store
  forest.create_tree('first')
  forest.create_tree('second')
  forest.create_tree('third')

  forest.switch('first')
  assert forest.active is not None
  assert forest.active.name == 'first'

  forest.switch('second')
  assert forest.active.name == 'second'

  forest.switch('third')
  assert forest.active.name == 'third'


def test_forest_cross_tree_query(integration_workspace_with_store) -> None:
  """Forest.query() returns nodes from all trees."""
  config, path_param, store, forest = integration_workspace_with_store

  tree_a = forest.create_tree('alpha')
  tree_b = forest.create_tree('beta')
  tree_c = forest.create_tree('gamma')

  _run_experiment(config, path_param, store, tree_a, 'cross-a', 0.3)
  _run_experiment(config, path_param, store, tree_b, 'cross-b', 0.6)
  _run_experiment(config, path_param, store, tree_c, 'cross-c', 0.8)

  all_nodes = forest.query().all()
  assert len(all_nodes) == 3

  completed = forest.query().completed().all()
  assert len(completed) == 3

  best = forest.query().completed().best('AccuracyMetric', higher_is_better=True)
  assert best is not None
  assert best.experiment.id == 'cross-c'

  worst = forest.query().completed().worst('AccuracyMetric', higher_is_better=True)
  assert worst is not None
  assert worst.experiment.id == 'cross-a'


def test_forest_persistence_across_reload(integration_workspace_with_store) -> None:
  """Forest state persists across FileForest reload."""
  config, path_param, store, forest = integration_workspace_with_store

  tree = forest.create_tree('persist-tree', description='test persistence')
  _run_experiment(config, path_param, store, tree, 'persist-exp', 0.75)
  forest.switch('persist-tree')

  forest2 = FileForest(store=store)
  assert len(forest2.list_trees()) == 1
  reloaded_tree = forest2.get_tree('persist-tree')
  assert reloaded_tree is not None
  assert reloaded_tree.description == 'test persistence'

  reloaded_nodes = reloaded_tree.query().all()
  assert len(reloaded_nodes) == 1
  assert reloaded_nodes[0].experiment.id == 'persist-exp'


def test_remove_tree_from_forest(integration_workspace_with_store) -> None:
  """Removing a tree from forest updates the collection."""
  _, _, _, forest = integration_workspace_with_store

  forest.create_tree('keep')
  forest.create_tree('remove-me')
  assert len(forest.list_trees()) == 2

  forest.remove_tree('remove-me')
  assert len(forest.list_trees()) == 1
  assert forest.get_tree('remove-me') is None
  assert forest.get_tree('keep') is not None


def test_forest_query_metric_filter(integration_workspace_with_store) -> None:
  """Forest-level metric filtering across trees."""
  config, path_param, store, forest = integration_workspace_with_store

  tree_a = forest.create_tree('filter-a')
  tree_b = forest.create_tree('filter-b')

  _run_experiment(config, path_param, store, tree_a, 'low-exp', 0.2)
  _run_experiment(config, path_param, store, tree_b, 'high-exp', 0.9)

  high_performers = forest.query().metric_gt('AccuracyMetric', 0.5).all()
  assert len(high_performers) == 1
  assert high_performers[0].experiment.id == 'high-exp'

  low_performers = forest.query().metric_lt('AccuracyMetric', 0.5).all()
  assert len(low_performers) == 1
  assert low_performers[0].experiment.id == 'low-exp'
