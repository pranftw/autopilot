"""Parallel experiment tests: root + child experiments with branching.

Verifies:
1. Root experiment completes via Trainer.fit
2. Child experiment branches from completed root
3. Both experiments complete successfully
4. QueryBuilder.best() returns the child with better metrics
5. Stabilize winner copies files to workspace
"""

from autopilot.ai.experiment import AutoPilotExperiment
from autopilot.core.callbacks.store import StoreCheckpointCallback
from autopilot.core.enums import Status
from autopilot.core.node import Node
from autopilot.core.trainer.trainer import Trainer
from tests.integration.doubles import MinimalPathModule, TwoBatchTrainDatamodule


def _run_experiment(config, path_param, store, tree, exp_id, accuracy, parent_node=None):
  """Create experiment, add to tree, train for 1 epoch, return (exp, node)."""
  exp = AutoPilotExperiment(experiment_id=exp_id)
  node = Node(experiment=exp, parent=parent_node)
  tree.add(node)
  module = MinimalPathModule(path_param, accuracy=accuracy)
  trainer = Trainer(
    callbacks=[StoreCheckpointCallback()],
    experiment=exp,
    store=store,
    config=config,
    tree=tree,
  )
  trainer.fit(module, datamodule=TwoBatchTrainDatamodule(), max_epochs=1)
  return exp, node


def test_root_and_child_experiment(integration_workspace_with_store) -> None:
  """Root experiment completes, child branches from it, both end up completed."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('experiment-tree')

  root_exp, root_node = _run_experiment(config, path_param, store, tree, 'root-exp', accuracy=0.5)
  assert root_exp.status == Status.completed

  child_exp, _ = _run_experiment(
    config,
    path_param,
    store,
    tree,
    'child-exp',
    accuracy=0.9,
    parent_node=root_node,
  )

  assert root_exp.status == Status.completed
  assert child_exp.status == Status.completed

  best_node = tree.query().completed().best('AccuracyMetric', higher_is_better=True)
  assert best_node is not None
  assert best_node.experiment.id == 'child-exp'

  copied = config.stabilize(best_node.experiment.id)
  assert len(copied) > 0
  for p in copied:
    assert p.exists()


def test_parallel_query_ordering(integration_workspace_with_store) -> None:
  """QueryBuilder returns experiments ordered by metric value."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('experiment-tree')

  experiments = []
  accuracies = [0.3, 0.8, 0.5]
  for i, acc in enumerate(accuracies):
    exp = AutoPilotExperiment(experiment_id=f'exp-{i}')
    node = Node(experiment=exp, parent=experiments[-1][1] if experiments else None)
    tree.add(node)

    module = MinimalPathModule(path_param, accuracy=acc)
    cb = StoreCheckpointCallback()
    trainer = Trainer(
      callbacks=[cb],
      experiment=exp,
      store=store,
      config=config,
      tree=tree,
    )
    trainer.fit(module, datamodule=TwoBatchTrainDatamodule(), max_epochs=1)

    experiments.append((exp, node))

  ordered = tree.query().completed().order_by_metric('AccuracyMetric').all()
  assert len(ordered) == 3
  first_acc = ordered[0].experiment.metrics['AccuracyMetric']
  second_acc = ordered[1].experiment.metrics['AccuracyMetric']
  assert first_acc >= second_acc


def test_worst_query(integration_workspace_with_store) -> None:
  """QueryBuilder.worst() returns the experiment with lowest metric."""
  config, path_param, store, forest = integration_workspace_with_store
  tree = forest.create_tree('experiment-tree')

  root_exp = AutoPilotExperiment(experiment_id='worst-root')
  root_node = Node(experiment=root_exp)
  tree.add(root_node)
  root_module = MinimalPathModule(path_param, accuracy=0.9)
  root_trainer = Trainer(
    callbacks=[StoreCheckpointCallback()],
    experiment=root_exp,
    store=store,
    config=config,
    tree=tree,
  )
  root_trainer.fit(root_module, datamodule=TwoBatchTrainDatamodule(), max_epochs=1)

  child_exp = AutoPilotExperiment(experiment_id='worst-child')
  child_node = Node(experiment=child_exp, parent=root_node)
  tree.add(child_node)
  child_module = MinimalPathModule(path_param, accuracy=0.2)
  child_trainer = Trainer(
    callbacks=[StoreCheckpointCallback()],
    experiment=child_exp,
    store=store,
    config=config,
    tree=tree,
  )
  child_trainer.fit(child_module, datamodule=TwoBatchTrainDatamodule(), max_epochs=1)

  worst_node = tree.query().completed().worst('AccuracyMetric', higher_is_better=True)
  assert worst_node is not None
  assert worst_node.experiment.id == 'worst-child'
