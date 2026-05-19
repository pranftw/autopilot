"""Tests for Forest.query cross-tree duplicate experiment id deduplication (BUG-044)."""

from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.node import Node
from unittest.mock import MagicMock


def test_forest_duplicate_experiment_id_deduplicates() -> None:
  """Cross-tree duplicate ids are deduplicated (first occurrence wins)."""
  store = MagicMock()
  forest = Forest(store)
  tree1 = forest.create_tree('tree-1')
  tree2 = forest.create_tree('tree-2')
  tree1.add(Node(experiment=Experiment(experiment_id='exp-a')))
  tree2.add(Node(experiment=Experiment(experiment_id='exp-a')))
  builder = forest.query()
  nodes = builder.all()
  ids = [n.experiment.id for n in nodes]
  assert ids.count('exp-a') == 1


def test_forest_unique_ids_across_trees_succeeds() -> None:
  """query() returns a builder when experiment ids are unique per tree."""
  store = MagicMock()
  forest = Forest(store)
  tree1 = forest.create_tree('tree-1')
  tree2 = forest.create_tree('tree-2')
  tree1.add(Node(experiment=Experiment(experiment_id='exp-a')))
  tree2.add(Node(experiment=Experiment(experiment_id='exp-b')))
  builder = forest.query()
  ids = {node.experiment.id for node in builder.all()}
  assert ids == {'exp-a', 'exp-b'}
