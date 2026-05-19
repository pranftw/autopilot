"""Tests for Tree add() and store.branch alignment (BUG-035, BUG-041).

BUG-015: branch failure tests pin the documented behavior where
Tree.add still inserts the node even when store.branch raises.
"""

from autopilot.core.enums import Status
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.tree import Tree
from unittest.mock import MagicMock


def test_tree_add_with_store_calls_branch_once_per_experiment() -> None:
  """Root add skips branch; child add calls store.branch once with the child id."""
  store = MagicMock()
  tree = Tree(name='test', store=store)
  parent_exp = Experiment(experiment_id='parent')
  parent_exp.status = Status.completed
  parent_node = Node(experiment=parent_exp)
  tree.add(parent_node)

  child_exp = Experiment(experiment_id='child')
  child_node = Node(experiment=child_exp, parent=parent_node)
  tree.add(child_node)

  store.branch.assert_called_once_with('child')


def test_tree_add_without_store_does_not_call_branch() -> None:
  """A root node (no parent) does not invoke store.branch."""
  store = MagicMock()
  tree = Tree(name='test', store=store)
  root_exp = Experiment(experiment_id='root-only')
  tree.add(Node(experiment=root_exp))
  store.branch.assert_not_called()


def test_tree_add_node_added_despite_branch_failure() -> None:
  """BUG-015/035: node is added even when store.branch raises StoreError."""
  store = MagicMock()
  store.branch.side_effect = StoreError('branch failed')
  tree = Tree(name='t', store=store)
  parent_exp = Experiment(experiment_id='parent')
  parent_exp.status = Status.completed
  parent_node = Node(experiment=parent_exp)
  tree.add(parent_node)

  child_exp = Experiment(experiment_id='child')
  tree.add(Node(experiment=child_exp, parent=parent_node))
  assert tree.get('child') is not None
  store.branch.assert_called_once_with('child')


def test_tree_add_node_added_despite_not_implemented() -> None:
  """BUG-015/035: node is added even when store.branch raises NotImplementedError."""
  store = MagicMock()
  store.branch.side_effect = NotImplementedError('branch not supported')
  tree = Tree(name='t', store=store)
  parent_exp = Experiment(experiment_id='parent')
  parent_exp.status = Status.completed
  parent_node = Node(experiment=parent_exp)
  tree.add(parent_node)

  child_exp = Experiment(experiment_id='child')
  tree.add(Node(experiment=child_exp, parent=parent_node))
  assert tree.get('child') is not None
  store.branch.assert_called_once_with('child')


def test_tree_add_root_does_not_call_branch() -> None:
  """Root nodes skip store.branch since there is no parent."""
  store = MagicMock()
  tree = Tree(name='t', store=store)
  tree.add(Node(experiment=Experiment(experiment_id='root')))
  store.branch.assert_not_called()
