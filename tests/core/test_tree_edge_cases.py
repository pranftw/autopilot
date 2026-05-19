"""Edge-case tests for Tree DAG operations."""

from autopilot.core.node import Node
from autopilot.core.store.base import Store
from autopilot.core.tree import Tree
from tests.core.conftest import completed_exp
from unittest.mock import MagicMock
import pytest


def _completed_exp(id_: str, metrics: dict | None = None):
  return completed_exp(id_, metrics=metrics, auto_hypothesis=False)


@pytest.fixture
def mock_store():
  return MagicMock(spec=Store)


@pytest.fixture
def tree(mock_store):
  return Tree(name='test', store=mock_store)


class TestRemoveCascadeFalseOnLeaf:
  def test_remove_leaf_no_cascade(self, tree) -> None:
    exp = _completed_exp('leaf')
    tree.add(Node(experiment=exp))
    tree.remove('leaf', cascade=False)
    assert tree.get('leaf') is None

  def test_remove_leaf_no_cascade_tree_still_has_other_nodes(self, tree) -> None:
    exp1 = _completed_exp('keep')
    exp2 = _completed_exp('remove')
    tree.add(Node(experiment=exp1))
    tree.add(Node(experiment=exp2))
    tree.remove('remove', cascade=False)
    assert tree.get('keep') is not None
    assert tree.get('remove') is None


class TestRemoveCascadeDeepLevels:
  def test_remove_cascade_5_levels(self, tree) -> None:
    parent = Node(experiment=_completed_exp('l0'))
    tree.add(parent)
    for i in range(1, 6):
      child = Node(experiment=_completed_exp(f'l{i}'), parent=parent)
      tree.add(child)
      parent = child
    tree.remove('l0', cascade=True)
    for i in range(6):
      assert tree.get(f'l{i}') is None
    assert len(tree.roots()) == 0

  def test_remove_cascade_10_levels(self, tree) -> None:
    parent = Node(experiment=_completed_exp('d0'))
    tree.add(parent)
    for i in range(1, 11):
      child = Node(experiment=_completed_exp(f'd{i}'), parent=parent)
      tree.add(child)
      parent = child
    tree.remove('d0', cascade=True)
    for i in range(11):
      assert tree.get(f'd{i}') is None


class TestRemoveHeadCleared:
  def test_remove_node_that_is_head(self, tree) -> None:
    exp = _completed_exp('head-node')
    tree.add(Node(experiment=exp))
    tree.checkout('head-node')
    assert tree.head == 'head-node'
    tree.remove('head-node')
    assert tree.head is None

  def test_remove_cascade_head_among_descendants(self, tree) -> None:
    parent = Node(experiment=_completed_exp('root'))
    tree.add(parent)
    child = Node(experiment=_completed_exp('child'), parent=parent)
    tree.add(child)
    tree.checkout('child')
    assert tree.head == 'child'
    tree.remove('root', cascade=True)
    assert tree.head is None


class TestRemoveLastNode:
  def test_remove_last_node_empties_tree(self, tree) -> None:
    exp = _completed_exp('only')
    tree.add(Node(experiment=exp))
    tree.remove('only')
    assert tree.get('only') is None
    assert len(tree.roots()) == 0
    assert tree.query().count() == 0

  def test_remove_last_node_head_cleared(self, tree) -> None:
    exp = _completed_exp('only')
    tree.add(Node(experiment=exp))
    tree.checkout('only')
    tree.remove('only')
    assert tree.head is None


class TestTreeWith100PlusNodes:
  def test_100_node_tree(self, tree) -> None:
    root = Node(experiment=_completed_exp('root'))
    tree.add(root)
    for i in range(100):
      child = Node(experiment=_completed_exp(f'child-{i}'), parent=root)
      tree.add(child)
    assert tree.query().count() == 101
    assert len(tree.roots()) == 1
    assert tree.get('child-99') is not None

  def test_100_node_chain(self, tree) -> None:
    parent = Node(experiment=_completed_exp('chain-0'))
    tree.add(parent)
    for i in range(1, 101):
      child = Node(experiment=_completed_exp(f'chain-{i}'), parent=parent)
      tree.add(child)
      parent = child
    assert tree.query().count() == 101
    tree.remove('chain-0', cascade=True)
    assert tree.query().count() == 0


class TestAddAfterRemove:
  def test_add_after_remove_fills_gap(self, tree) -> None:
    root = Node(experiment=_completed_exp('root'))
    tree.add(root)
    child = Node(experiment=_completed_exp('child'))
    tree.add(Node(experiment=child.experiment, parent=root))
    tree.remove('child')
    new_child = _completed_exp('child-new')
    tree.add(Node(experiment=new_child, parent=root))
    assert tree.get('child-new') is not None

  def test_add_same_id_after_remove(self, tree) -> None:
    exp = _completed_exp('reuse-id')
    tree.add(Node(experiment=exp))
    tree.remove('reuse-id')
    exp2 = _completed_exp('reuse-id')
    tree.add(Node(experiment=exp2))
    assert tree.get('reuse-id') is not None

  def test_add_root_after_removing_all(self, tree) -> None:
    exp = _completed_exp('gone')
    tree.add(Node(experiment=exp))
    tree.remove('gone')
    new_exp = _completed_exp('new-root')
    tree.add(Node(experiment=new_exp))
    assert len(tree.roots()) == 1
    assert tree.get('new-root') is not None


class TestCheckoutNonexistent:
  def test_checkout_nonexistent_raises_value_error(self, tree) -> None:
    with pytest.raises(ValueError, match='not found'):
      tree.checkout('does-not-exist')

  def test_checkout_removed_node_raises(self, tree) -> None:
    exp = _completed_exp('temp')
    tree.add(Node(experiment=exp))
    tree.remove('temp')
    with pytest.raises(ValueError, match='not found'):
      tree.checkout('temp')
