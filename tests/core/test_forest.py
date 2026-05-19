"""Tests for Forest multi-tree management."""

from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from autopilot.core.store.base import Store
from autopilot.core.tree import Tree
from tests.core.conftest import completed_exp as _completed_exp
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_store():
  return MagicMock(spec=Store)


@pytest.fixture
def forest(mock_store):
  return Forest(mock_store)


# --- create_tree ---


class TestCreateTree:
  def test_create_tree(self, forest):
    tree = forest.create_tree('tree-1', description='first tree')
    assert isinstance(tree, Tree)
    assert tree.name == 'tree-1'
    assert tree.description == 'first tree'

  def test_create_tree_duplicate_raises(self, forest):
    forest.create_tree('tree-1')
    with pytest.raises(ValueError, match='already exists'):
      forest.create_tree('tree-1')


# --- get_tree ---


class TestGetTree:
  def test_get_existing(self, forest):
    forest.create_tree('tree-1')
    tree = forest.get_tree('tree-1')
    assert tree is not None
    assert tree.name == 'tree-1'

  def test_get_nonexistent(self, forest):
    assert forest.get_tree('nonexistent') is None


# --- list_trees ---


class TestListTrees:
  def test_list_trees(self, forest):
    forest.create_tree('a')
    forest.create_tree('b')
    forest.create_tree('c')
    trees = forest.list_trees()
    names = {t.name for t in trees}
    assert names == {'a', 'b', 'c'}

  def test_list_trees_empty(self, forest):
    assert forest.list_trees() == []


# --- remove_tree ---


class TestRemoveTree:
  def test_remove_tree(self, forest):
    forest.create_tree('tree-1')
    forest.remove_tree('tree-1')
    assert forest.get_tree('tree-1') is None

  def test_remove_nonexistent_raises(self, forest):
    with pytest.raises(ValueError, match='not found'):
      forest.remove_tree('nonexistent')

  def test_remove_active_tree_clears_active(self, forest):
    forest.create_tree('tree-1')
    forest.switch('tree-1')
    forest.remove_tree('tree-1')
    assert forest.active is None


# --- active ---


class TestActive:
  def test_active_initially_none(self, forest):
    assert forest.active is None

  def test_active_after_switch(self, forest):
    forest.create_tree('tree-1')
    forest.switch('tree-1')
    assert forest.active is not None
    assert forest.active.name == 'tree-1'


# --- switch ---


class TestSwitch:
  def test_switch_changes_active(self, forest):
    forest.create_tree('a')
    forest.create_tree('b')
    forest.switch('a')
    assert forest.active.name == 'a'
    forest.switch('b')
    assert forest.active.name == 'b'

  def test_switch_nonexistent_raises(self, forest):
    with pytest.raises(ValueError, match='not found'):
      forest.switch('nonexistent')


# --- query ---


class TestQuery:
  def test_cross_tree_query(self, forest):
    tree1 = forest.create_tree('tree-1')
    tree2 = forest.create_tree('tree-2')

    exp1 = _completed_exp('exp-1', {'accuracy': 0.8})
    exp2 = _completed_exp('exp-2', {'accuracy': 0.9})
    tree1.add(Node(experiment=exp1))
    tree2.add(Node(experiment=exp2))

    qb = forest.query()
    assert isinstance(qb, QueryBuilder)
    assert qb.count() == 2

  def test_cross_tree_best(self, forest):
    tree1 = forest.create_tree('tree-1')
    tree2 = forest.create_tree('tree-2')

    tree1.add(Node(experiment=_completed_exp('exp-1', {'accuracy': 0.8})))
    tree2.add(Node(experiment=_completed_exp('exp-2', {'accuracy': 0.9})))

    best = forest.query().best('accuracy')
    assert best is not None
    assert best.experiment.id == 'exp-2'

  def test_cross_tree_completed(self, forest):
    tree1 = forest.create_tree('tree-1')
    tree2 = forest.create_tree('tree-2')

    tree1.add(Node(experiment=_completed_exp('exp-1')))
    pending = Experiment(experiment_id='exp-2')
    tree2.add(Node(experiment=pending))

    completed = forest.query().completed().all()
    assert len(completed) == 1
    assert completed[0].experiment.id == 'exp-1'


# --- state_dict / load_state_dict ---


class TestSerialization:
  def test_state_dict_structure(self, forest):
    forest.create_tree('a', description='alpha')
    forest.create_tree('b')
    forest.switch('a')

    state = forest.state_dict()
    assert state['active'] == 'a'
    assert len(state['trees']) == 2

  def test_round_trip(self, mock_store):
    forest1 = Forest(mock_store)
    tree_a = forest1.create_tree('a', description='alpha')
    tree_b = forest1.create_tree('b')
    forest1.switch('a')

    exp1 = _completed_exp('exp-1', {'accuracy': 0.8})
    tree_a.add(Node(experiment=exp1))

    exp2 = _completed_exp('exp-2', {'accuracy': 0.9})
    exp3 = Experiment(experiment_id='exp-3')
    tree_b.add(Node(experiment=exp2))
    tree_b.add(Node(experiment=exp3, parent=tree_b.get('exp-2')))

    state = forest1.state_dict()

    forest2 = Forest(mock_store)
    forest2.load_state_dict(state)

    assert forest2._active_name == 'a'
    assert len(forest2.list_trees()) == 2

    ta = forest2.get_tree('a')
    assert ta is not None
    assert ta.get('exp-1') is not None

    tb = forest2.get_tree('b')
    assert tb is not None
    assert tb.get('exp-2') is not None
    assert tb.get('exp-3') is not None
    exp3_node = tb.get('exp-3')
    assert exp3_node is not None
    assert exp3_node.parent is not None
    assert exp3_node.parent.experiment.id == 'exp-2'

  def test_round_trip_empty(self, mock_store):
    forest1 = Forest(mock_store)
    state = forest1.state_dict()
    forest2 = Forest(mock_store)
    forest2.load_state_dict(state)
    assert forest2.list_trees() == []
    assert forest2.active is None


# --- save/load convenience ---


class TestSaveLoad:
  def test_save_calls_store(self, forest, mock_store):
    forest.create_tree('tree-1')
    forest.save()
    mock_store.save_state_dict.assert_called_once()

  def test_load_from_store(self, mock_store):
    mock_store.load_state_dict.return_value = {
      'active': None,
      'trees': [
        {
          'name': 'loaded',
          'description': None,
          'head': None,
          'nodes': [],
        }
      ],
    }
    forest = Forest(mock_store)
    forest.load()
    assert len(forest.list_trees()) == 1
    assert forest.get_tree('loaded') is not None

  def test_load_none_is_noop(self, mock_store):
    mock_store.load_state_dict.return_value = None
    forest = Forest(mock_store)
    forest.load()
    assert forest.list_trees() == []
