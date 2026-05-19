"""Comprehensive tests for Forest multi-tree management.

Covers cross-tree queries, deduplication behavior, active tree lifecycle,
large forests, and state_dict/load_state_dict round-trips.
"""

from autopilot.core.forest import Forest
from autopilot.core.node import Node
from autopilot.core.store.base import Store
from tests.core.conftest import (
  completed_exp as _completed_exp,
)
from tests.core.conftest import (
  failed_exp as _failed_exp,
)
from tests.core.conftest import (
  pending_exp as _pending_exp,
)
from unittest.mock import MagicMock
import json
import pytest


@pytest.fixture
def mock_store():
  return MagicMock(spec=Store)


@pytest.fixture
def forest(mock_store):
  return Forest(mock_store)


# --- Cross-tree query ---


class TestCrossTreeQuery:
  def test_query_experiments_in_multiple_trees(self, forest):
    tree1 = forest.create_tree('t1')
    tree2 = forest.create_tree('t2')
    tree3 = forest.create_tree('t3')

    tree1.add(Node(experiment=_completed_exp('exp-1', {'accuracy': 0.7})))
    tree2.add(Node(experiment=_completed_exp('exp-2', {'accuracy': 0.8})))
    tree3.add(Node(experiment=_completed_exp('exp-3', {'accuracy': 0.9})))

    qb = forest.query()
    assert qb.count() == 3
    best = qb.best('accuracy')
    assert best is not None
    assert best.experiment.id == 'exp-3'

  def test_cross_tree_filter(self, forest):
    tree1 = forest.create_tree('t1')
    tree2 = forest.create_tree('t2')

    tree1.add(Node(experiment=_completed_exp('c1', {'accuracy': 0.8})))
    tree1.add(Node(experiment=_pending_exp('p1')))
    tree2.add(Node(experiment=_completed_exp('c2', {'accuracy': 0.9})))
    tree2.add(Node(experiment=_failed_exp('f1')))

    completed = forest.query().completed().all()
    assert len(completed) == 2
    ids = {n.experiment.id for n in completed}
    assert ids == {'c1', 'c2'}

  def test_cross_tree_structural_queries(self, forest):
    tree1 = forest.create_tree('t1')
    tree2 = forest.create_tree('t2')

    root1 = _completed_exp('root1', {'accuracy': 0.7})
    child1 = _completed_exp('child1', {'accuracy': 0.8})
    root1_node = Node(experiment=root1)
    tree1.add(root1_node)
    tree1.add(Node(experiment=child1, parent=root1_node))

    root2 = _completed_exp('root2', {'accuracy': 0.9})
    tree2.add(Node(experiment=root2))

    all_roots = forest.query().roots().all()
    root_ids = {n.experiment.id for n in all_roots}
    assert root_ids == {'root1', 'root2'}


# --- Forest.query() deduplication (Bug 54) ---


class TestQueryDeduplication:
  def test_same_experiment_id_in_two_trees_deduplicates(self, forest):
    """BUG-044: duplicate experiment IDs across trees are deduplicated."""
    tree1 = forest.create_tree('t1')
    tree2 = forest.create_tree('t2')

    exp1 = _completed_exp('shared-id', {'accuracy': 0.7})
    exp2 = _completed_exp('shared-id', {'accuracy': 0.9})
    tree1.add(Node(experiment=exp1))
    tree2.add(Node(experiment=exp2))

    nodes = forest.query().all()
    ids = [n.experiment.id for n in nodes]
    assert ids.count('shared-id') == 1
    matched = [n for n in nodes if n.experiment.id == 'shared-id']
    assert matched[0].experiment.metrics['accuracy'] == 0.7


# --- Remove active tree ---


class TestRemoveActiveTree:
  def test_remove_active_tree_clears_active(self, forest):
    forest.create_tree('t1')
    forest.create_tree('t2')
    forest.switch('t1')
    assert forest.active is not None
    forest.remove_tree('t1')
    assert forest.active is None
    assert forest._active_name is None

  def test_remove_non_active_tree_preserves_active(self, forest):
    forest.create_tree('t1')
    forest.create_tree('t2')
    forest.switch('t1')
    forest.remove_tree('t2')
    assert forest.active is not None
    assert forest.active.name == 't1'


# --- Forest with 10+ trees ---


class TestLargeForest:
  def test_forest_with_15_trees(self, forest):
    for i in range(15):
      tree = forest.create_tree(f'tree-{i}', description=f'tree number {i}')
      exp = _completed_exp(f'exp-{i}', {'accuracy': i / 15.0})
      tree.add(Node(experiment=exp))

    assert len(forest.list_trees()) == 15

    qb = forest.query()
    assert qb.count() == 15

    best = qb.best('accuracy')
    assert best is not None
    assert best.experiment.id == 'exp-14'

    worst = qb.worst('accuracy')
    assert worst is not None
    assert worst.experiment.id == 'exp-0'

  def test_forest_switch_among_many_trees(self, forest):
    for i in range(12):
      forest.create_tree(f'tree-{i}')

    for i in range(12):
      forest.switch(f'tree-{i}')
      assert forest.active.name == f'tree-{i}'

  def test_forest_remove_multiple_trees(self, forest):
    for i in range(10):
      forest.create_tree(f'tree-{i}')

    for i in range(0, 10, 2):
      forest.remove_tree(f'tree-{i}')

    remaining = [t.name for t in forest.list_trees()]
    assert len(remaining) == 5
    for i in range(1, 10, 2):
      assert f'tree-{i}' in remaining


# --- state_dict / load_state_dict with complex multi-tree state ---


class TestComplexStateDictRoundTrip:
  def test_multi_tree_with_nodes_and_active(self, mock_store):
    forest1 = Forest(mock_store)
    tree_a = forest1.create_tree('alpha', description='first direction')
    tree_b = forest1.create_tree('beta', description='second direction')
    forest1.create_tree('gamma')

    root_a = _completed_exp('a-root', {'accuracy': 0.7})
    child_a = _completed_exp('a-child', {'accuracy': 0.8})
    root_a_node = Node(experiment=root_a)
    tree_a.add(root_a_node)
    tree_a.add(Node(experiment=child_a, parent=root_a_node))

    root_b = _completed_exp('b-root', {'accuracy': 0.6})
    tree_b.add(Node(experiment=root_b))

    forest1.switch('beta')

    state = forest1.state_dict()

    forest2 = Forest(mock_store)
    forest2.load_state_dict(state)

    assert forest2._active_name == 'beta'
    assert len(forest2.list_trees()) == 3

    loaded_a = forest2.get_tree('alpha')
    assert loaded_a is not None
    assert loaded_a.get('a-root') is not None
    a_child = loaded_a.get('a-child')
    assert a_child is not None
    assert a_child.parent is not None
    assert a_child.parent.experiment.id == 'a-root'
    assert loaded_a.description == 'first direction'

    loaded_b = forest2.get_tree('beta')
    assert loaded_b is not None
    assert loaded_b.get('b-root') is not None

    loaded_c = forest2.get_tree('gamma')
    assert loaded_c is not None
    assert len(loaded_c.query().all()) == 0

  def test_state_dict_preserves_head(self, mock_store):
    forest1 = Forest(mock_store)
    tree = forest1.create_tree('t')
    exp = _completed_exp('e1')
    tree.add(Node(experiment=exp))
    tree._head = 'e1'

    state = forest1.state_dict()

    forest2 = Forest(mock_store)
    forest2.load_state_dict(state)
    loaded_tree = forest2.get_tree('t')
    assert loaded_tree is not None
    assert loaded_tree.head == 'e1'

  def test_state_dict_empty_forest(self, mock_store):
    forest1 = Forest(mock_store)
    state = forest1.state_dict()
    assert state == {'active': None, 'trees': []}

    forest2 = Forest(mock_store)
    forest2.load_state_dict(state)
    assert forest2.list_trees() == []
    assert forest2.active is None

  def test_state_dict_is_json_serializable(self, mock_store):
    forest1 = Forest(mock_store)
    tree = forest1.create_tree('t')
    tree.add(Node(experiment=_completed_exp('e1', {'accuracy': 0.8})))

    state = forest1.state_dict()
    serialized = json.dumps(state)
    deserialized = json.loads(serialized)
    assert deserialized['trees'][0]['name'] == 't'


# --- save / load to file round-trip ---


class TestSaveLoadRoundTrip:
  def test_save_then_load(self, mock_store):
    mock_store.load_state_dict.return_value = None
    forest1 = Forest(mock_store)
    tree = forest1.create_tree('t1', description='test tree')
    root_exp = _completed_exp('root', {'accuracy': 0.7})
    child_exp = _completed_exp('child', {'accuracy': 0.8})
    root_node = Node(experiment=root_exp)
    tree.add(root_node)
    tree.add(Node(experiment=child_exp, parent=root_node))
    forest1.switch('t1')

    forest1.save()
    saved_state = mock_store.save_state_dict.call_args[0][0]

    mock_store.load_state_dict.return_value = saved_state
    forest2 = Forest(mock_store)
    forest2.load()

    assert len(forest2.list_trees()) == 1
    assert forest2._active_name == 't1'
    loaded_tree = forest2.get_tree('t1')
    assert loaded_tree is not None
    assert loaded_tree.get('root') is not None
    child = loaded_tree.get('child')
    assert child is not None
    assert child.parent is not None
    assert child.parent.experiment.id == 'root'

  def test_load_when_store_returns_none(self, mock_store):
    mock_store.load_state_dict.return_value = None
    forest = Forest(mock_store)
    forest.load()
    assert forest.list_trees() == []

  def test_to_dict_matches_state_dict(self, forest):
    forest.create_tree('t1', description='alpha')
    forest.switch('t1')
    assert forest.to_dict() == forest.state_dict()


# --- repr ---


class TestRepr:
  def test_repr_empty(self, forest):
    assert 'Forest' in repr(forest)
    assert 'trees=0' in repr(forest)

  def test_repr_with_trees(self, forest):
    forest.create_tree('a')
    forest.create_tree('b')
    forest.switch('a')
    r = repr(forest)
    assert 'trees=2' in r
    assert "'a'" in r
