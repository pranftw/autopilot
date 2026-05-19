"""Tests for Tree DAG management."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.query import QueryBuilder
from autopilot.core.store.base import Store
from autopilot.core.tree import Tree
from pathlib import Path
from tests.core.conftest import (
  cancelled_exp as _cancelled_exp,
)
from tests.core.conftest import (
  completed_exp as _completed_exp,
)
from tests.core.conftest import (
  failed_exp as _failed_exp,
)
from tests.core.conftest import (
  pending_exp as _pending_exp,
)
from tests.core.conftest import (
  running_exp as _running_exp,
)
from unittest.mock import MagicMock
import pytest


@pytest.fixture
def mock_store():
  store = MagicMock(spec=Store)
  return store


@pytest.fixture
def tree(mock_store):
  return Tree(name='test-tree', store=mock_store, description='test tree')


# --- add ---


class TestAdd:
  def test_add_single_node(self, tree):
    exp = _completed_exp('exp-1')
    node = Node(experiment=exp)
    tree.add(node)
    assert tree.get('exp-1') is node

  def test_add_node_with_parent(self, tree):
    parent_exp = _completed_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)

    child_exp = Experiment(experiment_id='child')
    child_node = Node(experiment=child_exp, parent=parent_node)
    tree.add(child_node)
    assert tree.get('child') is child_node

  def test_add_multiple_roots(self, tree):
    for i in range(3):
      exp = _completed_exp(f'root-{i}')
      tree.add(Node(experiment=exp))
    assert len(tree.roots()) == 3

  def test_add_duplicate_raises(self, tree):
    exp = _completed_exp('dup')
    tree.add(Node(experiment=exp))
    with pytest.raises(ValueError, match='duplicate experiment id'):
      tree.add(Node(experiment=exp))

  def test_add_nonexistent_parent_raises(self, tree):
    parent_exp = _completed_exp('ghost')
    parent_node = Node(experiment=parent_exp)
    child_exp = Experiment(experiment_id='child')
    child_node = Node(experiment=child_exp, parent=parent_node)
    with pytest.raises(ValueError, match='not found in tree'):
      tree.add(child_node)

  def test_add_parent_completed_succeeds(self, tree):
    parent_exp = _completed_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child = Node(experiment=Experiment(experiment_id='child'), parent=parent_node)
    tree.add(child)
    assert tree.get('child') is child

  def test_add_parent_failed_succeeds(self, tree):
    parent_exp = _failed_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child = Node(experiment=Experiment(experiment_id='child'), parent=parent_node)
    tree.add(child)
    assert tree.get('child') is child

  def test_add_parent_cancelled_succeeds(self, tree):
    parent_exp = _cancelled_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child = Node(experiment=Experiment(experiment_id='child'), parent=parent_node)
    tree.add(child)
    assert tree.get('child') is child

  def test_add_parent_running_raises(self, tree):
    parent_exp = _running_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child = Node(experiment=Experiment(experiment_id='child'), parent=parent_node)
    with pytest.raises(ValueError, match='not terminal'):
      tree.add(child)

  def test_add_parent_pending_raises(self, tree):
    parent_exp = _pending_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child = Node(experiment=Experiment(experiment_id='child'), parent=parent_node)
    with pytest.raises(ValueError, match='not terminal'):
      tree.add(child)

  def test_add_root_node_any_status_succeeds(self, tree):
    for status_fn in [_pending_exp, _running_exp, _completed_exp, _failed_exp, _cancelled_exp]:
      t = Tree(name='t', store=tree.store)
      exp = status_fn(f'root-{status_fn.__name__}')
      t.add(Node(experiment=exp))
      assert t.get(exp.id) is not None


# --- update ---


class TestUpdate:
  def test_update_metrics(self, tree):
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp))
    tree.update('exp-1', metrics={'accuracy': 0.9})
    node = tree.get('exp-1')
    assert node is not None
    assert node.experiment.metrics == {'accuracy': 0.9}

  def test_update_error(self, tree):
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp))
    tree.update('exp-1', error='something broke')
    node = tree.get('exp-1')
    assert node is not None
    assert node.experiment.error == 'something broke'

  def test_update_status_raises_type_error(self, tree):
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp))
    with pytest.raises(TypeError, match='unknown keyword argument'):
      tree.update('exp-1', status=Status.failed)

  def test_update_unknown_kwarg_raises(self, tree):
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp))
    with pytest.raises(TypeError, match='unknown keyword argument'):
      tree.update('exp-1', foo='bar')

  def test_update_nonexistent_raises(self, tree):
    with pytest.raises(ValueError, match='not found'):
      tree.update('nonexistent', metrics={})


# --- remove ---


class TestRemove:
  def test_remove_leaf_node(self, tree):
    exp = _completed_exp('leaf')
    tree.add(Node(experiment=exp))
    tree.remove('leaf')
    assert tree.get('leaf') is None

  def test_remove_with_children_raises(self, tree):
    parent_exp = _completed_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child_exp = Experiment(experiment_id='child')
    tree.add(Node(experiment=child_exp, parent=parent_node))
    with pytest.raises(ValueError, match='cannot remove node with children'):
      tree.remove('parent')

  def test_remove_cascade(self, tree):
    parent_exp = _completed_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child_exp = _completed_exp('child')
    child_node = Node(experiment=child_exp, parent=parent_node)
    tree.add(child_node)
    gc_exp = Experiment(experiment_id='gc')
    tree.add(Node(experiment=gc_exp, parent=child_node))
    tree.remove('parent', cascade=True)
    assert tree.get('parent') is None
    assert tree.get('child') is None
    assert tree.get('gc') is None

  def test_remove_nonexistent_raises(self, tree):
    with pytest.raises(ValueError, match='not found'):
      tree.remove('nonexistent')

  def test_remove_no_cascade_leaf_succeeds(self, tree):
    """remove(cascade=False) on leaf node -- no error, node removed."""
    exp = _completed_exp('leaf')
    tree.add(Node(experiment=exp))
    tree.remove('leaf', cascade=False)
    assert tree.get('leaf') is None

  def test_remove_no_cascade_head_cleared(self, tree):
    """remove(cascade=False) on leaf that is head -- head cleared."""
    exp = _completed_exp('leaf')
    tree.add(Node(experiment=exp))
    tree.checkout('leaf')
    assert tree.head == 'leaf'
    tree.remove('leaf', cascade=False)
    assert tree.get('leaf') is None
    assert tree.head is None

  def test_remove_cascade_deep_chain(self, tree):
    """remove(cascade=True) with 5-level deep chain -- all descendants removed."""
    parent = Node(experiment=_completed_exp('level-0'))
    tree.add(parent)
    for i in range(1, 5):
      child = Node(experiment=_completed_exp(f'level-{i}'), parent=parent)
      tree.add(child)
      parent = child
    tree.remove('level-0', cascade=True)
    for i in range(5):
      assert tree.get(f'level-{i}') is None

  def test_remove_cascade_root_many_branches(self, tree):
    """remove(cascade=True) on root with many branches -- entire subtree gone."""
    root = Node(experiment=_completed_exp('root'))
    tree.add(root)
    for i in range(4):
      branch = Node(experiment=_completed_exp(f'branch-{i}'), parent=root)
      tree.add(branch)
      for j in range(2):
        leaf = Node(experiment=_completed_exp(f'branch-{i}-leaf-{j}'), parent=branch)
        tree.add(leaf)
    tree.remove('root', cascade=True)
    assert tree.get('root') is None
    for i in range(4):
      assert tree.get(f'branch-{i}') is None
      for j in range(2):
        assert tree.get(f'branch-{i}-leaf-{j}') is None
    assert len(tree.roots()) == 0

  def test_remove_last_node_empties_tree(self, tree):
    """Remove last node from tree -- tree is empty."""
    exp = _completed_exp('only')
    tree.add(Node(experiment=exp))
    tree.remove('only')
    assert tree.get('only') is None
    assert len(tree.roots()) == 0
    assert tree.query().count() == 0

  def test_add_after_remove_succeeds(self, tree):
    """Add a new node after removing a node -- succeeds at any valid parent."""
    root_exp = _completed_exp('root')
    root_node = Node(experiment=root_exp)
    tree.add(root_node)
    child_exp = _completed_exp('child')
    child_node = Node(experiment=child_exp, parent=root_node)
    tree.add(child_node)
    tree.remove('child')
    new_child_exp = _completed_exp('new-child')
    new_child_node = Node(experiment=new_child_exp, parent=root_node)
    tree.add(new_child_node)
    assert tree.get('new-child') is new_child_node
    new_root_exp = _completed_exp('new-root')
    tree.add(Node(experiment=new_root_exp))
    assert tree.get('new-root') is not None


# --- get ---


class TestGet:
  def test_get_existing(self, tree):
    exp = _completed_exp('exp-1')
    node = Node(experiment=exp)
    tree.add(node)
    assert tree.get('exp-1') is node

  def test_get_nonexistent(self, tree):
    assert tree.get('nonexistent') is None


# --- roots ---


class TestRoots:
  def test_roots_returns_parentless(self, tree):
    r1 = Node(experiment=_completed_exp('r1'))
    r2 = Node(experiment=_completed_exp('r2'))
    tree.add(r1)
    tree.add(r2)
    child = Node(experiment=Experiment(experiment_id='child'), parent=r1)
    tree.add(child)
    roots = tree.roots()
    ids = {n.experiment.id for n in roots}
    assert ids == {'r1', 'r2'}


# --- head ---


class TestHead:
  def test_head_initially_none(self, tree):
    assert tree.head is None

  def test_head_set_after_checkout(self, tree, mock_store):
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp))
    tree.checkout('exp-1')
    assert tree.head == 'exp-1'


# --- checkout ---


class TestCheckout:
  def test_checkout_sets_head(self, tree, mock_store):
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp))
    tree.checkout('exp-1')
    assert tree.head == 'exp-1'

  def test_checkout_calls_store(self, tree, mock_store):
    exp = _completed_exp('exp-1')
    exp.epoch = 3
    tree.add(Node(experiment=exp))
    tree.checkout('exp-1')
    mock_store.checkout.assert_called_once_with('exp-1', 3, context=None)

  def test_checkout_nonexistent_raises(self, tree):
    with pytest.raises(ValueError, match='not found'):
      tree.checkout('nonexistent')


# --- query ---


class TestQuery:
  def test_query_returns_querybuilder(self, tree):
    exp = _completed_exp('exp-1', {'accuracy': 0.8})
    tree.add(Node(experiment=exp))
    qb = tree.query()
    assert isinstance(qb, QueryBuilder)
    assert qb.count() == 1

  def test_query_scoped_to_tree(self, tree):
    for i in range(5):
      exp = _completed_exp(f'exp-{i}')
      tree.add(Node(experiment=exp))
    assert tree.query().count() == 5


# --- render ---


class TestRender:
  def test_render_empty_tree(self, tree):
    result = tree.render()
    assert 'test-tree' in result
    assert 'empty' in result

  def test_render_single_node(self, tree):
    tree.add(Node(experiment=_completed_exp('root')))
    result = tree.render()
    assert 'root' in result
    assert 'completed' in result

  def test_render_deep_tree(self, tree):
    parent = Node(experiment=_completed_exp('level-0'))
    tree.add(parent)
    for i in range(1, 6):
      exp = _completed_exp(f'level-{i}')
      node = Node(experiment=exp, parent=parent)
      tree.add(node)
      parent = node
    result = tree.render()
    assert 'level-0' in result
    assert 'level-5' in result
    lines = result.strip().split('\n')
    assert len(lines) >= 7

  def test_render_wide_tree(self, tree):
    parent = Node(experiment=_completed_exp('root'))
    tree.add(parent)
    for i in range(6):
      tree.add(Node(experiment=Experiment(experiment_id=f'child-{i}'), parent=parent))
    result = tree.render()
    for i in range(6):
      assert f'child-{i}' in result


# --- to_dict / state_dict / load_state_dict ---


class TestSerialization:
  def test_to_dict_structure(self, tree):
    root_exp = _completed_exp('root', {'accuracy': 0.8})
    root_node = Node(experiment=root_exp)
    tree.add(root_node)
    child_exp = Experiment(experiment_id='child')
    tree.add(Node(experiment=child_exp, parent=root_node))

    d = tree.to_dict()
    assert d['name'] == 'test-tree'
    assert d['description'] == 'test tree'
    assert len(d['nodes']) == 2

  def test_state_dict_matches_to_dict(self, tree):
    tree.add(Node(experiment=_completed_exp('exp-1')))
    assert tree.state_dict() == tree.to_dict()

  def test_round_trip(self, mock_store):
    tree1 = Tree(name='t1', store=mock_store, description='desc')
    root_exp = _completed_exp('root', {'accuracy': 0.8})
    root_node = Node(experiment=root_exp)
    tree1.add(root_node)
    child_exp = _completed_exp('child', {'accuracy': 0.9})
    child_node = Node(experiment=child_exp, parent=root_node)
    tree1.add(child_node)
    tree1._head = 'child'

    state = tree1.state_dict()

    tree2 = Tree(name='temp', store=mock_store)
    experiments = {
      'root': root_exp,
      'child': child_exp,
    }

    def resolver(id_str):
      if id_str in experiments:
        return experiments[id_str]
      msg = f'unknown id: {id_str}'
      raise ValueError(msg)

    tree2.load_state_dict(state, resolver)
    assert tree2.name == 't1'
    assert tree2.description == 'desc'
    assert tree2.head == 'child'
    assert tree2.get('root') is not None
    child = tree2.get('child')
    assert child is not None
    assert child.parent is not None
    assert child.parent.experiment.id == 'root'


# --- DAG integrity ---


class TestDagIntegrity:
  def test_parent_references_valid(self, tree):
    root_exp = _completed_exp('root')
    root_node = Node(experiment=root_exp)
    tree.add(root_node)
    child_exp = Experiment(experiment_id='child')
    child_node = Node(experiment=child_exp, parent=root_node)
    tree.add(child_node)

    assert child_node.parent is root_node
    assert child_node.parent.experiment.id == 'root'

  def test_self_reference_raises(self, tree):
    exp = _completed_exp('self-ref')
    node = Node(experiment=exp)
    node.parent = node
    with pytest.raises(ValueError, match='not found in tree'):
      tree.add(node)

  def test_load_state_dict_cycle_raises(self, mock_store):
    """Topological load detects cycles (A->B->A)."""
    tree = Tree(name='cycle-tree', store=mock_store)
    node_dicts = [
      {'experiment': 'a', 'parent': 'b', 'baseline': None, 'type': 'Node'},
      {'experiment': 'b', 'parent': 'a', 'baseline': None, 'type': 'Node'},
    ]
    exp_a = _completed_exp('a')
    exp_b = _completed_exp('b')
    exps = {'a': exp_a, 'b': exp_b}

    def resolver(id_str):
      return exps[id_str]

    state = {
      'name': 'cycle-tree',
      'description': None,
      'head': None,
      'nodes': node_dicts,
    }
    with pytest.raises(ValueError, match='cycle detected'):
      tree.load_state_dict(state, resolver)


# --- on_change callback ---


class TestOnChange:
  def test_add_triggers_callback(self, mock_store):
    callback = MagicMock()
    tree = Tree(name='t', store=mock_store, on_change=callback)
    tree.add(Node(experiment=_completed_exp('exp-1')))
    callback.assert_called_once()

  def test_update_triggers_callback(self, mock_store):
    callback = MagicMock()
    tree = Tree(name='t', store=mock_store, on_change=callback)
    tree.add(Node(experiment=_completed_exp('exp-1')))
    callback.reset_mock()
    tree.update('exp-1', metrics={'x': 1.0})
    callback.assert_called_once()

  def test_remove_triggers_callback(self, mock_store):
    callback = MagicMock()
    tree = Tree(name='t', store=mock_store, on_change=callback)
    tree.add(Node(experiment=_completed_exp('exp-1')))
    callback.reset_mock()
    tree.remove('exp-1')
    callback.assert_called_once()


class TestTreeCheckoutSyncsHead:
  """Tree.checkout sets Tree._head and persists refs HEAD via store.checkout."""

  def test_tree_checkout_syncs_head(self, tmp_path: Path) -> None:
    param_file = tmp_path / 'test.txt'
    param_file.write_text('hello', encoding='utf-8')

    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path = tmp_path / '.autopilot'
    param = PathParameter(source=str(param_file))
    store = FileStore(config)
    store.register_parameters({'source': param})

    exp = Experiment(experiment_id='exp-1')
    exp.start()
    exp.advance_epoch()
    store.snapshot('exp-1', 0)

    exp.complete()

    exp2 = Experiment(experiment_id='exp-2')
    exp2.start()
    exp2.advance_epoch()
    store.branch('exp-2')
    store.snapshot('exp-2', 1)

    tree = Tree(name='test-tree', store=store)
    tree.add(Node(experiment=exp))
    tree.add(Node(experiment=exp2, parent=tree.get('exp-1')))

    tree.checkout('exp-1')
    assert tree.head == 'exp-1'
    refs = store.load_refs()
    assert refs['HEAD'] == 'exp-1'

    tree.checkout('exp-2')
    assert tree.head == 'exp-2'
    refs = store.load_refs()
    assert refs['HEAD'] == 'exp-2'
