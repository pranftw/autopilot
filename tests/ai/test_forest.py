"""Tests for FileForest file-backed persistence."""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from tests.core.conftest import completed_exp as _completed_exp
from tests.core.conftest import failed_exp
from unittest.mock import patch
import pytest


@pytest.fixture
def config(tmp_path):
  return AutoPilotConfig(workspace=tmp_path)


@pytest.fixture
def store(config):
  return FileStore(config)


@pytest.fixture
def file_forest(store):
  return FileForest(store)


def _failed_exp(id_: str) -> Experiment:
  return failed_exp(id_, auto_hypothesis=False)


# --- persistence ---


class TestPersistence:
  def test_create_save_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('tree-1', description='first tree')
    exp1 = _completed_exp('exp-1', {'accuracy': 0.8})
    tree.add(Node(experiment=exp1))
    exp2 = _completed_exp('exp-2', {'accuracy': 0.9})
    tree.add(Node(experiment=exp2, parent=tree.get('exp-1')))

    ff2 = FileForest(store)
    assert len(ff2.list_trees()) == 1
    loaded_tree = ff2.get_tree('tree-1')
    assert loaded_tree is not None
    assert loaded_tree.description == 'first tree'
    assert loaded_tree.get('exp-1') is not None
    assert loaded_tree.get('exp-2') is not None

  def test_multiple_trees_persist(self, store):
    ff1 = FileForest(store)
    ff1.create_tree('a')
    ff1.create_tree('b')
    ff1.switch('a')

    ff2 = FileForest(store)
    assert len(ff2.list_trees()) == 2
    assert ff2._active_name == 'a'

  def test_node_parent_survives_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    parent_exp = _completed_exp('parent', {'accuracy': 0.7})
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child_exp = _completed_exp('child', {'accuracy': 0.8})
    tree.add(Node(experiment=child_exp, parent=parent_node))

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None
    assert loaded_tree.get('child') is not None

  def test_experiment_state_survives_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    exp = _completed_exp('exp-1', {'accuracy': 0.85})
    tree.add(Node(experiment=exp))

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None
    node_exp1 = loaded_tree.get('exp-1')
    assert node_exp1 is not None
    loaded_exp = node_exp1.experiment
    assert loaded_exp.status == Status.completed
    assert loaded_exp.metrics == {'accuracy': 0.85}
    assert loaded_exp.hypothesis == 'exp-1 hypothesis'

  def test_head_survives_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp))

    tree._head = 'exp-1'
    ff1.save()

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None
    assert loaded_tree.head == 'exp-1'


# --- auto-save ---


class TestAutoSave:
  def test_create_tree_triggers_save(self, store):
    ff = FileForest(store)
    with patch.object(ff, 'save', wraps=ff.save) as mock_save:
      ff.create_tree('t')
      mock_save.assert_called()

  def test_switch_triggers_save(self, store):
    ff = FileForest(store)
    ff.create_tree('t')
    with patch.object(ff, 'save', wraps=ff.save) as mock_save:
      ff.switch('t')
      mock_save.assert_called()

  def test_tree_add_triggers_save(self, store):
    ff = FileForest(store)
    tree = ff.create_tree('t')
    tree.add(Node(experiment=_completed_exp('e1')))
    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    node_e1 = tree_reloaded.get('e1')
    assert node_e1 is not None

  def test_tree_update_triggers_save(self, store):
    ff = FileForest(store)
    tree = ff.create_tree('t')
    tree.add(Node(experiment=_completed_exp('e1', {'accuracy': 0.7})))
    tree.update('e1', metrics={'accuracy': 0.9})
    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    node_e1 = tree_reloaded.get('e1')
    assert node_e1 is not None
    assert node_e1.experiment.metrics == {'accuracy': 0.9}

  def test_tree_remove_triggers_save(self, store):
    ff = FileForest(store)
    tree = ff.create_tree('t')
    tree.add(Node(experiment=_completed_exp('e1')))
    tree.remove('e1')
    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    assert tree_reloaded.get('e1') is None

  def test_remove_tree_triggers_save(self, store):
    ff = FileForest(store)
    ff.create_tree('t')
    with patch.object(ff, 'save', wraps=ff.save) as mock_save:
      ff.remove_tree('t')
      mock_save.assert_called()


# --- object resolution ---


class TestObjectResolution:
  def test_experiment_is_real_object(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    exp = _completed_exp('exp-1', {'accuracy': 0.8})
    tree.add(Node(experiment=exp))

    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    loaded = tree_reloaded.get('exp-1')
    assert loaded is not None
    assert isinstance(loaded.experiment, Experiment)
    assert loaded.experiment.id == 'exp-1'
    assert loaded.experiment.metrics == {'accuracy': 0.8}

  def test_parent_is_real_node(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    parent_exp = _completed_exp('parent')
    parent_node = Node(experiment=parent_exp)
    tree.add(parent_node)
    child_exp = _completed_exp('child')
    tree.add(Node(experiment=child_exp, parent=parent_node))

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None
    child = loaded_tree.get('child')
    assert child is not None
    assert isinstance(child.parent, Node)
    assert child.parent.experiment.id == 'parent'
    assert isinstance(child.parent.experiment, Experiment)

  def test_baseline_is_real_node(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    base_exp = _completed_exp('baseline')
    base_node = Node(experiment=base_exp)
    tree.add(base_node)
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp, parent=base_node, baseline=base_node))

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None
    n = loaded_tree.get('exp-1')
    assert n is not None
    assert n.baseline is not None
    assert n.baseline.experiment.id == 'baseline'

  def test_deep_tree_resolution(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    parent = None
    for i in range(5):
      exp = _completed_exp(f'level-{i}')
      node = Node(experiment=exp, parent=parent)
      tree.add(node)
      parent = node

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None
    leaf = loaded_tree.get('level-4')
    assert leaf is not None
    chain = []
    current = leaf
    while current is not None:
      chain.append(current.experiment.id)
      current = current.parent
    assert chain == ['level-4', 'level-3', 'level-2', 'level-1', 'level-0']

  def test_query_works_after_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    tree.add(Node(experiment=_completed_exp('exp-1', {'accuracy': 0.7})))
    tree.add(Node(experiment=_completed_exp('exp-2', {'accuracy': 0.9})))

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None
    best = loaded_tree.query().best('accuracy')
    assert best is not None
    assert best.experiment.id == 'exp-2'


# --- empty forest ---


class TestEmptyForest:
  def test_empty_forest_loads(self, store):
    ff = FileForest(store)
    assert ff.list_trees() == []
    assert ff.active is None

  def test_empty_forest_save_load(self, store):
    ff1 = FileForest(store)
    ff1.save()
    ff2 = FileForest(store)
    assert ff2.list_trees() == []
