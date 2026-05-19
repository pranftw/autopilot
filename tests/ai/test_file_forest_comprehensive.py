"""Comprehensive tests for FileForest file-backed persistence.

Covers experiment_state persistence, auto-save via on_change, missing
experiment_state defaults to pending, create/switch/remove cycle,
and FileForest surviving process restart (write, reload from new instance).
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.enums import Status
from autopilot.core.node import Node
from autopilot.tracking.io import atomic_write_json
from tests.core.conftest import (
  completed_exp as _completed_exp,
)
from tests.core.conftest import (
  failed_exp as _failed_exp,
)
from tests.core.conftest import (
  pending_exp as _pending_exp,
)
import pytest


@pytest.fixture
def config(tmp_path):
  return AutoPilotConfig(workspace=tmp_path)


@pytest.fixture
def store(config):
  return FileStore(config)


# --- experiment_state persistence ---


class TestExperimentStatePersistence:
  def test_completed_status_preserved_after_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    exp = _completed_exp('exp-1', {'accuracy': 0.85})
    tree.add(Node(experiment=exp))

    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    node_e1 = tree_reloaded.get('exp-1')
    assert node_e1 is not None
    loaded = node_e1.experiment
    assert loaded.status == Status.completed
    assert loaded.metrics == {'accuracy': 0.85}

  def test_failed_status_preserved_after_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    exp = _failed_exp('exp-1')
    tree.add(Node(experiment=exp))

    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    node_e1 = tree_reloaded.get('exp-1')
    assert node_e1 is not None
    loaded = node_e1.experiment
    assert loaded.status == Status.failed

  def test_pending_status_preserved_after_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    exp = _pending_exp('exp-1')
    tree.add(Node(experiment=exp))

    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    node_e1 = tree_reloaded.get('exp-1')
    assert node_e1 is not None
    loaded = node_e1.experiment
    assert loaded.status == Status.pending

  def test_hypothesis_preserved_after_reload(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    exp = _completed_exp('exp-1')
    tree.add(Node(experiment=exp))

    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    node_e1 = tree_reloaded.get('exp-1')
    assert node_e1 is not None
    loaded = node_e1.experiment
    assert loaded.hypothesis == 'exp-1 hypothesis'

  def test_multiple_experiments_state_preserved(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    exp1 = _completed_exp('e1', {'accuracy': 0.7})
    exp2 = _failed_exp('e2')
    exp3 = _pending_exp('e3')
    tree.add(Node(experiment=exp1))
    tree.add(Node(experiment=exp2))
    tree.add(Node(experiment=exp3))

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None
    ne1 = loaded_tree.get('e1')
    ne2 = loaded_tree.get('e2')
    ne3 = loaded_tree.get('e3')
    assert ne1 is not None
    assert ne2 is not None
    assert ne3 is not None
    assert ne1.experiment.status == Status.completed
    assert ne2.experiment.status == Status.failed
    assert ne3.experiment.status == Status.pending


# --- Auto-save via on_change ---


class TestAutoSaveOnChange:
  def test_add_node_updates_forest_file(self, store, config):
    ff = FileForest(store)
    tree = ff.create_tree('t')
    forest_file = config.forest_file

    mtime_before = forest_file.stat().st_mtime_ns

    tree.add(Node(experiment=_completed_exp('e1')))

    mtime_after = forest_file.stat().st_mtime_ns
    assert mtime_after >= mtime_before

    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    assert tree_reloaded.get('e1') is not None

  def test_remove_node_updates_forest_file(self, store):
    ff = FileForest(store)
    tree = ff.create_tree('t')
    tree.add(Node(experiment=_completed_exp('e1')))

    tree.remove('e1')

    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    assert tree_reloaded.get('e1') is None

  def test_update_node_triggers_save(self, store):
    ff = FileForest(store)
    tree = ff.create_tree('t')
    tree.add(Node(experiment=_completed_exp('e1', {'accuracy': 0.5})))
    tree.update('e1', metrics={'accuracy': 0.9})

    ff2 = FileForest(store)
    tree_reloaded = ff2.get_tree('t')
    assert tree_reloaded is not None
    node_e1 = tree_reloaded.get('e1')
    assert node_e1 is not None
    loaded = node_e1.experiment
    assert loaded.metrics == {'accuracy': 0.9}

  def test_create_tree_triggers_save(self, store):
    ff = FileForest(store)
    ff.create_tree('t')

    ff2 = FileForest(store)
    assert ff2.get_tree('t') is not None

  def test_switch_triggers_save(self, store):
    ff = FileForest(store)
    ff.create_tree('t1')
    ff.create_tree('t2')
    ff.switch('t2')

    ff2 = FileForest(store)
    assert ff2._active_name == 't2'


# --- Missing experiment_state defaults to pending ---


class TestMissingExperimentStateDefault:
  def test_missing_experiment_state_defaults_to_pending(self, store, config):
    state = {
      'active': None,
      'trees': [
        {
          'name': 't',
          'description': None,
          'head': None,
          'nodes': [
            {
              'type': 'Node',
              'experiment': 'e1',
              'parent': None,
              'baseline': None,
            }
          ],
        }
      ],
    }
    atomic_write_json(config.forest_file, state)

    ff = FileForest(store)
    tree_t = ff.get_tree('t')
    assert tree_t is not None
    node_e1 = tree_t.get('e1')
    assert node_e1 is not None
    loaded = node_e1.experiment
    assert loaded.status == Status.pending


# --- Create, switch, remove cycle ---


class TestCreateSwitchRemoveCycle:
  def test_full_lifecycle_cycle(self, store):
    ff = FileForest(store)

    tree1 = ff.create_tree('t1', description='first')
    tree2 = ff.create_tree('t2', description='second')

    tree1.add(Node(experiment=_completed_exp('e1', {'accuracy': 0.7})))
    tree2.add(Node(experiment=_completed_exp('e2', {'accuracy': 0.8})))

    ff.switch('t1')
    active_t1 = ff.active
    assert active_t1 is not None
    assert active_t1.name == 't1'

    ff.switch('t2')
    active_t2 = ff.active
    assert active_t2 is not None
    assert active_t2.name == 't2'

    ff.remove_tree('t1')
    assert ff.get_tree('t1') is None
    assert len(ff.list_trees()) == 1

    ff.remove_tree('t2')
    assert ff.active is None
    assert ff.list_trees() == []

    ff2 = FileForest(store)
    assert ff2.list_trees() == []
    assert ff2.active is None

  def test_create_after_remove_same_name(self, store):
    ff = FileForest(store)
    ff.create_tree('t')
    ff.remove_tree('t')
    tree = ff.create_tree('t', description='recreated')
    assert tree.name == 't'
    assert tree.description == 'recreated'
    assert len(tree.query().all()) == 0

  def test_switch_to_only_remaining_tree(self, store):
    ff = FileForest(store)
    ff.create_tree('t1')
    ff.create_tree('t2')
    ff.switch('t1')
    ff.remove_tree('t2')
    active_ff = ff.active
    assert active_ff is not None
    assert active_ff.name == 't1'

    ff2 = FileForest(store)
    active_ff2 = ff2.active
    assert active_ff2 is not None
    assert active_ff2.name == 't1'


# --- FileForest survives process restart ---


class TestProcessRestart:
  def test_write_reload_from_new_instance(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('experiment-tree', description='main exploration')
    root_exp = _completed_exp('root', {'accuracy': 0.7})
    root_node = Node(experiment=root_exp)
    tree.add(root_node)

    child_exp = _completed_exp('child', {'accuracy': 0.85})
    tree.add(Node(experiment=child_exp, parent=root_node))

    ff1.switch('experiment-tree')

    ff2 = FileForest(store)

    assert len(ff2.list_trees()) == 1
    assert ff2._active_name == 'experiment-tree'
    loaded_tree = ff2.get_tree('experiment-tree')
    assert loaded_tree is not None
    assert loaded_tree.description == 'main exploration'
    assert loaded_tree.get('root') is not None
    assert loaded_tree.get('child') is not None
    child_node = loaded_tree.get('child')
    root_node = loaded_tree.get('root')
    assert child_node is not None
    assert root_node is not None
    parent_of_child = child_node.parent
    assert parent_of_child is not None
    assert parent_of_child.experiment.id == 'root'
    assert root_node.experiment.status == Status.completed
    assert root_node.experiment.metrics == {'accuracy': 0.7}
    assert child_node.experiment.metrics == {'accuracy': 0.85}

  def test_multiple_restarts(self, store):
    ff1 = FileForest(store)
    ff1.create_tree('t')
    tree_ff1 = ff1.get_tree('t')
    assert tree_ff1 is not None
    tree_ff1.add(Node(experiment=_completed_exp('e1')))

    ff2 = FileForest(store)
    tree_ff2 = ff2.get_tree('t')
    assert tree_ff2 is not None
    tree_ff2.add(Node(experiment=_completed_exp('e2')))

    ff3 = FileForest(store)
    tree_ff3 = ff3.get_tree('t')
    assert tree_ff3 is not None
    tree_ff3.add(Node(experiment=_completed_exp('e3')))

    ff4 = FileForest(store)
    loaded_tree = ff4.get_tree('t')
    assert loaded_tree is not None
    assert loaded_tree.get('e1') is not None
    assert loaded_tree.get('e2') is not None
    assert loaded_tree.get('e3') is not None
    assert loaded_tree.query().count() == 3

  def test_restart_preserves_query_capability(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    tree.add(Node(experiment=_completed_exp('e1', {'accuracy': 0.7})))
    tree.add(Node(experiment=_completed_exp('e2', {'accuracy': 0.9})))
    tree.add(Node(experiment=_failed_exp('e3')))

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None

    assert loaded_tree.query().completed().count() == 2
    assert loaded_tree.query().failed().count() == 1

    best = loaded_tree.query().best('accuracy')
    assert best is not None
    assert best.experiment.id == 'e2'

  def test_restart_with_deep_tree(self, store):
    ff1 = FileForest(store)
    tree = ff1.create_tree('t')
    parent = None
    for i in range(7):
      exp = _completed_exp(f'level-{i}', {'accuracy': i * 0.1})
      node = Node(experiment=exp, parent=parent)
      tree.add(node)
      parent = node

    ff2 = FileForest(store)
    loaded_tree = ff2.get_tree('t')
    assert loaded_tree is not None

    leaf = loaded_tree.get('level-6')
    assert leaf is not None
    chain = []
    current = leaf
    while current is not None:
      chain.append(current.experiment.id)
      current = current.parent
    assert chain == [f'level-{i}' for i in range(6, -1, -1)]

  def test_restart_with_multiple_trees(self, store):
    ff1 = FileForest(store)
    for i in range(5):
      tree = ff1.create_tree(f't{i}')
      tree.add(Node(experiment=_completed_exp(f'exp-{i}', {'score': float(i)})))
    ff1.switch('t2')

    ff2 = FileForest(store)
    assert len(ff2.list_trees()) == 5
    assert ff2._active_name == 't2'
    for i in range(5):
      t = ff2.get_tree(f't{i}')
      assert t is not None
      assert t.get(f'exp-{i}') is not None

    best = ff2.query().best('score')
    assert best is not None
    assert best.experiment.id == 'exp-4'
