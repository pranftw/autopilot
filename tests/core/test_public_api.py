"""Tests for public API accessibility (sub-plan 03 + prior plan 15 tests).

Verifies that cross-boundary APIs are public, private patterns are gone,
and the public contract is correct.
"""

from autopilot.ai.environment import IsolatedEnvironment
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.module.module import Module
from autopilot.core.store.base import Store
from autopilot.core.trainer.trainer import Trainer
from autopilot.core.types import EvalDatum
from pathlib import Path
from typing import Any, cast
import inspect
import pytest


class _StubModule(Module):
  def forward(self, *args: Any, **kwargs: Any) -> EvalDatum:
    return EvalDatum(success=True)


class TestStoreABCAbstractMethods:
  """Store ABC: load_snapshot, read_object, load_refs are abstract methods."""

  def test_load_snapshot_is_abstract(self) -> None:
    assert hasattr(Store, 'load_snapshot')
    assert cast(Any, Store.load_snapshot).__isabstractmethod__

  def test_read_object_is_abstract(self) -> None:
    assert hasattr(Store, 'read_object')
    assert cast(Any, Store.read_object).__isabstractmethod__

  def test_load_refs_is_abstract(self) -> None:
    assert hasattr(Store, 'load_refs')
    assert cast(Any, Store.load_refs).__isabstractmethod__

  def test_store_init_raises_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError):
      Store(cast(Any, None))


def _make_config(tmp_path: Path) -> AutoPilotConfig:
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  return config


class TestFileStoreLoadSnapshot:
  """FileStore.load_snapshot returns SnapshotManifest for a valid experiment/epoch."""

  def test_load_snapshot_returns_manifest(self, tmp_path: Path) -> None:
    param_file = tmp_path / 'test.txt'
    param_file.write_text('hello', encoding='utf-8')

    config = _make_config(tmp_path)
    param = PathParameter(source=str(param_file))
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-1', 0)

    manifest = store.load_snapshot('exp-1', 0)
    assert manifest.epoch == 0
    assert len(manifest.entries) > 0


class TestFileStoreReadObject:
  """FileStore.read_object returns bytes for a valid hash."""

  def test_read_object_returns_bytes(self, tmp_path: Path) -> None:
    param_file = tmp_path / 'test.txt'
    param_file.write_text('content', encoding='utf-8')

    config = _make_config(tmp_path)
    param = PathParameter(source=str(param_file))
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-1', 0)

    manifest = store.load_snapshot('exp-1', 0)
    for entry in manifest.entries.values():
      data = store.read_object(entry.digest)
      assert isinstance(data, bytes)
      assert len(data) > 0


class TestFileStoreLoadRefs:
  """FileStore.load_refs returns dict."""

  def test_load_refs_returns_empty_dict_initially(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = FileStore(config)
    refs = store.load_refs()
    assert isinstance(refs, dict)

  def test_load_refs_returns_branches_after_snapshot(self, tmp_path: Path) -> None:
    param_file = tmp_path / 'test.txt'
    param_file.write_text('hi', encoding='utf-8')

    config = _make_config(tmp_path)
    param = PathParameter(source=str(param_file))
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-1', 0)

    refs = store.load_refs()
    assert 'branches' in refs
    assert 'exp-1' in refs['branches']
    assert refs['HEAD'] == 'exp-1'


class TestIsolatedEnvironmentUsesPublicStoreAPI:
  """IsolatedEnvironment._load_snapshot_content uses public store API."""

  def test_no_private_load_snapshot_call(self) -> None:
    source = inspect.getsource(IsolatedEnvironment._load_snapshot_content)
    assert '_load_snapshot(' not in source
    assert 'self._store.load_snapshot(' in source

  def test_no_private_read_object_call(self) -> None:
    source = inspect.getsource(IsolatedEnvironment._load_snapshot_content)
    assert '_read_object(' not in source
    assert 'self._store.read_object(' in source

  def test_get_snapshot_content_uses_public_load_refs(self) -> None:
    source = inspect.getsource(IsolatedEnvironment._get_snapshot_content)
    assert '_load_refs(' not in source
    assert 'self._store.load_refs()' in source


class TestTrainerDispatchCallbacks:
  """Trainer.dispatch_callbacks is callable (not _dispatch)."""

  def test_dispatch_callbacks_is_public(self) -> None:
    trainer = Trainer()
    assert hasattr(trainer, 'dispatch_callbacks')
    assert callable(trainer.dispatch_callbacks)

  def test_no_private_dispatch(self) -> None:
    assert not hasattr(Trainer, '_dispatch')

  def test_dispatch_callbacks_returns_list(self) -> None:
    trainer = Trainer()
    result = trainer.dispatch_callbacks('on_nonexistent_hook')
    assert isinstance(result, list)
    assert result == []


class TestTrainerStore:
  """Trainer(store=store) -> trainer.store returns the store."""

  def test_store_with_value(self, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    store = FileStore(config)
    trainer = Trainer(store=store)
    assert trainer.store is store

  def test_store_none(self) -> None:
    trainer = Trainer(store=None)
    assert trainer.store is None

  def test_store_default_is_none(self) -> None:
    trainer = Trainer()
    assert trainer.store is None


class TestExperimentPublicAttributes:
  """Experiment has store, last_accepted_epoch, rollback as real attributes."""

  def test_store_property_exists(self) -> None:
    exp = Experiment(experiment_id='t')
    assert hasattr(exp, 'store')
    assert exp.store is None

  def test_store_settable(self) -> None:
    exp = Experiment(experiment_id='t')
    sentinel = object()
    exp.store = sentinel
    assert exp.store is sentinel

  def test_last_accepted_epoch_property_exists(self) -> None:
    exp = Experiment(experiment_id='t')
    assert hasattr(exp, 'last_accepted_epoch')
    assert exp.last_accepted_epoch is None

  def test_last_accepted_epoch_settable(self) -> None:
    exp = Experiment(experiment_id='t')
    exp.last_accepted_epoch = 3
    assert exp.last_accepted_epoch == 3

  def test_rollback_callable(self) -> None:
    exp = Experiment(experiment_id='t')
    assert hasattr(exp, 'rollback')
    assert callable(exp.rollback)
    exp.rollback(0)

  def test_epoch_attribute(self) -> None:
    exp = Experiment(experiment_id='t')
    assert exp.epoch == -1

  def test_status_attribute(self) -> None:
    from autopilot.core.enums import Status

    exp = Experiment(experiment_id='t')
    assert exp.status == Status.pending


class TestExperimentNoOnLoopComplete:
  """Experiment does NOT have on_loop_complete method."""

  def test_no_on_loop_complete(self) -> None:
    exp = Experiment(experiment_id='t')
    assert not hasattr(exp, 'on_loop_complete')

  def test_no_on_loop_complete_on_class(self) -> None:
    assert 'on_loop_complete' not in Experiment.__dict__
    for cls in Experiment.__mro__:
      assert 'on_loop_complete' not in cls.__dict__


# --- Plan 03: Public API Accessibility ---


class TestOptimizerParametersProperty:
  """Optimizer.parameters read-only property (Plan 03, BFR-09)."""

  def test_optimizer_parameters_property(self) -> None:
    from autopilot.core.parameter import Parameter

    p1 = Parameter()
    p2 = Parameter()
    from tests.doubles import NoOpOptimizer

    opt = NoOpOptimizer([p1, p2])
    assert opt.parameters == [p1, p2]

  def test_optimizer_parameters_copy_semantics(self) -> None:
    from autopilot.core.parameter import Parameter
    from tests.doubles import NoOpOptimizer

    p1 = Parameter()
    opt = NoOpOptimizer([p1])
    returned = opt.parameters
    returned.append(Parameter())
    assert len(opt.parameters) == 1

  def test_optimizer_parameters_assignment_rejected(self) -> None:
    from autopilot.core.parameter import Parameter
    from tests.doubles import NoOpOptimizer

    opt = NoOpOptimizer([Parameter()])
    attr = 'parameters'
    with pytest.raises(AttributeError):
      setattr(opt, attr, [])

  def test_optimizer_parameters_empty(self) -> None:
    from tests.doubles import NoOpOptimizer

    opt = NoOpOptimizer([])
    assert opt.parameters == []

  def test_optimizer_parameters_preserves_order(self) -> None:
    from autopilot.core.parameter import Parameter
    from tests.doubles import NoOpOptimizer

    params = [Parameter() for _ in range(5)]
    opt = NoOpOptimizer(params)
    assert opt.parameters == params

  def test_zero_grad_uses_parameters_property(self) -> None:
    from autopilot.core.gradient import NumericGradient
    from autopilot.core.parameter import Parameter
    from tests.doubles import NoOpOptimizer

    p = Parameter()
    p.grad = NumericGradient(value=1.0)
    opt = NoOpOptimizer([p])
    opt.zero_grad()
    assert p.grad is None


class TestExperimentIdGetattr:
  """Experiment.__getattr__ guides experiment_id -> .id (Plan 03, BFR-11)."""

  def test_experiment_id_stored_on_id(self) -> None:
    exp = Experiment(experiment_id='slug')
    assert exp.id == 'slug'

  def test_experiment_experiment_id_getattr_raises(self) -> None:
    exp = Experiment(experiment_id='slug')
    with pytest.raises(AttributeError, match=r"use '\.id'"):
      _ = exp.experiment_id  # type: ignore[attr-defined]

  def test_experiment_other_missing_attr_raises(self) -> None:
    exp = Experiment(experiment_id='slug')
    with pytest.raises(AttributeError, match='no attribute'):
      _ = exp.nonexistent_field  # type: ignore[attr-defined]

  def test_experiment_getattr_message_contains_guidance(self) -> None:
    exp = Experiment(experiment_id='slug')
    with pytest.raises(AttributeError) as exc_info:
      _ = exp.experiment_id  # type: ignore[attr-defined]
    assert 'experiment_id' in str(exc_info.value)
    assert 'constructor argument' in str(exc_info.value)

  def test_experiment_hasattr_experiment_id_is_false(self) -> None:
    exp = Experiment(experiment_id='slug')
    assert not hasattr(exp, 'experiment_id')

  def test_experiment_existing_attrs_unaffected(self) -> None:
    exp = Experiment(experiment_id='slug')
    assert exp.status is not None
    assert exp.hypothesis is None
    assert exp.metrics == {}


class TestTreeNodesProperty:
  """Tree.nodes shallow-copy property (Plan 03)."""

  def test_tree_nodes_property(self) -> None:
    from autopilot.core.node import Node
    from autopilot.core.tree import Tree

    store = cast(Any, None)
    tree = Tree(name='test', store=store)
    exp = Experiment(experiment_id='e1')
    exp.status = _terminal_status()
    node = Node(experiment=exp)
    tree._nodes['e1'] = node
    assert tree.nodes['e1'] is tree.get('e1')

  def test_tree_nodes_dict_not_identity(self) -> None:
    from autopilot.core.node import Node
    from autopilot.core.tree import Tree

    store = cast(Any, None)
    tree = Tree(name='test', store=store)
    exp = Experiment(experiment_id='e1')
    exp.status = _terminal_status()
    node = Node(experiment=exp)
    tree._nodes['e1'] = node
    len_before = len(tree.nodes)
    nodes_copy = tree.nodes
    nodes_copy.popitem()
    assert len(tree.nodes) == len_before

  def test_tree_nodes_empty_tree(self) -> None:
    from autopilot.core.tree import Tree

    store = cast(Any, None)
    tree = Tree(name='empty', store=store)
    assert tree.nodes == {}

  def test_tree_nodes_multiple_entries(self) -> None:
    from autopilot.core.node import Node
    from autopilot.core.tree import Tree

    store = cast(Any, None)
    tree = Tree(name='test', store=store)
    for i in range(3):
      exp = Experiment(experiment_id=f'e{i}')
      tree._nodes[f'e{i}'] = Node(experiment=exp)
    assert len(tree.nodes) == 3


class TestForestTreesProperty:
  """Forest.trees property (Plan 03)."""

  def test_forest_trees_property(self) -> None:
    from autopilot.core.forest import Forest

    store = cast(Any, None)
    forest = Forest(store=store)
    forest.create_tree('alpha')
    forest.create_tree('beta')
    assert forest.trees == forest.list_trees()
    assert len(forest.trees) == 2

  def test_forest_trees_element_identity(self) -> None:
    from autopilot.core.forest import Forest

    store = cast(Any, None)
    forest = Forest(store=store)
    forest.create_tree('alpha')
    trees_prop = forest.trees
    trees_list = forest.list_trees()
    for a, b in zip(trees_prop, trees_list, strict=True):
      assert a is b

  def test_forest_trees_empty(self) -> None:
    from autopilot.core.forest import Forest

    store = cast(Any, None)
    forest = Forest(store=store)
    assert forest.trees == []


class TestTreeDuplicateIdRaises:
  """Tree.add() raises ValueError on duplicate id."""

  def test_tree_duplicate_id_warns(self) -> None:
    from autopilot.core.node import Node
    from autopilot.core.tree import Tree

    store = cast(Any, None)
    tree = Tree(name='test', store=store)
    exp = Experiment(experiment_id='dup')
    node = Node(experiment=exp)
    tree._nodes['dup'] = node

    dup_exp = Experiment(experiment_id='dup')
    dup_node = Node(experiment=dup_exp)
    with pytest.raises(ValueError, match='duplicate experiment id'):
      tree.add(dup_node)

  def test_tree_duplicate_id_error_contains_id(self) -> None:
    from autopilot.core.node import Node
    from autopilot.core.tree import Tree

    store = cast(Any, None)
    tree = Tree(name='my-tree', store=store)
    exp = Experiment(experiment_id='x')
    tree._nodes['x'] = Node(experiment=exp)

    with pytest.raises(ValueError, match='duplicate experiment id'):
      tree.add(Node(experiment=Experiment(experiment_id='x')))

  def test_tree_non_duplicate_no_warning(self) -> None:
    from autopilot.core.node import Node
    from autopilot.core.tree import Tree

    store = cast(Any, None)
    tree = Tree(name='test', store=store)
    exp1 = Experiment(experiment_id='a')
    node1 = Node(experiment=exp1)
    tree.add(node1)


def _terminal_status():
  """Return a terminal Status for test setup."""
  from autopilot.core.enums import Status

  return Status.completed
