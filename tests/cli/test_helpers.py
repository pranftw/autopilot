"""Tests for autopilot.cli.helpers: load_forest, require_active_tree,
require_experiment_node, store_vcs_arguments, with_store_vcs_arguments.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.store.file_store import FileStore
from autopilot.cli.helpers import (
  load_forest,
  require_active_tree,
  require_experiment_node,
  store_vcs_arguments,
  with_store_vcs_arguments,
)
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.forest import Forest
from autopilot.core.node import Node
from pathlib import Path
from tests.cli.conftest import make_ctx
from typing import Any, cast
from unittest.mock import patch
import pytest


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  ws = tmp_path / 'ws'
  ws.mkdir()
  return ws


@pytest.fixture
def ctx(ws: Path):
  return make_ctx(ws)


@pytest.fixture
def forest_with_tree(ws: Path) -> FileForest:
  """Forest with a single active tree named 'main'."""
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  store = FileStore(config)
  forest = FileForest(store)
  forest.create_tree('main')
  forest.switch('main')
  return forest


@pytest.fixture
def forest_with_experiment(forest_with_tree: FileForest) -> FileForest:
  """Forest with an active tree containing experiment 'exp-1'."""
  tree = forest_with_tree.active
  assert tree is not None
  exp = Experiment(experiment_id='exp-1', hypothesis='test hypothesis')
  node = Node(experiment=exp)
  tree.add(node)
  forest_with_tree.save()
  return forest_with_tree


class TestLoadForest:
  def test_creates_store_path(self, ctx, ws: Path) -> None:
    result = load_forest(ctx)
    assert ctx.config.store_path.is_dir()
    assert isinstance(result, Forest)

  def test_returns_file_forest(self, ctx) -> None:
    forest = load_forest(ctx)
    assert isinstance(forest, Forest)

  def test_invalid_config_propagates_error(self, ctx) -> None:
    with (
      patch.object(
        type(ctx.config.store_path),
        'mkdir',
        side_effect=PermissionError('denied'),
      ),
      pytest.raises(PermissionError),
    ):
      load_forest(ctx)


class TestRequireActiveTree:
  def test_returns_active_tree(self, ws: Path) -> None:
    ctx = make_ctx(ws)
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)
    forest.create_tree('main')
    forest.switch('main')

    tree = require_active_tree(ctx, forest)
    assert tree is forest.active
    assert tree.name == 'main'

  def test_fails_without_active_tree(self, ws: Path) -> None:
    ctx = make_ctx(ws)
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    with pytest.raises(SystemExit):
      require_active_tree(ctx, forest)

  def test_default_message(self, ws: Path) -> None:
    ctx = make_ctx(ws)
    ctx.fail = cast(Any, lambda msg, **kw: (_ for _ in ()).throw(RuntimeError(msg)))
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    with pytest.raises(RuntimeError, match='no active tree in forest'):
      require_active_tree(ctx, forest)

  def test_custom_message(self, ws: Path) -> None:
    ctx = make_ctx(ws)
    ctx.fail = cast(Any, lambda msg, **kw: (_ for _ in ()).throw(RuntimeError(msg)))
    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    forest = FileForest(store)

    with pytest.raises(RuntimeError, match='specify a name or switch first'):
      require_active_tree(ctx, forest, message='specify a name or switch first')


class TestRequireExperimentNode:
  def test_returns_node_when_found(self, ws: Path, forest_with_experiment: FileForest) -> None:
    ctx = make_ctx(ws)
    tree = forest_with_experiment.active
    node = require_experiment_node(ctx, cast(Any, tree), 'exp-1')
    assert node.experiment.id == 'exp-1'

  def test_fails_when_missing(self, ws: Path, forest_with_experiment: FileForest) -> None:
    ctx = make_ctx(ws)
    ctx.fail = cast(Any, lambda msg, **kw: (_ for _ in ()).throw(RuntimeError(msg)))
    tree = forest_with_experiment.active

    with pytest.raises(RuntimeError, match="Experiment 'ghost' not found in tree"):
      require_experiment_node(ctx, cast(Any, tree), 'ghost')

  def test_exits_nonzero_when_missing(self, ws: Path, forest_with_experiment: FileForest) -> None:
    ctx = make_ctx(ws)
    tree = forest_with_experiment.active

    with pytest.raises(SystemExit):
      require_experiment_node(ctx, cast(Any, tree), 'ghost')


class TestStoreVcsArguments:
  def test_returns_three_arguments(self) -> None:
    args = store_vcs_arguments()
    assert len(args) == 3
    flags = [a.flags[0] for a in args]
    assert '--source' in flags
    assert '--store' in flags
    assert '--pattern' in flags

  def test_source_is_required(self) -> None:
    args = store_vcs_arguments()
    source = next(a for a in args if a.flags[0] == '--source')
    assert source.kwargs['required'] is True

  def test_store_default_is_none(self) -> None:
    args = store_vcs_arguments()
    store = next(a for a in args if a.flags[0] == '--store')
    assert store.kwargs['default'] is None

  def test_pattern_default_is_glob(self) -> None:
    args = store_vcs_arguments()
    pattern = next(a for a in args if a.flags[0] == '--pattern')
    assert pattern.kwargs['default'] == '**/*'


class TestWithStoreVcsArguments:
  def test_stacks_arguments_on_subcommand_handler(self) -> None:
    from autopilot.cli.primitives import SubcommandMeta, subcommand

    @with_store_vcs_arguments
    @subcommand('test-cmd', help_text='test')
    def handler(ctx, args):
      pass

    meta = handler.subcommand_meta
    assert isinstance(meta, SubcommandMeta)
    arg_flags = [flags[0] for flags, _ in meta.arguments]
    assert '--source' in arg_flags
    assert '--store' in arg_flags
    assert '--pattern' in arg_flags
