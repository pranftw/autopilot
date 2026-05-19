"""Store, forest, and tree CLI integration regression tests.

Exercises merge workflow smoke, tree create/switch, store snapshot/checkout
integration, and forest persistence.  Uses shared fixtures from
``tests/conftest.py`` and ``tests/cli/conftest.py``.

Regression areas:
  - Forest lock contention (file locking during concurrent writes)
  - Store snapshot/checkout round-trip with PathParameter
  - Tree create / switch / HEAD tracking
  - Merge analysis / preview / apply pipeline
  - Store branch / reset_branch
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.experiment import Experiment
from autopilot.core.node import Node
from autopilot.core.store.types import MergeClassification, MergeStrategy
from pathlib import Path
import pytest


def _make_store_with_param(tmp_path: Path) -> tuple[FileStore, Path]:
  """Create a FileStore with a single PathParameter for testing."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.store_path.mkdir(parents=True, exist_ok=True)
  src = tmp_path / 'src'
  src.mkdir()
  (src / 'main.py').write_text('initial content')
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'code': param})
  return store, src


class TestStoreSnapshotCheckoutRoundTrip:
  """Snapshot -> checkout must restore file content exactly."""

  @pytest.mark.timeout(1)
  def test_snapshot_and_checkout_restores_content(self, tmp_path: Path) -> None:
    """Files restored by checkout must match what was snapshotted."""
    store, src = _make_store_with_param(tmp_path)
    (src / 'main.py').write_text('version 1')
    store.snapshot('exp-a', 0)

    (src / 'main.py').write_text('dirty')
    store.checkout('exp-a', 0)
    assert (src / 'main.py').read_text() == 'version 1'

  @pytest.mark.timeout(1)
  def test_multiple_epochs_snapshot_checkout(self, tmp_path: Path) -> None:
    """Checkout at different epochs restores the correct version."""
    store, src = _make_store_with_param(tmp_path)

    (src / 'main.py').write_text('epoch 0')
    store.snapshot('exp-a', 0)

    (src / 'main.py').write_text('epoch 1')
    store.snapshot('exp-a', 1)

    store.checkout('exp-a', 0)
    assert (src / 'main.py').read_text() == 'epoch 0'

    store.checkout('exp-a', 1)
    assert (src / 'main.py').read_text() == 'epoch 1'


class TestTreeCreateAndSwitch:
  """Tree lifecycle: create, add experiments, switch between trees."""

  @pytest.mark.timeout(1)
  def test_create_tree_and_add_experiment(self, tmp_path: Path) -> None:
    """Creating a tree and adding an experiment persists in forest."""
    store, _ = _make_store_with_param(tmp_path)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='exp-1', hypothesis='test')
    exp.start()
    exp.complete(metrics={'accuracy': 0.9})
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    reloaded = FileForest(store)
    active = reloaded.active
    assert active is not None
    assert active.name == 'main'
    node = active.get('exp-1')
    assert node is not None
    assert node.experiment.metrics['accuracy'] == 0.9

  @pytest.mark.timeout(1)
  def test_switch_between_trees(self, tmp_path: Path) -> None:
    """Switching trees changes the active tree HEAD."""
    store, _ = _make_store_with_param(tmp_path)
    forest = FileForest(store)
    forest.create_tree('alpha')
    forest.create_tree('beta')
    forest.switch('alpha')
    active_alpha = forest.active
    assert active_alpha is not None
    assert active_alpha.name == 'alpha'
    forest.switch('beta')
    active_beta = forest.active
    assert active_beta is not None
    assert active_beta.name == 'beta'

  @pytest.mark.timeout(1)
  def test_tree_head_tracks_explicit_set(self, tmp_path: Path) -> None:
    """Tree HEAD is set explicitly via tree.head = id (FRICTION-003)."""
    store, _ = _make_store_with_param(tmp_path)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    for i in range(3):
      exp = Experiment(experiment_id=f'exp-{i}', hypothesis=f'h{i}')
      tree.add(Node(experiment=exp))
      tree.head = f'exp-{i}'
    assert tree.head == 'exp-2'


class TestMergeWorkflowSmoke:
  """Three-step merge pipeline: analysis -> preview -> apply."""

  @pytest.mark.timeout(1)
  def test_merge_analysis_detects_divergence(self, tmp_path: Path) -> None:
    """When both branches modify the same file, analysis reports conflict."""
    store, src = _make_store_with_param(tmp_path)

    (src / 'main.py').write_text('base')
    store.snapshot('root', 0)
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'main.py').write_text('ours change')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'main.py').write_text('theirs change')
    store.snapshot('exp-b', 1)

    analysis = store.merge_analysis('exp-a', 'exp-b')
    assert analysis.has_conflicts
    assert analysis.conflict_count > 0
    assert analysis.classification == MergeClassification.conflict

  @pytest.mark.timeout(1)
  def test_merge_preview_materializes_conflicts(self, tmp_path: Path) -> None:
    """merge_preview produces a MergeIndex with unresolved conflicts."""
    store, src = _make_store_with_param(tmp_path)

    (src / 'main.py').write_text('base')
    store.snapshot('root', 0)
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'main.py').write_text('ours')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'main.py').write_text('theirs')
    store.snapshot('exp-b', 1)

    index = store.merge_preview('exp-a', 'exp-b')
    assert len(index.conflicts) > 0
    assert index.preview_token is not None
    assert not index.is_resolved()

  @pytest.mark.timeout(1)
  def test_merge_ours_strategy_auto_resolves(self, tmp_path: Path) -> None:
    """Strategy 'ours' should auto-resolve all conflicts during preview."""
    store, src = _make_store_with_param(tmp_path)

    (src / 'main.py').write_text('base')
    store.snapshot('root', 0)
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'main.py').write_text('ours')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'main.py').write_text('theirs')
    store.snapshot('exp-b', 1)

    index = store.merge_preview('exp-a', 'exp-b', strategy=MergeStrategy.ours)
    assert index.is_resolved()
    assert len(index.conflicts) == 0

  @pytest.mark.timeout(1)
  def test_merge_apply_produces_new_epoch(self, tmp_path: Path) -> None:
    """merge_apply persists a new epoch on the target branch."""
    store, src = _make_store_with_param(tmp_path)

    (src / 'main.py').write_text('base')
    store.snapshot('root', 0)
    store.branch('exp-a')
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'main.py').write_text('ours')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'main.py').write_text('theirs')
    store.snapshot('exp-b', 1)

    index = store.merge_preview('exp-a', 'exp-b', strategy=MergeStrategy.ours)
    manifest = store.merge_apply(index)
    assert manifest.epoch == 2
    assert len(manifest.entries) > 0


class TestStoreBranchReset:
  """Branch reset and re-snapshot behavior."""

  @pytest.mark.timeout(1)
  def test_reset_branch_allows_re_snapshot_at_epoch_zero(self, tmp_path: Path) -> None:
    """After reset_branch, snapshot at epoch 0 succeeds."""
    store, src = _make_store_with_param(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot('exp-a', 0)

    store.reset_branch('exp-a')
    (src / 'main.py').write_text('v2')
    store.snapshot('exp-a', 0)

    store.checkout('exp-a', 0)
    assert (src / 'main.py').read_text() == 'v2'


class TestForestPersistence:
  """Forest save/load round-trip integrity."""

  @pytest.mark.timeout(1)
  def test_forest_round_trip_preserves_trees_and_experiments(self, tmp_path: Path) -> None:
    """Save -> reload must preserve tree names, active tree, and node data."""
    store, _ = _make_store_with_param(tmp_path)
    forest = FileForest(store)
    tree = forest.create_tree('main')
    exp = Experiment(experiment_id='e1')
    tree.add(Node(experiment=exp))
    forest.switch('main')
    forest.save()

    reloaded = FileForest(store)
    tree_names = [t.name for t in reloaded.list_trees()]
    assert 'main' in tree_names
    reloaded_active = reloaded.active
    assert reloaded_active is not None
    assert reloaded_active.name == 'main'
    assert reloaded_active.get('e1') is not None
