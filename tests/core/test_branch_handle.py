"""Tests for BranchHandle and RefsView from autopilot.core.branch."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.branch import BranchHandle
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.store.base import Store
from pathlib import Path
import pytest


def _make_store(tmp_path: Path, files: dict[str, str] | None = None) -> tuple[FileStore, Path]:
  """Create a FileStore with a source directory and registered parameter."""
  src = tmp_path / 'src'
  src.mkdir(parents=True, exist_ok=True)
  for fname, content in (files or {'main.py': 'print("hello")'}).items():
    (src / fname).write_text(content)
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  return store, src


def _seed_branch(store: FileStore, exp_id: str, src: Path, epochs: int = 1) -> None:
  """Create a branch with ``epochs`` snapshots (0 through epochs-1)."""
  for epoch in range(epochs):
    store.snapshot(exp_id, epoch, context=f'epoch {epoch}')
    if epoch < epochs - 1:
      (src / 'main.py').write_text(f'# epoch {epoch + 1}')


class TestBranchHandleLatestEpoch:
  """test_branch_handle_latest_epoch: matches seeded refs.json."""

  def test_latest_epoch_matches_refs(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-1', src, epochs=3)

    handle = store.branch_handle('exp-1')
    assert handle.latest_epoch() == 2

  def test_latest_epoch_single_snapshot(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-1', src, epochs=1)

    handle = store.branch_handle('exp-1')
    assert handle.latest_epoch() == 0

  def test_latest_epoch_missing_branch(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path)

    handle = store.branch_handle('nonexistent')
    with pytest.raises(StoreError, match='nonexistent'):
      handle.latest_epoch()


class TestBranchHandleLatestEpochAfterReset:
  """test_branch_handle_latest_epoch_after_reset: reset_branch reflected."""

  def test_reflects_reset(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-1', src, epochs=2)

    handle = store.branch_handle('exp-1')
    assert handle.latest_epoch() == 1

    store.reset_branch('exp-1')
    assert handle.latest_epoch() == -1


class TestBranchHandleSnapshot:
  """test_branch_handle_snapshot: delegates to FileStore."""

  def test_snapshot_via_handle(self, tmp_path: Path) -> None:
    store, _src = _make_store(tmp_path)

    handle = store.branch_handle('exp-snap')
    manifest = handle.snapshot(0, context='via handle')
    assert manifest.epoch == 0

    refs = store.load_refs()
    assert refs['branches']['exp-snap']['latest_epoch'] == 0


class TestBranchHandleCheckout:
  """test_branch_handle_checkout: checkout mutates working tree."""

  def test_checkout_via_handle(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-co', src, epochs=2)

    handle = store.branch_handle('exp-co')
    handle.checkout(0)

    refs = store.load_refs()
    assert refs['HEAD'] == 'exp-co'


class TestBranchHandleLog:
  """test_branch_handle_log: list length matches epochs present."""

  def test_log_length(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-log', src, epochs=3)

    handle = store.branch_handle('exp-log')
    entries = handle.log()
    assert len(entries) == 3
    assert entries[0].epoch == 0
    assert entries[2].epoch == 2


class TestBranchHandleDiff:
  """test_branch_handle_diff: diff between two epochs non-trivial."""

  def test_diff_shows_changes(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-diff', src, epochs=2)

    handle = store.branch_handle('exp-diff')
    result = handle.diff(0, 1)
    assert len(result.entries) > 0


class TestBranchHandleCheckoutMissingEpoch:
  """test_branch_handle_checkout_missing_epoch: StoreError for nonexistent epoch."""

  def test_missing_epoch_raises(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-miss', src, epochs=1)

    handle = store.branch_handle('exp-miss')
    with pytest.raises(StoreError):
      handle.checkout(999)


class TestRefsViewGetitem:
  """test_refs_view_getitem: returns BranchHandle for existing id."""

  def test_getitem_returns_handle(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-view', src)

    view = store.refs_view
    handle = view['exp-view']
    assert isinstance(handle, BranchHandle)
    assert handle.experiment_id == 'exp-view'
    assert handle.store is store


class TestRefsViewContains:
  """test_refs_view_contains: True/False per branches map."""

  def test_contains_true(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'exp-exists', src)

    view = store.refs_view
    assert 'exp-exists' in view

  def test_contains_false(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path)
    view = store.refs_view
    assert 'missing' not in view

  def test_contains_non_string(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path)
    view = store.refs_view
    assert 42 not in view


class TestRefsViewIter:
  """test_refs_view_iter: yields expected branch names."""

  def test_iter_branches(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'alpha', src)
    (src / 'main.py').write_text('# beta')
    _seed_branch(store, 'beta', src)

    view = store.refs_view
    names = set(view)
    assert 'alpha' in names
    assert 'beta' in names


class TestRefsViewGetitemMissingBranch:
  """test_refs_view_getitem_missing_branch: StoreError."""

  def test_missing_branch_raises(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path)

    view = store.refs_view
    with pytest.raises(StoreError, match='Create the branch first'):
      view['nonexistent']


class TestRefsViewLen:
  """test_refs_view_len: matches number of branches."""

  def test_len_matches(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)
    _seed_branch(store, 'one', src)
    (src / 'main.py').write_text('# two')
    _seed_branch(store, 'two', src)

    view = store.refs_view
    assert len(view) == 2

  def test_len_empty(self, tmp_path: Path) -> None:
    store, _ = _make_store(tmp_path)
    view = store.refs_view
    assert len(view) == 0


class TestBranchHandleOnBaseStore:
  """test_branch_handle_on_base_store: raises NotImplementedError."""

  def test_base_store_branch_handle(self) -> None:
    with pytest.raises(NotImplementedError):
      Store.branch_handle(Store.__new__(Store), 'exp-1')

  def test_base_store_refs_view(self) -> None:
    with pytest.raises(NotImplementedError):
      _ = Store.refs_view.fget(Store.__new__(Store))


class TestRefsViewLifecycle:
  """test_refs_view_lifecycle: create branch via Store.branch(), snapshot, verify refs_view."""

  def test_lifecycle(self, tmp_path: Path) -> None:
    store, src = _make_store(tmp_path)

    store.snapshot('seed-exp', 0, context='seed')

    view = store.refs_view
    assert len(view) == 1
    assert 'exp-life' not in view

    store.branch('exp-life')

    view = store.refs_view
    assert len(view) == 2
    assert 'exp-life' in view

    handle = view['exp-life']
    assert handle.latest_epoch() == 0

    (src / 'main.py').write_text('# updated')
    store.snapshot('exp-life', 1, context='second', force=True)

    assert handle.latest_epoch() == 1

    entries = handle.log()
    assert len(entries) == 2


class TestBranchHandleSnapshotForce:
  """BranchHandle.snapshot forwards force= to store."""

  def test_snapshot_forwards_force(self, tmp_path: Path) -> None:
    """force=True accepted and forwarded without TypeError."""
    store, _src = _make_store(tmp_path)
    handle = store.branch_handle('exp-force')
    handle.snapshot(0, context='initial')
    manifest = handle.snapshot(1, context='forced', force=True)
    assert manifest.epoch == 1


class TestBranchHandleSnapshotExperiment:
  """BranchHandle.snapshot forwards experiment= to store."""

  def test_snapshot_forwards_experiment(self, tmp_path: Path) -> None:
    """experiment= accepted and forwarded without TypeError."""
    store, _src = _make_store(tmp_path)
    exp = Experiment(experiment_id='exp-with-exp')
    exp.start()
    exp.complete(metrics={'x': 1.0})
    handle = store.branch_handle('exp-with-exp')
    manifest = handle.snapshot(0, experiment=exp, context='with-exp')
    assert manifest.epoch == 0


class TestRefsViewMissingBranchGuidance:
  """RefsView.__getitem__ error includes remediation guidance."""

  def test_refs_view_missing_branch_error_has_guidance(self, tmp_path: Path) -> None:
    """Error message tells user how to create the branch."""
    store, _src = _make_store(tmp_path)
    view = store.refs_view
    with pytest.raises(StoreError, match='Create the branch first'):
      view['nonexistent']
