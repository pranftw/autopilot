"""Comprehensive tests for FileStore: content-addressed code versioning with worktrees."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store.store_io import atomic_write_json_safe
from autopilot.ai.store_lock import hash_content
from autopilot.core.errors import StoreError, TrackingError
from autopilot.core.parameter import Parameter
from autopilot.tracking.file_lock import ConcurrentMutationError
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
from unittest.mock import patch
import json
import pytest
import threading
import time


def _make_store(
  tmp_path: Path,
  slug: str = 'exp-001',
  files: dict[str, str] | None = None,
):
  seed = files if files is not None else {'main.py': 'print("hello")', 'util.py': 'x = 1'}
  src = make_source_dir(tmp_path, files=seed)
  config = make_store_config(tmp_path)
  params = [PathParameter(source=str(src), pattern='*')]
  store = FileStore(config)
  store.register_parameters({'source': params[0]})
  store.snapshot(slug, 0)
  return store, src, params


# init (constructor + first snapshot)


class TestInit:
  def test_creates_store_structure(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    assert store.config.objects_path.is_dir()
    assert store.config.snapshots_path.is_dir()
    assert store.config.refs_file.is_file()

  def test_stores_objects(self, tmp_path: Path) -> None:
    src = make_source_dir(tmp_path, files={'a.txt': 'hello'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})
    store.snapshot('exp', 0)

    content_hash = hash_content('hello')
    prefix = content_hash[:2]
    rest = content_hash[2:]
    assert (config.objects_path / prefix / rest).exists()

  def test_writes_epoch_0_snapshot(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    snap_path = store.config.snapshots_path / 'exp-001' / 'epoch_0.json'
    assert snap_path.exists()
    data = json.loads(snap_path.read_text())
    assert data['epoch'] == 0
    assert len(data['entries']) == 2

  def test_sets_head(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    refs = json.loads(store.config.refs_file.read_text())
    assert refs['HEAD'] == 'exp-001'

  def test_idempotent_reentry(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path)
    (src / 'main.py').write_text('print("v2")')
    store.snapshot('exp-001', 1)

    store2 = FileStore(store.config)
    store2.register_parameters({'source': params[0]})
    entries = store2.log('exp-001')
    assert entries[-1].epoch == 1

  def test_empty_parameters(self, tmp_path: Path) -> None:
    src = tmp_path / 'empty_src'
    src.mkdir()
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})
    store.snapshot('empty-exp', 0)
    snap_path = config.snapshots_path / 'empty-exp' / 'epoch_0.json'
    data = json.loads(snap_path.read_text())
    assert len(data['entries']) == 0

  def test_constructor_only_creates_dirs(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    FileStore(config)
    assert config.store_path.is_dir()
    assert config.objects_path.is_dir()
    assert config.snapshots_path.is_dir()


# snapshot


class TestSnapshot:
  def test_captures_current_state(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('print("updated")')
    snap = store.snapshot('exp-001', 1)
    key = next(k for k in snap.entries if 'main.py' in k)
    new_hash = hash_content('print("updated")')
    assert snap.entries[key].digest == new_hash

  def test_deduplicates_unchanged_files(self, tmp_path: Path) -> None:
    store, _src, _ = _make_store(tmp_path)
    old_hash = hash_content('x = 1')
    snap = store.snapshot('exp-001', 1)
    key = next(k for k in snap.entries if 'util.py' in k)
    assert snap.entries[key].digest == old_hash

    prefix = old_hash[:2]
    rest = old_hash[2:]
    obj_path = store.config.objects_path / prefix / rest
    assert obj_path.exists()

  def test_sequential_epochs(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot('exp-001', 1)
    (src / 'main.py').write_text('v2')
    store.snapshot('exp-001', 2)
    (src / 'main.py').write_text('v3')
    store.snapshot('exp-001', 3)
    entries = store.log('exp-001')
    assert entries[-1].epoch == 3

  def test_skipped_epoch_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='sequential'):
      store.snapshot('exp-001', 5)

  def test_updates_head(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.snapshot('exp-001', 1)
    refs = json.loads(store.config.refs_file.read_text())
    assert refs['HEAD'] == 'exp-001'

  def test_auto_creates_branch_on_epoch_0(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('new-exp', 0)
    entries = store.log('new-exp')
    assert len(entries) == 1
    assert entries[0].epoch == 0

  def test_new_branch_requires_epoch_0(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    with pytest.raises(StoreError, match='first snapshot must be epoch 0'):
      store.snapshot('new-exp', 3)


# checkout


class TestCheckout:
  def test_restores_files(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    original_content = (src / 'main.py').read_text()
    (src / 'main.py').write_text('completely changed')
    store.snapshot('exp-001', 1)

    store.checkout('exp-001', 0)
    assert (src / 'main.py').read_text() == original_content

  def test_checkout_removes_extraneous_working_tree_files(self, tmp_path: Path) -> None:
    """BUG-075: files matched by the parameter pattern but absent from the
    snapshot are removed on checkout to prevent stale leak-through."""
    store, src, _ = _make_store(tmp_path)
    (src / 'untracked.log').write_text('should be removed')
    store.checkout('exp-001', 0)
    assert not (src / 'untracked.log').exists()

  def test_updates_head(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v2')
    store.snapshot('exp-001', 1)
    store.checkout('exp-001', 0)
    refs = json.loads(store.config.refs_file.read_text())
    assert refs['HEAD'] == 'exp-001'

  def test_nonexistent_epoch_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='snapshot not found'):
      store.checkout('exp-001', 99)

  def test_nonexistent_experiment_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='not found'):
      store.checkout('nonexistent', 0)


# diff


class TestDiff:
  def test_identical_snapshots(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    result = store.diff('exp-001', 0, 0)
    assert len(result.entries) == 0

  def test_modified_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('updated content')
    store.snapshot('exp-001', 1)
    result = store.diff('exp-001', 0, 1)
    modified = result.modified()
    assert len(modified) == 1
    assert 'main.py' in modified[0].path

  def test_added_file(self, tmp_path: Path) -> None:
    src = make_source_dir(tmp_path, files={'a.txt': 'hello'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})
    store.snapshot('exp', 0)
    (src / 'b.txt').write_text('new file')
    store.snapshot('exp', 1)
    result = store.diff('exp', 0, 1)
    added = result.added()
    assert len(added) == 1
    assert 'b.txt' in added[0].path

  def test_deleted_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'util.py').unlink()
    store.snapshot('exp-001', 1)
    result = store.diff('exp-001', 0, 1)
    deleted = result.deleted()
    assert len(deleted) == 1
    assert 'util.py' in deleted[0].path

  def test_text_diff_content(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('line1\nline2\nline3\n')
    store.snapshot('exp-001', 1)
    (src / 'main.py').write_text('line1\nmodified\nline3\n')
    store.snapshot('exp-001', 2)
    result = store.diff('exp-001', 1, 2)
    modified = result.modified()
    assert len(modified) == 1
    assert '-line2' in modified[0].text_diff
    assert '+modified' in modified[0].text_diff

  def test_nonexistent_experiment_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='not found'):
      store.diff('nonexistent', 0, 1)


# branch


class TestBranch:
  def test_creates_new_experiment(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.branch('feature-1')
    refs = json.loads(store.config.refs_file.read_text())
    assert 'feature-1' in refs['branches']

  def test_shares_objects(self, tmp_path: Path) -> None:
    store, _src, _ = _make_store(tmp_path)
    objects_before = list(store.config.objects_path.rglob('*'))
    obj_files_before = [f for f in objects_before if f.is_file()]
    store.branch('feature-1')
    objects_after = list(store.config.objects_path.rglob('*'))
    obj_files_after = [f for f in objects_after if f.is_file()]
    assert len(obj_files_before) == len(obj_files_after)

  def test_records_parent(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.branch('child')
    refs = json.loads(store.config.refs_file.read_text())
    assert refs['branches']['child']['parent_id'] == 'exp-001'
    assert refs['branches']['child']['parent_epoch'] == 0

  def test_duplicate_experiment_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.branch('dup')
    with pytest.raises(StoreError, match='already exists'):
      store.branch('dup')

  def test_checkout_independent(self, tmp_path: Path) -> None:
    store, src, _params = _make_store(tmp_path)
    original = (src / 'main.py').read_text()
    store.branch('feature')
    (src / 'main.py').write_text('feature change')
    store.snapshot('feature', 1)
    store.checkout('exp-001', 0)
    assert (src / 'main.py').read_text() == original

  def test_branch_without_head_raises(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    with pytest.raises(StoreError, match='no HEAD'):
      store.branch('orphan')

  def test_branch_copies_latest_snapshot(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot('exp-001', 1)
    store.branch('fork')
    entries = store.log('fork')
    assert len(entries) == 1
    assert entries[0].epoch == 0
    assert entries[0].file_count == 2


# merge


class TestMergePreview:
  def test_no_conflicts(self, tmp_path: Path) -> None:
    store, src, _params = _make_store(
      tmp_path, files={'a.txt': 'original a', 'b.txt': 'original b'}
    )
    store.branch('feature')

    store.checkout('exp-001', 0)
    (src / 'a.txt').write_text('changed a on main')
    store.snapshot('exp-001', 1)

    store.checkout('feature', 0)
    (src / 'b.txt').write_text('changed b on feature')
    store.snapshot('feature', 1)

    result = store.merge_preview('exp-001', 'feature')
    assert result.is_resolved()
    assert len(result.conflicts) == 0

  def test_conflicting_changes(self, tmp_path: Path) -> None:
    store, src, _params = _make_store(tmp_path, files={'conflict.txt': 'line1\nline2\nline3\n'})
    store.branch('feature')

    store.checkout('exp-001', 0)
    (src / 'conflict.txt').write_text('line1\nmain change\nline3\n')
    store.snapshot('exp-001', 1)

    store.checkout('feature', 0)
    (src / 'conflict.txt').write_text('line1\nfeature change\nline3\n')
    store.snapshot('feature', 1)

    result = store.merge_preview('exp-001', 'feature')
    assert not result.is_resolved()
    assert len(result.conflicts) > 0

  def test_one_side_only(self, tmp_path: Path) -> None:
    store, src, _params = _make_store(tmp_path, files={'a.txt': 'original', 'b.txt': 'original'})
    store.branch('feature')

    (src / 'a.txt').write_text('only feature changed this')
    store.snapshot('feature', 1)

    result = store.merge_preview('exp-001', 'feature')
    assert result.is_resolved()

  def test_defaults_to_latest_epoch(self, tmp_path: Path) -> None:
    store, src, _params = _make_store(tmp_path)
    store.branch('feature')

    (src / 'main.py').write_text('feature v1')
    store.snapshot('feature', 1)
    (src / 'main.py').write_text('feature v2')
    store.snapshot('feature', 2)

    result = store.merge_preview('exp-001', 'feature')
    assert result.preview_token is not None

  def test_nonexistent_experiment_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='not found'):
      store.merge_preview('exp-001', 'nonexistent')


# log


class TestLog:
  def test_returns_all_epochs(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot('exp-001', 1)
    (src / 'main.py').write_text('v2')
    store.snapshot('exp-001', 2)
    entries = store.log('exp-001')
    assert len(entries) == 3
    assert entries[0].epoch == 0
    assert entries[1].epoch == 1
    assert entries[2].epoch == 2

  def test_single_epoch(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    entries = store.log('exp-001')
    assert len(entries) == 1
    assert entries[0].epoch == 0

  def test_chronological_order(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot('exp-001', 1)
    entries = store.log('exp-001')
    assert entries[0].epoch < entries[1].epoch

  def test_file_count(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    entries = store.log('exp-001')
    assert entries[0].file_count == 2

  def test_nonexistent_experiment_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='not found'):
      store.log('nonexistent')


# status


class TestStatus:
  def test_clean(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    result = store.status('exp-001')
    unchanged = result.unchanged()
    assert len(unchanged) == 2
    assert result.modified() == []
    assert result.added() == []
    assert result.deleted() == []

  def test_modified_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('changed')
    result = store.status('exp-001')
    modified = result.modified()
    assert len(modified) == 1
    assert 'main.py' in modified[0].path

  def test_deleted_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'util.py').unlink()
    result = store.status('exp-001')
    deleted = result.deleted()
    assert len(deleted) == 1
    assert 'util.py' in deleted[0].path

  def test_added_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'new.py').write_text('new file')
    result = store.status('exp-001')
    added = result.added()
    assert len(added) == 1
    assert 'new.py' in added[0].path


# promote


class TestPromote:
  def test_restores_files(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('epoch 1 version')
    store.snapshot('exp-001', 1)
    (src / 'main.py').write_text('epoch 2 version')
    store.snapshot('exp-001', 2)

    store.materialize('exp-001', 1)
    assert (src / 'main.py').read_text() == 'epoch 1 version'

  def test_updates_baseline(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('promoted version')
    store.snapshot('exp-001', 1)
    store.materialize('exp-001', 1)

    snap_path = store.config.snapshots_path / 'exp-001' / 'epoch_0.json'
    data = json.loads(snap_path.read_text())
    key = next(k for k in data['entries'] if 'main.py' in k)
    promoted_hash = hash_content('promoted version')
    assert data['entries'][key]['digest'] == promoted_hash

  def test_detects_external_modification(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('snapshot version')
    store.snapshot('exp-001', 1)
    (src / 'main.py').write_text('externally modified')
    store.snapshot('exp-001', 2)
    (src / 'main.py').write_text('tampered after snapshot 2')
    with pytest.raises(StoreError, match='external modification'):
      store.materialize('exp-001', 1)


# worktrees


class TestWorktrees:
  def test_create_worktree(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    wt_path = store.create_worktree('exp-001')
    assert wt_path.is_dir()
    assert 'exp-001' in str(wt_path)

  def test_create_worktree_empty_directory(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    wt_path = store.create_worktree('exp-001')
    contents = list(wt_path.iterdir())
    assert contents == []

  def test_create_worktree_registered_in_refs(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    wt_path = store.create_worktree('exp-001')
    refs = json.loads(store.config.refs_file.read_text())
    assert 'exp-001' in refs['worktrees']
    assert refs['worktrees']['exp-001'] == str(wt_path)

  def test_create_worktree_lock_removed_after(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.create_worktree('exp-001')
    lock_path = store.config.worktrees_path / 'exp-001.lock'
    assert not lock_path.exists()

  def test_create_worktree_nonexistent_experiment_raises(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    with pytest.raises(StoreError, match='not found'):
      store.create_worktree('nonexistent')

  def test_remove_worktree(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    wt_path = store.create_worktree('exp-001')
    assert wt_path.exists()
    store.remove_worktree('exp-001')
    assert not wt_path.exists()

  def test_remove_worktree_cleans_refs(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.create_worktree('exp-001')
    store.remove_worktree('exp-001')
    refs = json.loads(store.config.refs_file.read_text())
    assert 'exp-001' not in refs.get('worktrees', {})

  def test_remove_worktree_idempotent(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.create_worktree('exp-001')
    store.remove_worktree('exp-001')
    store.remove_worktree('exp-001')

  def test_remove_worktree_already_deleted_dir(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    wt_path = store.create_worktree('exp-001')
    import shutil

    shutil.rmtree(wt_path)
    store.remove_worktree('exp-001')

  def test_list_worktrees(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    assert store.list_worktrees() == []
    store.create_worktree('exp-001')
    assert store.list_worktrees() == ['exp-001']

  def test_two_worktrees(self, tmp_path: Path) -> None:
    store, _src, _ = _make_store(tmp_path)
    store.branch('exp-002')
    wt1 = store.create_worktree('exp-001')
    wt2 = store.create_worktree('exp-002')
    assert wt1.is_dir()
    assert wt2.is_dir()
    assert wt1 != wt2
    assert sorted(store.list_worktrees()) == ['exp-001', 'exp-002']


# resolve_path


class TestResolvePath:
  def test_experiment_scoped(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    path = store.resolve_path('exp-001')
    assert path.is_dir()
    assert 'experiments' in str(path)
    assert 'exp-001' in str(path)

  def test_epoch_scoped(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    path = store.resolve_path('exp-001', epoch=0)
    assert path.is_dir()
    assert 'epoch_0' in str(path)

  def test_creates_directory(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    path = store.resolve_path('exp-001', epoch=5)
    assert path.is_dir()

  def test_different_epochs_different_paths(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    p0 = store.resolve_path('exp-001', epoch=0)
    p1 = store.resolve_path('exp-001', epoch=1)
    assert p0 != p1


# forest persistence


class TestForestPersistence:
  def test_save_and_load_state_dict(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    state = {'trees': {'main': {'nodes': ['a', 'b']}}}
    store.save_state_dict(state)
    loaded = store.load_state_dict()
    assert loaded == state

  def test_load_state_dict_missing_file(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    assert store.load_state_dict() is None

  def test_round_trip_complex_state(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    state = {
      'trees': {
        'tree-1': {
          'name': 'tree-1',
          'nodes': [
            {'id': 'exp-001', 'parent': None},
            {'id': 'exp-002', 'parent': 'exp-001'},
          ],
        }
      },
      'active': 'tree-1',
    }
    store.save_state_dict(state)
    loaded = store.load_state_dict()
    assert loaded == state

  def test_save_overwrites(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.save_state_dict({'v': 1})
    store.save_state_dict({'v': 2})
    loaded = store.load_state_dict()
    assert loaded == {'v': 2}


# round-trip integration


class TestIntegration:
  def test_full_lifecycle(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    original_main = (src / 'main.py').read_text()
    original_util = (src / 'util.py').read_text()

    (src / 'main.py').write_text('epoch 1')
    store.snapshot('exp-001', 1)
    (src / 'main.py').write_text('epoch 2')
    store.snapshot('exp-001', 2)

    store.checkout('exp-001', 0)
    assert (src / 'main.py').read_text() == original_main
    assert (src / 'util.py').read_text() == original_util

    store.checkout('exp-001', 1)
    assert (src / 'main.py').read_text() == 'epoch 1'

  def test_branch_and_merge_lifecycle(self, tmp_path: Path) -> None:
    store, src, _params = _make_store(tmp_path, files={'a.txt': 'base a', 'b.txt': 'base b'})

    store.branch('feature')

    store.checkout('exp-001', 0)
    (src / 'a.txt').write_text('main changed a')
    store.snapshot('exp-001', 1)

    store.checkout('feature', 0)
    (src / 'b.txt').write_text('feature changed b')
    store.snapshot('feature', 1)

    result = store.merge_preview('exp-001', 'feature')
    assert result.is_resolved()

  def test_parallel_experiments(self, tmp_path: Path) -> None:
    src = make_source_dir(tmp_path, files={'shared.py': 'base'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]

    store = FileStore(config)
    store.register_parameters({'source': params[0]})
    store.snapshot('exp-a', 0)

    (src / 'shared.py').write_text('exp-a version')
    store.snapshot('exp-a', 1)

    store.checkout('exp-a', 0)
    store.snapshot('exp-b', 0)

    (src / 'shared.py').write_text('exp-b version')
    store.snapshot('exp-b', 1)

    store.checkout('exp-a', 1)
    assert (src / 'shared.py').read_text() == 'exp-a version'

    store.checkout('exp-b', 1)
    assert (src / 'shared.py').read_text() == 'exp-b version'

  def test_rollback_scenario(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('good version')
    store.snapshot('exp-001', 1)
    (src / 'main.py').write_text('bad regression')
    store.snapshot('exp-001', 2)

    store.checkout('exp-001', 1)
    assert (src / 'main.py').read_text() == 'good version'

  def test_idempotent_reentry_preserves_state(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot('exp-001', 1)

    store2 = FileStore(store.config)
    store2.register_parameters({'source': params[0]})
    entries = store2.log('exp-001')
    assert len(entries) == 2


# atomic writes and locking


class TestSnapshotLockRelease:
  def test_snapshot_releases_lock_when_build_raises(self, tmp_path: Path) -> None:
    from unittest.mock import patch

    src = make_source_dir(tmp_path)
    config = make_store_config(tmp_path)
    store = FileStore(config)
    store.register_parameters({'source': PathParameter(source=str(src), pattern='*')})
    store.snapshot('exp', 0)
    lock_path = config.store_path / '.lock'
    with (
      patch('autopilot.ai.store.snapshot.build_snapshot', side_effect=RuntimeError('boom')),
      pytest.raises(RuntimeError, match='boom'),
    ):
      store.snapshot('exp', 1)
    assert not lock_path.exists()
    store.snapshot('exp', 1)
    assert not lock_path.exists()


class TestAtomicAndLocking:
  def test_lock_file_created_and_released(self, tmp_path: Path) -> None:
    store, _src, _ = _make_store(tmp_path)
    lock_path = store.config.store_path / '.lock'
    assert not lock_path.exists()
    store.snapshot('exp-001', 1)
    assert not lock_path.exists()

  def test_lock_prevents_concurrent_access(self, tmp_path: Path) -> None:
    store, _src, _ = _make_store(tmp_path)
    barrier = threading.Barrier(2)
    err: list[BaseException | None] = [None]

    def holder() -> None:
      store._backend.acquire_lock()
      barrier.wait()
      time.sleep(0.15)
      store._backend.release_lock()

    def contender() -> None:
      barrier.wait()
      try:
        store._backend.acquire_lock()
      except ConcurrentMutationError as exc:
        err[0] = exc
      else:
        store._backend.release_lock()

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=contender)
    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    assert isinstance(err[0], ConcurrentMutationError)
    assert 'concurrent mutation' in str(err[0])

  def test_refs_json_not_corrupted_after_operations(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    for i in range(1, 6):
      (src / 'main.py').write_text(f'version {i}')
      store.snapshot('exp-001', i)
    refs = json.loads(store.config.refs_file.read_text())
    assert refs['branches']['exp-001']['latest_epoch'] == 5
    assert refs['HEAD'] == 'exp-001'

  def test_snapshot_files_valid_json(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot('exp-001', 1)
    for snap_file in (store.config.snapshots_path / 'exp-001').iterdir():
      data = json.loads(snap_file.read_text())
      assert 'epoch' in data
      assert 'entries' in data
      assert 'timestamp' in data


# snapshot/restore decoupling tests


class TestSnapshotRestoreDecoupling:
  def test_filestore_snapshot_via_param_snapshot(self, tmp_path: Path) -> None:

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    config = make_store_config(tmp_path)
    param = PromptParameter('hello world')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('test-slug', 0)
    snap = store.load_snapshot('test-slug', 0)
    assert len(snap.entries) == 1
    key = next(iter(snap.entries.keys()))
    assert key == 'source/prompt'

  def test_filestore_checkout_via_param_restore(self, tmp_path: Path) -> None:

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    config = make_store_config(tmp_path)
    param = PromptParameter('v1')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('test-slug', 0)

    param._text = 'v2'
    store.snapshot('test-slug', 1)

    store.checkout('test-slug', 0)
    assert param._text == 'v1'

  def test_filestore_empty_snapshot(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    param = Parameter()
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('empty-snap', 0)
    snap = store.load_snapshot('empty-snap', 0)
    assert len(snap.entries) == 0

  def test_filestore_diff_uses_param_snapshot(self, tmp_path: Path) -> None:

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    config = make_store_config(tmp_path)
    param = PromptParameter('original')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('diff-test', 0)

    param._text = 'modified'
    store.snapshot('diff-test', 1)

    result = store.diff('diff-test', 0, 1)
    assert len(result.modified()) == 1
    assert 'prompt' in result.modified()[0].path

  def test_filestore_multiple_param_types(self, tmp_path: Path) -> None:

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    src = make_source_dir(tmp_path, files={'code.py': 'x = 1'})
    config = make_store_config(tmp_path)
    path_param = PathParameter(source=str(src), pattern='*')
    prompt_param = PromptParameter('system prompt v1')
    store = FileStore(config)
    store.register_parameters({'p0': path_param, 'p1': prompt_param})
    store.snapshot('mixed', 0)

    snap = store.load_snapshot('mixed', 0)
    assert len(snap.entries) == 2

    (src / 'code.py').write_text('x = 2')
    prompt_param._text = 'system prompt v2'
    store.snapshot('mixed', 1)

    store.checkout('mixed', 0)
    assert (src / 'code.py').read_text() == 'x = 1'
    assert prompt_param._text == 'system prompt v1'

  def test_filestore_pathparameter_import_limited_to_narrowing(self) -> None:
    """PathParameter used only for import, isinstance checks, type annotations, and delegation."""
    import autopilot.ai.store.file_store as store_module

    source = Path(store_module.__file__).read_text(encoding='utf-8')
    code_lines_with_pp = [
      line.strip()
      for line in source.splitlines()
      if 'PathParameter' in line and not line.strip().startswith('#')
    ]
    for line in code_lines_with_pp:
      is_import = 'import' in line
      is_annotation = "'" in line and 'PathParameter' in line
      assert is_import or is_annotation, f'non-import/annotation PathParameter usage: {line}'

  def test_filestore_composite_keys(self, tmp_path: Path) -> None:

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'system': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['system']

    config = make_store_config(tmp_path)
    param = PromptParameter('hello')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('keys-test', 0)
    snap = store.load_snapshot('keys-test', 0)
    keys = list(snap.entries.keys())
    assert len(keys) == 1
    assert keys[0] == 'source/system'

  def test_filestore_content_addressed_dedup(self, tmp_path: Path) -> None:

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    config = make_store_config(tmp_path)
    p1 = PromptParameter('same content')
    p2 = PromptParameter('same content')
    store = FileStore(config)
    store.register_parameters({'p0': p1, 'p1': p2})
    store.snapshot('dedup', 0)

    snap = store.load_snapshot('dedup', 0)
    hashes = [entry.digest for entry in snap.entries.values()]
    assert hashes[0] == hashes[1]
    obj_hash = hashes[0]
    prefix = obj_hash[:2]
    rest = obj_hash[2:]
    assert (config.objects_path / prefix / rest).exists()


# common ancestor


class TestCommonAncestor:
  def test_common_ancestor_branch_from_branch(self, tmp_path: Path) -> None:
    store, _src, _ = _make_store(tmp_path)
    store.branch('mid')
    store.checkout('mid', 0)
    store.branch('leaf')
    refs = store.load_refs()
    lca_exp, lca_epoch = store._find_lca('leaf', 'exp-001', refs)
    assert lca_exp is not None
    assert lca_epoch is not None
    base = store.load_snapshot(lca_exp, lca_epoch)
    ours = store.load_snapshot('exp-001', refs['branches']['exp-001']['latest_epoch'])
    assert len(base.entries) == len(ours.entries)

  def test_common_ancestor_sibling_fork(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.branch('left')
    store.branch('right')
    refs = store.load_refs()
    lca_exp, lca_epoch = store._find_lca('left', 'right', refs)
    assert lca_exp is not None
    assert lca_epoch is not None

  def test_common_ancestor_divergent_snapshots(self, tmp_path: Path) -> None:
    """two branches evolve different epochs; ancestor still aligns with walker rules."""
    store, src, _ = _make_store(tmp_path)
    store.branch('slow')
    (src / 'main.py').write_text('slow-1')
    store.snapshot('slow', 1)
    store.branch('fast')
    (src / 'main.py').write_text('fast-1')
    store.snapshot('fast', 1)
    (src / 'main.py').write_text('fast-2')
    store.snapshot('fast', 2)
    refs = store.load_refs()
    lca_exp, lca_epoch = store._find_lca('slow', 'fast', refs)
    assert lca_exp is not None
    assert isinstance(lca_epoch, int)


# merge preview-only


class TestMergePreviewNoDiskMutation:
  def test_merge_preview_no_disk_mutation(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path, files={'a.txt': 'x', 'b.txt': 'y'})
    store.branch('feature')
    store.checkout('exp-001', 0)
    (src / 'a.txt').write_text('main')
    store.snapshot('exp-001', 1)
    store.checkout('feature', 0)
    (src / 'b.txt').write_text('feat')
    store.snapshot('feature', 1)
    refs_before = json.loads(store.config.refs_file.read_text(encoding='utf-8'))
    snap_paths_before = sorted((store.config.snapshots_path / 'exp-001').glob('epoch_*.json'))
    result = store.merge_preview('exp-001', 'feature')
    assert result.preview_token is not None
    refs_after = json.loads(store.config.refs_file.read_text(encoding='utf-8'))
    snap_paths_after = sorted((store.config.snapshots_path / 'exp-001').glob('epoch_*.json'))
    assert refs_after == refs_before
    assert snap_paths_after == snap_paths_before


# repr


class TestRepr:
  def test_repr(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    r = repr(store)
    assert 'FileStore' in r
    assert 'parameters=1' in r


class TestSequenceParameterConstructor:
  """FileStore accepts Sequence[Parameter] so subtype lists type-check."""

  def test_list_of_path_parameters(self, tmp_path: Path) -> None:
    src = make_source_dir(tmp_path)
    config = make_store_config(tmp_path)
    params: list[PathParameter] = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})
    snap = store.snapshot('test', 0)
    assert len(snap.entries) > 0

  def test_mixed_parameter_types(self, tmp_path: Path) -> None:

    class PromptParameter(Parameter):
      def __init__(self) -> None:
        super().__init__()
        self._text = 'hello'

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    config = make_store_config(tmp_path)
    params: list[Parameter] = [PromptParameter()]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})
    snap = store.snapshot('test', 0)
    assert 'source/prompt' in snap.entries


class TestMalformedJsonGuards:
  """JSON validation raises StoreError for non-object payloads."""

  def test_load_state_dict_list_json_raises(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text('[]', encoding='utf-8')
    with pytest.raises(StoreError, match='must contain a JSON object'):
      store.load_state_dict()

  def test_load_state_dict_none_returns_none(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    assert store.load_state_dict() is None

  def test_load_refs_list_json_raises(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    store._refs_file.parent.mkdir(parents=True, exist_ok=True)
    store._refs_file.write_text('[]', encoding='utf-8')
    with pytest.raises(StoreError, match='must contain a JSON object'):
      store.load_refs()

  def test_load_refs_missing_returns_empty(self, tmp_path: Path) -> None:
    config = make_store_config(tmp_path)
    store = FileStore(config)
    assert store.load_refs() == {}

  def test_load_snapshot_list_json_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    snap_path = store._snapshots_dir / 'exp-001' / 'epoch_0.json'
    snap_path.write_text('[]', encoding='utf-8')
    with pytest.raises(StoreError, match='must contain a JSON object'):
      store.load_snapshot('exp-001', 0)

  def test_load_snapshot_string_json_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    snap_path = store._snapshots_dir / 'exp-001' / 'epoch_0.json'
    snap_path.write_text('"just a string"', encoding='utf-8')
    with pytest.raises(StoreError, match='must contain a JSON object'):
      store.load_snapshot('exp-001', 0)


class TestResolveOriginalPath:
  """_resolve_original_path narrows via isinstance(param, PathParameter)."""

  def test_returns_relative_path_for_path_param(self, tmp_path: Path) -> None:
    src = make_source_dir(tmp_path)
    config = make_store_config(tmp_path)
    param = PathParameter(source=str(src), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    result = store._resolve_original_path(param, 'main.py')
    assert result is not None
    assert 'main.py' in result

  def test_returns_none_for_non_path_param(self, tmp_path: Path) -> None:

    class PromptParam(Parameter):
      def snapshot(self) -> dict[str, str]:
        return {'p': 'text'}

      def restore(self, content: dict[str, str]) -> None:
        pass

    config = make_store_config(tmp_path)
    param = PromptParam()
    store = FileStore(config)
    store.register_parameters({'source': param})
    assert store._resolve_original_path(param, 'key') is None

  def test_returns_none_for_path_outside_workspace(self, tmp_path: Path) -> None:
    """_resolve_original_path returns None when path is outside workspace."""
    outside = Path('/tmp/outside_workspace_param')
    outside.mkdir(parents=True, exist_ok=True)
    (outside / 'file.py').write_text('x = 1')
    config = make_store_config(tmp_path)
    param = PathParameter(source=str(outside), pattern='*')
    store = FileStore(config)
    store.register_parameters({'source': param})
    result = store._resolve_original_path(param, 'file.py')
    assert result is None


# -- atomic_write_json_safe coverage --


class TestAtomicWriteJsonSafe:
  def test_raises_store_error_on_tracking_error(self, tmp_path: Path) -> None:
    """atomic_write_json_safe wraps TrackingError in StoreError."""
    target = tmp_path / 'test.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    with patch(
      'autopilot.ai.store.store_io.atomic_write_json',
      side_effect=TrackingError('disk full'),
    ):
      with pytest.raises(StoreError) as exc_info:
        atomic_write_json_safe(target, {'key': 'val'})
      assert 'disk full' in str(exc_info.value)
      assert exc_info.value.__cause__ is not None


# -- merge missing branches --


class TestMergeMissingBranches:
  def test_merge_missing_experiment_id(self, tmp_path: Path) -> None:
    """merge_preview raises StoreError when experiment_id not found."""
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='not found'):
      store.merge_preview('nonexistent', 'exp-001')

  def test_merge_missing_from_experiment_id(self, tmp_path: Path) -> None:
    """merge_preview raises StoreError when from_experiment_id not found."""
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='not found'):
      store.merge_preview('exp-001', 'nonexistent')


# -- _three_way_merge coverage --


class TestThreeWayMerge:
  def test_binary_content_returns_none(self, tmp_path: Path) -> None:
    """_three_way_merge_text returns None for non-UTF-8 content."""
    store, _, _ = _make_store(tmp_path)
    result = store._three_way_merge_text(b'\x80\x81', b'\x90\x91', b'\xa0\xa1')
    assert result is None

  def test_trivial_ours_unchanged(self, tmp_path: Path) -> None:
    """When ours matches base, theirs wins."""
    store, _, _ = _make_store(tmp_path)
    base = b'line1\nline2\n'
    ours = b'line1\nline2\n'
    theirs = b'line1\nchanged\n'
    result = store._three_way_merge_text(base, ours, theirs)
    assert result == 'line1\nchanged\n'

  def test_trivial_theirs_unchanged(self, tmp_path: Path) -> None:
    """When theirs matches base, ours wins."""
    store, _, _ = _make_store(tmp_path)
    base = b'line1\nline2\n'
    ours = b'line1\nours change\n'
    theirs = b'line1\nline2\n'
    result = store._three_way_merge_text(base, ours, theirs)
    assert result == 'line1\nours change\n'

  def test_non_overlapping_edits_merge_cleanly(self, tmp_path: Path) -> None:
    """Non-overlapping changes produce merged text."""
    store, _, _ = _make_store(tmp_path)
    base = b'line1\nline2\nline3\nline4\n'
    ours = b'ours1\nline2\nline3\nline4\n'
    theirs = b'line1\nline2\nline3\ntheirs4\n'
    result = store._three_way_merge_text(base, ours, theirs)
    assert result is not None
    assert 'ours1' in result
    assert 'theirs4' in result

  def test_overlapping_edits_produce_conflict(self, tmp_path: Path) -> None:
    """Overlapping changes on same line produce None (conflict)."""
    store, _, _ = _make_store(tmp_path)
    base = b'line1\nline2\nline3\n'
    ours = b'line1\nours_change\nline3\n'
    theirs = b'line1\ntheirs_change\nline3\n'
    result = store._three_way_merge_text(base, ours, theirs)
    assert result is None


# -- _text_diff binary coverage --


class TestTextDiffBinary:
  def test_binary_diff_returns_marker(self, tmp_path: Path) -> None:
    """_text_diff returns '(binary files differ)' for non-UTF-8 content."""
    store, _, _ = _make_store(tmp_path)
    result = store._text_diff('file.bin', b'\x80\x81\x82', b'\x90\x91\x92')
    assert result == '(binary files differ)'

  def test_text_diff_produces_unified_diff(self, tmp_path: Path) -> None:
    """_text_diff produces unified diff for valid UTF-8."""
    store, _, _ = _make_store(tmp_path)
    result = store._text_diff('f.py', b'line1\nline2\n', b'line1\nmodified\n')
    assert '-line2' in result
    assert '+modified' in result


# -- status unknown experiment --


class TestStatusUnknownExperiment:
  def test_status_missing_experiment_raises(self, tmp_path: Path) -> None:
    """status with unknown experiment_id raises StoreError."""
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='not found'):
      store.status('unknown-exp')


# -- create_worktree lock held --


class TestCreateWorktreeLockHeld:
  def test_lock_held_raises_store_error(self, tmp_path: Path) -> None:
    """create_worktree raises StoreError when lock file already exists."""
    store, _, _ = _make_store(tmp_path)
    worktrees_dir = store._config.worktrees_path
    worktrees_dir.mkdir(parents=True, exist_ok=True)
    lock_path = worktrees_dir / 'exp-001.lock'
    lock_path.write_text('')
    with pytest.raises(StoreError, match='locked'):
      store.create_worktree('exp-001')


# -- load_state_dict bad JSON --


class TestLoadStateDictBadJson:
  def test_invalid_json_raises_store_error(self, tmp_path: Path) -> None:
    """load_state_dict with malformed JSON raises StoreError with chained cause."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text('{invalid json', encoding='utf-8')
    with pytest.raises(StoreError) as exc_info:
      store.load_state_dict()
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, TrackingError)


# -- load_refs parse error --


class TestLoadRefsParseError:
  def test_corrupt_refs_raises_store_error(self, tmp_path: Path) -> None:
    """load_refs with corrupt JSON raises StoreError."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    store._refs_file.parent.mkdir(parents=True, exist_ok=True)
    store._refs_file.write_text('{broken', encoding='utf-8')
    with pytest.raises(StoreError, match='failed to load refs'):
      store.load_refs()


# -- load_snapshot errors --


class TestLoadSnapshotErrors:
  def test_corrupt_snapshot_raises_store_error(self, tmp_path: Path) -> None:
    """load_snapshot with corrupt JSON raises StoreError."""
    store, _, _ = _make_store(tmp_path)
    snap_path = store._snapshots_dir / 'exp-001' / 'epoch_0.json'
    snap_path.write_text('{broken json', encoding='utf-8')
    with pytest.raises(StoreError, match='invalid JSON'):
      store.load_snapshot('exp-001', 0)

  def test_missing_snapshot_raises_store_error(self, tmp_path: Path) -> None:
    """load_snapshot for non-existent epoch raises StoreError."""
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='snapshot not found'):
      store.load_snapshot('exp-001', 99)


# -- _extract_changed_line_numbers coverage --


class TestExtractChangedLineNumbers:
  def test_replace_delete_insert_opcodes(self, tmp_path: Path) -> None:
    """_extract_changed_line_numbers reports base line indices touched by edits."""
    store, _, _ = _make_store(tmp_path)
    base_lines = ['line1\n', 'line2\n', 'line3\n', 'line4\n']
    modified_lines = ['replaced1\n', 'line3\n', 'line4\n', 'inserted\n']
    changed = store._extract_changed_line_numbers(base_lines, modified_lines)
    assert len(changed) > 0
    assert 0 in changed or 1 in changed


# -- _find_lca edge cases --


class TestFindLcaEdge:
  def test_no_common_ancestor_returns_none(self, tmp_path: Path) -> None:
    """When no common ancestor exists, _find_lca returns (None, None)."""
    store, _, _ = _make_store(tmp_path)
    refs = store.load_refs()
    refs['branches']['orphan'] = {'latest_epoch': 0, 'parent_id': 'ghost', 'parent_epoch': 0}
    store.save_refs(refs)
    lca_exp, lca_epoch = store._find_lca('exp-001', 'orphan', refs)
    assert lca_exp is None
    assert lca_epoch is None

  def test_ancestor_chain_parent_not_in_branches(self, tmp_path: Path) -> None:
    """_ancestor_chain_ordered stops when parent_id references a missing branch."""
    store, _, _ = _make_store(tmp_path)
    branches = store.load_refs().get('branches', {})
    chain = store._ancestor_chain_ordered('exp-001', branches)
    assert len(chain) >= 1
    assert chain[0] == ('exp-001', 0)

  def test_shared_parent_returns_correct_epoch(self, tmp_path: Path) -> None:
    """Two branches sharing a parent return ancestor with expected epoch."""
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot('exp-001', 1)
    store.branch('left')
    store.branch('right')
    refs = store.load_refs()
    lca_exp, lca_epoch = store._find_lca('left', 'right', refs)
    assert lca_exp is not None
    assert isinstance(lca_epoch, int)
    assert lca_epoch >= 0


# -- merge three-way integration (full merge path) --


class TestMergeThreeWayIntegration:
  def test_merge_ours_unchanged_takes_theirs(self, tmp_path: Path) -> None:
    """When ours hash matches base, resolved entries use theirs."""
    store, src, _ = _make_store(tmp_path, files={'f.txt': 'base content'})
    store.branch('feature')
    store.checkout('feature', 0)
    (src / 'f.txt').write_text('feature content')
    store.snapshot('feature', 1)
    result = store.merge_preview('exp-001', 'feature')
    assert result.is_resolved()
    key = next(k for k in result.resolved if 'f.txt' in k)
    content_hash = result.resolved[key].digest
    data = store.read_object(content_hash)
    assert data.decode('utf-8') == 'feature content'

  def test_merge_theirs_unchanged_takes_ours(self, tmp_path: Path) -> None:
    """When theirs hash matches base, resolved entries use ours."""
    store, src, _ = _make_store(tmp_path, files={'f.txt': 'base content'})
    store.branch('feature')
    store.checkout('exp-001', 0)
    (src / 'f.txt').write_text('main content')
    store.snapshot('exp-001', 1)
    result = store.merge_preview('exp-001', 'feature')
    assert result.is_resolved()
    key = next(k for k in result.resolved if 'f.txt' in k)
    content_hash = result.resolved[key].digest
    data = store.read_object(content_hash)
    assert data.decode('utf-8') == 'main content'

  def test_merge_conflict_produces_non_empty_conflicts(self, tmp_path: Path) -> None:
    """Conflicting edits to same line produce non-empty conflicts dict."""
    store, src, _ = _make_store(tmp_path, files={'c.txt': 'line1\nline2\nline3\n'})
    store.branch('feature')
    store.checkout('exp-001', 0)
    (src / 'c.txt').write_text('line1\nmain edit\nline3\n')
    store.snapshot('exp-001', 1)
    store.checkout('feature', 0)
    (src / 'c.txt').write_text('line1\nfeature edit\nline3\n')
    store.snapshot('feature', 1)
    result = store.merge_preview('exp-001', 'feature')
    assert not result.is_resolved()
    assert len(result.conflicts) > 0
    assert any('c.txt' in key for key in result.conflicts)

  def test_merge_clean_three_way_round_trips(self, tmp_path: Path) -> None:
    """Non-overlapping edits produce clean merge with resolved content."""
    store, src, _ = _make_store(tmp_path, files={'a.txt': 'line1\nline2\nline3\nline4\n'})
    store.branch('feature')
    store.checkout('exp-001', 0)
    (src / 'a.txt').write_text('ours1\nline2\nline3\nline4\n')
    store.snapshot('exp-001', 1)
    store.checkout('feature', 0)
    (src / 'a.txt').write_text('line1\nline2\nline3\ntheirs4\n')
    store.snapshot('feature', 1)
    result = store.merge_preview('exp-001', 'feature')
    assert result.is_resolved()
    assert len(result.conflicts) == 0
    key = next(k for k in result.resolved if 'a.txt' in k)
    merged_data = store.read_object(result.resolved[key].digest)
    merged_text = merged_data.decode('utf-8')
    assert 'ours1' in merged_text
    assert 'theirs4' in merged_text
