"""Integration tests for FileStore: parallel isolation, content-addressed dedup,
worktree lifecycle, and Store+Trainer interaction.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store_lock import hash_content
from autopilot.core.errors import StoreError
from autopilot.core.parameter import Parameter
from autopilot.core.store.types import MergeIndex
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
import pytest


def _make_store(
  tmp_path: Path,
  slug: str = 'exp-001',
  files: dict[str, str] | None = None,
  source_name: str = 'src',
) -> tuple[FileStore, Path, list[PathParameter]]:
  src = make_source_dir(tmp_path, name=source_name, files=files)
  config = make_store_config(tmp_path)
  params = [PathParameter(source=str(src), pattern='*')]
  store = FileStore(config)
  store.register_parameters({'source': params[0]})
  store.snapshot(slug, 0)
  return store, src, params


# -- Parallel isolation: two experiment_ids, snapshot each independently --


class TestParallelIsolation:
  def test_two_experiments_no_interference(self, tmp_path: Path) -> None:
    """Two experiment_ids snapshot independently with no interference."""
    src = make_source_dir(tmp_path, files={'data.txt': 'shared base'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('exp-a', 0)
    store.snapshot('exp-b', 0)

    (src / 'data.txt').write_text('exp-a version')
    store.snapshot('exp-a', 1)

    (src / 'data.txt').write_text('exp-b version')
    store.snapshot('exp-b', 1)

    store.checkout('exp-a', 1)
    assert (src / 'data.txt').read_text() == 'exp-a version'

    store.checkout('exp-b', 1)
    assert (src / 'data.txt').read_text() == 'exp-b version'

    store.checkout('exp-a', 0)
    assert (src / 'data.txt').read_text() == 'shared base'

    store.checkout('exp-b', 0)
    assert (src / 'data.txt').read_text() == 'shared base'

  def test_independent_epoch_sequences(self, tmp_path: Path) -> None:
    """Each experiment tracks its own epoch independently."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})

    store.snapshot('alpha', 0, force=True)
    store.snapshot('alpha', 1, force=True)
    store.snapshot('alpha', 2, force=True)

    store.snapshot('beta', 0, force=True)
    store.snapshot('beta', 1, force=True)

    log_alpha = store.log('alpha')
    log_beta = store.log('beta')

    assert [e.epoch for e in log_alpha] == [0, 1, 2]
    assert [e.epoch for e in log_beta] == [0, 1]

  def test_snapshot_one_does_not_affect_other_log(self, tmp_path: Path) -> None:
    """Snapshotting one experiment does not change another's log."""
    src = make_source_dir(tmp_path, files={'f.txt': 'base'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('x', 0)
    store.snapshot('y', 0)

    (src / 'f.txt').write_text('x-only change')
    store.snapshot('x', 1)

    log_y = store.log('y')
    assert len(log_y) == 1
    assert log_y[0].epoch == 0


# -- Content-addressed dedup --


class TestContentAddressedDedup:
  def test_same_content_across_experiments_same_hash(self, tmp_path: Path) -> None:
    """Same file content across experiments produces the same hash in object store."""
    src = make_source_dir(tmp_path, files={'shared.txt': 'identical content'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('exp-1', 0)
    store.snapshot('exp-2', 0)

    snap1 = store.load_snapshot('exp-1', 0)
    snap2 = store.load_snapshot('exp-2', 0)

    hashes1 = {entry.digest for entry in snap1.entries.values()}
    hashes2 = {entry.digest for entry in snap2.entries.values()}
    assert hashes1 == hashes2

    expected_hash = hash_content('identical content')
    assert expected_hash in hashes1

  def test_unchanged_files_between_epochs_same_hash(self, tmp_path: Path) -> None:
    """Unchanged files between epochs produce the same hash."""
    src = make_source_dir(tmp_path, files={'a.txt': 'stable', 'b.txt': 'changes'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('exp', 0)
    (src / 'b.txt').write_text('changed')
    store.snapshot('exp', 1)

    snap0 = store.load_snapshot('exp', 0)
    snap1 = store.load_snapshot('exp', 1)

    a_key = next(k for k in snap0.entries if 'a.txt' in k)
    assert snap0.entries[a_key].digest == snap1.entries[a_key].digest

    b_key = next(k for k in snap0.entries if 'b.txt' in k)
    assert snap0.entries[b_key].digest != snap1.entries[b_key].digest

  def test_dedup_only_stores_one_object(self, tmp_path: Path) -> None:
    """Content-addressed dedup means the same content isn't stored twice."""
    config = make_store_config(tmp_path)

    class DupParam(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'data': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['data']

    p1 = DupParam('same')
    p2 = DupParam('same')
    store = FileStore(config)
    store.register_parameters({'p0': p1, 'p1': p2})
    store.snapshot('dedup-test', 0)

    snap = store.load_snapshot('dedup-test', 0)
    hashes = [entry.digest for entry in snap.entries.values()]
    assert hashes[0] == hashes[1]

    obj_hash = hashes[0]
    obj_path = config.objects_path / obj_hash[:2] / obj_hash[2:]
    assert obj_path.exists()


# -- Empty parameters --


class TestEmptyParameters:
  def test_empty_parameters_snapshot_has_empty_entries(self, tmp_path: Path) -> None:
    """Snapshot without register_parameters raises StoreError."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    with pytest.raises(StoreError, match='register_parameters'):
      store.snapshot('empty-exp', 0)

  def test_empty_pathparam_snapshot(self, tmp_path: Path) -> None:
    """PathParameter pointing at an empty directory produces empty entries."""
    empty_dir = tmp_path / 'empty_dir'
    empty_dir.mkdir()
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(empty_dir), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})
    snap = store.snapshot('empty-path', 0)
    assert snap.entries == {}

  def test_empty_custom_param_snapshot(self, tmp_path: Path) -> None:
    """Custom Parameter with empty snapshot() produces empty entries."""
    config = make_store_config(tmp_path)
    param = Parameter()
    store = FileStore(config)
    store.register_parameters({'source': param})
    snap = store.snapshot('empty-custom', 0)
    assert snap.entries == {}


# -- Very long experiment ID --


class TestLongExperimentId:
  def test_long_experiment_id(self, tmp_path: Path) -> None:
    """200+ char experiment ID works correctly."""
    long_id = 'a' * 220
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    snap = store.snapshot(long_id, 0)
    assert snap.epoch == 0

    log = store.log(long_id)
    assert len(log) == 1
    assert log[0].epoch == 0

    refs = store.load_refs()
    assert long_id in refs['branches']

  def test_long_id_snapshot_and_checkout(self, tmp_path: Path) -> None:
    """Long experiment ID roundtrips through snapshot/checkout."""
    long_id = 'experiment-' + 'x' * 200
    src = make_source_dir(tmp_path, files={'f.txt': 'content'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot(long_id, 0)
    (src / 'f.txt').write_text('changed')
    store.snapshot(long_id, 1)
    store.checkout(long_id, 0)
    assert (src / 'f.txt').read_text() == 'content'


# -- Special characters in experiment ID --


class TestSpecialCharExperimentId:
  def test_special_characters_in_id(self, tmp_path: Path) -> None:
    """Special characters in experiment ID work."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})

    special_ids = [
      'exp-with-dashes',
      'exp_with_underscores',
      'exp.with.dots',
      'exp@v2',
      'user/branch',
    ]
    for exp_id in special_ids:
      store.snapshot(exp_id, 0)
      log = store.log(exp_id)
      assert len(log) == 1, f'Failed for ID: {exp_id}'

  def test_numeric_experiment_id(self, tmp_path: Path) -> None:
    """Numeric-looking experiment ID works."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('12345', 0)
    log = store.log('12345')
    assert len(log) == 1


# -- Sequential epoch enforcement --


class TestSequentialEpochEnforcement:
  def test_skip_epoch_raises_store_error(self, tmp_path: Path) -> None:
    """Skipping an epoch raises StoreError."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('seq-exp', 0)
    with pytest.raises(StoreError, match='sequential'):
      store.snapshot('seq-exp', 5)

  def test_duplicate_epoch_raises_store_error(self, tmp_path: Path) -> None:
    """Duplicate epoch raises StoreError."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('dup-exp', 0)
    with pytest.raises(StoreError, match='sequential'):
      store.snapshot('dup-exp', 0)

  def test_reverse_epoch_raises(self, tmp_path: Path) -> None:
    """Backwards epoch raises StoreError."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('rev-exp', 0, force=True)
    store.snapshot('rev-exp', 1, force=True)
    store.snapshot('rev-exp', 2, force=True)
    with pytest.raises(StoreError, match='sequential'):
      store.snapshot('rev-exp', 1, force=True)


# -- Checkout non-existent experiment/epoch --


class TestCheckoutErrors:
  def test_checkout_nonexistent_experiment_raises(self, tmp_path: Path) -> None:
    """Checkout non-existent experiment raises StoreError."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    with pytest.raises(StoreError, match='not found'):
      store.checkout('ghost-experiment', 0)

  def test_checkout_nonexistent_epoch_raises(self, tmp_path: Path) -> None:
    """Checkout non-existent epoch raises StoreError."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    src = tmp_path / 'src'
    src.mkdir(exist_ok=True)
    (src / 'dummy.txt').write_text('x', encoding='utf-8')
    param = PathParameter(source=str(src), pattern='**/*')
    store.register_parameters({'source': param})
    store.snapshot('real-exp', 0)
    with pytest.raises(StoreError, match='snapshot not found'):
      store.checkout('real-exp', 99)


# -- Worktree lifecycle --


class TestWorktreeLifecycle:
  def test_create_list_remove_verify(self, tmp_path: Path) -> None:
    """Full worktree lifecycle: create, list, remove, verify cleanup."""
    store, _, _ = _make_store(tmp_path)

    assert store.list_worktrees() == []

    wt_path = store.create_worktree('exp-001')
    assert wt_path.is_dir()
    assert store.list_worktrees() == ['exp-001']

    refs = store.load_refs()
    assert 'exp-001' in refs.get('worktrees', {})

    store.remove_worktree('exp-001')
    assert not wt_path.exists()
    assert store.list_worktrees() == []

    refs_after = store.load_refs()
    assert 'exp-001' not in refs_after.get('worktrees', {})

  def test_multiple_worktrees(self, tmp_path: Path) -> None:
    """Multiple worktrees can coexist."""
    store, _, _ = _make_store(tmp_path)
    store.branch('exp-002')
    store.branch('exp-003')

    wt1 = store.create_worktree('exp-001')
    wt2 = store.create_worktree('exp-002')
    wt3 = store.create_worktree('exp-003')

    assert wt1 != wt2 != wt3
    assert sorted(store.list_worktrees()) == ['exp-001', 'exp-002', 'exp-003']

    store.remove_worktree('exp-002')
    assert sorted(store.list_worktrees()) == ['exp-001', 'exp-003']
    assert not wt2.exists()
    assert wt1.exists()
    assert wt3.exists()

  def test_worktree_lock_cleaned_up(self, tmp_path: Path) -> None:
    """Lock file is cleaned up after worktree creation."""
    store, _, _ = _make_store(tmp_path)
    store.create_worktree('exp-001')
    lock_path = store.config.worktrees_path / 'exp-001.lock'
    assert not lock_path.exists()

  def test_remove_nonexistent_worktree_idempotent(self, tmp_path: Path) -> None:
    """Removing a non-existent worktree is idempotent."""
    store, _, _ = _make_store(tmp_path)
    store.remove_worktree('never-created')


# -- save_state_dict / load_state_dict round-trip --


class TestStateDictRoundTrip:
  def test_save_and_load(self, tmp_path: Path) -> None:
    """save_state_dict / load_state_dict round-trip."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    state = {
      'trees': {
        'main': {
          'name': 'main',
          'nodes': [
            {'id': 'root', 'parent': None},
            {'id': 'child', 'parent': 'root'},
          ],
        }
      },
      'active': 'main',
    }
    store.save_state_dict(state)
    loaded = store.load_state_dict()
    assert loaded == state

  def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
    """load_state_dict returns None when no forest file exists."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    assert store.load_state_dict() is None

  def test_save_overwrites_previous(self, tmp_path: Path) -> None:
    """save_state_dict overwrites previous state."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    store.save_state_dict({'v': 1})
    store.save_state_dict({'v': 2})
    assert store.load_state_dict() == {'v': 2}

  def test_empty_state_dict(self, tmp_path: Path) -> None:
    """Empty dict round-trips."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    store.save_state_dict({})
    assert store.load_state_dict() == {}

  def test_nested_state_dict(self, tmp_path: Path) -> None:
    """Deeply nested state dict round-trips."""
    config = make_store_config(tmp_path)
    store = FileStore(config)
    state = {
      'level1': {
        'level2': {
          'level3': ['a', 'b', 'c'],
        }
      }
    }
    store.save_state_dict(state)
    assert store.load_state_dict() == state


# -- Merge with real conflicts --


class TestMergeWithConflicts:
  def test_conflicting_same_file_different_content(self, tmp_path: Path) -> None:
    """Two experiments modify the same file with different content -> conflict."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.md').write_text('base content\n')
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(prompts_dir), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (prompts_dir / 'main.md').write_text('exp-a changed\n')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (prompts_dir / 'main.md').write_text('exp-b changed\n')
    store.snapshot('exp-b', 1)

    result = store.merge_preview('exp-a', 'exp-b')
    assert isinstance(result, MergeIndex)
    assert not result.is_resolved()
    assert isinstance(result.conflicts, dict)
    assert len(result.conflicts) > 0
    conflict_keys = list(result.conflicts)
    has_main_md = any('main.md' in c for c in conflict_keys)
    assert has_main_md, f'Expected main.md in conflicts, got {conflict_keys}'

  def test_no_conflict_different_files(self, tmp_path: Path) -> None:
    """Two experiments modify different files -> no conflict."""
    src = make_source_dir(tmp_path, files={'a.txt': 'base a', 'b.txt': 'base b'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'a.txt').write_text('exp-a changed a')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'b.txt').write_text('exp-b changed b')
    store.snapshot('exp-b', 1)

    result = store.merge_preview('exp-a', 'exp-b')
    assert result.is_resolved()
    assert result.conflicts == {}

  def test_merge_index_fields(self, tmp_path: Path) -> None:
    """MergeIndex has conflicts: dict and resolved: dict fields."""
    src = make_source_dir(tmp_path, files={'f.txt': 'base\n'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('exp-a', 0)
    store.branch('exp-b')

    store.checkout('exp-a', 0)
    (src / 'f.txt').write_text('change a\n')
    store.snapshot('exp-a', 1)

    store.checkout('exp-b', 0)
    (src / 'f.txt').write_text('change b\n')
    store.snapshot('exp-b', 1)

    result = store.merge_preview('exp-a', 'exp-b')
    assert hasattr(result, 'conflicts')
    assert hasattr(result, 'resolved')
    assert isinstance(result.conflicts, dict)
    assert isinstance(result.resolved, dict)
    for key in result.conflicts:
      assert isinstance(key, str)


# -- Cross-experiment full lifecycle --


class TestCrossExperimentLifecycle:
  def test_full_parallel_lifecycle(self, tmp_path: Path) -> None:
    """Full lifecycle: two experiments created, snapshotted, checked out independently."""
    src = make_source_dir(tmp_path, files={'code.py': 'v0'})
    config = make_store_config(tmp_path)
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(config)
    store.register_parameters({'source': params[0]})

    store.snapshot('main', 0)

    (src / 'code.py').write_text('main-v1')
    store.snapshot('main', 1)

    store.checkout('main', 0)
    store.snapshot('branch', 0)

    (src / 'code.py').write_text('branch-v1')
    store.snapshot('branch', 1)

    store.checkout('main', 1)
    assert (src / 'code.py').read_text() == 'main-v1'

    store.checkout('branch', 1)
    assert (src / 'code.py').read_text() == 'branch-v1'

    main_log = store.log('main')
    branch_log = store.log('branch')
    assert len(main_log) == 2
    assert len(branch_log) == 2

  def test_many_epochs_two_experiments(self, tmp_path: Path) -> None:
    """Two experiments each run 5 epochs independently."""
    config = make_store_config(tmp_path)

    class CounterParam(Parameter):
      def __init__(self) -> None:
        super().__init__()
        self._value = 0

      def snapshot(self) -> dict[str, str]:
        return {'count': str(self._value)}

      def restore(self, content: dict[str, str]) -> None:
        self._value = int(content['count'])

    p = CounterParam()
    store = FileStore(config)
    store.register_parameters({'source': p})

    for epoch in range(5):
      p._value = epoch * 10
      store.snapshot('exp-a', epoch)

    p._value = 0
    for epoch in range(5):
      p._value = epoch * 100
      store.snapshot('exp-b', epoch)

    log_a = store.log('exp-a')
    log_b = store.log('exp-b')
    assert [e.epoch for e in log_a] == [0, 1, 2, 3, 4]
    assert [e.epoch for e in log_b] == [0, 1, 2, 3, 4]

    store.checkout('exp-a', 3)
    assert p._value == 30

    store.checkout('exp-b', 2)
    assert p._value == 200


class TestFileStoreSnapshotNotDict:
  def test_snapshot_array_raises_store_error(self, tmp_path: Path) -> None:
    """StoreError when snapshot file holds a JSON array instead of object."""
    store, _src, _params = _make_store(tmp_path)
    snap_path = tmp_path / '.autopilot' / 'snapshots' / 'exp-001' / 'epoch_0.json'
    snap_path.write_text('[1, 2, 3]')
    with pytest.raises(StoreError) as exc_info:
      store.load_snapshot('exp-001', 0)
    assert 'list' in str(exc_info.value)
