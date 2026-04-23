"""Comprehensive tests for FileStore: content-addressed code versioning."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store import FileStore, _hash_content
from autopilot.core.errors import StoreError
from autopilot.core.parameter import Parameter
from pathlib import Path
import json
import pytest


def _make_source(tmp_path: Path, name: str = 'src', files: dict[str, str] | None = None) -> Path:
  """Create a source directory with files for PathParameter."""
  src = tmp_path / name
  src.mkdir(parents=True, exist_ok=True)
  for fname, content in (files or {'main.py': 'print("hello")', 'util.py': 'x = 1'}).items():
    (src / fname).write_text(content)
  return src


def _make_store(tmp_path: Path, slug: str = 'exp-001', files: dict[str, str] | None = None):
  """Create a FileStore with a source directory and parameters."""
  src = _make_source(tmp_path, files=files)
  store_path = tmp_path / '.store'
  params = [PathParameter(source=str(src), pattern='*')]
  store = FileStore(store_path, slug, params)
  return store, src, params


# init (idempotent constructor)


class TestInit:
  def test_creates_store_structure(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    assert (store.path / 'objects').is_dir()
    assert (store.path / 'snapshots').is_dir()
    assert (store.path / 'refs.json').is_file()

  def test_stores_objects(self, tmp_path: Path) -> None:
    src = _make_source(tmp_path, files={'a.txt': 'hello'})
    store_path = tmp_path / '.store'
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(store_path, 'exp', params)

    content_hash = _hash_content('hello')
    prefix = content_hash[:2]
    rest = content_hash[2:]
    assert (store.path / 'objects' / prefix / rest).exists()

  def test_writes_epoch_0_snapshot(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    snap_path = store.path / 'snapshots' / 'exp-001' / 'epoch_0.json'
    assert snap_path.exists()
    data = json.loads(snap_path.read_text())
    assert data['epoch'] == 0
    assert len(data['entries']) == 2

  def test_sets_head(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    refs = json.loads((store.path / 'refs.json').read_text())
    assert refs['HEAD'] == {'slug': 'exp-001', 'epoch': 0}

  def test_idempotent_reentry(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path)
    store.snapshot(1)
    assert store.epoch == 1

    store2 = FileStore(store.path, 'exp-001', params)
    assert store2.epoch == 1
    assert store2.slug == 'exp-001'

  def test_empty_parameters(self, tmp_path: Path) -> None:
    src = tmp_path / 'empty_src'
    src.mkdir()
    store_path = tmp_path / '.store'
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(store_path, 'empty-exp', params)
    assert store.epoch == 0
    snap_path = store.path / 'snapshots' / 'empty-exp' / 'epoch_0.json'
    data = json.loads(snap_path.read_text())
    assert len(data['entries']) == 0

  def test_slug_and_epoch_properties(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    assert store.slug == 'exp-001'
    assert store.epoch == 0


# snapshot


class TestSnapshot:
  def test_captures_current_state(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('print("updated")')
    snap = store.snapshot(1)
    key = [k for k in snap.entries if 'main.py' in k][0]
    new_hash = _hash_content('print("updated")')
    assert snap.entries[key].hash == new_hash

  def test_deduplicates_unchanged_files(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    old_hash = _hash_content('x = 1')
    snap = store.snapshot(1)
    key = [k for k in snap.entries if 'util.py' in k][0]
    assert snap.entries[key].hash == old_hash

    prefix = old_hash[:2]
    rest = old_hash[2:]
    obj_path = store.path / 'objects' / prefix / rest
    assert obj_path.exists()

  def test_sequential_epochs(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    store.snapshot(1)
    assert store.epoch == 1
    store.snapshot(2)
    assert store.epoch == 2
    store.snapshot(3)
    assert store.epoch == 3

  def test_skipped_epoch_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='sequential'):
      store.snapshot(5)

  def test_updates_head(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.snapshot(1)
    refs = json.loads((store.path / 'refs.json').read_text())
    assert refs['HEAD']['epoch'] == 1

  def test_updates_epoch_property(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    assert store.epoch == 0
    store.snapshot(1)
    assert store.epoch == 1


# checkout


class TestCheckout:
  def test_restores_files(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    original_content = (src / 'main.py').read_text()
    (src / 'main.py').write_text('completely changed')
    store.snapshot(1)

    store.checkout(0)
    assert (src / 'main.py').read_text() == original_content

  def test_only_restores_tracked_files(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'untracked.log').write_text('should remain')
    store.checkout(0)
    assert (src / 'untracked.log').read_text() == 'should remain'

  def test_updates_head(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v2')
    store.snapshot(1)
    store.checkout(0)
    refs = json.loads((store.path / 'refs.json').read_text())
    assert refs['HEAD']['epoch'] == 0

  def test_nonexistent_epoch_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='snapshot not found'):
      store.checkout(99)


# diff


class TestDiff:
  def test_identical_snapshots(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    result = store.diff(0, store.slug, 0)
    assert len(result.entries) == 0

  def test_modified_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('updated content')
    store.snapshot(1)
    result = store.diff(0, store.slug, 1)
    modified = result.modified()
    assert len(modified) == 1
    assert 'main.py' in modified[0].path

  def test_added_file(self, tmp_path: Path) -> None:
    src = _make_source(tmp_path, files={'a.txt': 'hello'})
    store_path = tmp_path / '.store'
    params = [PathParameter(source=str(src), pattern='*')]
    store = FileStore(store_path, 'exp', params)
    (src / 'b.txt').write_text('new file')
    store.snapshot(1)
    result = store.diff(0, store.slug, 1)
    added = result.added()
    assert len(added) == 1
    assert 'b.txt' in added[0].path

  def test_deleted_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'util.py').unlink()
    store.snapshot(1)
    result = store.diff(0, store.slug, 1)
    deleted = result.deleted()
    assert len(deleted) == 1
    assert 'util.py' in deleted[0].path

  def test_across_slugs(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path)
    store.branch('branch-1', 0)
    branch_store = FileStore(store.path, 'branch-1', params)
    (src / 'main.py').write_text('branch version')
    branch_store.snapshot(1)

    result = store.diff(0, 'branch-1', 1)
    assert len(result.modified()) == 1

  def test_text_diff_content(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('line1\nline2\nline3\n')
    store.snapshot(1)
    (src / 'main.py').write_text('line1\nmodified\nline3\n')
    store.snapshot(2)
    result = store.diff(1, store.slug, 2)
    modified = result.modified()
    assert len(modified) == 1
    assert '-line2' in modified[0].text_diff
    assert '+modified' in modified[0].text_diff


# branch


class TestBranch:
  def test_creates_new_slug(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.branch('feature-1', 0)
    refs = json.loads((store.path / 'refs.json').read_text())
    assert 'feature-1' in refs

  def test_shares_objects(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    objects_before = list((store.path / 'objects').rglob('*'))
    obj_files_before = [f for f in objects_before if f.is_file()]
    store.branch('feature-1', 0)
    objects_after = list((store.path / 'objects').rglob('*'))
    obj_files_after = [f for f in objects_after if f.is_file()]
    assert len(obj_files_before) == len(obj_files_after)

  def test_records_parent(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.branch('child', 0)
    refs = json.loads((store.path / 'refs.json').read_text())
    assert refs['child']['parent_slug'] == 'exp-001'
    assert refs['child']['parent_epoch'] == 0

  def test_duplicate_slug_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    store.branch('dup', 0)
    with pytest.raises(StoreError, match='already exists'):
      store.branch('dup', 0)

  def test_checkout_independent(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path)
    original = (src / 'main.py').read_text()
    store.branch('feature', 0)

    feature = FileStore(store.path, 'feature', params)
    (src / 'main.py').write_text('feature change')
    feature.snapshot(1)

    store.checkout(0)
    assert (src / 'main.py').read_text() == original


# merge


class TestMerge:
  def test_no_conflicts(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path, files={'a.txt': 'original a', 'b.txt': 'original b'})
    store.branch('feature', 0)

    (src / 'a.txt').write_text('changed a on main')
    store.snapshot(1)

    store.checkout(0)
    feature = FileStore(store.path, 'feature', params)
    (src / 'b.txt').write_text('changed b on feature')
    feature.snapshot(1)

    result = store.merge('feature')
    assert result.merged is True
    assert len(result.conflicts) == 0

  def test_conflicting_changes(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path, files={'conflict.txt': 'line1\nline2\nline3\n'})
    store.branch('feature', 0)

    (src / 'conflict.txt').write_text('line1\nmain change\nline3\n')
    store.snapshot(1)

    store.checkout(0)
    feature = FileStore(store.path, 'feature', params)
    (src / 'conflict.txt').write_text('line1\nfeature change\nline3\n')
    feature.snapshot(1)

    result = store.merge('feature')
    assert result.merged is False
    assert len(result.conflicts) > 0

  def test_one_side_only(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path, files={'a.txt': 'original', 'b.txt': 'original'})
    store.branch('feature', 0)

    feature = FileStore(store.path, 'feature', params)
    (src / 'a.txt').write_text('only feature changed this')
    feature.snapshot(1)

    result = store.merge('feature')
    assert result.merged is True

  def test_defaults_to_latest_epochs(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path)
    store.branch('feature', 0)

    feature = FileStore(store.path, 'feature', params)
    (src / 'main.py').write_text('feature v1')
    feature.snapshot(1)
    (src / 'main.py').write_text('feature v2')
    feature.snapshot(2)

    result = store.merge('feature')
    assert result.merged_snapshot is not None

  def test_nonexistent_slug_raises(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    with pytest.raises(StoreError, match='not found'):
      store.merge('nonexistent')


# log


class TestLog:
  def test_returns_all_epochs(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot(1)
    (src / 'main.py').write_text('v2')
    store.snapshot(2)
    entries = store.log()
    assert len(entries) == 3
    assert entries[0].epoch == 0
    assert entries[1].epoch == 1
    assert entries[2].epoch == 2

  def test_single_epoch(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    entries = store.log()
    assert len(entries) == 1
    assert entries[0].epoch == 0

  def test_chronological_order(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot(1)
    entries = store.log()
    assert entries[0].epoch < entries[1].epoch

  def test_file_count(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    entries = store.log()
    assert entries[0].file_count == 2


# status


class TestStatus:
  def test_clean(self, tmp_path: Path) -> None:
    store, _, _ = _make_store(tmp_path)
    result = store.status()
    unchanged = result.unchanged()
    assert len(unchanged) == 2
    assert result.modified() == []
    assert result.added() == []
    assert result.deleted() == []

  def test_modified_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('changed')
    result = store.status()
    modified = result.modified()
    assert len(modified) == 1
    assert 'main.py' in modified[0].path

  def test_deleted_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'util.py').unlink()
    result = store.status()
    deleted = result.deleted()
    assert len(deleted) == 1
    assert 'util.py' in deleted[0].path

  def test_added_file(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'new.py').write_text('new file')
    result = store.status()
    added = result.added()
    assert len(added) == 1
    assert 'new.py' in added[0].path


# promote


class TestPromote:
  def test_restores_files(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('epoch 1 version')
    store.snapshot(1)
    (src / 'main.py').write_text('epoch 2 version')
    store.snapshot(2)

    store.promote(1)
    assert (src / 'main.py').read_text() == 'epoch 1 version'

  def test_updates_baseline(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('promoted version')
    store.snapshot(1)
    store.promote(1)

    snap_path = store.path / 'snapshots' / 'exp-001' / 'epoch_0.json'
    data = json.loads(snap_path.read_text())
    key = [k for k in data['entries'] if 'main.py' in k][0]
    promoted_hash = _hash_content('promoted version')
    assert data['entries'][key]['hash'] == promoted_hash

  def test_detects_external_modification(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('snapshot version')
    store.snapshot(1)
    (src / 'main.py').write_text('externally modified')
    store.snapshot(2)
    (src / 'main.py').write_text('tampered after snapshot 2')
    with pytest.raises(StoreError, match='external modification'):
      store.promote(1)


# round-trip integration


class TestIntegration:
  def test_full_lifecycle(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    original_main = (src / 'main.py').read_text()
    original_util = (src / 'util.py').read_text()

    (src / 'main.py').write_text('epoch 1')
    store.snapshot(1)
    (src / 'main.py').write_text('epoch 2')
    store.snapshot(2)

    store.checkout(0)
    assert (src / 'main.py').read_text() == original_main
    assert (src / 'util.py').read_text() == original_util

    store.checkout(1)
    assert (src / 'main.py').read_text() == 'epoch 1'

  def test_branch_and_merge_lifecycle(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path, files={'a.txt': 'base a', 'b.txt': 'base b'})

    store.branch('feature', 0)
    feature = FileStore(store.path, 'feature', params)

    (src / 'a.txt').write_text('main changed a')
    store.snapshot(1)

    store.checkout(0)
    (src / 'b.txt').write_text('feature changed b')
    feature.snapshot(1)

    result = store.merge('feature')
    assert result.merged is True

  def test_parallel_experiments(self, tmp_path: Path) -> None:
    src = _make_source(tmp_path, files={'shared.py': 'base'})
    store_path = tmp_path / '.store'
    params = [PathParameter(source=str(src), pattern='*')]

    store_a = FileStore(store_path, 'exp-a', params)
    store_b = FileStore(store_path, 'exp-b', params)

    (src / 'shared.py').write_text('exp-a version')
    store_a.snapshot(1)

    store_b.checkout(0)
    (src / 'shared.py').write_text('exp-b version')
    store_b.snapshot(1)

    store_a.checkout(1)
    assert (src / 'shared.py').read_text() == 'exp-a version'

    store_b.checkout(1)
    assert (src / 'shared.py').read_text() == 'exp-b version'

  def test_rollback_scenario(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('good version')
    store.snapshot(1)
    (src / 'main.py').write_text('bad regression')
    store.snapshot(2)

    store.checkout(1)
    assert (src / 'main.py').read_text() == 'good version'

  def test_idempotent_reentry_preserves_state(self, tmp_path: Path) -> None:
    store, src, params = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot(1)

    store2 = FileStore(store.path, 'exp-001', params)
    assert store2.epoch == 1
    entries = store2.log()
    assert len(entries) == 2


# atomic writes and locking


class TestAtomicAndLocking:
  def test_lock_file_created_and_released(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    lock_path = store.path / '.lock'
    assert not lock_path.exists()
    store.snapshot(1)
    assert not lock_path.exists()

  def test_lock_prevents_concurrent_access(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    lock_path = store.path / '.lock'
    lock_path.write_text('')
    with pytest.raises(StoreError, match='locked'):
      store.snapshot(1)
    lock_path.unlink()

  def test_refs_json_not_corrupted_after_operations(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    for i in range(1, 6):
      (src / 'main.py').write_text(f'version {i}')
      store.snapshot(i)
    refs = json.loads((store.path / 'refs.json').read_text())
    assert refs['exp-001']['latest_epoch'] == 5
    assert refs['HEAD']['epoch'] == 5

  def test_snapshot_files_valid_json(self, tmp_path: Path) -> None:
    store, src, _ = _make_store(tmp_path)
    (src / 'main.py').write_text('v1')
    store.snapshot(1)
    for snap_file in (store.path / 'snapshots' / 'exp-001').iterdir():
      data = json.loads(snap_file.read_text())
      assert 'epoch' in data
      assert 'entries' in data
      assert 'timestamp' in data


# snapshot/restore decoupling tests


class TestSnapshotRestoreDecoupling:
  def test_filestore_snapshot_via_param_snapshot(self, tmp_path: Path) -> None:
    """Custom Parameter subclass with snapshot() works with FileStore."""

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    store_path = tmp_path / '.store'
    param = PromptParameter('hello world')
    store = FileStore(store_path, 'test-slug', [param])
    snap = store._load_snapshot('test-slug', 0)
    assert len(snap.entries) == 1
    key = list(snap.entries.keys())[0]
    assert key == 'param_0/prompt'

  def test_filestore_checkout_via_param_restore(self, tmp_path: Path) -> None:
    """Checkout calls param.restore() with correct content."""

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    store_path = tmp_path / '.store'
    param = PromptParameter('v1')
    store = FileStore(store_path, 'test-slug', [param])

    param._text = 'v2'
    store.snapshot(1)

    store.checkout(0)
    assert param._text == 'v1'

  def test_filestore_empty_snapshot(self, tmp_path: Path) -> None:
    """Parameter with default empty snapshot() causes no crash."""
    store_path = tmp_path / '.store'
    param = Parameter()
    store = FileStore(store_path, 'empty-snap', [param])
    snap = store._load_snapshot('empty-snap', 0)
    assert len(snap.entries) == 0

  def test_filestore_diff_uses_param_snapshot(self, tmp_path: Path) -> None:
    """Diff detects changes from param.snapshot() output."""

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    store_path = tmp_path / '.store'
    param = PromptParameter('original')
    store = FileStore(store_path, 'diff-test', [param])

    param._text = 'modified'
    store.snapshot(1)

    result = store.diff(0, 'diff-test', 1)
    assert len(result.modified()) == 1
    assert 'prompt' in result.modified()[0].path

  def test_filestore_multiple_param_types(self, tmp_path: Path) -> None:
    """Mix of PathParameter and custom parameter both snapshot/checkout correctly."""

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    src = _make_source(tmp_path, files={'code.py': 'x = 1'})
    store_path = tmp_path / '.store'
    path_param = PathParameter(source=str(src), pattern='*')
    prompt_param = PromptParameter('system prompt v1')
    store = FileStore(store_path, 'mixed', [path_param, prompt_param])

    snap = store._load_snapshot('mixed', 0)
    assert len(snap.entries) == 2

    (src / 'code.py').write_text('x = 2')
    prompt_param._text = 'system prompt v2'
    store.snapshot(1)

    store.checkout(0)
    assert (src / 'code.py').read_text() == 'x = 1'
    assert prompt_param._text == 'system prompt v1'

  def test_filestore_no_pathparameter_import(self) -> None:
    """Verify ai/store.py does not import PathParameter."""
    import autopilot.ai.store as store_module

    source = Path(store_module.__file__).read_text()
    assert 'PathParameter' not in source

  def test_filestore_composite_keys(self, tmp_path: Path) -> None:
    """Snapshot keys use param_name/state_key format."""

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'system': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['system']

    store_path = tmp_path / '.store'
    param = PromptParameter('hello')
    store = FileStore(store_path, 'keys-test', [param])
    snap = store._load_snapshot('keys-test', 0)
    keys = list(snap.entries.keys())
    assert len(keys) == 1
    assert keys[0] == 'param_0/system'

  def test_filestore_content_addressed_dedup(self, tmp_path: Path) -> None:
    """Two parameters with identical snapshot content share one object."""

    class PromptParameter(Parameter):
      def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

      def snapshot(self) -> dict[str, str]:
        return {'prompt': self._text}

      def restore(self, content: dict[str, str]) -> None:
        self._text = content['prompt']

    store_path = tmp_path / '.store'
    p1 = PromptParameter('same content')
    p2 = PromptParameter('same content')
    store = FileStore(store_path, 'dedup', [p1, p2])

    snap = store._load_snapshot('dedup', 0)
    hashes = [entry.hash for entry in snap.entries.values()]
    assert hashes[0] == hashes[1]
    obj_hash = hashes[0]
    prefix = obj_hash[:2]
    rest = obj_hash[2:]
    assert (store.path / 'objects' / prefix / rest).exists()
