"""Tests for FileStore stash / WIP snapshots (Plan 18).

Covers stash creation, stash_list ordering, stash_pop LIFO/explicit,
empty-pop error, reflog entries, corrupt manifest handling, prune_orphans
with stash-only blobs, concurrent stash under lock, non-numeric junk files,
parameter re-registration, and stash_pop after registration changed.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.snapshot import SnapshotManifest
from autopilot.tracking.file_lock import ConcurrentMutationError
from autopilot.tracking.io import read_jsonl
from pathlib import Path
import json
import pytest
import threading


def _make_store(
  tmp_path: Path,
  files: dict[str, str] | None = None,
) -> tuple[FileStore, Path, PathParameter]:
  """Create a FileStore with a single PathParameter for testing."""
  if files is None:
    files = {'main.py': 'print("hello")\n'}
  src = tmp_path / 'src'
  src.mkdir(parents=True, exist_ok=True)
  for name, content in files.items():
    (src / name).write_text(content)
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  return store, src, param


def _read_reflog(store: FileStore) -> list[dict]:
  """Read all reflog entries from the store."""
  path = store.config.store_path / 'reflog.jsonl'
  return read_jsonl(path, strict=False)


class TestStashCreation:
  """stash() writes stash/0000.json with valid manifest."""

  def test_stash_writes_first_file(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    manifest = store.stash()
    stash_dir = store.config.store_path / 'stash'
    assert (stash_dir / '0000.json').exists()
    assert manifest.epoch == -1
    assert len(manifest.entries) > 0

  def test_stash_manifest_parses_round_trip(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    manifest = store.stash()
    stash_path = store.config.store_path / 'stash' / '0000.json'
    data = json.loads(stash_path.read_text())
    reloaded = SnapshotManifest.from_dict(data)
    assert reloaded.epoch == -1
    assert len(reloaded.entries) == len(manifest.entries)
    assert reloaded.timestamp == manifest.timestamp

  def test_stash_with_context(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    manifest = store.stash(context='WIP before refactor')
    assert manifest.context == 'WIP before refactor'

  def test_stash_increments_index(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash()
    store.stash()
    store.stash()
    stash_dir = store.config.store_path / 'stash'
    assert (stash_dir / '0000.json').exists()
    assert (stash_dir / '0001.json').exists()
    assert (stash_dir / '0002.json').exists()


class TestStashListOrder:
  """stash_list() returns oldest-to-newest order by index."""

  def test_stash_list_empty(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    assert store.stash_list() == []

  def test_stash_list_order(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.stash(context='first')
    (src / 'main.py').write_text('print("v2")\n')
    store.stash(context='second')
    manifests = store.stash_list()
    assert len(manifests) == 2
    assert manifests[0].context == 'first'
    assert manifests[1].context == 'second'


class TestStashPopLIFO:
  """stash_pop() restores latest pushed content; file removed."""

  def test_stash_pop_lifo(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.stash(context='first')
    (src / 'main.py').write_text('print("v2")\n')
    store.stash(context='second')

    (src / 'main.py').write_text('print("dirty")\n')
    manifest = store.stash_pop()
    assert manifest.context == 'second'
    assert (src / 'main.py').read_text() == 'print("v2")\n'

    remaining = store.stash_list()
    assert len(remaining) == 1

  def test_stash_pop_explicit_index(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    (src / 'main.py').write_text('version_0\n')
    store.stash(context='stash-0')
    (src / 'main.py').write_text('version_1\n')
    store.stash(context='stash-1')
    (src / 'main.py').write_text('version_2\n')
    store.stash(context='stash-2')

    (src / 'main.py').write_text('dirty\n')
    manifest = store.stash_pop(index=0)
    assert manifest.context == 'stash-0'
    assert (src / 'main.py').read_text() == 'version_0\n'

    remaining = store.stash_list()
    assert len(remaining) == 2


class TestMidStackPopRenumber:
  """Pop from middle of 3 stashes, verify dense renumbering."""

  def test_mid_stack_pop_renumbers(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    (src / 'main.py').write_text('v0\n')
    store.stash(context='stash-0')
    (src / 'main.py').write_text('v1\n')
    store.stash(context='stash-1')
    (src / 'main.py').write_text('v2\n')
    store.stash(context='stash-2')

    store.stash_pop(index=1)

    stash_dir = store.config.store_path / 'stash'
    assert (stash_dir / '0000.json').exists()
    assert (stash_dir / '0001.json').exists()
    assert not (stash_dir / '0002.json').exists()

    remaining = store.stash_list()
    assert len(remaining) == 2
    assert remaining[0].context == 'stash-0'
    assert remaining[1].context == 'stash-2'


class TestEmptyPop:
  """stash_pop on empty stack raises StoreError."""

  def test_empty_pop_raises(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    with pytest.raises(StoreError, match='stash stack is empty'):
      store.stash_pop()


class TestPopIndexOutOfRange:
  """stash_pop with invalid index raises StoreError."""

  def test_invalid_index_raises(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash()
    with pytest.raises(StoreError, match='stash index 99 not found'):
      store.stash_pop(index=99)


class TestStashReflog:
  """Stash and stash_pop append reflog rows."""

  def test_stash_reflog_entry(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash(context='wip save')
    entries = _read_reflog(store)
    stash_entries = [e for e in entries if e['operation'] == 'stash']
    assert len(stash_entries) == 1
    assert stash_entries[0]['experiment_id'] == '_'
    assert stash_entries[0]['new_epoch'] == 0
    assert stash_entries[0]['old_epoch'] is None
    assert stash_entries[0]['context'] == 'wip save'

  def test_stash_pop_reflog_entry(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash()
    store.stash_pop()
    entries = _read_reflog(store)
    pop_entries = [e for e in entries if e['operation'] == 'stash_pop']
    assert len(pop_entries) == 1
    assert pop_entries[0]['experiment_id'] == '_'
    assert pop_entries[0]['old_epoch'] == 0
    assert pop_entries[0]['new_epoch'] is None


class TestStashDoesNotAdvanceBranchTip:
  """stash() does not change branch latest_epoch or HEAD."""

  def test_stash_preserves_branch_tip(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    refs_before = store.load_refs()
    tip_before = refs_before['branches']['exp-1']['latest_epoch']

    store.stash()

    refs_after = store.load_refs()
    tip_after = refs_after['branches']['exp-1']['latest_epoch']
    assert tip_before == tip_after


class TestStashSurvivesSnapshotCheckout:
  """Stash manifests are independent from branch snapshots."""

  def test_stash_survives_after_snapshot(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.stash(context='saved')
    (src / 'main.py').write_text('v2\n')
    store.snapshot('exp-1', 1)

    manifests = store.stash_list()
    assert len(manifests) == 1
    assert manifests[0].context == 'saved'


class TestStashSurvivesCheckout:
  """Stash manifests survive unrelated checkout operations."""

  def test_stash_survives_after_checkout(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    (src / 'main.py').write_text('v2\n')
    store.snapshot('exp-1', 1)
    store.stash(context='saved-before-checkout')

    store.checkout('exp-1', 0)

    manifests = store.stash_list()
    assert len(manifests) == 1
    assert manifests[0].context == 'saved-before-checkout'


class TestSnapshotSequentialAfterStash:
  """Normal snapshot() epoch sequencing works after stash."""

  def test_snapshot_works_after_stash(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.stash()
    (src / 'main.py').write_text('epoch 1\n')
    manifest = store.snapshot('exp-1', 1)
    assert manifest.epoch == 1


class TestCorruptStashManifest:
  """stash_list raises StoreError on corrupt manifest."""

  def test_corrupt_json_raises_store_error(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash()

    stash_dir = store.config.store_path / 'stash'
    (stash_dir / '0001.json').write_text('NOT VALID JSON {{{')

    with pytest.raises(StoreError, match='corrupt stash manifest'):
      store.stash_list()


class TestNonNumericStashFiles:
  """Non-numeric files in stash/ are ignored during index scan."""

  def test_non_numeric_files_ignored(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash()

    stash_dir = store.config.store_path / 'stash'
    (stash_dir / 'notes.json').write_text('{"garbage": true}')
    (stash_dir / '.DS_Store').write_text('')
    (stash_dir / 'temp.txt').write_text('nothing')

    manifests = store.stash_list()
    assert len(manifests) == 1


class TestPruneOrphansWithStashBlobs:
  """Blobs referenced only by stash manifests must NOT be deleted by prune_orphans."""

  def test_stash_blobs_not_pruned(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    (src / 'main.py').write_text('stash-only content\n')
    store.stash()

    (src / 'main.py').write_text('print("hello")\n')

    removed = store.prune_orphans()
    assert removed == []

    manifests = store.stash_list()
    assert len(manifests) == 1
    for entry in manifests[0].entries.values():
      obj_data = store.read_object(entry.digest)
      assert obj_data is not None


class TestConcurrentStashUnderLock:
  """Second thread gets StoreError or lock failure during concurrent stash."""

  def test_concurrent_stash_no_overwrite(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    errors: list[Exception] = []
    successes = []
    lock = threading.Lock()

    def stash_worker() -> None:
      try:
        store.stash()
        with lock:
          successes.append(True)
      except (StoreError, ConcurrentMutationError, OSError) as exc:
        with lock:
          errors.append(exc)

    threads = [threading.Thread(target=stash_worker) for _ in range(10)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    stash_dir = store.config.store_path / 'stash'
    stash_files = sorted(stash_dir.glob('*.json'))
    actual_count = len(successes)
    assert len(stash_files) == actual_count


class TestStashAfterParameterReRegistration:
  """Different registered param names produce valid stash manifest."""

  def test_re_register_then_stash(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash()

    new_src = tmp_path / 'new_src'
    new_src.mkdir()
    (new_src / 'config.yaml').write_text('key: value\n')
    new_param = PathParameter(source=str(new_src), pattern='*')
    store.register_parameters({'config': new_param})

    manifest = store.stash()
    assert len(manifest.entries) > 0
    has_config = any(k.startswith('config/') for k in manifest.entries)
    assert has_config


class TestStashPopAfterRegistrationChanged:
  """stash_pop raises StoreError when param registration diverged from stash."""

  def test_pop_with_changed_registration_raises(self, tmp_path: Path) -> None:
    store, src, param = _make_store(tmp_path)
    (src / 'main.py').write_text('original content\n')
    store.stash(context='original-params')

    new_src = tmp_path / 'src2'
    new_src.mkdir()
    (new_src / 'app.py').write_text('new file\n')
    new_param = PathParameter(source=str(new_src), pattern='*')
    store.register_parameters({'source': param, 'extra': new_param})

    with pytest.raises(StoreError, match='registered parameters'):
      store.stash_pop()


class TestStashEpochSentinel:
  """Stash manifest uses epoch=-1 as sentinel."""

  def test_epoch_is_minus_one(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    manifest = store.stash()
    assert manifest.epoch == -1

  def test_stash_list_all_epoch_minus_one(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash()
    store.stash()
    for m in store.stash_list():
      assert m.epoch == -1


class TestStashWorkingTreeUnchanged:
  """Working parameter files remain unchanged after stash (capture-only)."""

  def test_files_unchanged_after_stash(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    (src / 'main.py').write_text('before stash\n')
    store.stash()
    assert (src / 'main.py').read_text() == 'before stash\n'


class TestStashPopMissingParams:
  """stash_pop raises StoreError when registered parameters missing from manifest."""

  def test_stash_pop_raises_missing_params(self, tmp_path: Path) -> None:
    store, _src, param = _make_store(tmp_path)
    store.stash(context='before-new-param')

    new_src = tmp_path / 'new_src'
    new_src.mkdir()
    (new_src / 'extra.txt').write_text('extra\n')
    extra_param = PathParameter(source=str(new_src), pattern='*')
    store.register_parameters({'source': param, 'extra': extra_param})

    with pytest.raises(StoreError, match='registered parameters') as exc_info:
      store.stash_pop()
    assert 'extra' in str(exc_info.value)

  def test_stash_pop_succeeds_when_all_present(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    (src / 'main.py').write_text('stashed content\n')
    store.stash()
    (src / 'main.py').write_text('dirty\n')
    store.stash_pop()
    assert (src / 'main.py').read_text() == 'stashed content\n'

  def test_stash_pop_raises_all_missing(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.stash(context='original')

    new_a = tmp_path / 'a'
    new_a.mkdir()
    (new_a / 'a.txt').write_text('a\n')
    new_b = tmp_path / 'b'
    new_b.mkdir()
    (new_b / 'b.txt').write_text('b\n')
    param_a = PathParameter(source=str(new_a), pattern='*')
    param_b = PathParameter(source=str(new_b), pattern='*')
    store.register_parameters({'alpha': param_a, 'beta': param_b})

    with pytest.raises(StoreError, match='registered parameters') as exc_info:
      store.stash_pop()
    msg = str(exc_info.value)
    assert 'alpha' in msg
    assert 'beta' in msg


class TestStashPopReflogContext:
  """stash_pop records CLI-supplied context in reflog entries."""

  def test_stash_pop_reflog_includes_context(self, tmp_path: Path) -> None:
    """After stash_pop(context='test'), the reflog entry carries the context."""
    store, _src, _param = _make_store(tmp_path)
    store.stash()
    store.stash_pop(context='test')
    entries = _read_reflog(store)
    pop_entries = [e for e in entries if e['operation'] == 'stash_pop']
    assert len(pop_entries) == 1
    assert pop_entries[0]['context'] == 'test'

  def test_stash_pop_reflog_context_none(self, tmp_path: Path) -> None:
    """Omitting context yields null/absent field consistent with other operations."""
    store, _src, _param = _make_store(tmp_path)
    store.stash()
    store.stash_pop()
    entries = _read_reflog(store)
    pop_entries = [e for e in entries if e['operation'] == 'stash_pop']
    assert len(pop_entries) == 1
    assert pop_entries[0].get('context') is None


class TestDoctorIncludesStashHealth:
  """doctor() includes stash blobs in reachability check."""

  def test_doctor_healthy_with_stash(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    (src / 'main.py').write_text('stash content\n')
    store.stash()
    store.save_state_dict({'trees': []})
    report = store.doctor_report()
    assert report['healthy'] is True
    assert report['missing_blobs'] == []
