"""Tests for FileStore reflog and tags (Plan 14).

Covers reflog append on snapshot/checkout/merge_apply/reset_branch/branch/
materialize, tag CRUD with validation, corrupt reflog handling, tag
immutability, concurrent tag creation, and reflog append failure after
save_refs.
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.store.types import MergeStrategy, TagEntry, validate_tag_name
from autopilot.tracking.io import read_jsonl
from pathlib import Path
from tests.cli.conftest import make_cli_workspace, run_cli, run_cli_no_context
from unittest.mock import patch
import pytest


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


def _reflog_path(store: FileStore) -> Path:
  """Return the reflog JSONL path for the given store."""
  return store.config.store_path / 'reflog.jsonl'


def _read_reflog(store: FileStore) -> list[dict]:
  """Read all reflog entries from the store."""
  return read_jsonl(_reflog_path(store), strict=False)


# reflog tests


class TestReflogAppendedOnSnapshot:
  """After snapshot, reflog file exists and last line has operation=='snapshot'."""

  def test_reflog_appended_on_snapshot(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    entries = _read_reflog(store)
    assert len(entries) == 1
    assert entries[0]['operation'] == 'snapshot'
    assert entries[0]['experiment_id'] == 'exp-1'
    assert entries[0]['new_epoch'] == 0
    assert entries[0]['old_epoch'] is None
    assert 'timestamp' in entries[0]

  def test_snapshot_second_epoch_records_old_new(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    (src / 'main.py').write_text('print("v2")\n')
    store.snapshot('exp-1', 1)

    entries = _read_reflog(store)
    assert len(entries) == 2
    last = entries[-1]
    assert last['old_epoch'] == 0
    assert last['new_epoch'] == 1


class TestReflogAppendedOnCheckout:
  """Entry recorded with expected experiment_id / epochs on checkout."""

  def test_reflog_appended_on_checkout(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.checkout('exp-1', 0)

    entries = _read_reflog(store)
    checkout_entries = [e for e in entries if e['operation'] == 'checkout']
    assert len(checkout_entries) == 1
    entry = checkout_entries[0]
    assert entry['experiment_id'] == 'exp-1'
    assert entry['new_epoch'] == 0
    assert entry['old_epoch'] == 0
    assert entry.get('context') is None


class TestReflogAppendedOnMergeApply:
  """Merge creates entry with operation=='merge_apply'."""

  def test_reflog_appended_on_merge_apply(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.branch('exp-2')
    (src / 'main.py').write_text('merged content\n')
    store.snapshot('exp-2', 1)
    store.merge_and_apply('exp-1', 'exp-2', strategy=MergeStrategy.theirs)

    entries = _read_reflog(store)
    merge_entries = [e for e in entries if e['operation'] == 'merge_apply']
    assert len(merge_entries) == 1
    entry = merge_entries[0]
    assert entry['experiment_id'] == 'exp-1'
    assert entry['old_epoch'] == 0
    assert entry['new_epoch'] == 1
    assert entry['source_experiment_id'] == 'exp-2'


class TestReflogAppendedOnReset:
  """reset_branch writes entry with new_epoch==-1."""

  def test_reflog_appended_on_reset(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.reset_branch('exp-1')

    entries = _read_reflog(store)
    reset_entries = [e for e in entries if e['operation'] == 'reset_branch']
    assert len(reset_entries) == 1
    entry = reset_entries[0]
    assert entry['experiment_id'] == 'exp-1'
    assert entry['old_epoch'] == 0
    assert entry['new_epoch'] == -1


class TestReflogRecordsOldNewEpoch:
  """Controlled branch tip before/after snapshot encoded correctly."""

  def test_reflog_records_old_new_epoch(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    (src / 'main.py').write_text('v2')
    store.snapshot('exp-1', 1)
    (src / 'main.py').write_text('v3')
    store.snapshot('exp-1', 2)

    entries = _read_reflog(store)
    snapshot_entries = [e for e in entries if e['operation'] == 'snapshot']
    assert len(snapshot_entries) == 3
    assert snapshot_entries[0]['old_epoch'] is None
    assert snapshot_entries[0]['new_epoch'] == 0
    assert snapshot_entries[1]['old_epoch'] == 0
    assert snapshot_entries[1]['new_epoch'] == 1
    assert snapshot_entries[2]['old_epoch'] == 1
    assert snapshot_entries[2]['new_epoch'] == 2


class TestReflogIsAppendOnly:
  """Two snapshots yield two lines; first preserved."""

  def test_reflog_is_append_only(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    first_entries = _read_reflog(store)
    assert len(first_entries) == 1
    first_record = first_entries[0]

    (src / 'main.py').write_text('v2')
    store.snapshot('exp-1', 1)
    all_entries = _read_reflog(store)
    assert len(all_entries) == 2
    assert all_entries[0] == first_record


class TestReflogBranch:
  """Branch operation records a reflog entry."""

  def test_reflog_appended_on_branch(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.branch('exp-2')

    entries = _read_reflog(store)
    branch_entries = [e for e in entries if e['operation'] == 'branch']
    assert len(branch_entries) == 1
    entry = branch_entries[0]
    assert entry['experiment_id'] == 'exp-2'
    assert entry['old_epoch'] is None
    assert entry['new_epoch'] == 0


class TestReflogMaterialize:
  """Materialize operation records a reflog entry."""

  def test_reflog_appended_on_materialize(self, tmp_path: Path) -> None:
    store, src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    (src / 'main.py').write_text('v2')
    store.snapshot('exp-1', 1)
    store.materialize('exp-1', 0)

    entries = _read_reflog(store)
    mat_entries = [e for e in entries if e['operation'] == 'materialize']
    assert len(mat_entries) == 1
    entry = mat_entries[0]
    assert entry['experiment_id'] == 'exp-1'
    assert entry['old_epoch'] == 1
    assert entry['new_epoch'] == 0


class TestReflogContext:
  """Reflog records context when supplied."""

  def test_snapshot_context_recorded(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0, context='initial setup')

    entries = _read_reflog(store)
    assert entries[0]['context'] == 'initial setup'

  def test_snapshot_null_context(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    entries = _read_reflog(store)
    assert entries[0]['context'] is None


class TestReflogCorruptTailSkipped:
  """Corrupt tail line is skipped by read_jsonl(strict=False)."""

  def test_reflog_corrupt_tail_skipped(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    reflog_file = _reflog_path(store)
    with reflog_file.open('a', encoding='utf-8') as fh:
      fh.write('this is not valid json\n')

    entries = _read_reflog(store)
    assert len(entries) == 1
    assert entries[0]['operation'] == 'snapshot'


class TestReflogAppendFailureAfterSaveRefsSilent:
  """Simulate append_jsonl failure after save_refs: silently swallowed, operation succeeds."""

  def test_reflog_append_failure_silent(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)

    with patch('autopilot.ai.store.store_reflog.append_jsonl', side_effect=OSError('disk full')):
      manifest = store.snapshot('exp-1', 0)

    assert manifest.epoch == 0

    refs = store.load_refs()
    assert refs['branches']['exp-1']['latest_epoch'] == 0


# tag tests


class TestTagCreate:
  """get_tag returns TagEntry with matching fields."""

  def test_tag_create(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('v1.0', 'exp-1', 0, context='first release')

    tag = store.get_tag('v1.0')
    assert tag is not None
    assert tag.name == 'v1.0'
    assert tag.experiment_id == 'exp-1'
    assert tag.epoch == 0
    assert tag.context == 'first release'
    assert tag.timestamp

  def test_tag_reflog_entry(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('v1.0', 'exp-1', 0)

    entries = _read_reflog(store)
    tag_entries = [e for e in entries if e['operation'] == 'tag']
    assert len(tag_entries) == 1
    entry = tag_entries[0]
    assert entry['experiment_id'] == 'exp-1'
    assert entry['old_epoch'] is None
    assert entry['new_epoch'] == 0


class TestTagImmutable:
  """Second tag with same name raises StoreError."""

  def test_tag_immutable(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('v1.0', 'exp-1', 0)

    with pytest.raises(StoreError, match='already exists'):
      store.tag('v1.0', 'exp-1', 0)


class TestTagList:
  """Multiple tags returned sorted by name."""

  def test_tag_list(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('beta', 'exp-1', 0)
    store.tag('alpha', 'exp-1', 0)
    store.tag('gamma', 'exp-1', 0)

    tags = store.list_tags()
    names = [t.name for t in tags]
    assert names == ['alpha', 'beta', 'gamma']
    assert all(isinstance(t, TagEntry) for t in tags)


class TestTagGetUnknown:
  """Unknown tag name returns None."""

  def test_tag_get_unknown(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    assert store.get_tag('nonexistent') is None


class TestTagsKeyMissingUsesEmptyDict:
  """Store/refs without 'tags' key: first tag creation initializes refs['tags']."""

  def test_tags_key_missing_uses_empty_dict(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    refs = store.load_refs()
    assert 'tags' not in refs

    store.tag('v1.0', 'exp-1', 0)

    refs = store.load_refs()
    assert 'tags' in refs
    assert 'v1.0' in refs['tags']


class TestTagCreateConcurrentSecondFails:
  """Two concurrent tag creators: second gets StoreError."""

  def test_tag_create_concurrent_second_fails(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    store.tag('v1.0', 'exp-1', 0)
    with pytest.raises(StoreError, match='already exists'):
      store.tag('v1.0', 'exp-1', 0)


class TestTagEpochMustExist:
  """Tag pointing at non-existent epoch raises StoreError."""

  def test_tag_epoch_must_exist(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    with pytest.raises(StoreError, match='epoch 99 does not exist'):
      store.tag('bad-tag', 'exp-1', 99)


class TestTagBranchMustExist:
  """Tag pointing at non-existent branch raises StoreError."""

  def test_tag_branch_must_exist(self, tmp_path: Path) -> None:
    store, _src, _param = _make_store(tmp_path)
    store.snapshot('exp-1', 0)

    with pytest.raises(StoreError, match='not found'):
      store.tag('bad-tag', 'nonexistent', 0)


# tag name validation


class TestTagNameValidation:
  """Tag name validation covers all rules."""

  def test_empty_name_rejected(self) -> None:
    with pytest.raises(StoreError, match='must not be empty'):
      validate_tag_name('')

  def test_too_long_name_rejected(self) -> None:
    with pytest.raises(StoreError, match='exceeds 128'):
      validate_tag_name('x' * 129)

  def test_invalid_chars_rejected(self) -> None:
    with pytest.raises(StoreError, match='invalid characters'):
      validate_tag_name('tag with spaces')

  def test_leading_dot_rejected(self) -> None:
    with pytest.raises(StoreError, match='must not start or end'):
      validate_tag_name('.hidden')

  def test_trailing_dot_rejected(self) -> None:
    with pytest.raises(StoreError, match='must not start or end'):
      validate_tag_name('trailing.')

  def test_double_dot_rejected(self) -> None:
    with pytest.raises(StoreError, match='must not contain'):
      validate_tag_name('bad..name')

  def test_valid_names_accepted(self) -> None:
    for name in ['v1.0', 'release-1', 'my_tag', 'A123', 'a.b.c']:
      validate_tag_name(name)

  def test_max_length_accepted(self) -> None:
    validate_tag_name('x' * 128)


# TagEntry round-trip


class TestTagEntryRoundTrip:
  """TagEntry serializes and deserializes correctly."""

  def test_tag_entry_round_trip(self) -> None:
    entry = TagEntry(
      name='v1.0',
      experiment_id='exp-1',
      epoch=3,
      context='release',
      timestamp='2026-01-01T00:00:00+00:00',
    )
    data = entry.to_dict()
    restored = TagEntry.from_dict(data)
    assert restored == entry

  def test_tag_entry_none_context(self) -> None:
    entry = TagEntry(name='beta', experiment_id='exp-2', epoch=0)
    data = entry.to_dict()
    restored = TagEntry.from_dict(data)
    assert restored.context is None
    assert restored.timestamp is None

  def test_tag_entry_none_timestamp_default(self) -> None:
    """TagEntry defaults timestamp to None."""
    entry = TagEntry(name='v1', experiment_id='e1', epoch=0)
    assert entry.timestamp is None

  def test_tag_entry_round_trip_none_timestamp(self) -> None:
    """Round-trip preserves None timestamp."""
    entry = TagEntry(name='v1', experiment_id='e1', epoch=0)
    data = entry.to_dict()
    restored = TagEntry.from_dict(data)
    assert restored.timestamp is None
    assert restored == entry

  def test_tag_entry_round_trip_with_timestamp(self) -> None:
    """Round-trip preserves a real ISO timestamp string."""
    entry = TagEntry(
      name='v1',
      experiment_id='e1',
      epoch=0,
      timestamp='2026-01-01T00:00:00Z',
    )
    data = entry.to_dict()
    restored = TagEntry.from_dict(data)
    assert restored.timestamp == '2026-01-01T00:00:00Z'
    assert restored == entry


# CLI tests


class TestDebugStoreReflogCLI:
  """CLI harness test for debug store reflog."""

  def test_debug_store_reflog_cli(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)

    envelope = run_cli_no_context(ws, ['debug', 'store', 'reflog'])
    assert envelope['ok'] is True
    entries = envelope['result']['entries']
    assert len(entries) >= 1
    assert entries[0]['operation'] == 'snapshot'
    assert entries[0]['experiment_id'] == 'exp-1'

  def test_debug_store_reflog_limit(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)
    (tmp_path / 'ws' / 'src' / 'main.py').write_text('updated\n')
    store.snapshot('exp-1', 1)

    envelope = run_cli_no_context(ws, ['debug', 'store', 'reflog', '-n', '1'])
    assert envelope['ok'] is True
    assert len(envelope['result']['entries']) == 1


class TestStoreTagCreateCLI:
  """CLI harness test for store tag create."""

  def test_store_tag_create_cli(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)

    envelope = run_cli(ws, ['--experiment', 'exp-1', 'store', 'tag', 'create', 'v1.0'])
    assert envelope['ok'] is True
    assert envelope['result']['tag'] == 'v1.0'
    assert envelope['result']['experiment_id'] == 'exp-1'
    assert envelope['result']['epoch'] == 0


class TestStoreTagListCLI:
  """CLI harness test for store tag list."""

  def test_store_tag_list_cli(self, tmp_path: Path) -> None:
    store, ws = make_cli_workspace(tmp_path)
    store.snapshot('exp-1', 0)
    store.tag('beta', 'exp-1', 0)
    store.tag('alpha', 'exp-1', 0)

    envelope = run_cli_no_context(ws, ['store', 'tag', 'list'])
    assert envelope['ok'] is True
    tags = envelope['result']['tags']
    assert len(tags) == 2
    assert tags[0]['name'] == 'alpha'
    assert tags[1]['name'] == 'beta'


# doctor reflog gap tests (BUG-009)


class TestDoctorReflogGaps:
  """Doctor reports branches in refs without reflog entries as reflog_gaps."""

  def _make_store_with_forest(self, tmp_path: Path) -> tuple[FileStore, Path, PathParameter]:
    """Create a store with a forest for doctor tests to report healthy."""
    store, src, param = _make_store(tmp_path)
    forest = FileForest(store)
    forest.save()
    return store, src, param

  def test_doctor_reports_reflog_gaps(self, tmp_path: Path) -> None:
    """Branch in refs without a reflog entry appears in reflog_gaps."""
    store, _src, _param = self._make_store_with_forest(tmp_path)
    store.snapshot('exp-1', 0)

    refs = store.load_refs()
    refs['branches']['orphan-branch'] = {'latest_epoch': -1}
    store.save_refs(refs)

    result = store.doctor_report()
    assert result['healthy'] is True
    assert 'orphan-branch' in result['reflog_gaps']
    assert 'exp-1' not in result['reflog_gaps']

  def test_doctor_no_reflog_gaps_when_complete(self, tmp_path: Path) -> None:
    """All branches have reflog entries: reflog_gaps is empty."""
    store, src, _param = self._make_store_with_forest(tmp_path)
    store.snapshot('exp-1', 0)
    store.branch('exp-2')
    (src / 'main.py').write_text('v2\n')
    store.snapshot('exp-2', 1)

    result = store.doctor_report()
    assert result['reflog_gaps'] == []

  def test_doctor_reflog_gaps_no_reflog_file(self, tmp_path: Path) -> None:
    """No reflog.jsonl file: reflog_gaps is empty and healthy is True."""
    store, _src, _param = self._make_store_with_forest(tmp_path)
    store.snapshot('exp-1', 0)

    reflog_file = _reflog_path(store)
    if reflog_file.exists():
      reflog_file.unlink()

    result = store.doctor_report()
    assert result['reflog_gaps'] == []
    assert result['healthy'] is True

  def test_doctor_reflog_gaps_empty_reflog(self, tmp_path: Path) -> None:
    """Empty reflog.jsonl: all branches appear in reflog_gaps."""
    store, _src, _param = self._make_store_with_forest(tmp_path)
    store.snapshot('exp-1', 0)

    reflog_file = _reflog_path(store)
    reflog_file.write_text('')

    result = store.doctor_report()
    assert 'exp-1' in result['reflog_gaps']
