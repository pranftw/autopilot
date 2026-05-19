"""Tests for FileStore reflog lifecycle APIs.

Covers:
  1. test_reflog_expire_removes_old -- entries older than cutoff removed
  2. test_reflog_expire_keeps_recent -- no recent entry is dropped
  3. test_reflog_expire_empty -- when no line qualifies, returns 0
  4. test_reflog_expire_count -- returned int equals removed count
  5. test_recover_from_reflog -- restores refs/latest_epoch
  6. test_recover_invalid_index_raises -- StoreError for out-of-range
  7. test_reflog_corrupt_line -- corrupt line skipped, stderr warning
  8. test_expire_reflog_unreadable_file -- bad permissions -> StoreError
  9. test_recover_malformed_entry -- missing fields -> StoreError
  10. test_recover_empty_reflog_error -- empty reflog says 'empty' in message
  11. test_recover_out_of_range_shows_valid_range -- non-empty reflog shows range
  12. test_recover_negative_index_error -- negative index says 'out of range'
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.tracking.io import append_jsonl, utc_now_iso
from datetime import UTC, datetime, timedelta
from pathlib import Path
import json
import pytest


def _make_store(tmp_path: Path) -> FileStore:
  """Create a FileStore with a parameter and a branch for testing."""
  prompts_dir = tmp_path / 'prompts'
  prompts_dir.mkdir()
  (prompts_dir / 'main.txt').write_text('hello')

  config = AutoPilotConfig(workspace=tmp_path)
  param = PathParameter(source=str(prompts_dir), pattern='*.txt')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('exp-001', 0)
  return store


def _inject_reflog_entry(
  store: FileStore,
  *,
  operation: str = 'snapshot',
  experiment_id: str = 'exp-001',
  old_epoch: int | None = None,
  new_epoch: int = 0,
  timestamp: str | None = None,
) -> None:
  """Append a synthetic reflog entry with a given timestamp."""
  record = {
    'timestamp': timestamp or utc_now_iso(),
    'operation': operation,
    'experiment_id': experiment_id,
    'old_epoch': old_epoch,
    'new_epoch': new_epoch,
    'context': None,
  }
  append_jsonl(store.reflog_path, record)


class TestReflogExpireRemovesOld:
  """test_reflog_expire_removes_old -- entries older than cutoff removed."""

  def test_old_entries_are_removed(self, tmp_path: Path) -> None:
    """Entries with timestamps older than cutoff are removed from reflog."""
    store = _make_store(tmp_path)
    old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    _inject_reflog_entry(store, timestamp=old_ts, experiment_id='exp-old', new_epoch=1)

    removed = store.expire_reflog(timedelta(days=30))
    assert removed >= 1

    remaining = list(store.iter_reflog())
    for entry in remaining:
      ts = datetime.fromisoformat(entry['timestamp'])
      cutoff = datetime.now(UTC) - timedelta(days=30)
      assert ts >= cutoff


class TestReflogExpireKeepsRecent:
  """test_reflog_expire_keeps_recent -- no recent entry dropped."""

  def test_recent_entries_preserved(self, tmp_path: Path) -> None:
    """Entries younger than cutoff remain in reflog after expire."""
    store = _make_store(tmp_path)
    recent_ts = (datetime.now(UTC) - timedelta(days=5)).isoformat()
    _inject_reflog_entry(store, timestamp=recent_ts, experiment_id='exp-recent', new_epoch=2)

    store.expire_reflog(timedelta(days=30))
    final_entries = list(store.iter_reflog())

    recent_ids = [
      e['experiment_id'] for e in final_entries if e.get('experiment_id') == 'exp-recent'
    ]
    assert len(recent_ids) == 1


class TestReflogExpireEmpty:
  """test_reflog_expire_empty -- no qualifying entry -> 0."""

  def test_no_old_entries_returns_zero(self, tmp_path: Path) -> None:
    """When no entry is older than cutoff, returns 0 and file unchanged."""
    store = _make_store(tmp_path)
    entries_before = list(store.iter_reflog())
    removed = store.expire_reflog(timedelta(days=365))
    assert removed == 0
    entries_after = list(store.iter_reflog())
    assert len(entries_after) == len(entries_before)

  def test_missing_reflog_returns_zero(self, tmp_path: Path) -> None:
    """When reflog file does not exist, returns 0."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    assert store.expire_reflog(timedelta(days=1)) == 0


class TestReflogExpireCount:
  """test_reflog_expire_count -- returned int equals number removed."""

  def test_count_matches_removed(self, tmp_path: Path) -> None:
    """Returned count exactly matches number of old entries removed."""
    store = _make_store(tmp_path)
    old_ts = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    _inject_reflog_entry(store, timestamp=old_ts, experiment_id='old-1', new_epoch=1)
    _inject_reflog_entry(store, timestamp=old_ts, experiment_id='old-2', new_epoch=2)

    recent_ts = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    _inject_reflog_entry(store, timestamp=recent_ts, experiment_id='new-1', new_epoch=3)

    removed = store.expire_reflog(timedelta(days=30))
    assert removed == 2


class TestRecoverFromReflog:
  """test_recover_from_reflog -- restores refs and latest_epoch."""

  def test_recover_restores_branch_tip(self, tmp_path: Path) -> None:
    """After recover, refs show the entry's experiment and epoch."""
    store = _make_store(tmp_path)

    prompts_dir = tmp_path / 'prompts'
    (prompts_dir / 'main.txt').write_text('updated content')
    store.snapshot('exp-001', 1, force=True)

    entries = list(store.iter_reflog())
    assert len(entries) >= 2

    store.recover_from_reflog(0)

    refs = store.load_refs()
    entry = entries[0]
    assert refs['HEAD'] == entry['experiment_id']
    assert refs['branches'][entry['experiment_id']]['latest_epoch'] == entry['new_epoch']


class TestRecoverInvalidIndexRaises:
  """test_recover_invalid_index_raises -- StoreError for out-of-range."""

  def test_negative_index(self, tmp_path: Path) -> None:
    """Negative index raises StoreError."""
    store = _make_store(tmp_path)
    with pytest.raises(StoreError, match='out of range'):
      store.recover_from_reflog(-1)

  def test_too_large_index(self, tmp_path: Path) -> None:
    """Index beyond entry count raises StoreError."""
    store = _make_store(tmp_path)
    entries = list(store.iter_reflog())
    with pytest.raises(StoreError, match='out of range'):
      store.recover_from_reflog(len(entries) + 100)


class TestReflogCorruptLine:
  """test_reflog_corrupt_line -- corrupt line skipped with stderr message."""

  def test_corrupt_line_skipped(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Corrupt JSONL line is skipped; stderr contains warning."""
    store = _make_store(tmp_path)
    reflog_path = store.reflog_path
    with reflog_path.open('a', encoding='utf-8') as fh:
      fh.write('this is not json\n')

    entries = list(store.iter_reflog())
    captured = capsys.readouterr()
    assert 'corrupt' in captured.err.lower() or 'skipping' in captured.err.lower()
    for entry in entries:
      assert isinstance(entry, dict)


class TestExpireReflogUnreadableFile:
  """test_expire_reflog_unreadable_file -- bad permissions -> StoreError."""

  def test_unreadable_reflog_raises(self, tmp_path: Path) -> None:
    """Reflog file that cannot be read raises StoreError."""
    store = _make_store(tmp_path)
    reflog_path = store.reflog_path
    assert reflog_path.exists()
    reflog_path.chmod(0o000)
    try:
      with pytest.raises(StoreError, match='cannot read reflog'):
        store.expire_reflog(timedelta(days=1))
    finally:
      reflog_path.chmod(0o644)


class TestRecoverMalformedEntry:
  """test_recover_malformed_entry -- missing fields -> StoreError."""

  def test_missing_experiment_id(self, tmp_path: Path) -> None:
    """Entry without experiment_id raises StoreError with guidance."""
    store = _make_store(tmp_path)
    reflog_path = store.reflog_path
    with reflog_path.open('a', encoding='utf-8') as fh:
      fh.write(json.dumps({'timestamp': utc_now_iso(), 'new_epoch': 5}) + '\n')

    entries = list(store.iter_reflog())
    last_idx = len(entries) - 1
    with pytest.raises(StoreError, match='missing "experiment_id"'):
      store.recover_from_reflog(last_idx)

  def test_missing_new_epoch(self, tmp_path: Path) -> None:
    """Entry without new_epoch raises StoreError with guidance."""
    store = _make_store(tmp_path)
    reflog_path = store.reflog_path
    with reflog_path.open('a', encoding='utf-8') as fh:
      fh.write(json.dumps({'timestamp': utc_now_iso(), 'experiment_id': 'exp-x'}) + '\n')

    entries = list(store.iter_reflog())
    last_idx = len(entries) - 1
    with pytest.raises(StoreError, match='missing "new_epoch"'):
      store.recover_from_reflog(last_idx)


class TestRecoverEmptyReflogError:
  """test_recover_empty_reflog_error -- empty reflog says 'empty' in message."""

  def test_recover_empty_reflog_error_says_empty(self, tmp_path: Path) -> None:
    """Empty reflog produces 'empty' in the StoreError message."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('hello')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.reflog_path.parent.mkdir(parents=True, exist_ok=True)
    store.reflog_path.write_text('')

    with pytest.raises(StoreError, match=r'(?i)empty'):
      store.recover_from_reflog(0)


class TestRecoverOutOfRangeShowsValidRange:
  """test_recover_out_of_range_shows_valid_range -- non-empty reflog shows range."""

  def test_shows_valid_range(self, tmp_path: Path) -> None:
    """Store with 3 reflog entries reports valid: 0..2 for out-of-range index."""
    store = _make_store(tmp_path)
    _inject_reflog_entry(store, new_epoch=1)
    _inject_reflog_entry(store, new_epoch=2)

    entries = list(store.iter_reflog())
    assert len(entries) >= 3

    with pytest.raises(StoreError, match=r'valid: 0\.\.\d+'):
      store.recover_from_reflog(len(entries) + 5)

  def test_negative_index_shows_out_of_range(self, tmp_path: Path) -> None:
    """Negative index on non-empty reflog includes 'out of range'."""
    store = _make_store(tmp_path)
    with pytest.raises(StoreError, match='out of range'):
      store.recover_from_reflog(-1)
