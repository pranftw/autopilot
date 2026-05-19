"""Tests for timeline error handling (QUALITY-003, QUALITY-004, QUALITY-005).

QUALITY-003: ``_load_filtered_reflog`` propagates ``iter_reflog`` errors
  instead of swallowing them; store-open failures still return ``[]``.
QUALITY-004: ``_entry_sort_key`` uses strict ``STREAM_ORDER[entry.stream]``
  so unknown streams fail loudly with ``KeyError``.
QUALITY-005: ``_execution_to_entry`` uses strict ``record['command']`` for
  dict records and direct ``record.args`` (no ``hasattr`` fallback) for
  typed objects.
"""

from autopilot.cli.commands.experiment.timeline import _load_filtered_reflog
from autopilot.core.errors import StoreError
from autopilot.core.timeline import (
  STREAM_ORDER,
  TimelineEntry,
  _entry_sort_key,
  _execution_to_entry,
)
from autopilot.tracking.executions import ExecutionRecord
from autopilot.tracking.io import utc_now_iso
from unittest.mock import MagicMock, patch
import pytest


class TestLoadFilteredReflogPropagation:
  """_load_filtered_reflog propagates iter_reflog failures (QUALITY-003)."""

  def test_load_filtered_reflog_propagates_store_error_from_iter_reflog(
    self, tmp_path: object
  ) -> None:
    """StoreError from iter_reflog bubbles up, not silently returns []."""
    mock_store = MagicMock()
    mock_store.iter_reflog.side_effect = StoreError('corrupt reflog')

    mock_ctx = MagicMock()

    with (
      patch(
        'autopilot.cli.commands.experiment.timeline.open_forest_store',
        return_value=mock_store,
      ),
      pytest.raises(StoreError, match='corrupt reflog'),
    ):
      _load_filtered_reflog(mock_ctx, 'exp-test')

  def test_load_filtered_reflog_open_store_failure_returns_empty(self, tmp_path: object) -> None:
    """Store-open failure returns [] (intentional silent fallback)."""
    mock_ctx = MagicMock()

    with patch(
      'autopilot.cli.commands.experiment.timeline.open_forest_store',
      side_effect=StoreError('no store configured'),
    ):
      result = _load_filtered_reflog(mock_ctx, 'exp-test')

    assert result == []

  def test_load_filtered_reflog_open_oserror_returns_empty(self, tmp_path: object) -> None:
    """OSError on store open returns [] (file-not-found, permissions)."""
    mock_ctx = MagicMock()

    with patch(
      'autopilot.cli.commands.experiment.timeline.open_forest_store',
      side_effect=OSError('permission denied'),
    ):
      result = _load_filtered_reflog(mock_ctx, 'exp-test')

    assert result == []


class TestStreamOrderStrict:
  """_entry_sort_key uses strict STREAM_ORDER[entry.stream] (QUALITY-004)."""

  def test_stream_order_strict_indexing(self) -> None:
    """All valid TimelineStream literal values exist in STREAM_ORDER."""
    expected_streams = ('context', 'execution', 'reflog')
    for stream in expected_streams:
      assert stream in STREAM_ORDER, f'{stream!r} missing from STREAM_ORDER'

  def test_stream_order_invalid_stream_raises_key_error(self) -> None:
    """Unknown stream value raises KeyError during sort key computation."""
    entry = TimelineEntry(
      timestamp=utc_now_iso(),
      stream='bogus',  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
      source=None,
      reason='test',
    )
    with pytest.raises(KeyError, match='bogus'):
      _entry_sort_key(entry)

  def test_all_valid_streams_sort_without_error(self) -> None:
    """Entries with all valid stream values sort without raising."""
    ts = utc_now_iso()
    entries = [
      TimelineEntry(timestamp=ts, stream='reflog', source=None, reason='c'),
      TimelineEntry(timestamp=ts, stream='context', source=None, reason='a'),
      TimelineEntry(timestamp=ts, stream='execution', source=None, reason='b'),
    ]
    sorted_entries = sorted(entries, key=_entry_sort_key)
    assert [e.stream for e in sorted_entries] == ['context', 'execution', 'reflog']


class TestExecutionToEntryStrict:
  """_execution_to_entry uses strict field access (QUALITY-005)."""

  def test_execution_to_entry_dict_requires_command(self) -> None:
    """Dict without 'command' raises KeyError (required field)."""
    record = {'timestamp': utc_now_iso(), 'args': ['--flag']}
    with pytest.raises(KeyError, match='command'):
      _execution_to_entry(record)

  def test_execution_to_entry_dict_with_command_works(self) -> None:
    """Dict with 'command' converts successfully."""
    record = {
      'timestamp': utc_now_iso(),
      'command': 'optimize train',
      'args': ['--max-epochs', '5'],
    }
    entry = _execution_to_entry(record)
    assert entry.stream == 'execution'
    assert entry.metadata['command'] == 'optimize train'
    assert entry.metadata['args'] == ['--max-epochs', '5']
    assert 'optimize train --max-epochs 5' in entry.reason

  def test_execution_to_entry_dict_optional_fields_default(self) -> None:
    """Dict missing optional fields (args, duration_ms, exit_code) uses defaults."""
    record = {'timestamp': utc_now_iso(), 'command': 'query'}
    entry = _execution_to_entry(record)
    assert entry.metadata['args'] == []
    assert entry.metadata['duration_ms'] == 0
    assert entry.metadata['exit_code'] == 0

  def test_execution_to_entry_typed_record_uses_args(self) -> None:
    """Typed ExecutionRecord carries args directly (no hasattr fallback)."""
    record = ExecutionRecord(
      timestamp=utc_now_iso(),
      command='execute',
      args=['-c', 'print(1)'],
      duration_ms=42.0,
      exit_code=0,
    )
    entry = _execution_to_entry(record)
    assert entry.stream == 'execution'
    assert entry.metadata['command'] == 'execute'
    assert entry.metadata['args'] == ['-c', 'print(1)']
    assert entry.metadata['duration_ms'] == 42.0
    assert 'execute -c print(1)' in entry.reason
