"""Tests for ContextEntry and ContextLog in core/context.py."""

from autopilot.core.context import ContextEntry, ContextLog
from dataclasses import dataclass

# -- ContextEntry tests --


def test_context_entry_create_sets_timestamp():
  """create() sets a non-empty ISO-shaped timestamp."""
  entry = ContextEntry.create('test reason')
  assert entry.timestamp
  assert 'T' in entry.timestamp


def test_context_entry_create_reason_and_source():
  """reason and source match arguments."""
  entry = ContextEntry.create('fix bug', source='user')
  assert entry.reason == 'fix bug'
  assert entry.source == 'user'


def test_context_entry_create_defaults():
  """source, command, epoch are None; metadata is {}."""
  entry = ContextEntry.create('just a reason')
  assert entry.source is None
  assert entry.command is None
  assert entry.epoch is None
  assert entry.metadata == {}


def test_context_entry_create_uses_cls():
  """Subclass of ContextEntry; create() returns subclass instance."""

  @dataclass
  class CustomEntry(ContextEntry):
    custom_field: str = 'extra'

  entry = CustomEntry.create('subclass reason')
  assert isinstance(entry, CustomEntry)
  assert entry.custom_field == 'extra'
  assert entry.reason == 'subclass reason'


def test_context_entry_to_dict_from_dict_roundtrip():
  """All fields populated; from_dict(to_dict()) equals original for comparable fields."""
  entry = ContextEntry(
    timestamp='2024-01-01T00:00:00+00:00',
    reason='rollback triggered',
    source='policy',
    command='optimize',
    epoch=3,
    metadata={'accuracy': 0.72, 'gate': 'fail'},
  )
  data = entry.to_dict()
  restored = ContextEntry.from_dict(data)
  assert restored.timestamp == entry.timestamp
  assert restored.reason == entry.reason
  assert restored.source == entry.source
  assert restored.command == entry.command
  assert restored.epoch == entry.epoch
  assert restored.metadata == entry.metadata


def test_context_entry_to_dict_from_dict_minimal():
  """Only required serialization fields; round-trip stable."""
  entry = ContextEntry(
    timestamp='2024-06-01T12:00:00+00:00',
    reason='minimal',
  )
  data = entry.to_dict()
  restored = ContextEntry.from_dict(data)
  assert restored.timestamp == entry.timestamp
  assert restored.reason == entry.reason
  assert restored.source is None
  assert restored.metadata == {}


def test_context_entry_metadata_nested():
  """Nested dicts in metadata round-trip via to_dict/from_dict."""
  entry = ContextEntry(
    timestamp='2024-01-01T00:00:00+00:00',
    reason='nested test',
    metadata={'outer': {'inner': [1, 2, 3]}, 'flag': True},
  )
  data = entry.to_dict()
  restored = ContextEntry.from_dict(data)
  assert restored.metadata == {'outer': {'inner': [1, 2, 3]}, 'flag': True}


# -- ContextLog tests --


def test_context_log_empty():
  """New log: len 0, entries is [], to_list() is []."""
  log = ContextLog()
  assert len(log) == 0
  assert log.entries == []
  assert log.to_list() == []


def test_context_log_append_returns_entry():
  """append returns ContextEntry with matching reason, source, etc."""
  log = ContextLog()
  entry = log.append('test reason', source='user', epoch=1)
  assert entry is not None
  assert entry.reason == 'test reason'
  assert entry.source == 'user'
  assert entry.epoch == 1


def test_context_log_append_increments_length():
  """Each append increases len by 1 when accepted."""
  log = ContextLog()
  log.append('first')
  assert len(log) == 1
  log.append('second')
  assert len(log) == 2
  log.append('third')
  assert len(log) == 3


def test_context_log_append_rejected_by_accept():
  """Subclass with accept always False; append returns None; len stays 0."""

  class RejectAll(ContextLog):
    def accept(self, entry):
      return False

  log = RejectAll()
  result = log.append('should be rejected')
  assert result is None
  assert len(log) == 0


def test_context_log_record_pre_built_entry():
  """record(ContextEntry(...)) with manual timestamp; entry appears in entries."""
  log = ContextLog()
  entry = ContextEntry(
    timestamp='2024-05-01T10:00:00+00:00',
    reason='pre-built entry',
    source='test',
  )
  log.record(entry)
  assert len(log) == 1
  assert log.entries[0] is entry


def test_context_log_record_rejected_by_accept():
  """accept False; record leaves log empty."""

  class RejectAll(ContextLog):
    def accept(self, entry):
      return False

  log = RejectAll()
  entry = ContextEntry(
    timestamp='2024-05-01T10:00:00+00:00',
    reason='will be rejected',
  )
  log.record(entry)
  assert len(log) == 0


def test_context_log_search_substring_match():
  """Reasons containing query returned."""
  log = ContextLog()
  log.append('policy gate rejected epoch 3')
  log.append('optimizer applied changes')
  log.append('policy gate accepted epoch 4')
  results = log.search('policy gate')
  assert len(results) == 2
  assert all('policy gate' in e.reason for e in results)


def test_context_log_search_no_match():
  """Empty list when no match."""
  log = ContextLog()
  log.append('some reason')
  results = log.search('nonexistent')
  assert results == []


def test_context_log_search_empty_log():
  """Empty list on empty log."""
  log = ContextLog()
  results = log.search('anything')
  assert results == []


def test_context_log_filter_by_source():
  """Only matching source."""
  log = ContextLog()
  log.append('user action', source='user')
  log.append('trainer action', source='trainer')
  log.append('another user action', source='user')
  results = log.filter_by_source('user')
  assert len(results) == 2
  assert all(e.source == 'user' for e in results)


def test_context_log_filter_by_source_no_match():
  """Empty list when no source match."""
  log = ContextLog()
  log.append('action', source='trainer')
  results = log.filter_by_source('policy')
  assert results == []


def test_context_log_after_filters_correctly():
  """Entries at or after a cutoff (inclusive >=)."""
  log = ContextLog()
  e1 = ContextEntry(timestamp='2024-01-01T10:00:00+00:00', reason='early')
  e2 = ContextEntry(timestamp='2024-01-01T12:00:00+00:00', reason='midday')
  e3 = ContextEntry(timestamp='2024-01-01T14:00:00+00:00', reason='afternoon')
  log.record(e1)
  log.record(e2)
  log.record(e3)
  results = log.after('2024-01-01T12:00:00+00:00')
  assert len(results) == 2
  assert results[0].reason == 'midday'
  assert results[1].reason == 'afternoon'


def test_context_log_after_inclusive():
  """Entry at exact threshold iso_timestamp included."""
  log = ContextLog()
  entry = ContextEntry(timestamp='2024-06-15T08:30:00+00:00', reason='exact match')
  log.record(entry)
  results = log.after('2024-06-15T08:30:00+00:00')
  assert len(results) == 1
  assert results[0].reason == 'exact match'


def test_context_log_between_filters_correctly():
  """Window filtering."""
  log = ContextLog()
  e1 = ContextEntry(timestamp='2024-01-01T08:00:00+00:00', reason='before')
  e2 = ContextEntry(timestamp='2024-01-01T10:00:00+00:00', reason='inside')
  e3 = ContextEntry(timestamp='2024-01-01T12:00:00+00:00', reason='also inside')
  e4 = ContextEntry(timestamp='2024-01-01T14:00:00+00:00', reason='after')
  log.record(e1)
  log.record(e2)
  log.record(e3)
  log.record(e4)
  results = log.between('2024-01-01T09:00:00+00:00', '2024-01-01T13:00:00+00:00')
  assert len(results) == 2
  assert results[0].reason == 'inside'
  assert results[1].reason == 'also inside'


def test_context_log_between_inclusive_both():
  """Entries at start and end included."""
  log = ContextLog()
  e1 = ContextEntry(timestamp='2024-01-01T10:00:00+00:00', reason='at start')
  e2 = ContextEntry(timestamp='2024-01-01T12:00:00+00:00', reason='at end')
  log.record(e1)
  log.record(e2)
  results = log.between('2024-01-01T10:00:00+00:00', '2024-01-01T12:00:00+00:00')
  assert len(results) == 2


def test_context_log_to_list_from_list_roundtrip():
  """from_list(to_list()) preserves order and field content."""
  log = ContextLog()
  log.append('first', source='user', epoch=0, metadata={'key': 'val'})
  log.append('second', source='trainer', epoch=1)
  serialized = log.to_list()
  restored = ContextLog.from_list(serialized)
  assert len(restored) == 2
  entries = restored.entries
  assert entries[0].reason == 'first'
  assert entries[0].source == 'user'
  assert entries[0].epoch == 0
  assert entries[0].metadata == {'key': 'val'}
  assert entries[1].reason == 'second'
  assert entries[1].source == 'trainer'


def test_context_log_to_list_empty():
  """Empty log serializes to []."""
  log = ContextLog()
  assert log.to_list() == []


def test_context_log_iter():
  """Iteration order matches append order."""
  log = ContextLog()
  log.append('alpha')
  log.append('beta')
  log.append('gamma')
  reasons = [e.reason for e in log]
  assert reasons == ['alpha', 'beta', 'gamma']


def test_context_log_entries_is_copy():
  """Mutating log.entries list does not change len(log) or internal order."""
  log = ContextLog()
  log.append('original')
  entries_copy = log.entries
  entries_copy.clear()
  assert len(log) == 1
  assert log.entries[0].reason == 'original'


# -- ContextLog.append(ContextEntry) rejection tests (BUG-001) --


def test_append_context_entry_rejected_returns_none():
  """Subclass with accept() returning False; append(ContextEntry) returns None."""

  class RejectAll(ContextLog):
    def accept(self, entry):
      return False

  log = RejectAll()
  entry = ContextEntry.create('should reject', source='test')
  result = log.append(entry)
  assert result is None
  assert len(log) == 0


def test_append_context_entry_accepted_returns_entry():
  """Default log; append(ContextEntry) returns the same entry."""
  log = ContextLog()
  entry = ContextEntry.create('accepted entry', source='test')
  result = log.append(entry)
  assert result is entry
  assert len(log) == 1


def test_append_string_rejected_returns_none():
  """Regression: rejecting subclass; append(string) returns None."""

  class RejectAll(ContextLog):
    def accept(self, entry):
      return False

  log = RejectAll()
  result = log.append('rejected string', source='user')
  assert result is None
  assert len(log) == 0
