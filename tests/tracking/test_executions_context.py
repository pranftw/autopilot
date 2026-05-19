"""Tests for the context field on ExecutionRecord and related functions."""

from autopilot.tracking.executions import (
  ExecutionRecord,
  create_execution_record,
  filter_executions,
)


class TestExecutionRecordContextField:
  """ExecutionRecord.context field behavior."""

  def test_execution_record_context_field_default_none(self):
    """ExecutionRecord without explicit context has context=None."""
    rec = ExecutionRecord(timestamp='t', command='c')
    assert rec.context is None

  def test_execution_record_context_field_set(self):
    """ExecutionRecord with explicit context holds the value."""
    rec = ExecutionRecord(timestamp='t', command='c', context='reason')
    assert rec.context == 'reason'

  def test_execution_record_to_dict_includes_context(self):
    """to_dict() includes the context key regardless of value."""
    rec_none = ExecutionRecord(timestamp='t', command='c')
    d_none = rec_none.to_dict()
    assert 'context' in d_none
    assert d_none['context'] is None

    rec_set = ExecutionRecord(timestamp='t', command='c', context='testing')
    d_set = rec_set.to_dict()
    assert d_set['context'] == 'testing'

  def test_execution_record_from_dict_with_context(self):
    """from_dict restores a non-None context value."""
    data = {
      'timestamp': 't',
      'command': 'c',
      'args': [],
      'duration_ms': 0.0,
      'exit_code': 0,
      'stdout': None,
      'stderr': None,
      'experiment': None,
      'project': None,
      'extra': {},
      'context': 'restored-reason',
    }
    rec = ExecutionRecord.from_dict(data)
    assert rec.context == 'restored-reason'

  def test_execution_record_roundtrip_with_context(self):
    """Full to_dict/from_dict roundtrip preserves context."""
    original = ExecutionRecord(
      timestamp='2025-01-01T00:00:00+00:00',
      command='optimize train',
      args=['--epochs=5'],
      duration_ms=123.4,
      exit_code=0,
      stdout='ok',
      stderr=None,
      experiment='exp-1',
      project='proj-1',
      extra={'key': 'val'},
      context='reason for this run',
    )
    restored = ExecutionRecord.from_dict(original.to_dict())
    assert restored.context == original.context
    assert restored.command == original.command
    assert restored.experiment == original.experiment


class TestCreateExecutionRecord:
  """create_execution_record factory with context parameter."""

  def test_create_execution_record_with_context(self):
    """Factory passes context onto the record."""
    rec = create_execution_record(
      'test-cmd',
      ['arg1'],
      100.0,
      0,
      context='reason',
    )
    assert rec.context == 'reason'
    assert rec.command == 'test-cmd'

  def test_create_execution_record_without_context(self):
    """Factory omits context; result has context=None."""
    rec = create_execution_record('test-cmd', [], 0.0, 0)
    assert rec.context is None


class TestFilterExecutionsByContext:
  """filter_executions context filtering behavior."""

  def _make_records(self):
    """Build a set of records with varying context values."""
    return [
      ExecutionRecord(timestamp='t1', command='a', context='keep'),
      ExecutionRecord(timestamp='t2', command='b', context='drop'),
      ExecutionRecord(timestamp='t3', command='c', context='keep'),
      ExecutionRecord(timestamp='t4', command='d', context=None),
    ]

  def test_filter_executions_by_context_predicate(self):
    """Predicate-based filtering on context works."""
    records = self._make_records()
    result = filter_executions(
      records,
      predicate=lambda r: r.context == 'keep',
    )
    assert len(result) == 2
    assert all(r.context == 'keep' for r in result)

  def test_filter_executions_by_context_exact_match(self):
    """context= kwarg filters by exact string match."""
    records = self._make_records()
    result = filter_executions(records, context='keep')
    assert len(result) == 2
    assert all(r.context == 'keep' for r in result)

  def test_filter_executions_by_context_exact_no_substring(self):
    """context= uses exact match, not substring."""
    records = self._make_records()
    result = filter_executions(records, context='kee')
    assert len(result) == 0

  def test_filter_executions_context_combined_with_other_filters(self):
    """context= combined with command= applies both."""
    records = self._make_records()
    result = filter_executions(records, context='keep', command='a')
    assert len(result) == 1
    assert result[0].command == 'a'
    assert result[0].context == 'keep'

  def test_filter_executions_context_none_returns_all(self):
    """context=None does not filter by context."""
    records = self._make_records()
    result = filter_executions(records, context=None)
    assert len(result) == len(records)

  def test_filter_executions_context_no_match(self):
    """context= with a value matching no records returns empty."""
    records = self._make_records()
    result = filter_executions(records, context='nonexistent')
    assert len(result) == 0

  def test_filter_executions_context_matches_none_valued_records(self):
    """context=None (default) preserves records with context=None."""
    records = self._make_records()
    result = filter_executions(records)
    none_records = [r for r in result if r.context is None]
    assert len(none_records) == 1
