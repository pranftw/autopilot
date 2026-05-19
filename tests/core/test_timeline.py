"""Tests for autopilot.core.timeline: TimelineEntry and build_timeline."""

from autopilot.core.context import ContextEntry
from autopilot.core.timeline import TimelineEntry, build_timeline
from autopilot.tracking.executions import ExecutionRecord
from autopilot.tracking.io import parse_timestamp
import pytest


class TestTimelineEntryRoundTrip:
  """TimelineEntry.to_dict() -> from_dict() equals original."""

  def test_timeline_entry_round_trip(self) -> None:
    """Round-trip with representative row including nonempty metadata."""
    entry = TimelineEntry(
      timestamp='2025-06-01T12:00:00+00:00',
      stream='context',
      source='trainer',
      reason='experiment completed successfully',
      epoch=3,
      metadata={'final_metrics': {'accuracy': 0.95}},
    )
    serialized = entry.to_dict()
    restored = TimelineEntry.from_dict(serialized)

    assert restored.timestamp == entry.timestamp
    assert restored.stream == entry.stream
    assert restored.source == entry.source
    assert restored.reason == entry.reason
    assert restored.epoch == entry.epoch
    assert restored.metadata == entry.metadata


class TestBuildTimelineMergesSources:
  """Seed three synthetic lists (one row each); assert merged length 3."""

  def test_build_timeline_merges_sources(self) -> None:
    """All three streams contribute one entry each."""
    ctx_entry = ContextEntry.create(reason='gate accepted', source='trainer', epoch=0)
    exec_record = ExecutionRecord(
      timestamp='2025-06-01T12:01:00+00:00',
      command='optimize train',
      args=['--max-epochs', '5'],
      duration_ms=1500.0,
      exit_code=0,
      experiment='exp-001',
    )
    reflog_entry = {
      'timestamp': '2025-06-01T12:02:00+00:00',
      'operation': 'snapshot',
      'experiment_id': 'exp-001',
      'new_epoch': 0,
    }

    result = build_timeline(
      experiment_id='exp-001',
      context_log=[ctx_entry],
      execution_records=[exec_record],
      reflog_entries=[reflog_entry],
    )

    assert len(result) == 3
    streams = {e.stream for e in result}
    assert streams == {'context', 'execution', 'reflog'}


class TestBuildTimelineSorted:
  """Timestamps out of order input; assert output monotonic."""

  def test_build_timeline_sorted(self) -> None:
    """Output is sorted by parsed timestamp ascending."""
    late = {
      'timestamp': '2025-06-01T15:00:00+00:00',
      'reason': 'late context',
      'source': 'user',
      'epoch': None,
      'metadata': {},
    }
    early = {
      'timestamp': '2025-06-01T10:00:00+00:00',
      'reason': 'early context',
      'source': 'user',
      'epoch': None,
      'metadata': {},
    }
    mid = ExecutionRecord(
      timestamp='2025-06-01T12:00:00+00:00',
      command='execute',
      args=['-c', 'print(1)'],
      duration_ms=100.0,
      exit_code=0,
    )

    result = build_timeline(
      experiment_id='exp-001',
      context_log=[late, early],
      execution_records=[mid],
      reflog_entries=[],
    )

    timestamps = [parse_timestamp(e.timestamp) for e in result]
    assert timestamps == sorted(timestamps)
    assert result[0].reason == 'early context'
    assert result[-1].reason == 'late context'


class TestBuildTimelineStreamField:
  """Context -> 'context'; execution -> 'execution'; reflog -> 'reflog'."""

  def test_build_timeline_stream_field(self) -> None:
    """Each source maps to the correct stream literal."""
    ctx_entry = {
      'timestamp': '2025-06-01T12:00:00+00:00',
      'reason': 'ctx reason',
      'source': 'cost',
      'epoch': None,
      'metadata': {},
    }
    exec_record = ExecutionRecord(
      timestamp='2025-06-01T12:01:00+00:00',
      command='track',
      args=['--', 'ruff', 'check'],
      duration_ms=200.0,
      exit_code=0,
    )
    reflog = {
      'timestamp': '2025-06-01T12:02:00+00:00',
      'operation': 'checkout',
      'experiment_id': 'exp-001',
    }

    result = build_timeline(
      experiment_id='exp-001',
      context_log=[ctx_entry],
      execution_records=[exec_record],
      reflog_entries=[reflog],
    )

    assert result[0].stream == 'context'
    assert result[1].stream == 'execution'
    assert result[2].stream == 'reflog'


class TestBuildTimelineEmpty:
  """All inputs empty -> []."""

  def test_build_timeline_empty(self) -> None:
    """Empty inputs produce empty timeline."""
    result = build_timeline(
      experiment_id='exp-001',
      context_log=[],
      execution_records=[],
      reflog_entries=[],
    )
    assert result == []


class TestBuildTimelineInvalidTimestampRaises:
  """Upstream row with unparseable timestamp raises ValueError."""

  def test_build_timeline_invalid_timestamp_raises(self) -> None:
    """ValueError identifies the stream with the bad timestamp."""
    bad_ctx = {
      'timestamp': 'not-a-timestamp',
      'reason': 'bad',
      'source': None,
      'epoch': None,
      'metadata': {},
    }

    with pytest.raises(ValueError, match='context'):
      build_timeline(
        experiment_id='exp-001',
        context_log=[bad_ctx],
        execution_records=[],
        reflog_entries=[],
      )

  def test_invalid_execution_timestamp_raises(self) -> None:
    """ValueError from execution stream mentions 'execution'."""
    bad_exec = ExecutionRecord(
      timestamp='garbage',
      command='test',
      duration_ms=0,
      exit_code=0,
    )

    with pytest.raises(ValueError, match='execution'):
      build_timeline(
        experiment_id='exp-001',
        context_log=[],
        execution_records=[bad_exec],
        reflog_entries=[],
      )

  def test_invalid_reflog_timestamp_raises(self) -> None:
    """ValueError from reflog stream mentions 'reflog'."""
    bad_reflog = {
      'timestamp': '???',
      'operation': 'snapshot',
      'experiment_id': 'exp-001',
    }

    with pytest.raises(ValueError, match='reflog'):
      build_timeline(
        experiment_id='exp-001',
        context_log=[],
        execution_records=[],
        reflog_entries=[bad_reflog],
      )


class TestBuildTimelineTieBreakDeterministic:
  """Rows sharing identical timestamp are ordered context < execution < reflog."""

  def test_build_timeline_tie_break_deterministic(self) -> None:
    """Same timestamp: context < execution < reflog, then lexical reason."""
    ts = '2025-06-01T12:00:00+00:00'

    ctx_entry = {
      'timestamp': ts,
      'reason': 'context event',
      'source': 'trainer',
      'epoch': 0,
      'metadata': {},
    }
    exec_record = ExecutionRecord(
      timestamp=ts,
      command='execute',
      args=[],
      duration_ms=0,
      exit_code=0,
    )
    reflog = {
      'timestamp': ts,
      'operation': 'snapshot',
      'experiment_id': 'exp-001',
    }

    result = build_timeline(
      experiment_id='exp-001',
      context_log=[ctx_entry],
      execution_records=[exec_record],
      reflog_entries=[reflog],
    )

    assert len(result) == 3
    assert result[0].stream == 'context'
    assert result[1].stream == 'execution'
    assert result[2].stream == 'reflog'

  def test_tie_break_lexical_reason_within_stream(self) -> None:
    """Within same stream and timestamp, lexical reason order applies."""
    ts = '2025-06-01T12:00:00+00:00'

    ctx_b = {
      'timestamp': ts,
      'reason': 'zebra event',
      'source': 'trainer',
      'epoch': None,
      'metadata': {},
    }
    ctx_a = {
      'timestamp': ts,
      'reason': 'alpha event',
      'source': 'trainer',
      'epoch': None,
      'metadata': {},
    }

    result = build_timeline(
      experiment_id='exp-001',
      context_log=[ctx_b, ctx_a],
      execution_records=[],
      reflog_entries=[],
    )

    assert result[0].reason == 'alpha event'
    assert result[1].reason == 'zebra event'


class TestTimelineEntryOptionalFields:
  """TimelineEntry with epoch=None and metadata={} round-trips unchanged."""

  def test_timeline_entry_optional_fields(self) -> None:
    """Minimal entry round-trips with None epoch and empty metadata."""
    entry = TimelineEntry(
      timestamp='2025-06-01T12:00:00+00:00',
      stream='execution',
      source=None,
      reason='simple command',
      epoch=None,
      metadata={},
    )
    serialized = entry.to_dict()
    restored = TimelineEntry.from_dict(serialized)

    assert restored.epoch is None
    assert restored.metadata == {}
    assert restored.source is None
    assert restored.stream == 'execution'
    assert restored.reason == 'simple command'
