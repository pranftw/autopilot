"""Unified experiment timeline: merge context, execution, and reflog streams.

TimelineEntry is a DictMixin row type representing one chronological event
from any stream (context log, execution record, or store reflog). The
build_timeline function merges pre-filtered streams into a single sorted
list.

Mapping rules:
  - Context log entries -> stream='context' (including cost rows with source='cost').
  - Execution records -> stream='execution'. Metadata includes command, args,
    duration_ms, and exit_code. stdout/stderr are never included (avoids bloat).
  - Reflog entries -> stream='reflog'. The full structured entry is retained
    as metadata minus the timestamp (which is promoted to the top level).

Sorting:
  Primary: parsed timestamp ascending (via parse_timestamp).
  Tie-break: stream order (context < execution < reflog), then lexical reason.
"""

from autopilot.core.serialization import DictMixin
from autopilot.tracking.io import parse_timestamp
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

TimelineStream = Literal['context', 'execution', 'reflog']

STREAM_ORDER: dict[str, int] = {
  'context': 0,
  'execution': 1,
  'reflog': 2,
}


@dataclass
class TimelineEntry(DictMixin):
  """Single unified timeline row across experiment-related streams.

  Attributes:
    timestamp: ISO 8601 timestamp string (sort key).
    stream: Logical stream name (``context``, ``execution``, or ``reflog``); cost-related
      rows use ``stream='context'`` with ``source='cost'`` (see module docstring).
    source: Optional finer-grained source (e.g. ``'trainer'``, ``'user'``, exec argv).
    reason: Human-readable primary line (context reason, command summary, reflog op).
    epoch: Trainer/store epoch when applicable.
    metadata: Opaque structured payload; never None (empty dict default).
  """

  timestamp: str
  stream: TimelineStream
  source: str | None
  reason: str
  epoch: int | None = None
  metadata: dict[str, Any] = field(default_factory=dict)


def _parse_ts_safe(timestamp: str, stream: str) -> datetime:
  """Parse a timestamp, raising ValueError with stream context on failure.

  Args:
    timestamp: ISO 8601 string to parse.
    stream: Stream name for error context.

  Returns:
    Parsed aware datetime.

  Raises:
    ValueError: When timestamp is unparseable; message identifies the stream.
  """
  try:
    return parse_timestamp(timestamp)
  except (ValueError, TypeError) as exc:
    msg = f'unparseable timestamp {timestamp!r} in {stream!r} stream: {exc}'
    raise ValueError(msg) from exc


def _entry_sort_key(entry: TimelineEntry) -> tuple[datetime, int, str]:
  """Build a composite sort key for deterministic ordering.

  Args:
    entry: Timeline entry to produce a key for.

  Returns:
    Tuple of (parsed_timestamp, stream_order, reason) for stable sorting.
  """
  return (
    _parse_ts_safe(entry.timestamp, entry.stream),
    STREAM_ORDER[entry.stream],
    entry.reason,
  )


def _context_to_entry(record: Any) -> TimelineEntry:
  """Convert a context log entry (ContextEntry or dict) to a TimelineEntry.

  Args:
    record: Context entry as a ContextEntry-like object or dict.

  Returns:
    TimelineEntry with stream='context'.
  """
  if isinstance(record, dict):
    return TimelineEntry(
      timestamp=record['timestamp'],
      stream='context',
      source=record.get('source'),
      reason=record['reason'],
      epoch=record.get('epoch'),
      metadata=dict(record.get('metadata') or {}),
    )
  return TimelineEntry(
    timestamp=record.timestamp,
    stream='context',
    source=record.source,
    reason=record.reason,
    epoch=record.epoch,
    metadata=dict(record.metadata) if record.metadata else {},
  )


def _execution_to_entry(record: Any) -> TimelineEntry:
  """Convert an execution record to a TimelineEntry.

  Args:
    record: ExecutionRecord-like object or dict.

  Returns:
    TimelineEntry with stream='execution'.
  """
  if isinstance(record, dict):
    command = record['command']
    args = record.get('args', [])
    reason = f'{command} {" ".join(args)}'.strip() if args else command
    return TimelineEntry(
      timestamp=record['timestamp'],
      stream='execution',
      source=command,
      reason=reason,
      epoch=None,
      metadata={
        'command': command,
        'args': args,
        'duration_ms': record.get('duration_ms', 0),
        'exit_code': record.get('exit_code', 0),
      },
    )
  command = record.command
  args = record.args
  reason = f'{command} {" ".join(args)}'.strip() if args else command
  return TimelineEntry(
    timestamp=record.timestamp,
    stream='execution',
    source=command,
    reason=reason,
    epoch=None,
    metadata={
      'command': command,
      'args': list(args),
      'duration_ms': record.duration_ms,
      'exit_code': record.exit_code,
    },
  )


def _reflog_to_entry(record: dict[str, Any]) -> TimelineEntry:
  """Convert a reflog entry (dict) to a TimelineEntry.

  Args:
    record: Reflog dict with at least a 'timestamp' key.

  Returns:
    TimelineEntry with stream='reflog'.
  """
  operation = record.get('operation', 'unknown')
  experiment_id = record.get('experiment_id') or ''
  epoch = record.get('new_epoch')
  if epoch is None:
    epoch = record.get('epoch')

  parts = [operation]
  if experiment_id:
    parts.append(experiment_id)
  if epoch is not None:
    parts.append(f'epoch={epoch}')
  reason = ' '.join(parts)

  metadata = {k: v for k, v in record.items() if k != 'timestamp'}

  return TimelineEntry(
    timestamp=record['timestamp'],
    stream='reflog',
    source=operation,
    reason=reason,
    epoch=epoch if isinstance(epoch, int) else None,
    metadata=metadata,
  )


def build_timeline(
  experiment_id: str,
  context_log: list[Any],
  execution_records: list[Any],
  reflog_entries: list[Any],
) -> list[TimelineEntry]:
  """Merge pre-filtered streams into a sorted timeline.

  Sorting uses ``parse_timestamp()`` which propagates ``ValueError`` for
  unparseable timestamps (fail closed; message identifies the stream).

  Args:
    experiment_id: Owning experiment id (retained for API symmetry).
    context_log: Already-loaded context entries (domain objects or dicts).
    execution_records: Already filtered to this experiment.
    reflog_entries: Already filtered to branch ``experiment_id``.

  Returns:
    Entries sorted by ``parse_timestamp(entry.timestamp)`` ascending,
    with deterministic tie-breaking by stream order then reason.
  """
  _ = experiment_id
  entries = [_context_to_entry(record) for record in context_log]
  entries.extend(_execution_to_entry(record) for record in execution_records)
  entries.extend(_reflog_to_entry(record) for record in reflog_entries)

  entries.sort(key=_entry_sort_key)
  return entries
