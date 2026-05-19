"""Context data model for audit-grade traceability.

ContextEntry is the immutable atom of decision context -- a timestamped reason
with optional provenance and metadata. ContextLog is an append-only, searchable
collection with an accept hook for policy-driven gating.

Classes:
  ContextEntry -- single audit record (DictMixin dataclass).
  ContextLog -- ordered journal with search, time-filtering, and accept gate.
"""

from autopilot.core.serialization import DictMixin
from autopilot.tracking.io import parse_timestamp, utc_now_iso
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Self


@dataclass
class ContextEntry(DictMixin):
  """One append-only context record: reason, optional provenance, and metadata.

  Use :meth:`create` for new entries (sole clock-based factory).
  Use ``from_dict`` only for deserialization of persisted data.

  Conventional source values (not enforced):
    'user', 'agent-optimizer', 'policy', 'trainer', 'early-stopping', 'plateau'

  Attributes:
    timestamp: ISO 8601 time the entry was recorded.
    reason: Human- or machine-readable explanation of what happened.
    source: Origin of the entry (e.g. 'user', 'policy'), or None.
    command: CLI command that triggered this entry, or None.
    epoch: Training epoch associated with this entry, or None.
    metadata: Arbitrary key-value context data.

  Example:
    >>> entry = ContextEntry.create(  # doctest: +SKIP
    ...   reason='experiment completed',
    ...   source='trainer',
    ...   metadata={'final_accuracy': 0.95},
    ... )
  """

  timestamp: str
  reason: str
  source: str | None = None
  command: str | None = None
  epoch: int | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  @classmethod
  def create(
    cls,
    reason: str,
    *,
    source: str | None = None,
    command: str | None = None,
    epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> Self:
    """Build a new entry with timestamp from :func:`~autopilot.tracking.io.utc_now_iso`.

    Args:
      reason: Human- or machine-readable explanation of what happened.
      source: Origin of the entry (e.g. 'user', 'policy', 'trainer').
      command: CLI command that triggered this entry.
      epoch: Training epoch associated with this entry.
      metadata: Arbitrary key-value context data.

    Returns:
      New ContextEntry with an auto-generated UTC timestamp.
    """
    return cls(
      timestamp=utc_now_iso(),
      reason=reason,
      source=source,
      command=command,
      epoch=epoch,
      metadata=dict(metadata) if metadata is not None else {},
    )


class ContextLog:
  """Append-only decision journal with search and time-filtering.

  Two entry-addition methods:
    ``append(reason, ...)``  -- when ``reason`` is a string, builds a new
      ``ContextEntry`` via ``ContextEntry.create`` and records it (DRY-01).
      When ``reason`` is a ``ContextEntry``, calls ``accept()`` and appends
      if accepted; returns the entry on success or ``None`` on rejection
      (keyword args are ignored in that case).
    ``record(entry)``        -- accepts a pre-built ``ContextEntry`` (e.g.
      from the callback path). Also calls ``accept()`` before recording.

  Entries are stored in append order and iteration preserves that order.
  Entries are validated through :meth:`accept` before appending. Override
  ``accept`` in subclasses for custom rejection rules.

  Serialization: :meth:`to_list` / :meth:`from_list` round-trip through
  ``list[dict]``. ``from_list`` bypasses ``accept`` (entries were validated
  on original append).

  Override points:
    accept(entry) -- pre-append validation hook (default: accept all).
    search(query) -- substring match on reason (default: case-sensitive).
    filter_by_source(source) -- exact match on entry.source.
  """

  def __init__(self) -> None:
    """Initialize an empty context log."""
    self._entries: list[ContextEntry] = []

  @property
  def entries(self) -> list[ContextEntry]:
    """Shallow copy of internal entries list.

    Returns:
      New list of ContextEntry objects (mutations do not affect internal state).
    """
    return list(self._entries)

  def __len__(self) -> int:
    """Return the number of entries in the log."""
    return len(self._entries)

  def __iter__(self) -> Iterator[ContextEntry]:
    """Iterate over entries in append order.

    Returns:
      Iterator over ContextEntry objects.
    """
    return iter(self._entries)

  def accept(self, entry: ContextEntry) -> bool:
    """Pre-append validation hook. Override to reject entries.

    Args:
      entry: The entry about to be appended.

    Returns:
      True to accept, False to reject.
    """
    return True

  def append(
    self,
    reason: str | ContextEntry,
    *,
    source: str | None = None,
    command: str | None = None,
    epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> ContextEntry | None:
    """Append a context entry.

    If ``reason`` is a :class:`ContextEntry`, calls :meth:`accept` and
    appends directly if accepted (keyword args are ignored in that case).
    If ``reason`` is a string, creates a new entry via
    ``ContextEntry.create`` (DRY-01).

    Args:
      reason: Human-readable explanation string, or a pre-built ContextEntry.
      source: Origin of the entry (ignored when reason is ContextEntry).
      command: CLI command (ignored when reason is ContextEntry).
      epoch: Training epoch (ignored when reason is ContextEntry).
      metadata: Arbitrary context data (ignored when reason is ContextEntry).

    Returns:
      The appended ContextEntry, or None if accept() rejected it.
    """
    if isinstance(reason, ContextEntry):
      if not self.accept(reason):
        return None
      self._entries.append(reason)
      return reason
    entry = ContextEntry.create(
      reason, source=source, command=command, epoch=epoch, metadata=metadata
    )
    if not self.accept(entry):
      return None
    self._entries.append(entry)
    return entry

  def record(self, entry: ContextEntry) -> None:
    """Accept a pre-built entry (e.g. from callback path). Calls accept().

    Args:
      entry: Pre-built ContextEntry to record.
    """
    if self.accept(entry):
      self._entries.append(entry)

  def search(self, query: str) -> list[ContextEntry]:
    """Substring search on entry reason (case-sensitive).

    Args:
      query: Substring to match against each entry's reason field.

    Returns:
      Entries whose reason contains the query string.
    """
    return [e for e in self._entries if query in e.reason]

  def filter_by_source(self, source: str) -> list[ContextEntry]:
    """Filter entries by exact source match.

    Args:
      source: Source string to match (None entries do not match).

    Returns:
      Entries whose source equals the given string.
    """
    return [e for e in self._entries if e.source == source]

  def after(self, iso_timestamp: str) -> list[ContextEntry]:
    """Entries at or after a timestamp (inclusive lower bound).

    Args:
      iso_timestamp: ISO 8601 threshold string.

    Returns:
      Entries with parsed timestamp >= threshold.
    """
    threshold = parse_timestamp(iso_timestamp)
    return [e for e in self._entries if parse_timestamp(e.timestamp) >= threshold]

  def between(self, start: str, end: str) -> list[ContextEntry]:
    """Entries within a time window (inclusive both ends).

    Args:
      start: ISO 8601 lower bound (inclusive).
      end: ISO 8601 upper bound (inclusive).

    Returns:
      Entries with start <= parsed timestamp <= end.
    """
    start_dt = parse_timestamp(start)
    end_dt = parse_timestamp(end)
    return [e for e in self._entries if start_dt <= parse_timestamp(e.timestamp) <= end_dt]

  def to_list(self) -> list[dict[str, Any]]:
    """Serialize all entries to a list of dicts.

    Returns:
      List of serialized entry dicts suitable for JSON persistence.
    """
    return [e.to_dict() for e in self._entries]

  @classmethod
  def from_list(cls, data: list[dict[str, Any]]) -> 'ContextLog':
    """Deserialize entries without re-validating through accept().

    Entries were validated on original append; re-running accept() on
    deserialization could reject previously-valid entries if accept()
    semantics change in a subclass.

    Args:
      data: List of serialized entry dicts.

    Returns:
      New ContextLog populated with deserialized entries.
    """
    log = cls()
    log._entries = [ContextEntry.from_dict(d) for d in data]
    return log
