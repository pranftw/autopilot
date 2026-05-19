"""Tests for Traceable in core/traceable.py."""

from autopilot.core.context import ContextEntry, ContextLog
from autopilot.core.traceable import Traceable
from typing import Any


def test_traceable_init_creates_context_log():
  """context_log exists and is empty."""
  obj = Traceable()
  assert isinstance(obj.context_log, ContextLog)
  assert len(obj.context_log) == 0


def test_traceable_add_context_appends_entry():
  """add_context adds to context_log."""
  obj = Traceable()
  obj.add_context('first reason', source='user')
  assert len(obj.context_log) == 1
  assert obj.context_log.entries[0].reason == 'first reason'


def test_traceable_add_context_returns_entry():
  """Returned entry fields match call."""
  obj = Traceable()
  entry = obj.add_context(
    'test action',
    source='trainer',
    command='optimize',
    epoch=2,
    metadata={'key': 'value'},
  )
  assert entry is not None
  assert isinstance(entry, ContextEntry)
  assert entry.reason == 'test action'
  assert entry.source == 'trainer'
  assert entry.command == 'optimize'
  assert entry.epoch == 2
  assert entry.metadata == {'key': 'value'}


def test_traceable_create_context_log_override():
  """Subclass returns custom ContextLog subtype from create_context_log."""

  class LimitedLog(ContextLog):
    """Log that only accepts up to 5 entries."""

    def accept(self, entry):
      return len(self._entries) < 5

  class LimitedTraceable(Traceable):
    def create_context_log(self) -> ContextLog:
      return LimitedLog()

  obj = LimitedTraceable()
  assert isinstance(obj.context_log, LimitedLog)
  for i in range(7):
    obj.add_context(f'entry {i}')
  assert len(obj.context_log) == 5


def test_traceable_add_context_override():
  """Subclass overrides add_context to enrich; verify enrichment on stored entry."""

  class EnrichedTraceable(Traceable):
    def add_context(
      self,
      reason: str,
      *,
      source: str | None = None,
      command: str | None = None,
      epoch: int | None = None,
      metadata: dict[str, Any] | None = None,
    ) -> ContextEntry | None:
      enriched_metadata = dict(metadata) if metadata is not None else {}
      enriched_metadata['enriched'] = True
      return super().add_context(
        reason,
        source=source or 'auto',
        command=command,
        epoch=epoch,
        metadata=enriched_metadata,
      )

  obj = EnrichedTraceable()
  entry = obj.add_context('test', metadata={'original': 'data'})
  assert entry is not None
  assert entry.source == 'auto'
  assert entry.metadata['enriched'] is True
  assert entry.metadata['original'] == 'data'
