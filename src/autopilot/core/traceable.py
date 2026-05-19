"""Traceable base class for objects with decision journals.

Traceable attaches a :class:`~autopilot.core.context.ContextLog` to any object,
providing a single choke-point for appending context entries. Subclasses can
override :meth:`create_context_log` for custom log types and :meth:`add_context`
for entry pre-processing (e.g. injecting default source or metadata).
"""

from autopilot.core.context import ContextEntry, ContextLog
from typing import Any


class Traceable:
  """Base class attaching a :class:`ContextLog` for traceability.

  Override points:
    create_context_log() -- return a custom ContextLog subclass.
    add_context() -- pre-process entries before appending.
  """

  def __init__(self) -> None:
    """Initialize the traceable with an empty context log."""
    self._context_log = self.create_context_log()

  def create_context_log(self) -> ContextLog:
    """Factory for the log instance; override to use a custom ContextLog subtype.

    Returns:
      A new ContextLog (or subclass) instance.
    """
    return ContextLog()

  @property
  def context_log(self) -> ContextLog:
    """The context log attached to this object.

    Returns:
      The ContextLog instance for this traceable.
    """
    return self._context_log

  def add_context(
    self,
    reason: str,
    *,
    source: str | None = None,
    command: str | None = None,
    epoch: int | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> ContextEntry | None:
    """Append a context entry via the attached log.

    Override to enrich entries (e.g. inject default source or metadata)
    before delegating to super().add_context().

    Args:
      reason: Human- or machine-readable explanation.
      source: Origin of the entry.
      command: CLI command that triggered this entry.
      epoch: Training epoch associated with this entry.
      metadata: Arbitrary key-value context data.

    Returns:
      The appended ContextEntry, or None if accept() rejected it.
    """
    return self._context_log.append(
      reason,
      source=source,
      command=command,
      epoch=epoch,
      metadata=metadata,
    )
