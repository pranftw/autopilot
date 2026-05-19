"""Logger base class, JSONLogger, and canonical experiment event helpers.

Like Lightning's fabric.loggers.Logger. The module-level functions
``create_event``, ``append_event``, and ``load_events`` are the canonical
experiment JSONL event API. ``JSONLogger`` delegates to these functions
internally. All experiment event I/O should import from this module.
"""

from autopilot.core.artifacts.experiment import EventsArtifact
from autopilot.core.models import Event
from autopilot.tracking.io import utc_now_iso
from pathlib import Path


def create_event(
  event_type: str,
  message: str | None = None,
  metadata: dict | None = None,
) -> Event:
  """Build a timestamped experiment event object.

  Args:
    event_type: Short category label.
    message: Optional human-readable detail.
    metadata: Optional structured fields.

  Returns:
    Event ready for ``append_event``.
  """
  return Event(
    timestamp=utc_now_iso(),
    event_type=event_type,
    message=message,
    metadata=dict(metadata) if metadata is not None else {},
  )


def append_event(path: Path, event: Event) -> None:
  """Persist ``event`` to the experiment event log under ``path``.

  Args:
    path: Experiment directory containing events artifact storage.
    event: Event instance to append.
  """
  EventsArtifact().append(event, path)


def load_events(path: Path) -> list[Event]:
  """Load all events previously recorded for an experiment directory.

  Args:
    path: Experiment directory root.

  Returns:
    Parsed Event list (possibly empty).
  """
  return EventsArtifact().read(path)


class Logger:
  """Base experiment logger. Subclass for different backends.

  Follows Lightning's Logger API: name, version, log_metrics,
  log_hyperparams, log, finalize.
  """

  @property
  def name(self) -> str | None:
    """Logger name for run identification (subclasses may override)."""
    return None

  @property
  def version(self) -> str | int | None:
    """Optional version/tag string for this logger run."""
    return None

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    """Record scalar metrics at an optional global step.

    Raises:
      NotImplementedError: Subclasses must implement logging.
    """
    raise NotImplementedError

  def log_hyperparams(self, params: dict) -> None:
    """Persist hyperparameters or run configuration.

    Raises:
      NotImplementedError: Subclasses must implement logging.
    """
    raise NotImplementedError

  def log(self, event_type: str, message: str | None = None, metadata: dict | None = None) -> None:
    """Emit a structured log event.

    Raises:
      NotImplementedError: Subclasses must implement logging.
    """
    raise NotImplementedError

  def finalize(self, status: str) -> None:
    """End-of-run cleanup. Called with 'success', 'failed', or 'interrupted'.

    Args:
      status: Terminal status label for the run.
    """


class JSONLogger(Logger):
  """Append-only JSONL logger. Default implementation."""

  def __init__(self, path: Path) -> None:
    """Create a logger writing events under ``path``.

    Args:
      path: Directory where ``events.jsonl`` (or equivalent) is stored.
    """
    self._dir = Path(path)

  @property
  def name(self) -> str:
    """Fixed name ``json`` for this backend."""
    return 'json'

  def log(self, event_type: str, message: str | None = None, metadata: dict | None = None) -> None:
    """Append one event record to the JSONL log."""
    self._dir.mkdir(parents=True, exist_ok=True)
    event = create_event(event_type, message, metadata)
    append_event(self._dir, event)

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    """Log metrics via a structured ``metrics`` event."""
    self.log('metrics', metadata={'metrics': metrics, 'step': step})

  def log_hyperparams(self, params: dict) -> None:
    """Log hyperparameters as metadata on a ``hyperparams`` event."""
    self.log('hyperparams', metadata=params)

  def load_events(self) -> list[Event]:
    """Read all persisted events from disk.

    Returns:
      Chronological ``Event`` list from the backing JSONL.
    """
    return load_events(self._dir)

  def finalize(self, status: str) -> None:
    """Write a final ``finalize`` event with the given status string."""
    self.log('finalize', message=status)
