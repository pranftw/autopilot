"""Logger base class and JSONLogger. Like Lightning's fabric.loggers.Logger."""

from autopilot.core.models import Event
from autopilot.tracking.events import append_event, create_event, load_events
from pathlib import Path


class Logger:
  """Base experiment logger. Subclass for different backends.

  Follows Lightning's Logger API: name, version, log_metrics,
  log_hyperparams, log, finalize.
  """

  @property
  def name(self) -> str | None:
    return None

  @property
  def version(self) -> str | int | None:
    return None

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    raise NotImplementedError

  def log_hyperparams(self, params: dict) -> None:
    raise NotImplementedError

  def log(self, event_type: str, message: str = '', metadata: dict | None = None) -> None:
    raise NotImplementedError

  def finalize(self, status: str) -> None:
    """End-of-run cleanup. Called with 'success', 'failed', or 'interrupted'."""
    pass


class JSONLogger(Logger):
  """Append-only JSONL logger. Default implementation."""

  def __init__(self, experiment_dir: Path) -> None:
    self._dir = Path(experiment_dir)

  @property
  def name(self) -> str:
    return 'json'

  def log(self, event_type: str, message: str = '', metadata: dict | None = None) -> None:
    self._dir.mkdir(parents=True, exist_ok=True)
    event = create_event(event_type, message, metadata)
    append_event(self._dir, event)

  def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
    self.log('metrics', metadata={'metrics': metrics, 'step': step})

  def log_hyperparams(self, params: dict) -> None:
    self.log('hyperparams', metadata=params)

  def load_events(self) -> list[Event]:
    return load_events(self._dir)

  def finalize(self, status: str) -> None:
    self.log('finalize', message=status)
