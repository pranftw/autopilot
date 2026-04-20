"""Append-only experiment event log (JSON Lines)."""

from autopilot.core.errors import TrackingError
from autopilot.core.models import Event
from autopilot.tracking.io import append_jsonl, read_jsonl
from datetime import datetime, timezone
from pathlib import Path


def _events_path(experiment_dir: Path) -> Path:
  return experiment_dir / 'events.jsonl'


def create_event(
  event_type: str,
  message: str = '',
  metadata: dict | None = None,
) -> Event:
  ts = datetime.now(timezone.utc).isoformat()
  return Event(
    timestamp=ts,
    event_type=event_type,
    message=message,
    metadata=dict(metadata) if metadata is not None else {},
  )


def append_event(experiment_dir: Path, event: Event) -> None:
  path = _events_path(experiment_dir)
  append_jsonl(path, event.to_dict())


def load_events(experiment_dir: Path) -> list[Event]:
  path = _events_path(experiment_dir)
  records = read_jsonl(path, strict=True)
  events: list[Event] = []
  for line_no, data in enumerate(records, start=1):
    try:
      events.append(Event.from_dict(data))
    except (TypeError, ValueError, KeyError) as exc:
      raise TrackingError(
        f'invalid event fields on line {line_no} of {path}: {exc}',
      ) from exc
  return events
