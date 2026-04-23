"""Append-only experiment event log (JSON Lines)."""

from autopilot.core.artifacts.experiment import EventsArtifact
from autopilot.core.models import Event
from datetime import datetime, timezone
from pathlib import Path


def create_event(
  event_type: str,
  message: str | None = None,
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
  EventsArtifact().append(event, experiment_dir)


def load_events(experiment_dir: Path) -> list[Event]:
  return EventsArtifact().read(experiment_dir)
