"""Append-only checkpoint storage and orchestration for AI workflow runs."""

from autopilot.ai.models import CheckpointEvent, CheckpointHeader
from autopilot.core.errors import AIError
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Protocol
import json


class Checkpointable(Protocol):
  """Protocol for anything that can be checkpointed."""

  def state_dict(self) -> dict[str, Any]: ...

  def load_state_dict(self, state: dict[str, Any]) -> None: ...


class CheckpointIO:
  """Storage backend for checkpoints. Default: append-only JSONL."""

  def save_event(self, path: Path, event: BaseModel) -> None:
    """Append a single event to the checkpoint file (incremental)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = event.model_dump_json() + '\n'
    try:
      with path.open('a', encoding='utf-8') as fh:
        fh.write(line)
    except OSError as exc:
      raise AIError(f'failed to append checkpoint to {path}: {exc}') from exc

  def load(self, path: Path) -> list[dict]:
    """Load all events from checkpoint file."""
    if not path.is_file():
      return []
    events: list[dict] = []
    try:
      with path.open('r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, start=1):
          stripped = line.strip()
          if not stripped:
            continue
          try:
            data = json.loads(stripped)
          except json.JSONDecodeError as exc:
            raise AIError(f'invalid JSON on line {line_no} of {path}: {exc}') from exc
          if not isinstance(data, dict):
            raise AIError(f'checkpoint line {line_no} of {path} must be an object')
          events.append(data)
    except OSError as exc:
      raise AIError(f'failed to read checkpoint at {path}: {exc}') from exc
    return events

  def remove(self, path: Path) -> None:
    """Delete a checkpoint file. No error if missing."""
    try:
      path.unlink(missing_ok=True)
    except OSError as exc:
      raise AIError(f'failed to remove checkpoint at {path}: {exc}') from exc


class CheckpointManager:
  """Orchestrates checkpointing for a workflow run."""

  def __init__(self, path: Path, io: CheckpointIO | None = None) -> None:
    """Create or resume from checkpoint. Uses default CheckpointIO if not provided."""
    self._path = path
    self._io = io if io is not None else CheckpointIO()
    self._header: dict[str, Any] | None = None
    self._completed_ids: set[str] = set()
    self._args: dict[str, Any] = {}
    self._states: dict[str, dict[str, Any]] = {}
    self._summary_counts: dict[str, int] = {}
    raw = self._io.load(self._path)
    for d in raw:
      self._apply_event(d)

  def _apply_event(self, d: dict[str, Any]) -> None:
    t = d.get('type')
    if t == 'header':
      self._header = dict(d)
      self._args = dict(d.get('args') or {})
    elif t == 'result':
      item_id = d.get('item_id', '')
      if isinstance(item_id, str):
        self._completed_ids.add(item_id)
    elif t == 'state':
      payload = d.get('payload') or {}
      if not isinstance(payload, dict):
        payload = {}
      key = payload.get('key')
      if key is not None:
        st = payload.get('state')
        self._states[str(key)] = dict(st) if isinstance(st, dict) else {}
    elif t == 'args_update':
      upd = d.get('payload') or {}
      if isinstance(upd, dict):
        self._args.update(upd)

    if t in ('header', 'state', 'args_update'):
      return
    if isinstance(t, str):
      self._summary_counts[t] = self._summary_counts.get(t, 0) + 1

  def save_header(
    self,
    config_hash: str,
    subsystem: str,
    args: dict[str, Any] | None = None,
    **kwargs: Any,
  ) -> None:
    """Write checkpoint header."""
    merged_args: dict[str, Any] = dict(args) if args is not None else {}
    merged_args.update(kwargs)
    ts = datetime.now(timezone.utc).isoformat()
    header = CheckpointHeader(
      subsystem=subsystem,
      config_hash=config_hash,
      created_at=ts,
      args=merged_args,
    )
    self._io.save_event(self._path, header)
    self._apply_event(header.model_dump())

  def save_event(
    self,
    event_type: str,
    item_id: str,
    payload: dict | None = None,
  ) -> None:
    """Incrementally save a single event."""
    ts = datetime.now(timezone.utc).isoformat()
    pl = dict(payload) if payload is not None else {}
    event = CheckpointEvent(
      type=event_type,
      item_id=item_id,
      timestamp=ts,
      payload=pl,
    )
    self._io.save_event(self._path, event)
    self._apply_event(event.model_dump())

  def save_state(self, key: str, state: dict[str, Any]) -> None:
    """Save arbitrary state."""
    self.save_event(
      'state',
      item_id='',
      payload={'key': key, 'state': state},
    )

  def update_args(self, args: dict[str, Any]) -> None:
    """Update the run args. Saves 'args_update' event."""
    ts = datetime.now(timezone.utc).isoformat()
    event = CheckpointEvent(
      type='args_update',
      item_id='',
      timestamp=ts,
      payload=dict(args),
    )
    self._io.save_event(self._path, event)
    self._apply_event(event.model_dump())

  def is_completed(self, item_id: str) -> bool:
    """Check if item was already processed."""
    return item_id in self._completed_ids

  def completed_ids(self) -> set[str]:
    """All item IDs with type='result'."""
    return set(self._completed_ids)

  def load_state(self, key: str) -> dict[str, Any] | None:
    """Load saved state by key. Returns None if not found."""
    if key not in self._states:
      return None
    return dict(self._states[key])

  @property
  def args(self) -> dict[str, Any]:
    """Current run args (original from header, merged with any updates)."""
    return dict(self._args)

  def summary(self) -> dict[str, int]:
    """Counts by event type (excluding 'header', 'state', 'args_update')."""
    return dict(self._summary_counts)

  def load_events(self) -> list[dict]:
    """Load all raw events from the checkpoint file."""
    return self._io.load(self._path)

  @property
  def header(self) -> dict | None:
    """The checkpoint header, if present."""
    return dict(self._header) if self._header is not None else None
