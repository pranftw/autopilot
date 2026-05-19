"""Append-only checkpoint storage and orchestration for AI workflow runs."""

from autopilot.ai.evaluation.schemas import CheckpointEvent, CheckpointHeader
from autopilot.core.errors import AIError, TrackingError
from autopilot.tracking.io import append_jsonl, read_jsonl, utc_now_iso
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Protocol


class Checkpointable(Protocol):
  """Protocol for anything that can be checkpointed."""

  def state_dict(self) -> dict[str, Any]:
    """Return a serializable snapshot of checkpointable state."""
    ...

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Restore state previously produced by :meth:`state_dict`."""
    ...


class CheckpointIO:
  """Storage backend for checkpoints. Default: append-only JSONL."""

  def save_event(self, path: Path, event: BaseModel) -> None:
    """Append a single event to the checkpoint file (incremental).

    Raises:
      AIError: When JSONL append fails with :class:`TrackingError`.
    """
    try:
      append_jsonl(path, event.model_dump(by_alias=True))
    except TrackingError as exc:
      raise AIError(str(exc)) from exc

  def load(self, path: Path) -> list[dict]:
    """Load all events from checkpoint file.

    Returns:
      List of deserialized JSON objects, one per line in the file.

    Raises:
      AIError: When JSONL read fails with :class:`TrackingError`.
    """
    try:
      return read_jsonl(path, strict=True)
    except TrackingError as exc:
      raise AIError(str(exc)) from exc

  def remove(self, path: Path) -> None:
    """Delete a checkpoint file. No error if missing.

    Raises:
      AIError: When file removal fails with :class:`OSError`.
    """
    try:
      path.unlink(missing_ok=True)
    except OSError as exc:
      msg = f'failed to remove checkpoint at {path}: {exc}'
      raise AIError(msg) from exc


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
    event_type = d.get('type')
    if event_type == 'header':
      self._header = dict(d)
      self._args = dict(d.get('args', {}))
    elif event_type == 'result':
      eid = d['id']
      if isinstance(eid, str):
        self._completed_ids.add(eid)
    elif event_type == 'state':
      payload = d.get('payload', {})
      if not isinstance(payload, dict):
        payload = {}
      key = payload.get('key')
      if key is not None:
        st = payload.get('state')
        self._states[str(key)] = dict(st) if isinstance(st, dict) else {}
    elif event_type == 'args_update':
      upd = d.get('payload', {})
      if isinstance(upd, dict):
        self._args.update(upd)

    if event_type in {'header', 'state', 'args_update'}:
      return
    if isinstance(event_type, str):
      self._summary_counts[event_type] = self._summary_counts.get(event_type, 0) + 1

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
    header = CheckpointHeader(
      subsystem=subsystem,
      config_hash=config_hash,
      created_at=utc_now_iso(),
      args=merged_args,
    )
    self._io.save_event(self._path, header)
    self._apply_event(header.model_dump(by_alias=True))

  def save_event(
    self,
    event_type: str,
    item_id: str,
    payload: dict | None = None,
  ) -> None:
    """Incrementally save a single event."""
    pl = dict(payload) if payload is not None else {}
    event = CheckpointEvent(
      type=event_type,
      id=item_id,
      timestamp=utc_now_iso(),
      payload=pl,
    )
    self._io.save_event(self._path, event)
    self._apply_event(event.model_dump(by_alias=True))

  def save_state(self, key: str, state: dict[str, Any]) -> None:
    """Save arbitrary state."""
    self.save_event(
      'state',
      item_id='',
      payload={'key': key, 'state': state},
    )

  def update_args(self, args: dict[str, Any]) -> None:
    """Update the run args. Saves 'args_update' event."""
    event = CheckpointEvent(
      type='args_update',
      id='',
      timestamp=utc_now_iso(),
      payload=dict(args),
    )
    self._io.save_event(self._path, event)
    self._apply_event(event.model_dump(by_alias=True))

  def is_completed(self, item_id: str) -> bool:
    """Check if item was already processed.

    Returns:
      True if a prior ``result`` event recorded ``item_id`` as complete.
    """
    return item_id in self._completed_ids

  def completed_ids(self) -> set[str]:
    """All IDs with type='result'.

    Returns:
      Copy of completed item id strings.
    """
    return set(self._completed_ids)

  def load_state(self, key: str) -> dict[str, Any] | None:
    """Load saved state by key. Returns None if not found.

    Returns:
      State dict for ``key``, or ``None`` when no ``state`` event exists.
    """
    if key not in self._states:
      return None
    return dict(self._states[key])

  @property
  def args(self) -> dict[str, Any]:
    """Current run args (original from header, merged with any updates)."""
    return dict(self._args)

  def summary(self) -> dict[str, int]:
    """Counts by event type (excluding 'header', 'state', 'args_update').

    Returns:
      Mapping from event type string to occurrence count.
    """
    return dict(self._summary_counts)

  def load_events(self) -> list[dict]:
    """Load all raw events from the checkpoint file.

    Returns:
      Event dicts as returned by the underlying :class:`CheckpointIO`.
    """
    return self._io.load(self._path)

  @property
  def header(self) -> dict | None:
    """The checkpoint header, if present."""
    return dict(self._header) if self._header is not None else None
