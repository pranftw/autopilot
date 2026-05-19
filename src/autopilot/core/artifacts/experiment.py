"""Experiment-scoped artifact classes."""

from autopilot.core.artifacts.artifact import JSONArtifact, JSONLArtifact, TextArtifact
from autopilot.core.models import Event
from autopilot.tracking.io import read_json
from pathlib import Path
from typing import Any


class EventsArtifact(JSONLArtifact):
  """events.jsonl -- append-only experiment event log."""

  def __init__(self) -> None:
    """Create an events log artifact writing to ``events.jsonl``."""
    super().__init__('events.jsonl')

  def schema(self) -> dict:
    """Return a schema describing Event records in this log.

    Returns:
      Dict with ``record_type`` and ``fields`` metadata for tooling.
    """
    return {
      'record_type': 'Event',
      'fields': {
        'timestamp': 'str (ISO 8601)',
        'event_type': 'str',
        'message': 'str | None',
        'metadata': 'dict[str, Any]',
      },
    }

  def validate(self, data: Any) -> None:
    """Ensure ``data`` is an ``Event`` or a dict with required keys.

    Args:
      data: ``Event`` instance or mapping to validate.

    Raises:
      TypeError: When ``data`` is neither ``Event`` nor a ``dict``.
      ValueError: When ``data`` is a dict missing ``timestamp`` or ``event_type``.
    """
    if isinstance(data, Event):
      return
    if not isinstance(data, dict):
      msg = f'event must be Event or dict, got type={type(data).__name__!r}'
      raise TypeError(
        msg,
      )
    if 'timestamp' not in data or 'event_type' not in data:
      msg = f'event requires timestamp and event_type; keys={sorted(data.keys())!r}'
      raise ValueError(
        msg,
      )

  def serialize(self, data: Any) -> dict:
    """Convert ``Event`` or dict to a JSON-serializable dict.

    Args:
      data: ``Event`` or plain dict.

    Returns:
      Dict suitable for JSONL storage.
    """
    if isinstance(data, Event):
      return data.to_dict()
    return data

  def deserialize(self, raw: dict) -> Event:
    """Parse a dict row into an ``Event``.

    Args:
      raw: Mapping from disk.

    Returns:
      Reconstructed ``Event`` instance.
    """
    return Event.from_dict(raw)


class RunStateArtifact(JSONArtifact):
  """run_state.json -- tracks current run status for crash detection."""

  def __init__(self) -> None:
    """Create a run-state artifact writing to ``run_state.json``."""
    super().__init__('run_state.json')

  def schema(self) -> dict:
    """Return a schema for run state fields.

    Returns:
      Dict describing expected ``run_state.json`` shape.
    """
    return {
      'fields': {
        'epoch': 'int',
        'timestamp': 'str (ISO 8601)',
        'status': "str ('running' | 'completed' | 'failed')",
        'stop_reason': 'str | None',
        'last_good_epoch': 'int | None',
      },
    }

  def validate(self, data: Any) -> None:
    """Validate ``status`` when present on a dict payload.

    Args:
      data: Candidate run state; only dicts with ``status`` are checked.

    Raises:
      ValueError: When ``status`` is not one of the allowed literals.
    """
    if isinstance(data, dict) and 'status' in data:
      valid = {'running', 'completed', 'failed'}
      if data['status'] not in valid:
        msg = f'status must be one of {valid}'
        raise ValueError(msg)

  def merge(self, existing: dict, new: dict) -> dict:
    """Shallow-merge ``new`` into a copy of ``existing``.

    Args:
      existing: Current dict on disk.
      new: Patch to apply (keys in ``new`` overwrite).

    Returns:
      Merged dict for atomic write.
    """
    merged = dict(existing)
    merged.update(new)
    return merged


class CostArtifact(JSONArtifact):
  """cost_summary.json -- aggregate cost information for an experiment."""

  def __init__(self) -> None:
    """Create a cost summary artifact writing to ``cost_summary.json``."""
    super().__init__('cost_summary.json')


class SummaryArtifact(JSONArtifact):
  """summary.json -- experiment summary at experiment scope."""

  def __init__(self) -> None:
    """Create a summary artifact writing to ``summary.json``."""
    super().__init__('summary.json')


class ResultArtifact(JSONArtifact):
  """result.json -- experiment evaluation result."""

  def __init__(self) -> None:
    """Create a result artifact writing to ``result.json``."""
    super().__init__('result.json')


class CommandsArtifact(JSONArtifact):
  """commands.json -- experiment command history (JSON array)."""

  def __init__(self) -> None:
    """Create a command history artifact writing to ``commands.json``."""
    super().__init__('commands.json')

  def merge(self, existing: dict, new: dict) -> dict:
    """Replace prior content with ``new`` (commands history is authoritative).

    Args:
      existing: Previous on-disk dict (ignored).
      new: Full replacement payload.

    Returns:
      ``new`` unchanged.
    """
    return new

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> list[dict]:
    """Read ``commands.json`` as a list of dict rows.

    Args:
      base_dir: Experiment directory root.
      epoch: Unused (experiment-scoped file).

    Returns:
      List from JSON when the file holds an array; otherwise ``[]``.
    """
    raw = read_json(self.resolve_path(base_dir, epoch))
    if raw is None:
      return []
    if not isinstance(raw, list):
      return []
    return raw

  def append_record(self, record: dict, base_dir: Path) -> Path:
    """Load existing array, append record, write back atomically.

    WARNING: Not safe for concurrent writers. This uses a read-modify-write
    pattern. If two processes call append_record simultaneously, one write
    will be lost. Single-writer only.

    Args:
      record: Command record dict to append.
      base_dir: Experiment directory root.

    Returns:
      Path to the written ``commands.json`` file.
    """
    existing = self.read_raw(base_dir)
    existing.append(record)
    return self.write(existing, base_dir)


class ReportArtifact(TextArtifact):
  """report.md -- human-readable experiment report."""

  def __init__(self, filename: str = 'report.md') -> None:
    """Create a report artifact with the given filename.

    Args:
      filename: Relative report path (default ``report.md``).
    """
    super().__init__(filename)

  def serialize(self, data: Any) -> str:
    """Render ``data`` as markdown text.

    Args:
      data: ``str``, ``dict`` (sectioned report), or other (``str(...)``).

    Returns:
      UTF-8 markdown body.
    """
    if isinstance(data, str):
      return data
    if isinstance(data, dict):
      lines = ['# Experiment Report', '']
      for key, value in data.items():
        lines.extend([f'## {key}', str(value), ''])
      return '\n'.join(lines)
    return str(data)

  def deserialize(self, raw: str) -> str:
    """Identity: report text is stored and loaded as ``str``.

    Args:
      raw: Raw markdown from disk.

    Returns:
      Same string as ``raw``.
    """
    return raw

  def update(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Append a new section to the report.

    Args:
      data: Content to append (serialized then appended).
      base_dir: Experiment directory root.
      epoch: Optional epoch scope for path resolution.

    Returns:
      Path to the report file after append.
    """
    return self.append(data, base_dir, epoch)
