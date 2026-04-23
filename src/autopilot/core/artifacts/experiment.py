"""Experiment-scoped artifact classes."""

from autopilot.core.artifacts.artifact import JSONArtifact, JSONLArtifact, TextArtifact
from autopilot.core.models import Event
from autopilot.tracking.io import read_json
from pathlib import Path
from typing import Any


class EventsArtifact(JSONLArtifact):
  """events.jsonl -- append-only experiment event log."""

  def __init__(self) -> None:
    super().__init__('events.jsonl')

  def schema(self) -> dict:
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
    if isinstance(data, Event):
      return
    if not isinstance(data, dict):
      raise ValueError('event must be Event or dict')
    if 'timestamp' not in data or 'event_type' not in data:
      raise ValueError('event requires timestamp and event_type')

  def serialize(self, data: Any) -> dict:
    if isinstance(data, Event):
      return data.to_dict()
    return data

  def deserialize(self, raw: dict) -> Event:
    return Event.from_dict(raw)


class BaselineArtifact(JSONArtifact):
  """best_baseline.json -- tracks the best validation metrics seen."""

  def __init__(self) -> None:
    super().__init__('best_baseline.json')

  def schema(self) -> dict:
    return {
      'fields': {
        'epoch': 'int',
        'metrics': 'dict[str, float]',
      },
    }

  def validate(self, data: Any) -> None:
    if isinstance(data, dict):
      if 'epoch' not in data or 'metrics' not in data:
        raise ValueError('baseline requires epoch and metrics')

  def merge(self, existing: dict, new: dict) -> dict:
    return new


class RunStateArtifact(JSONArtifact):
  """run_state.json -- tracks current run status for crash detection."""

  def __init__(self) -> None:
    super().__init__('run_state.json')

  def schema(self) -> dict:
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
    if isinstance(data, dict) and 'status' in data:
      valid = {'running', 'completed', 'failed'}
      if data['status'] not in valid:
        raise ValueError(f'status must be one of {valid}')

  def merge(self, existing: dict, new: dict) -> dict:
    merged = dict(existing)
    merged.update(new)
    return merged


class CostArtifact(JSONArtifact):
  """cost_summary.json -- aggregate cost information for an experiment."""

  def __init__(self) -> None:
    super().__init__('cost_summary.json')


class SummaryArtifact(JSONArtifact):
  """summary.json -- experiment summary at experiment scope."""

  def __init__(self) -> None:
    super().__init__('summary.json')


class ResultArtifact(JSONArtifact):
  """result.json -- experiment evaluation result."""

  def __init__(self) -> None:
    super().__init__('result.json')


class PromotionArtifact(JSONArtifact):
  """promotion.json -- promotion decision record."""

  def __init__(self) -> None:
    super().__init__('promotion.json')


class CommandsArtifact(JSONArtifact):
  """commands.json -- experiment command history (JSON array)."""

  def __init__(self) -> None:
    super().__init__('commands.json')

  def merge(self, existing: dict, new: dict) -> dict:
    return new

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> list[dict]:
    raw = read_json(self.resolve_path(base_dir, epoch))
    if raw is None:
      return []
    if not isinstance(raw, list):
      return []
    return raw

  def append_record(self, record: dict, base_dir: Path) -> Path:
    """Load existing array, append record, write back atomically."""
    existing = self.read_raw(base_dir)
    existing.append(record)
    return self.write(existing, base_dir)


class ReportArtifact(TextArtifact):
  """report.md -- human-readable experiment report."""

  def __init__(self, filename: str = 'report.md') -> None:
    super().__init__(filename)

  def serialize(self, data: Any) -> str:
    if isinstance(data, str):
      return data
    if isinstance(data, dict):
      lines = ['# Experiment Report', '']
      for key, value in data.items():
        lines.append(f'## {key}')
        lines.append(str(value))
        lines.append('')
      return '\n'.join(lines)
    return str(data)

  def deserialize(self, raw: str) -> str:
    return raw

  def update(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Append a new section to the report."""
    return self.append(data, base_dir, epoch)
