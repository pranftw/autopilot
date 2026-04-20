"""Core data models for AutoPilot.

All shared types used across CLI, tracking, and policies.
"""

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any
import json


class GateResult(str, Enum):
  PASS = 'pass'
  FAIL = 'fail'
  WARN = 'warn'
  SKIP = 'skip'


@dataclass
class Manifest:
  slug: str
  title: str = ''
  current_epoch: int = 0
  idea: str = ''
  hypothesis: str = ''
  hyperparams: dict[str, Any] = field(default_factory=dict)
  decision: str = ''
  decision_reason: str = ''
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
      'slug': self.slug,
      'title': self.title,
      'current_epoch': self.current_epoch,
      'idea': self.idea,
      'hypothesis': self.hypothesis,
      'hyperparams': self.hyperparams,
      'decision': self.decision,
      'decision_reason': self.decision_reason,
      'metadata': self.metadata,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Manifest':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), indent=2)


@dataclass
class Event:
  timestamp: str
  event_type: str
  message: str = ''
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
      'timestamp': self.timestamp,
      'event_type': self.event_type,
      'message': self.message,
      'metadata': self.metadata,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Event':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class CommandRecord:
  timestamp: str
  command: str
  args: list[str] = field(default_factory=list)
  redacted_args: list[str] = field(default_factory=list)
  exit_code: int | None = None
  duration_seconds: float | None = None

  def to_dict(self) -> dict[str, Any]:
    return {
      'timestamp': self.timestamp,
      'command': self.command,
      'args': self.args,
      'redacted_args': self.redacted_args,
      'exit_code': self.exit_code,
      'duration_seconds': self.duration_seconds,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'CommandRecord':
    data = dict(data)
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class Result:
  """Evaluation result: metrics, gate outcomes, and overall pass/fail."""

  metrics: dict[str, float] = field(default_factory=dict)
  gates: dict[str, str] = field(default_factory=dict)
  passed: bool = False
  summary: str = ''

  def __bool__(self) -> bool:
    return self.passed

  def to_dict(self) -> dict[str, Any]:
    return {
      'metrics': self.metrics,
      'gates': self.gates,
      'passed': self.passed,
      'summary': self.summary,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Result':
    return cls(**data)


@dataclass
class DatasetEntry:
  name: str
  split: str
  path: str
  format: str = 'jsonl'
  rows: int = 0
  content_hash: str = ''

  def to_dict(self) -> dict[str, Any]:
    return {
      'name': self.name,
      'split': self.split,
      'path': self.path,
      'format': self.format,
      'rows': self.rows,
      'content_hash': self.content_hash,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'DatasetEntry':
    data = dict(data)
    return cls(**data)


@dataclass
class DatasetSnapshot:
  created_at: str
  entries: list[DatasetEntry] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    return {
      'created_at': self.created_at,
      'entries': [e.to_dict() for e in self.entries],
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'DatasetSnapshot':
    data = dict(data)
    data['entries'] = [DatasetEntry.from_dict(e) for e in data.get('entries', [])]
    return cls(**data)


@dataclass
class HyperparamSet:
  version: int = 1
  values: dict[str, Any] = field(default_factory=dict)
  schema: dict[str, Any] = field(default_factory=dict)
  locked: bool = False

  def to_dict(self) -> dict[str, Any]:
    return {
      'version': self.version,
      'values': self.values,
      'schema': self.schema,
      'locked': self.locked,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'HyperparamSet':
    return cls(**data)


@dataclass
class Datum:
  """Universal data container. AutoPilot's Tensor."""

  split: str | None = None
  epoch: int = 0
  metrics: dict[str, float] = field(default_factory=dict)
  metadata: dict[str, Any] = field(default_factory=dict)
  success: bool = True
  error_message: str = ''
  item_id: str = ''
  feedback: str = ''
  items: list['Datum'] = field(default_factory=list)

  def __post_init__(self) -> None:
    object.__setattr__(self, 'grad_fn', None)

  def __bool__(self) -> bool:
    return self.success

  def to_dict(self) -> dict[str, Any]:
    return {
      'split': self.split,
      'epoch': self.epoch,
      'metrics': self.metrics,
      'metadata': self.metadata,
      'success': self.success,
      'error_message': self.error_message,
      'item_id': self.item_id,
      'feedback': self.feedback,
      'items': [item.to_dict() for item in self.items],
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Datum':
    data = dict(data)
    data['items'] = [cls.from_dict(item) for item in data.get('items', [])]
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class Promotion:
  timestamp: str
  decision: str
  reason: str
  reviewer: str = ''
  metadata: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
      'timestamp': self.timestamp,
      'decision': self.decision,
      'reason': self.reason,
      'reviewer': self.reviewer,
      'metadata': self.metadata,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Promotion':
    return cls(**data)
