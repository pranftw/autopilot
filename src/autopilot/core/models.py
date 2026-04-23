"""Shared data models: Manifest, Result, Event, Dataset records."""

from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field, fields
from typing import Any
import json


@dataclass
class Manifest(DictMixin):
  """Experiment manifest: slug, epoch counter, hypothesis, decision, and metadata."""

  slug: str
  title: str | None = None
  current_epoch: int = 0
  idea: str | None = None
  hypothesis: str | None = None
  hyperparams: dict[str, Any] = field(default_factory=dict)
  decision: str | None = None
  decision_reason: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)

  @property
  def is_decided(self) -> bool:
    return bool(self.decision)

  def to_json(self) -> str:
    return json.dumps(self.to_dict(), indent=2)


@dataclass
class Event(DictMixin):
  """Append-only lifecycle event within an experiment."""

  timestamp: str
  event_type: str
  message: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandRecord(DictMixin):
  """Logged CLI command with optional arg redaction."""

  timestamp: str
  command: str
  args: list[str] = field(default_factory=list)
  redacted_args: list[str] = field(default_factory=list)


@dataclass
class Result(DictMixin):
  """Evaluation result: metrics, gate outcomes, and overall pass/fail."""

  metrics: dict[str, float] = field(default_factory=dict)
  gates: dict[str, str] = field(default_factory=dict)
  passed: bool = False
  summary: str | None = None

  def __bool__(self) -> bool:
    return self.passed


@dataclass
class DatasetEntry(DictMixin):
  """Single dataset split entry with path, format, and content hash."""

  name: str
  split: str
  path: str
  format: str = 'jsonl'
  rows: int = 0
  content_hash: str | None = None


@dataclass
class DatasetSnapshot(DictMixin):
  """Point-in-time snapshot of all dataset entries for reproducibility."""

  created_at: str
  entries: list[DatasetEntry] = field(default_factory=list)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'DatasetSnapshot':
    data = dict(data)
    data['entries'] = [DatasetEntry.from_dict(e) for e in data.get('entries', [])]
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class HyperparamSet(DictMixin):
  """Versioned hyperparameter set with optional schema and lock."""

  version: int = 1
  values: dict[str, Any] = field(default_factory=dict)
  schema: dict[str, Any] = field(default_factory=dict)
  locked: bool = False


@dataclass
class Promotion(DictMixin):
  """Promote/reject decision record for an experiment."""

  timestamp: str
  decision: str
  reason: str
  reviewer: str | None = None
  metadata: dict[str, Any] = field(default_factory=dict)
