"""Foundational types used across all AutoPilot modules.

Datum is the universal data container (AutoPilot's Tensor).
GateResult is the pass/fail/warn/skip enum for policy gates.

These types have zero imports from autopilot, making them the safe
bottom of the dependency graph.
"""

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any
from uuid import uuid4


class GateResult(str, Enum):
  """Outcome of a gate evaluation: pass, fail, warn, or skip."""

  PASS = 'pass'
  FAIL = 'fail'
  WARN = 'warn'
  SKIP = 'skip'


@dataclass
class Datum:
  """Universal data container. AutoPilot's Tensor."""

  split: str | None = None
  epoch: int = 0
  metrics: dict[str, float] = field(default_factory=dict)
  metadata: dict[str, Any] = field(default_factory=dict)
  success: bool = True
  error_message: str | None = None
  feedback: str | None = None
  items: list['Datum'] = field(default_factory=list)

  def __post_init__(self) -> None:
    if not hasattr(self, '_id'):
      object.__setattr__(self, '_id', uuid4().hex[:12])
    object.__setattr__(self, 'grad_fn', None)

  @property
  def id(self) -> str:
    return self._id

  def __bool__(self) -> bool:
    return self.success

  def to_dict(self) -> dict[str, Any]:
    return {
      'id': self._id,
      'split': self.split,
      'epoch': self.epoch,
      'metrics': self.metrics,
      'metadata': self.metadata,
      'success': self.success,
      'error_message': self.error_message,
      'feedback': self.feedback,
      'items': [item.to_dict() for item in self.items],
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Datum':
    data = dict(data)
    stored_id = data.pop('id', None)
    data['items'] = [cls.from_dict(item) for item in data.get('items', [])]
    names = {f.name for f in fields(cls)}
    instance = cls(**{k: v for k, v in data.items() if k in names})
    if stored_id:
      object.__setattr__(instance, '_id', stored_id)
    return instance
