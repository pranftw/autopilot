"""Shared data models: Event, Result, CommandRecord, DatasetEntry, DatasetSnapshot.

HyperparamSet lives in core/hyperparams.py.
"""

from autopilot.core.constraint import ConstraintResult
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field, fields
from typing import Any


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
  """Evaluation result: metrics, structured constraint outcomes, and computed pass/fail.

  Breaking change: ``gates`` is ``list[ConstraintResult]`` (was ``dict[str, str]``).
  ``passed`` is a computed property: ``all(c.passed for c in self.gates)``.
  Passing a dict for ``gates`` raises ``TypeError``.
  """

  metrics: dict[str, float] = field(default_factory=dict)
  gates: list[ConstraintResult] = field(default_factory=list)
  summary: str | None = None

  def __post_init__(self) -> None:
    """Validate gates type; reject old dict format.

    Raises:
      TypeError: When ``gates`` is a dict (old format) instead of a list.
    """
    if isinstance(self.gates, dict):
      msg = (
        'Result.gates must be list[ConstraintResult], got dict. '
        'The dict[str, str] format is removed. '
        'Use ConstraintResult instances or gate_to_constraint().'
      )
      raise TypeError(msg)

  @property
  def passed(self) -> bool:
    """Whether all constraint gates passed (vacuous truth when empty).

    Returns:
      True when every ``ConstraintResult`` in ``gates`` has ``passed=True``,
      or when ``gates`` is empty.
    """
    return all(c.passed for c in self.gates)

  def __bool__(self) -> bool:
    """Truthiness follows overall pass/fail, not metric presence.

    Returns:
      The ``passed`` property value.
    """
    return self.passed

  def to_dict(self) -> dict[str, Any]:
    """Serialize including the computed ``passed`` property.

    Returns:
      Dict with ``metrics``, ``gates``, ``summary``, and ``passed`` keys.
    """
    result = super().to_dict()
    result['passed'] = self.passed
    return result

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Result':
    """Deserialize from dict, hydrating nested ConstraintResult entries.

    Args:
      data: Raw mapping with optional ``gates`` list of dicts.

    Returns:
      Result with hydrated ``ConstraintResult`` gates.

    Raises:
      TypeError: When ``gates`` is a dict (legacy format).
    """
    data = dict(data)
    raw_gates = data.pop('gates', [])
    if isinstance(raw_gates, dict):
      msg = (
        'Result.gates must be list[ConstraintResult], got dict. '
        'The dict[str, str] format is removed. '
        'Use ConstraintResult instances or gate_to_constraint().'
      )
      raise TypeError(msg)
    gates_list = [] if raw_gates is None else raw_gates
    hydrated_gates = [
      ConstraintResult.from_dict(g) if isinstance(g, dict) else g for g in gates_list
    ]
    names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in names}
    filtered.pop('passed', None)
    return cls(gates=hydrated_gates, **filtered)


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
    """Deserialize from dict, handling null -> empty container coercion for collection fields.

    Args:
      data: Raw mapping possibly containing null ``entries``.

    Returns:
      DatasetSnapshot with coerced and parsed ``entries``.
    """
    data = dict(data)
    raw_entries = data.get('entries')
    seq = [] if raw_entries is None else raw_entries
    data['entries'] = [DatasetEntry.from_dict(e) for e in seq]
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})
