"""Structured constraint model for policy gate results.

ConstraintResult records the pass/fail outcome, metric, value, threshold,
and optional message for a single policy constraint check. Used by
Result.gates as the structured replacement for the old dict[str, str] format.

gate_to_constraint converts a GateResult enum to a ConstraintResult.
"""

from autopilot.core.serialization import DictMixin
from autopilot.core.types import GateResult
from dataclasses import dataclass
from typing import Any


@dataclass
class ConstraintResult(DictMixin):
  """Structured result of a single policy constraint check.

  Attributes:
    name: Gate or constraint identifier.
    passed: Whether the constraint was satisfied.
    metric: Metric key evaluated by this constraint.
    value: Observed metric value, or None when the metric was missing.
    threshold: Human-readable threshold description (e.g. '>= 0.8').
    message: Optional diagnostic message (typically set on failure).
  """

  name: str
  passed: bool
  metric: str
  value: float | None
  threshold: str
  message: str | None = None

  def __post_init__(self) -> None:
    """Validate field types at construction time.

    Raises:
      TypeError: When ``name`` is not a str, ``passed`` is not a bool,
        ``metric`` is not a str, ``threshold`` is not a str,
        ``value`` is not a numeric type (int/float) or None, or
        ``message`` is not a str or None. Bool is explicitly rejected
        for ``value`` despite being an int subclass.
    """
    if not isinstance(self.name, str):
      msg = (
        f'ConstraintResult.name must be str, got {type(self.name).__name__}.'
        ' Pass a string identifier for the constraint.'
      )
      raise TypeError(msg)
    if not isinstance(self.passed, bool):
      msg = (
        f'ConstraintResult.passed must be bool, got {type(self.passed).__name__}.'
        ' Pass True or False.'
      )
      raise TypeError(msg)
    if not isinstance(self.metric, str):
      msg = (
        f'ConstraintResult.metric must be str, got {type(self.metric).__name__}.'
        ' Pass the metric key name.'
      )
      raise TypeError(msg)
    if not isinstance(self.threshold, str):
      msg = (
        f'ConstraintResult.threshold must be str, got {type(self.threshold).__name__}.'
        ' Pass a human-readable threshold description.'
      )
      raise TypeError(msg)
    if isinstance(self.value, bool):
      msg = (
        f'ConstraintResult.value must be float, int, or None, '
        f'got bool ({self.value!r}). Pass a numeric value, not a boolean.'
      )
      raise TypeError(msg)
    if self.value is not None and not isinstance(self.value, (int, float)):
      msg = (
        f'ConstraintResult.value must be float, int, or None, '
        f'got {type(self.value).__name__} ({self.value!r}).'
        ' Pass a numeric metric value or None.'
      )
      raise TypeError(msg)
    if self.message is not None and not isinstance(self.message, str):
      msg = (
        f'ConstraintResult.message must be str or None, '
        f'got {type(self.message).__name__} ({self.message!r}).'
        ' Pass a string description or None.'
      )
      raise TypeError(msg)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ConstraintResult':
    """Deserialize from dict, validating required keys.

    Args:
      data: Mapping with at least ``name``, ``passed``, ``metric``,
        ``threshold`` keys.

    Returns:
      ConstraintResult instance.

    Raises:
      KeyError: When a required key is missing.
    """
    required = ('name', 'passed', 'metric', 'threshold')
    missing = [k for k in required if k not in data]
    if missing:
      msg = (
        f'ConstraintResult.from_dict missing required keys: {missing}.'
        f' Provided keys: {sorted(data.keys())}.'
      )
      raise KeyError(msg)
    return cls(
      name=data['name'],
      passed=data['passed'],
      metric=data['metric'],
      value=data.get('value'),
      threshold=data['threshold'],
      message=data.get('message'),
    )


def gate_to_constraint(
  gate_name: str,
  gate_result: GateResult,
  metric: str,
  value: float | None,
  threshold: str,
) -> ConstraintResult:
  """Convert a GateResult enum to a ConstraintResult.

  Mapping: PASSED -> passed=True; all other members -> passed=False.

  Args:
    gate_name: Gate identifier string.
    gate_result: Enum outcome from the gate evaluation.
    metric: Metric key the gate evaluated.
    value: Observed metric value (None when missing).
    threshold: Human-readable threshold description.

  Returns:
    ConstraintResult with appropriate passed flag and failure message.
  """
  passed = gate_result == GateResult.PASSED
  message = None if passed else f'{gate_name} failed'
  return ConstraintResult(
    name=gate_name,
    passed=passed,
    metric=metric,
    value=value,
    threshold=threshold,
    message=message,
  )
