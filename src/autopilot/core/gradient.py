"""Gradient base class. Extends Datum like Parameter extends Datum.

Gradient is the structured feedback type that accumulates on Parameter.grad
during graph backward traversal. Loss.backward() seeds a gradient into the
computation graph; AccumulateGrad leaf nodes materialize param.grad using
Gradient.accumulate(). Unlike numeric tensors, gradients here carry semantic
information about WHERE and WHAT to fix.
"""

from autopilot.core.types import Datum, _hydrate_datum_base
from dataclasses import dataclass
from typing import Any

TODO_LINE_MIN_LEN = 15
TODO_ITEM_MAX_CHARS = 200


@dataclass
class Gradient(Datum):
  """Base gradient type.

  Materialized on Parameter.grad by AccumulateGrad during graph backward
  traversal.

  Subclass and override:
    accumulate(other) -> Gradient  -- combine two gradients (for grad accumulation)
    render() -> str                -- describe for prompt inclusion (AgentOptimizer reads this)

  AccumulateGrad (in core/operator.py) calls accumulate() during backward traversal.
  AgentOptimizer calls render() to build the optimization prompt.

  Built-in subclass: TextGradient (ai/gradient.py) for LLM-oriented gradients
  with text, attribution, severity, and evidence items.

  accumulate contract:
    Same runtime type required at fan-in. Cross-type accumulate raises TypeError
    instructing callers to insert a conversion operator before fan-in. Subclasses
    implement merge semantics after the type check.

  transform(ctx):
    Opt-in semantic transform during Operator.backward() bodies. Default is the
    identity (returns self). The backward engine does NOT call transform
    automatically; it is invoked only from inside Operator.backward()
    implementations when a semantic transform is desired.

  render():
    Abstract -- subclasses must override to produce a string representation
    suitable for prompt inclusion.

  Example:
    Use a concrete subclass such as :class:`NumericGradient` for simple numeric
    feedback:

    >>> from autopilot.core.gradient import NumericGradient
    >>>
    >>> NumericGradient(value=-0.25).render()
    'gradient: -0.25'
  """

  def accumulate(self, other: 'Gradient') -> 'Gradient':
    """Combine this gradient with another (subclasses implement merge).

    Args:
      other: Gradient of the same runtime type at fan-in.

    Raises:
      TypeError: When ``other`` is not the exact same type as ``self``.
      NotImplementedError: On the base class; subclasses must override.
    """
    if type(other) is not type(self):
      msg = (
        f'Cannot accumulate {type(self).__name__} with {type(other).__name__}. '
        f'Insert a conversion operator to coerce types before fan-in.'
      )
      raise TypeError(msg)
    raise NotImplementedError

  def render(self) -> str:
    """Human-readable gradient for prompts (subclasses must implement).

    Raises:
      NotImplementedError: On the base class.
    """
    raise NotImplementedError

  def todo_items(self) -> list[str]:
    """Extract actionable todo items from this gradient.

    Default: returns the first substantive line from render() that is
    longer than 15 characters and not a header. Subclasses should override
    for structured extraction.

    Returns:
      List of actionable item strings for the optimization todo.
    """
    try:
      rendered = self.render()
    except NotImplementedError:
      return []
    items: list[str] = []
    for raw_line in rendered.split('\n'):
      stripped = raw_line.strip()
      if stripped and not stripped.startswith('#') and len(stripped) > TODO_LINE_MIN_LEN:
        items.append(stripped[:TODO_ITEM_MAX_CHARS])
        break
    return items

  def transform(self, ctx: Any) -> 'Gradient':
    """Opt-in semantic transform. Default is identity (returns self).

    Called only from inside Operator.backward() implementations when a
    semantic transform is desired. The backward engine does not call this
    automatically.

    Args:
      ctx: Backward context supplied by the operator.

    Returns:
      Transformed gradient, or ``self`` when unchanged.
    """
    return self


@dataclass
class NumericGradient(Gradient):
  """Numeric gradient for testing and programmatic losses.

  Constructor takes a single ``value`` field (float, default 0.0).
  There is no ``direction`` parameter; directionality is a concern of
  ``TextGradient`` in the AI layer.
  """

  value: float = 0.0

  def accumulate(self, other: 'Gradient') -> 'NumericGradient':
    """Add numeric gradient values.

    Args:
      other: Must be a ``NumericGradient``.

    Raises:
      TypeError: When ``other`` is not ``NumericGradient``.

    Returns:
      New ``NumericGradient`` with summed ``value``.
    """
    if not isinstance(other, NumericGradient):
      msg = (
        f'Cannot accumulate NumericGradient with {type(other).__name__}. '
        f'Insert a conversion operator to coerce types before fan-in.'
      )
      raise TypeError(msg)
    return NumericGradient(value=self.value + other.value)

  def render(self) -> str:
    """Return a short string describing the numeric value.

    Returns:
      Text like ``gradient: <value>``.
    """
    return f'gradient: {self.value}'

  def to_dict(self) -> dict[str, Any]:
    """Serialize this gradient including the numeric ``value`` field.

    Returns:
      Dict from ``Datum.to_dict`` plus ``value``.
    """
    payload = super().to_dict()
    payload['value'] = self.value
    return payload

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'NumericGradient':
    """Deserialize a ``NumericGradient`` from a dict payload.

    Args:
      data: Mapping with optional ``id``, ``type``, ``value``, and ``items``.

    Returns:
      Reconstructed ``NumericGradient`` instance.
    """
    data = dict(data)
    value = data.pop('value', 0.0)
    instance = _hydrate_datum_base(
      cls,
      data,
      hydrate_child=Datum.from_dict,
      pop_type=True,
    )
    instance.value = value
    return instance
