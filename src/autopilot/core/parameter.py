"""Parameter base class. Like nn.Parameter.

Parameter is a Datum that the optimizer is allowed to modify.
Assigned as Module attributes, auto-registered by Module.__setattr__
into _parameters. module.parameters() collects all.

Datum.id: every Datum (including Parameter) gets an auto-generated,
internal, immutable id (12-char hex from uuid4). Used by CollationResult
to key gradients to parameters.
"""

from autopilot.core.gradient import Gradient
from autopilot.core.types import Datum
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass
class Parameter(Datum):
  """Declared mutable scope for the optimizer.

  Like nn.Parameter IS-A Tensor, Parameter IS-A Datum.
  requires_grad controls whether the optimizer targets this parameter.
  grad holds a Gradient instance (or None) after Loss.backward().

  Two versioning protocols (separate concerns):
    snapshot() / restore()           -- content versioning via Store
    state_dict() / load_state_dict() -- checkpoint serialization

  Public extension methods (subclasses override for domain behavior):
    render() -> str         -- describe for prompt inclusion (AgentOptimizer reads this)
    snapshot() -> dict      -- capture managed content as {key: text} pairs
    restore(content: dict)  -- restore from snapshot (inverse of snapshot)

  Built-in subclass: PathParameter (ai/parameter.py) for filesystem scope.
  """

  requires_grad: bool = True
  grad: Gradient | None = field(default=None, repr=False)

  def __post_init__(self) -> None:
    super().__post_init__()
    object.__setattr__(self, '_grad_accumulator', None)

  def render(self) -> str:
    """Describe this parameter for prompt inclusion.

    Subclasses override to provide domain-specific descriptions.
    Default returns empty string.
    """
    return ''

  def snapshot(self) -> dict[str, str]:
    """Capture this parameter's managed content for versioning.

    Subclasses override to export their content. Keys and values are
    domain-specific: file-based params use relative paths as keys and
    file content as values; prompt params use descriptive keys.
    Default returns empty dict (parameter has no external content).
    """
    return {}

  def restore(self, content: dict[str, str]) -> None:
    """Restore this parameter's managed content from a version snapshot.

    Inverse of snapshot(). Subclasses override to restore their content.
    Default is a no-op.
    """

  def to_dict(self) -> dict[str, Any]:
    d = super().to_dict()
    d['requires_grad'] = self.requires_grad
    return d

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Parameter':
    data = dict(data)
    stored_id = data.pop('id', None)
    requires_grad = data.pop('requires_grad', True)
    items_raw = data.pop('items', [])
    items = [Datum.from_dict(item) for item in items_raw]
    names = {f.name for f in fields(cls)}
    param = cls(**{k: v for k, v in data.items() if k in names}, items=items)
    param.requires_grad = requires_grad
    if stored_id:
      object.__setattr__(param, '_id', stored_id)
    return param
