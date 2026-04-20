"""Parameter base class. Like nn.Parameter.

Parameter is a Datum that the optimizer is allowed to modify.
Assigned as Module attributes, auto-registered by Module.__setattr__
into _parameters. module.parameters() collects all.
"""

from autopilot.core.models import Datum
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Parameter(Datum):
  """Declared mutable scope for the optimizer.

  Like nn.Parameter IS-A Tensor, Parameter IS-A Datum.
  requires_grad controls whether the optimizer targets this parameter.
  grad holds the gradient (generic type -- text, structured, numerical).
  """

  requires_grad: bool = True
  grad: Any = field(default=None, repr=False)

  def __post_init__(self) -> None:
    super().__post_init__()
    object.__setattr__(self, '_grad_accumulator', None)

  def to_dict(self) -> dict[str, Any]:
    d = super().to_dict()
    d['requires_grad'] = self.requires_grad
    return d

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'Parameter':
    data = dict(data)
    requires_grad = data.pop('requires_grad', True)
    items_raw = data.pop('items', [])
    items = [Datum.from_dict(item) for item in items_raw]
    param = cls(**data, items=items)
    param.requires_grad = requires_grad
    return param
