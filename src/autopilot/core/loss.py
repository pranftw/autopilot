"""Loss base class. Extends Module like nn.CrossEntropyLoss extends nn.Module."""

from autopilot.core.models import Datum
from autopilot.core.module import Module
from autopilot.core.parameter import Parameter
from typing import Any


class Loss(Module):
  """Base loss. Extends Module so it auto-registers as a child module.

  forward() accumulates per batch. backward() aggregates and fills param.grad.
  reset() clears the accumulation window.
  """

  def __init__(self, parameters: list[Parameter] | None = None) -> None:
    super().__init__()
    self._loss_parameters = list(parameters) if parameters else []

  def forward(self, data: Datum, targets: Any = None) -> None:
    raise NotImplementedError

  def backward(self) -> None:
    raise NotImplementedError

  @property
  def gradients(self) -> Any:
    return None

  def reset(self) -> None:
    pass
