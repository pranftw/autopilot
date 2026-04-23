"""Loss base class. Extends Module like nn.CrossEntropyLoss extends nn.Module.

Loss is a Module subclass, so assigning it as an attribute on a parent Module
auto-registers it into _modules. Trainer.fit() discovers the first Loss via
module.modules() walk.
"""

from autopilot.core.module import Module
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from typing import Any


class Loss(Module):
  """Base loss. Extends Module so it auto-registers as a child module.

  Three-phase contract per accumulation window:
    forward(data, targets)  -- accumulate per-item feedback from one batch
    backward()              -- aggregate feedback, fill param.grad with Gradient instances
    reset()                 -- clear internal accumulation state

  Optional _loss_parameters scoping:
    Pass parameters= to __init__ to restrict which Parameters receive gradients.
    When empty, the Loss applies to all parameters discovered by the optimizer.

  Gradient flow:
    forward() accumulates feedback -> backward() produces Gradient instances
    (either directly for programmatic losses, or via a GradientCollator for
    JudgeLoss) -> param.grad is set -> Optimizer.step() reads param.grad.render()

  Built-in subclass: JudgeLoss (ai/loss.py) wraps a Judge + GradientCollator.
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
