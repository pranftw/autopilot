"""JudgeLoss: wraps JudgeAgent as a Loss for the training loop."""

from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.gradient import CollationResult, GradientCollator
from autopilot.core.loss import Loss
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum
from typing import Any


class JudgeLoss(Loss):
  """Loss that uses a JudgeAgent and GradientCollator to compute per-parameter gradients."""

  def __init__(
    self,
    judge: JudgeAgent,
    collator: GradientCollator,
    parameters: list[Parameter] | None = None,
  ) -> None:
    super().__init__(parameters)
    self._judge = judge
    self._collator = collator
    self._accumulated: list[dict[str, Any]] = []
    self._last_collation: CollationResult | None = None

  def forward(self, data: Datum, targets: Any = None) -> None:
    self._accumulated.append(
      {
        'data': data,
        'targets': targets,
      }
    )

  def backward(self) -> None:
    if not self._accumulated:
      return
    result = self._collator.collate(self._accumulated, self._loss_parameters)
    self._last_collation = result
    for param in self._loss_parameters:
      if param.requires_grad:
        grad = result.gradients.get(param.id)
        if grad is not None:
          param.grad = grad

  @property
  def gradients(self) -> CollationResult | None:
    return self._last_collation

  def reset(self) -> None:
    self._accumulated = []
