"""JudgeLoss: wraps Judge as a Loss for the training loop."""

from autopilot.ai.judge import Judge
from autopilot.core.loss import Loss
from autopilot.core.models import Datum
from autopilot.core.parameter import Parameter
from typing import Any


class JudgeLoss(Loss):
  """Loss that uses a Judge to compute gradients."""

  def __init__(self, judge: Judge, parameters: list[Parameter] | None = None) -> None:
    super().__init__(parameters)
    self._judge = judge
    self._accumulated: list[dict[str, Any]] = []

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
    feedback = self._build_gradient(self._accumulated)
    for param in self._loss_parameters:
      if param.requires_grad:
        param.grad = feedback

  def _build_gradient(self, accumulated: list[dict[str, Any]]) -> str:
    """Aggregate accumulated batch results into a structured text gradient."""
    parts = []
    for entry in accumulated:
      data = entry['data']
      if data.feedback:
        parts.append(data.feedback)
      if data.error_message:
        parts.append(f'error: {data.error_message}')
    return '\n'.join(parts) if parts else 'no feedback'

  @property
  def gradients(self) -> list[dict[str, Any]]:
    return list(self._accumulated)

  def reset(self) -> None:
    self._accumulated = []
