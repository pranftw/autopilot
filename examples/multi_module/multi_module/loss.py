"""SimpleLoss: trivial Loss subclass using graph.backward() for the multi-module example.

Compares model output metadata to batch target via a simple rule. Produces a
TextGradient seed that enters the computation graph -- no direct param.grad
assignment.
"""

from autopilot.ai.gradient import TextGradient
from autopilot.core.loss import Loss
from autopilot.core.types import Datum, EvalDatum
from typing import Any


class SimpleLoss(Loss):
  """Loss that seeds graph backward with a TextGradient.

  forward() accumulates per-item feedback by comparing output metadata
  to the batch target. compute_seed_gradient() produces a TextGradient
  summarizing what went wrong. backward() calls graph.backward() via the
  base class -- no manual param.grad assignment.
  """

  def __init__(self) -> None:
    super().__init__()
    self._feedback_items: list[str] = []

  def forward(self, data: Datum, targets: Any = None) -> None:
    super().forward(data, targets)
    if isinstance(data, EvalDatum) and isinstance(targets, EvalDatum):
      expected = targets.metadata.get('expected')
      actual = data.metadata.get('stage')
      if expected and actual != expected:
        self._feedback_items.append(
          f'Expected output quality "{expected}" but got stage "{actual}"'
        )

  def compute_seed_gradient(self) -> TextGradient:
    if self._feedback_items:
      text_content = '; '.join(self._feedback_items)
    else:
      text_content = 'Output does not meet expected quality'
    return TextGradient(
      text=text_content,
      attribution='Pipeline output needs improvement',
      severity=0.7,
    )

  def reset(self) -> None:
    super().reset()
    self._feedback_items = []
