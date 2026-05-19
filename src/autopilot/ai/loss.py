"""JudgeLoss: wraps JudgeAgent as a Loss for the training loop.

Graph-based backward: JudgeLoss.backward() collates accumulated feedback via
GradientCollator, builds a seed TextGradient, and injects it into the
computation graph via get_current_graph().backward(). The graph distributes
attribution to parameters through AccumulateGrad leaf nodes. JudgeLoss never
sets param.grad directly.

When parameters=None: produces a single unattributed seed TextGradient. The
graph distributes via ModuleCallOperator.backward() broadcast to all reachable
parameters.

When parameters=[p1, p2, ...]: collator produces per-parameter attributions
feeding the seed content. The seed still enters graph.backward() at the root --
attribution is in the seed / operator behavior.
"""

from autopilot.ai.evaluation.judge import JudgeAgent
from autopilot.ai.gradient import CollationResult, GradientCollator, TextGradient
from autopilot.core.loss import Loss
from autopilot.core.parameter import Parameter
from autopilot.core.types import Datum


class JudgeLoss(Loss):
  """Loss that seeds gradients from evaluation feedback via a collation strategy.

  The ``judge`` parameter is stored for caller access (e.g. to re-run
  evaluations externally) but is not invoked by the loss itself.
  Gradient seeding uses the ``collator`` to transform accumulated
  feedback into ``TextGradient`` instances.

  backward() performs three steps:
    1. Collate accumulated feedback via self._collator.collate()
    2. Build a seed TextGradient from the collation result
    3. Inject seed into computation graph: get_current_graph().backward(root, seed)

  The graph distributes attribution to parameters via AccumulateGrad leaf nodes.

  Attributes:
    judge: The judge agent, stored for external access.
  """

  def __init__(
    self,
    judge: JudgeAgent,
    collator: GradientCollator,
    parameters: list[Parameter] | None = None,
  ) -> None:
    """Wire a judge workflow and collator for gradient seeding.

    The ``judge`` is stored as a public attribute for caller access
    (e.g. to re-run evaluations externally) but is not called by the
    loss internally. The loss is a gradient-seeding component, not an
    evaluation runner.

    Args:
      judge: Judge agent stored for external access, not called by loss.
      collator: Strategy for turning feedback into gradients.
      parameters: Optional subset to collate against; ``None`` means unattributed.
    """
    super().__init__(parameters)
    self.judge = judge
    self._collator = collator
    self._last_collation: CollationResult | None = None

  def compute_seed_gradient(self) -> TextGradient:
    """Collate feedback and produce a seed TextGradient.

    When _loss_parameters is populated, collator produces per-parameter
    attributions. When empty, produces an unattributed direction-only seed.

    Returns:
      Aggregated :class:`TextGradient` injected into the autograd graph.
    """
    result = self._collator.collate(self._accumulated, self._loss_parameters)
    self._last_collation = result

    if self._loss_parameters and result.gradients:
      directions: list[str] = []
      attributions: list[str] = []
      all_items: list[Datum] = []
      max_severity = 0.0
      for grad in result.gradients.values():
        if isinstance(grad, TextGradient):
          if grad.text:
            directions.append(grad.text)
          if grad.attribution:
            attributions.append(grad.attribution)
          all_items.extend(grad.items)
          max_severity = max(max_severity, grad.severity)
      return TextGradient(
        text='; '.join(directions) if directions else result.context,
        attribution='; '.join(attributions) if attributions else None,
        items=all_items,
        severity=max_severity,
      )

    return TextGradient(text=result.context)

  @property
  def gradients(self) -> CollationResult | None:
    """Last collation result from :meth:`compute_seed_gradient`, if any."""
    return self._last_collation

  def reset(self) -> None:
    """Clear accumulated loss state and last collation snapshot."""
    super().reset()
    self._last_collation = None
