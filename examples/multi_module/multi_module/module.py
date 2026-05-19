"""Multi-agent pipeline: explicit forward wiring with per-module attribution.

This example demonstrates the multi-agent composition pattern using explicit
forward() wiring. Each AgentModule owns a single Parameter (prompt) for clarity.

Backward traversal follows ModuleCallOperator edges in reverse of forward order:
writer <- researcher <- planner. The Pipeline is a container module; the inner
graph built by child module calls is authoritative for backward propagation.

CustomAttributionOperator is wired after each child module's output in the
Pipeline.forward() method. During backward, it transforms TextGradient to
include per-module attribution labels, illustrating the extension pattern
for attribution without replacing the core loss.

Expected gradient flow (CustomAttributionOperator sits between each module):

  Input Datum
      |
      v
  [Planner] -- prompt param
      |
      v
  CustomAttributionOperator('planner')
      |
      v
  [Researcher] -- prompt param
      |
      v
  CustomAttributionOperator('researcher')
      |
      v
  [Writer] -- prompt param
      |
      v
  CustomAttributionOperator('writer')
      |
      v
  Output Datum (grad_fn -> CustomAttributionOperator for Writer)
      |
      v
  Loss.backward() seeds TextGradient at Output.grad_fn
      |
      v
  Graph.backward() propagates:
    CustomAttribution('writer') tags gradient with writer attribution
    Writer.AccumulateGrad <- gradient
    CustomAttribution('researcher') tags gradient with researcher attribution
    Researcher.AccumulateGrad <- gradient
    CustomAttribution('planner') tags gradient with planner attribution
    Planner.AccumulateGrad <- gradient
"""

from autopilot.ai.gradient import TextGradient
from autopilot.core.module.module import Module
from autopilot.core.ops import AttributionOperator
from autopilot.core.parameter import Parameter
from autopilot.core.types import EvalDatum


class AgentModule(Module):
  """Leaf module representing a single agent in the pipeline.

  Owns one Parameter (prompt). forward() returns an EvalDatum carrying
  the agent's name in metadata so tests can trace the execution path.
  """

  def __init__(self, name: str) -> None:
    super().__init__()
    self.prompt = Parameter(items=[])
    self._name = name

  def forward(self, x):
    items = list(x.items) if hasattr(x, 'items') else []
    return EvalDatum(
      items=items,
      metadata={'agent': self._name, 'stage': self._name},
    )


class Pipeline(Module):
  """Container module: planner -> researcher -> writer.

  Explicit forward wiring only -- no composition sugar (>>, |).
  The inner graph built by child module __call__ invocations is authoritative;
  ModuleCallOperator on Pipeline detects the inner grad_fn and preserves it
  (container transparency).
  """

  def __init__(self) -> None:
    super().__init__()
    self.planner = AgentModule('planner')
    self.researcher = AgentModule('researcher')
    self.writer = AgentModule('writer')

  def forward(self, x):
    plan = self.planner(x)
    plan = CustomAttributionOperator.apply(plan, module_name='planner')
    research = self.researcher(plan)
    research = CustomAttributionOperator.apply(research, module_name='researcher')
    result = self.writer(research)
    return CustomAttributionOperator.apply(result, module_name='writer')


class CustomAttributionOperator(AttributionOperator):
  """Per-module attribution operator that tags gradients with module identity.

  Extends AttributionOperator (plan 06) to transform TextGradient during
  backward, prepending the module name to the text and adding an
  attribution label. Non-TextGradient types pass through unchanged.
  """

  @staticmethod
  def forward(ctx, datum, module_name=None):
    ctx.save_for_backward(module_name)
    return datum.clone()

  @staticmethod
  def backward(ctx, grad_output):
    module_name = ctx.saved[0]
    if isinstance(grad_output, TextGradient):
      _d = grad_output.text or ''
      return (
        TextGradient(
          text=f'Fix {module_name}: {_d}',
          attribution=f'Error attributed to {module_name}',
          severity=grad_output.severity,
        ),
      )
    return (grad_output,)
