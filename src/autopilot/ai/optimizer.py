"""AgentOptimizer: wraps an Agent to apply code changes based on param.grad."""

from autopilot.ai.agents.agent import Agent
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from typing import Any


class AgentOptimizer(Optimizer):
  """Optimizer that uses an Agent to apply updates from gradients.

  Calls param.render() and param.grad.render() to build the prompt.
  Zero isinstance checks against concrete parameter or gradient types.

  Public extension methods:
    build_prompt() -> str       -- assemble the full optimization prompt
    build_context() -> dict     -- provide context dict to the Agent
    update_context(**kwargs)    -- refresh epoch/metrics/collation_context between epochs

  step() flow:
    1. Check if any parameter has a gradient
    2. Build prompt from param.render() + param.grad.render()
    3. Build context from self._context
    4. Call agent.run(prompt, context=ctx)
    5. Clear gradients on success

  Context keys used by build_prompt():
    epoch, metrics, collation_context (from CollationResult.context)
  """

  def __init__(
    self,
    agent: Agent,
    parameters: list[Parameter],
    lr: float = 1.0,
    context: dict[str, Any] | None = None,
  ) -> None:
    super().__init__(parameters, lr)
    self._agent = agent
    self._context = context if context is not None else {}

  def step(self) -> None:
    has_grads = False
    for param in self._parameters:
      if param.requires_grad and param.grad is not None:
        has_grads = True
        break
    if not has_grads:
      return

    prompt = self.build_prompt()
    ctx = self.build_context()
    if self._agent.limiter:
      self._agent.limiter.acquire()
    result = self._agent.run(prompt, context=ctx)

    if result and result.output:
      for param in self._parameters:
        if param.requires_grad:
          param.grad = None

  def build_prompt(self) -> str:
    parts: list[str] = ['Apply the following improvements based on feedback:']

    epoch = self._context.get('epoch')
    metrics = self._context.get('metrics')
    collation_context = self._context.get('collation_context')

    if epoch is not None:
      parts.append(f'\nCurrent epoch: {epoch}')
    if metrics:
      parts.append(f'Current metrics: {metrics}')
    if collation_context:
      parts.append(f'\n## Overall Direction\n{collation_context}')

    for param in self._parameters:
      if not param.requires_grad or param.grad is None:
        continue
      parts.append(f'\n--- Parameter {param.id} ---')
      desc = param.render()
      if desc:
        parts.append(desc)
      parts.append(param.grad.render())

    return '\n'.join(parts)

  def build_context(self) -> dict[str, Any]:
    return dict(self._context)

  def update_context(self, **kwargs: Any) -> None:
    """Update optimizer context between epochs (e.g. new metrics)."""
    self._context.update(kwargs)
