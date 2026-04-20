"""AgentOptimizer: wraps an Agent to apply code changes based on param.grad."""

from autopilot.ai.agent import Agent
from autopilot.ai.parameter import PathParameter
from autopilot.core.optimizer import Optimizer
from autopilot.core.parameter import Parameter
from typing import Any


class AgentOptimizer(Optimizer):
  """Optimizer that uses an Agent to apply updates from gradients.

  Passes context (workspace, epoch, metrics, file paths) to the agent
  and clears param.grad for PathParameters whose files changed on disk.
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
    self._context = context or {}

  def step(self) -> None:
    grads = {}
    for param in self._parameters:
      if param.requires_grad and param.grad is not None:
        grads[param.item_id or id(param)] = param.grad
    if not grads:
      return

    prompt = self._build_prompt(grads)
    ctx = self._build_context()
    result = self._agent.forward(prompt, context=ctx)

    if result and result.output:
      for param in self._parameters:
        if isinstance(param, PathParameter) and param.requires_grad:
          param.grad = None

  def _build_prompt(self, grads: dict[str, Any]) -> str:
    parts = ['Apply the following improvements based on feedback:']

    epoch = self._context.get('epoch')
    metrics = self._context.get('metrics')
    if epoch is not None:
      parts.append(f'\nCurrent epoch: {epoch}')
    if metrics:
      parts.append(f'Current metrics: {metrics}')

    for param in self._parameters:
      if isinstance(param, PathParameter):
        files = param.matched_files()
        if files:
          parts.append(f'\nEditable files ({param.source}):')
          for f in files[:20]:
            parts.append(f'  - {f}')

    for param_id, grad in grads.items():
      parts.append(f'\n--- Parameter {param_id} ---')
      parts.append(str(grad))
    return '\n'.join(parts)

  def _build_context(self) -> dict[str, Any]:
    ctx: dict[str, Any] = dict(self._context)
    for param in self._parameters:
      if isinstance(param, PathParameter):
        if 'cwd' not in ctx and param.source:
          ctx['cwd'] = param.source
        files = param.matched_files()
        if files:
          ctx.setdefault('allowed_files', []).extend(str(f) for f in files)
    return ctx

  def update_context(self, **kwargs: Any) -> None:
    """Update optimizer context between epochs (e.g. new metrics)."""
    self._context.update(kwargs)
