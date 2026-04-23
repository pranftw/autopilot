"""Optimizer base class. Like torch.optim.Optimizer (NOT a Module).

Optimizer is not a Module -- it does not participate in the module tree
or auto-registration. It is instantiated separately and passed to the
Trainer via AutoPilotModule.configure_optimizers().
"""

from autopilot.core.parameter import Parameter


class Optimizer:
  """Base optimizer. step() applies updates, zero_grad() clears gradients.

  Core protocol:
    step()       -- read param.grad, apply changes (subclass implements)
    zero_grad()  -- set param.grad = None on all parameters

  Strategy blocklist API (used by MemoryCallback to prevent retrying failed strategies):
    block_strategy(name)          -- add to blocklist
    unblock_strategy(name)        -- remove from blocklist
    is_strategy_blocked(name)     -- check membership
    blocked_strategies            -- frozenset property of all blocked names

  Built-in subclass: AgentOptimizer (ai/optimizer.py) uses an Agent to apply
  LLM-driven changes. Deterministic subclasses (e.g. RuleOptimizer) skip the LLM.
  """

  def __init__(self, parameters: list[Parameter], lr: float = 1.0) -> None:
    self._parameters = list(parameters)
    self.lr = lr
    self._blocked_strategies: set[str] = set()

  def block_strategy(self, name: str) -> None:
    self._blocked_strategies.add(name)

  def unblock_strategy(self, name: str) -> None:
    self._blocked_strategies.discard(name)

  def is_strategy_blocked(self, name: str) -> bool:
    return name in self._blocked_strategies

  @property
  def blocked_strategies(self) -> frozenset[str]:
    return frozenset(self._blocked_strategies)

  def step(self) -> None:
    raise NotImplementedError

  def zero_grad(self) -> None:
    for param in self._parameters:
      if param.requires_grad:
        param.grad = None
