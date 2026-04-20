"""Optimizer base class. Like torch.optim.Optimizer (NOT a Module)."""

from autopilot.core.parameter import Parameter


class Optimizer:
  """Base optimizer. step() applies updates, zero_grad() clears gradients."""

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
