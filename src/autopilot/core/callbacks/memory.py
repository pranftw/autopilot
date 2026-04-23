"""MemoryCallback: auto-records structured learnings and syncs blocked strategies."""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.memory import Memory
from autopilot.core.models import Result
from typing import Any


class MemoryCallback(Callback):
  """Auto-records structured learnings and syncs blocked strategies."""

  def __init__(self, memory: Memory, default_category: str = 'epoch_result') -> None:
    self._memory = memory
    self._default_category = default_category

  def on_epoch_end(
    self,
    trainer: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    metrics = result.metrics if result is not None else {}
    outcome = 'worked' if result is not None and result.passed else 'failed'

    if trainer.experiment and trainer.experiment.should_rollback:
      outcome = 'rollback'

    strategy = trainer.fit_context.get('strategy')

    self._memory.learn(
      epoch=epoch,
      outcome=outcome,
      category=self._default_category,
      strategy=strategy,
      metrics=metrics,
    )

  def on_before_optimizer_step(self, trainer: Any) -> None:
    if trainer.optimizer:
      for strategy in self._memory.blocked_strategies():
        trainer.optimizer.block_strategy(strategy)

  def state_dict(self) -> dict[str, Any]:
    return self._memory.state_dict()

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    self._memory.load_state_dict(state_dict)
