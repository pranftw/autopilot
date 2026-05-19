"""LR Scheduler base class and built-in LambdaScheduler.

Schedulers adjust per-group learning rates in an Optimizer after each epoch.
They are wired into the Trainer loop via ``configure_optimizers()`` dict return
and stepped automatically at the end of each epoch (after ``on_train_epoch_end``).

Usage::

    from autopilot.core.optimizer import Optimizer
    from autopilot.core.scheduler import LambdaScheduler

    optimizer = MyOptimizer(module.parameters(), lr=0.1)
    scheduler = LambdaScheduler(optimizer, lr_lambda=lambda epoch: 0.9**epoch)


    # Wire via configure_optimizers dict return:
    class MyModule(AutoPilotModule):
      def configure_optimizers(self):
        return {'optimizer': optimizer, 'scheduler': scheduler}

Scheduler is a plain class (not a dataclass), consistent with Optimizer style.
"""

from autopilot.core.optimizer import Optimizer
from collections.abc import Callable
from typing import Any


class Scheduler:
  """Base LR scheduler. Subclass and override ``get_lr()``.

  Captures ``base_lrs`` from the optimizer's param_groups at construction time.
  Multiplicative schedules use these frozen base values to prevent drift from
  repeated multiplication against current LR.

  Attributes:
    optimizer: The optimizer whose param_group LRs are adjusted.
    last_epoch: Index of the last epoch stepped (starts at -1, meaning no step
      has occurred yet).
    base_lrs: Frozen snapshot of per-group LR values at construction time.

  Checkpoint hooks:
    state_dict()             -- serialize last_epoch and base_lrs
    load_state_dict(state)   -- restore from a previously serialized dict
  """

  def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
    """Initialize scheduler with optimizer reference and base LRs.

    Args:
      optimizer: Optimizer whose param_group LRs will be adjusted on step.
      last_epoch: Index of the last epoch. -1 means no step has occurred.
    """
    self.optimizer = optimizer
    self.last_epoch = last_epoch
    self.base_lrs: list[float] = [float(group['lr']) for group in optimizer.param_groups]

  def step(self, epoch: int | None = None) -> None:
    """Advance the scheduler and update optimizer param_group LRs.

    When called with an explicit epoch, sets ``last_epoch`` to that value.
    When called without an argument, increments ``last_epoch`` by 1.
    Then applies ``get_lr()`` values to each param_group.

    Args:
      epoch: Explicit epoch index to set. If None, auto-increments.
    """
    if epoch is None:
      self.last_epoch += 1
    else:
      self.last_epoch = epoch
    values = self.get_lr()
    for group, lr in zip(self.optimizer.param_groups, values, strict=True):
      group['lr'] = lr

  def get_lr(self) -> list[float]:
    """Compute per-group LR values for the current epoch.

    Subclasses must override this method.

    Returns:
      List of floats, one per param_group.

    Raises:
      NotImplementedError: Base class does not define a schedule.
    """
    raise NotImplementedError

  def state_dict(self) -> dict[str, Any]:
    """Serialize scheduler state for checkpointing.

    Returns:
      Dict with ``last_epoch`` and ``base_lrs``.
    """
    return {'last_epoch': self.last_epoch, 'base_lrs': list(self.base_lrs)}

  def load_state_dict(self, state: dict[str, Any]) -> None:
    """Restore scheduler state from a checkpoint dict.

    Args:
      state: Dict previously returned by :meth:`state_dict`.
    """
    self.last_epoch = state['last_epoch']
    self.base_lrs = list(state['base_lrs'])


class LambdaScheduler(Scheduler):
  """Scheduler using a user-supplied function to compute LR factors.

  The function receives the current epoch index and returns a multiplicative
  factor applied to each group's base LR::

      scheduler = LambdaScheduler(optimizer, lr_lambda=lambda epoch: 0.95**epoch)

  At epoch ``e``, each group's LR becomes ``base_lr * lr_lambda(e)``.
  """

  def __init__(self, optimizer: Optimizer, lr_lambda: Callable[[int], float]) -> None:
    """Initialize with optimizer and lambda function.

    Args:
      optimizer: Optimizer whose param_group LRs will be adjusted.
      lr_lambda: Function mapping epoch index to a multiplicative factor.
    """
    super().__init__(optimizer=optimizer)
    self.lr_lambda = lr_lambda

  def get_lr(self) -> list[float]:
    """Compute per-group LRs by multiplying base_lrs by lr_lambda(last_epoch).

    Returns:
      List of floats, one per param_group.
    """
    factor = self.lr_lambda(self.last_epoch)
    return [base_lr * factor for base_lr in self.base_lrs]
