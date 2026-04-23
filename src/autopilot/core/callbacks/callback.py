"""Callback base class for cross-cutting trainer concerns.

Hook naming follows PyTorch Lightning conventions:
  on_fit_start / on_fit_end
  on_train_epoch_start / on_train_epoch_end
  on_validation_epoch_start / on_validation_epoch_end
  on_test_epoch_start / on_test_epoch_end

Plus framework-specific hooks:
  on_epoch_start / on_epoch_end  (generic loop)
  on_loop_start / on_loop_end
"""

from autopilot.core.models import Result
from typing import Any


class Callback:
  """Base callback. Override any hook method.

  Callbacks are composable -- the Trainer calls all registered callbacks
  in order for each hook. Callbacks observe; they don't control flow.

  Hook categories:

    Fit lifecycle:
      on_fit_start(trainer)
      on_fit_end(trainer)

    Training epoch:
      on_train_epoch_start(trainer, epoch)
      on_train_epoch_end(trainer, epoch)

    Training batch:
      on_train_batch_start(trainer, batch_idx)
      on_train_batch_end(trainer, batch_idx, data)

    Optimizer:
      on_before_backward(trainer)
      on_after_backward(trainer)
      on_before_optimizer_step(trainer)
      on_before_zero_grad(trainer)

    Validation:
      on_validation_epoch_start(trainer, epoch)
      on_validation_epoch_end(trainer, epoch)

    Test:
      on_test_epoch_start(trainer, epoch)
      on_test_epoch_end(trainer, epoch)

    Generic loop:
      on_epoch_start(trainer, epoch)
      on_epoch_end(trainer, epoch, result)
      on_loop_start(trainer, max_epochs)
      on_loop_end(trainer, result)

    State persistence:
      state_dict() -> dict
      load_state_dict(state_dict)

  The ``trainer`` parameter is typed as ``Any`` to avoid a circular import
  with ``core.trainer``.  At runtime it is always a ``Trainer`` instance.
  """

  # Lightning-style fit hooks

  def on_fit_start(self, trainer: Any) -> None:
    pass

  def on_fit_end(self, trainer: Any) -> None:
    pass

  # Lightning-style train hooks

  def on_train_epoch_start(self, trainer: Any, epoch: int) -> None:
    pass

  def on_train_epoch_end(self, trainer: Any, epoch: int) -> None:
    pass

  def on_train_batch_start(self, trainer: Any, batch_idx: int = 0) -> None:
    pass

  def on_train_batch_end(self, trainer: Any, batch_idx: int = 0, data: Any = None) -> None:
    pass

  def on_before_backward(self, trainer: Any) -> None:
    pass

  def on_after_backward(self, trainer: Any) -> None:
    pass

  def on_before_optimizer_step(self, trainer: Any) -> None:
    pass

  def on_before_zero_grad(self, trainer: Any) -> None:
    pass

  # Lightning-style validation hooks

  def on_validation_epoch_start(self, trainer: Any, epoch: int) -> None:
    pass

  def on_validation_epoch_end(self, trainer: Any, epoch: int) -> None:
    pass

  # Lightning-style test hooks

  def on_test_epoch_start(self, trainer: Any, epoch: int) -> None:
    pass

  def on_test_epoch_end(self, trainer: Any, epoch: int) -> None:
    pass

  # framework-specific hooks (generic loop)

  def on_epoch_start(self, trainer: Any, epoch: int) -> None:
    pass

  def on_epoch_end(self, trainer: Any, epoch: int, result: Result | None = None) -> None:
    pass

  def on_loop_start(self, trainer: Any, max_epochs: int) -> None:
    pass

  def on_loop_end(self, trainer: Any, result: dict[str, Any]) -> None:
    pass

  # checkpointing

  def state_dict(self) -> dict[str, Any]:
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    pass
