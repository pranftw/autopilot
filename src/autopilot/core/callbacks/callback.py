"""Callback base class for cross-cutting trainer concerns.

All hooks receive (trainer, module, ...) matching Lightning convention.
Override hooks you need; unimplemented hooks are no-ops.

Hook naming follows PyTorch Lightning conventions:
  on_fit_start / on_fit_end
  on_train_epoch_start / on_train_epoch_end
  on_validation_epoch_start / on_validation_epoch_end
  on_test_epoch_start / on_test_epoch_end

Plus Lightning-aligned lifecycle hooks:
  setup / teardown (with Stage parameter)
  on_exception
  on_save_checkpoint / on_load_checkpoint
  on_sanity_check_start / on_sanity_check_end (dispatched by Trainer._run_sanity_check)
  on_validation_batch_start / on_validation_batch_end (dispatched by run_eval_phase)
  on_test_batch_start / on_test_batch_end (dispatched by run_eval_phase)
  on_predict_start / on_predict_end (dispatched by Trainer.predict)
  on_predict_batch_start / on_predict_batch_end (dispatched by Trainer.predict)

Plus framework-specific hooks:
  on_epoch_start / on_epoch_end  (generic loop)
  on_loop_start / on_loop_end
  on_context_emit  (context traceability)

Note: on_fit_end(trainer, module) intentionally does NOT receive the loop's
result dict. For metrics at fit-end, use experiment.complete(metrics) or
on_loop_end(trainer, module, result). This avoids confusion for Lightning users
who expect on_fit_end to receive pl_module (it does receive module here) but
might also expect the result dict (it does not).
"""

from autopilot.core.models import Result
from autopilot.data.datamodule import Stage
from typing import Any


class Callback:
  """Base callback. Override any hook method.

  All hooks receive (trainer, module, ...) matching Lightning convention.
  Override hooks you need; unimplemented hooks are no-ops.

  Callbacks are composable -- the Trainer calls all registered callbacks
  in order for each hook. Callbacks observe; they don't control flow.

  Hook categories:

    Lifecycle (Lightning-aligned):
      setup(trainer, module, stage)  -- called before fit/validate/test/predict
      teardown(trainer, module, stage)  -- called after fit/validate/test/predict
      on_exception(trainer, module, exception)  -- called on unhandled exception

    Fit lifecycle:
      on_fit_start(trainer, module)
      on_fit_end(trainer, module)

    Training epoch:
      on_train_epoch_start(trainer, module, epoch)
      on_train_epoch_end(trainer, module, epoch)

    Training batch:
      on_train_batch_start(trainer, module, batch_idx)
      on_train_batch_end(trainer, module, batch_idx, data)

    Optimizer:
      on_before_backward(trainer, module, loss_fn)
      on_after_backward(trainer, module)
      on_before_optimizer_step(trainer, module)
      on_before_zero_grad(trainer, module)

    Validation:
      on_validation_epoch_start(trainer, module, epoch)
      on_validation_epoch_end(trainer, module, epoch)

    Validation batch (dispatched by run_eval_phase for validate/test/fit):
      on_validation_batch_start(trainer, module, batch, batch_idx)
      on_validation_batch_end(trainer, module, batch, batch_idx)

    Test:
      on_test_epoch_start(trainer, module, epoch)
      on_test_epoch_end(trainer, module, epoch)

    Test batch (dispatched by run_eval_phase for test/fit-tail):
      on_test_batch_start(trainer, module, batch, batch_idx)
      on_test_batch_end(trainer, module, batch, batch_idx)

    Predict (dispatched by Trainer.predict):
      on_predict_start(trainer, module)
      on_predict_end(trainer, module)
      on_predict_batch_start(trainer, module, batch, batch_idx)
      on_predict_batch_end(trainer, module, batch, batch_idx)

    Sanity check:
      on_sanity_check_start(trainer, module)
      on_sanity_check_end(trainer, module)

    Checkpoint mutation:
      on_save_checkpoint(trainer, module, checkpoint)  -- mutate dict in-place
      on_load_checkpoint(trainer, module, checkpoint)  -- observe/modify raw dict

    Generic loop:
      on_epoch_start(trainer, module, epoch)
      on_epoch_end(trainer, module, epoch, result)
      on_loop_start(trainer, module, max_epochs)
      on_loop_end(trainer, module, result)

    Context traceability:
      on_context_emit(trainer, module, entry)

    State persistence:
      state_dict() -> dict
      load_state_dict(state_dict)

  The ``trainer`` parameter is typed as ``Any`` to avoid a circular import
  with ``core.trainer``.  At runtime it is always a ``Trainer`` instance.

  Note: on_fit_end(trainer, module) does NOT receive the loop's result dict.
  For metrics at fit-end, use experiment.complete(metrics) or
  on_loop_end(trainer, module, result).

  Example:
    >>> from autopilot.core.callbacks.callback import Callback
    >>>
    >>> class EpochTracer(Callback):
    ...   def on_train_epoch_start(self, trainer, module, epoch):
    ...     self.last_epoch = epoch
    >>>
    >>> callback = EpochTracer()
    >>> callback.on_train_epoch_start(None, None, 0)
    >>> callback.last_epoch
    0
  """

  # Lightning-style lifecycle hooks

  def setup(self, trainer: Any, module: Any, stage: Stage) -> None:
    """Called before fit/validate/test/predict begins.

    Trainer dispatches this after PathParameter bind and environment
    activation (fit only), before ``on_fit_start``. Also dispatched for
    validate, test, and predict stages.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being trained/evaluated.
      stage: Lifecycle stage (``Stage.fit``, ``Stage.validate``, etc.).
    """

  def teardown(self, trainer: Any, module: Any, stage: Stage) -> None:
    """Called after fit/validate/test/predict completes (success or failure).

    Trainer dispatches this in the outer ``finally`` of ``fit()``, after
    module teardown. Also dispatched for validate, test, and predict stages.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module that was trained/evaluated.
      stage: Lifecycle stage that is ending.
    """

  def on_exception(self, trainer: Any, module: Any, exception: BaseException) -> None:
    """Called when an unhandled exception occurs during fit.

    Dispatched in the ``except`` block of ``Trainer.fit``, before
    ``_fit_failure_path``. Exceptions raised inside this hook propagate
    naturally (no inner swallow).

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being trained when the error occurred.
      exception: The exception that was raised.
    """

  def on_save_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
    """Called during ``Trainer.save_checkpoint`` to mutate the checkpoint dict.

    The callback receives the assembled state dict and may add or modify
    entries in-place before the dict is written to storage.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Current module.
      checkpoint: Mutable checkpoint dict (``experiment``, ``module``,
        ``optimizer``, ``callbacks``, ``datamodule`` keys).
    """

  def on_load_checkpoint(self, trainer: Any, module: Any, checkpoint: dict[str, Any]) -> None:
    """Called during checkpoint resume before component state is restored.

    The callback observes (or modifies) the raw JSON envelope before
    ``_restore_from_checkpoint`` distributes state to components.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being restored.
      checkpoint: Raw checkpoint dict loaded from storage.
    """

  def on_sanity_check_start(self, trainer: Any, module: Any) -> None:
    """Called before the sanity-check validation run.

    Dispatched by ``Trainer._run_sanity_check`` before the capped
    validation pass. ``trainer.sanity_checking`` is ``True`` when this
    fires.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being validated.
    """

  def on_sanity_check_end(self, trainer: Any, module: Any) -> None:
    """Called after the sanity-check validation run.

    Dispatched by ``Trainer._run_sanity_check`` after the capped
    validation pass but before metric ``reset()``.
    ``trainer.sanity_checking`` is ``True`` when this fires.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being validated.
    """

  def on_validation_batch_start(
    self, trainer: Any, module: Any, batch: Any, batch_idx: int
  ) -> None:
    """Called before processing a validation batch.

    Dispatched by ``run_eval_phase`` during validate, fit-loop validation,
    and sanity check. Receives the raw batch and its zero-based index.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being validated.
      batch: Current validation batch.
      batch_idx: Zero-based batch index.
    """

  def on_validation_batch_end(self, trainer: Any, module: Any, batch: Any, batch_idx: int) -> None:
    """Called after processing a validation batch.

    Dispatched by ``run_eval_phase`` after ``validation_step`` and metric
    update for the batch.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being validated.
      batch: Current validation batch.
      batch_idx: Zero-based batch index.
    """

  def on_test_batch_start(self, trainer: Any, module: Any, batch: Any, batch_idx: int) -> None:
    """Called before processing a test batch.

    Dispatched by ``run_eval_phase`` during test and fit-tail test phase.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being tested.
      batch: Current test batch.
      batch_idx: Zero-based batch index.
    """

  def on_test_batch_end(self, trainer: Any, module: Any, batch: Any, batch_idx: int) -> None:
    """Called after processing a test batch.

    Dispatched by ``run_eval_phase`` after ``test_step`` and metric update
    for the batch.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module being tested.
      batch: Current test batch.
      batch_idx: Zero-based batch index.
    """

  def on_predict_start(self, trainer: Any, module: Any) -> None:
    """Called before the prediction loop begins.

    Dispatched by ``Trainer.predict()`` before the first predict batch.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module running prediction.
    """

  def on_predict_end(self, trainer: Any, module: Any) -> None:
    """Called after the prediction loop finishes.

    Dispatched by ``Trainer.predict()`` after the last predict batch.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module running prediction.
    """

  def on_predict_batch_start(self, trainer: Any, module: Any, batch: Any, batch_idx: int) -> None:
    """Called before processing a predict batch.

    Dispatched by ``Trainer.predict()`` before ``predict_step`` for each
    batch.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module running prediction.
      batch: Current predict batch.
      batch_idx: Zero-based batch index.
    """

  def on_predict_batch_end(self, trainer: Any, module: Any, batch: Any, batch_idx: int) -> None:
    """Called after processing a predict batch.

    Dispatched by ``Trainer.predict()`` after ``predict_step`` for each
    batch.

    Args:
      trainer: Active ``Trainer`` instance.
      module: Module running prediction.
      batch: Current predict batch.
      batch_idx: Zero-based batch index.
    """

  # Lightning-style fit hooks

  def on_fit_start(self, trainer: Any, module: Any) -> None:
    """Hook at the start of ``Trainer.fit``."""

  def on_fit_end(self, trainer: Any, module: Any) -> None:
    """Hook at the end of ``Trainer.fit`` (no loop result dict)."""

  # Lightning-style train hooks

  def on_train_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    """Hook before the training phase of an epoch."""

  def on_train_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    """Hook after the training phase of an epoch."""

  def on_train_batch_start(self, trainer: Any, module: Any, batch_idx: int = 0) -> None:
    """Hook before processing a training batch."""

  def on_train_batch_end(
    self,
    trainer: Any,
    module: Any,
    batch_idx: int = 0,
    data: Any | None = None,
  ) -> None:
    """Hook after processing a training batch."""

  def on_before_backward(self, trainer: Any, module: Any, loss_fn: Any) -> None:
    """Hook immediately before ``loss_fn.backward``."""

  def on_after_backward(self, trainer: Any, module: Any) -> None:
    """Hook immediately after the backward pass completes."""

  def on_before_optimizer_step(self, trainer: Any, module: Any) -> None:
    """Hook before the optimizer applies an update."""

  def on_before_zero_grad(self, trainer: Any, module: Any) -> None:
    """Hook before gradients are zeroed."""

  # Lightning-style validation hooks

  def on_validation_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    """Hook before the validation pass for ``epoch``."""

  def on_validation_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    """Hook after the validation pass for ``epoch``."""

  # Lightning-style test hooks

  def on_test_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    """Hook before a test epoch (symmetry with Lightning)."""

  def on_test_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    """Hook after a test epoch (symmetry with Lightning)."""

  # framework-specific hooks (generic loop)

  def on_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    """Hook at epoch start for the framework's generic training loop."""

  def on_epoch_end(
    self,
    trainer: Any,
    module: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    """Hook at epoch end, optionally carrying a structured ``Result``."""

  def on_loop_start(self, trainer: Any, module: Any, max_epochs: int) -> None:
    """Hook when the epoch loop begins."""

  def on_loop_end(self, trainer: Any, module: Any, result: dict[str, Any]) -> None:
    """Hook when the epoch loop finishes with aggregated ``result``."""

  # framework-specific hooks (context traceability)

  def on_context_emit(self, trainer: Any, module: Any, entry: Any) -> None:
    """Hook when a context entry is emitted via Trainer.emit_context()."""

  # checkpointing

  def state_dict(self) -> dict[str, Any]:
    """Serialize callback state for checkpoints.

    Returns:
      Payload dict; default empty dict.
    """
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Restore callback state from ``state_dict``.

    Args:
      state_dict: Data previously returned by ``state_dict``.
    """
