"""AutoPilotModule -- Module with step methods and lifecycle hooks.

Like LightningModule. Extends Module with the Trainer integration surface.

Import from terminal modules directly::

  from autopilot.core.module.autopilot_module import AutoPilotModule
"""

from autopilot.core.module.module import Module
from typing import Any


class AutoPilotModule(Module):
  """Module with step methods and lifecycle hooks. Like LightningModule.

  Extends ``Module`` with the Trainer integration surface; see ``Module`` for
  registration and ``__call__`` graph capture. ``Trainer`` orchestrates
  training and invokes the hooks below.

  Step methods (override for custom behavior):
    training_step(batch, batch_idx) -- called per train batch
    validation_step(batch, batch_idx) -- called per validation batch
    test_step(batch, batch_idx)    -- called per test batch (standalone test() or fit test phase)
    predict_step(batch, batch_idx) -- called per predict batch (Trainer.predict())
    configure_optimizers()     -- return an Optimizer (or dict with 'optimizer' key)

  Lifecycle hooks (called by Trainer/EpochLoop):
    setup()                    -- before training starts
    teardown()                 -- after training ends
    on_train_start()           -- before first train batch
    on_train_end()             -- after last train batch
    on_validation_start()      -- before first val batch
    on_validation_end()        -- after last val batch
    on_test_start()            -- before first test batch
    on_test_end()              -- after last test batch

  Attributes:
    trainer: Reference to the ``Trainer``. Set during ``Trainer.fit()``; ``None``
      before fitting.

  Example::

    class MyModule(AutoPilotModule):
      def __init__(self):
        super().__init__()
        self.backend = BackendModule(...)
        self.loss = JudgeLoss(judge=MyJudge(), collator=ConcatCollator())
        self.metrics = AccuracyMetric()

      def forward(self, batch):
        return self.backend(batch)

      def training_step(self, batch, batch_idx):
        return self.forward(batch)

      def configure_optimizers(self):
        return AgentOptimizer(
          agent=ClaudeCodeAgent(), params=list(self.parameters())
        )
  """

  def __init__(self) -> None:
    """Initialize module state with no trainer wired yet."""
    super().__init__()
    self.trainer = None

  def training_step(self, batch: Any, batch_idx: int) -> Any:
    """Override to define the training step.

    Args:
      batch: One item from the training dataloader.
      batch_idx: 0-based index of the batch within the current epoch.

    Implementations should call ``self(batch)`` to ensure graph recording via
    ModuleCallOperator, not ``self.forward(batch)``. Calling ``forward()``
    directly bypasses the computation graph, producing a Datum with no grad_fn.
    ``Loss.backward()`` will raise ``RuntimeError('cannot backward: data has
    no grad_fn')`` if this happens.
    """
    raise NotImplementedError

  def validation_step(self, batch: Any, batch_idx: int) -> Any:
    """Override to define the validation step.

    Args:
      batch: One item from the validation dataloader.
      batch_idx: 0-based index of the batch within the current validation pass.

    Returns:
      Validation output (typically a ``Datum``); passed to metrics for update.

    Raises:
      NotImplementedError: If not overridden in a concrete ``AutoPilotModule``.
    """
    raise NotImplementedError

  def test_step(self, batch: Any, batch_idx: int) -> Any:
    """Override to define the test step.

    Called by ``Trainer.test()`` and by the fit-tail test phase when
    ``test_dataloaders`` are provided.

    Args:
      batch: One item from the test dataloader.
      batch_idx: 0-based index of the batch within the test pass.

    Returns:
      Test output (typically a ``Datum``); passed to metrics for update.

    Raises:
      NotImplementedError: If not overridden in a concrete ``AutoPilotModule``.
    """
    raise NotImplementedError

  def configure_optimizers(self) -> Any:
    """Return the optimizer wiring used by ``Trainer.fit``.

    Returns:
      An ``Optimizer`` instance, or a dict with ``'optimizer'`` key (required)
      and optional ``'scheduler'`` key mapping to a ``Scheduler`` instance.
      The Trainer steps the scheduler automatically after each epoch.

    Raises:
      NotImplementedError: If not overridden in a concrete ``AutoPilotModule``.
    """
    raise NotImplementedError

  # lifecycle hooks

  def setup(self) -> None:
    """Hook invoked by the trainer before the first batch."""

  def teardown(self) -> None:
    """Hook invoked after training ends or aborts."""

  def on_train_start(self) -> None:
    """Hook before the first training batch."""

  def on_train_end(self) -> None:
    """Hook after the last training batch."""

  def on_validation_start(self) -> None:
    """Hook before the first validation batch."""

  def on_validation_end(self) -> None:
    """Hook after the last validation batch."""

  def predict_step(self, batch: Any, batch_idx: int) -> Any:
    """Override to define the predict step.

    Called by ``Trainer.predict()`` per batch under ``no_grad()``.
    Predict is output-only: no metrics, no loss, no optimizer.
    Implementations should return prediction output (arbitrary type);
    the Trainer collects outputs into the list returned by ``predict()``.

    Args:
      batch: One item from the predict dataloader.
      batch_idx: 0-based index of the batch within the predict pass.

    Raises:
      NotImplementedError: If not overridden in a concrete ``AutoPilotModule``.
        Override predict_step(batch, batch_idx) to define prediction logic.
    """
    msg = (
      f'{type(self).__name__} does not implement predict_step(batch, batch_idx). '
      f'Override predict_step in your AutoPilotModule subclass to define '
      f'prediction logic. predict_step should return the prediction output '
      f'for each batch.'
    )
    raise NotImplementedError(msg)

  def on_test_start(self) -> None:
    """Hook before the first test batch."""

  def on_test_end(self) -> None:
    """Hook after the last test batch."""
