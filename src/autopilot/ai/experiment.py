"""Experiment with lifecycle hooks for Trainer integration.

AutoPilotExperiment extends Experiment with hooks that fire automatically
when lifecycle methods are called. Like AutoPilotModule extends Module.

Context manager behavior (``with experiment:``) is inherited from
``Experiment`` -- no overrides needed. ``__enter__`` calls ``start()``
(which triggers ``on_start``), and ``__exit__`` calls ``complete()``
or ``fail()`` (which trigger ``on_complete`` / ``on_fail``).

Override hooks in subclasses for custom behavior. Trainer calls lifecycle
methods during fit(), which triggers the hooks automatically.

Hook surface:
  on_start()                      -- after start() transitions to running
  on_epoch_complete(epoch, metrics, **kwargs) -- called by EpochLoop after each
      epoch. NOT called by advance_epoch(); the loop is the single caller.
  on_validation_complete(epoch, metrics, **kwargs) -- called by EpochLoop after
      validation phase. Accepts arbitrary kwargs (e.g. metric_metadata).
  on_complete()                   -- after complete() transitions to completed
  on_fail(error)                  -- after fail() transitions to failed
  on_cancel()                     -- after cancel() transitions to cancelled
  build_result()                  -- called by complete() when no metrics provided
"""

from autopilot.core.experiment import Experiment
from typing import Any


class AutoPilotExperiment(Experiment):
  """Experiment with lifecycle hooks for Trainer integration.

  Overrides lifecycle methods to call hooks after state transitions.
  All hooks are public no-ops by default -- override in subclasses.
  """

  def start(self) -> None:
    """Transition pending -> running, then fire on_start hook."""
    super().start()
    self.on_start()

  def complete(self, metrics: dict[str, float] | None = None) -> None:
    """Transition running -> completed, then fire on_complete hook.

    If metrics is None, calls build_result() to produce the metrics dict.
    """
    if metrics is None:
      metrics = self.build_result()
    super().complete(metrics)
    self.on_complete()

  def fail(self, error: str | None = None, metrics: dict[str, Any] | None = None) -> None:
    """Transition running -> failed, then fire on_fail hook."""
    super().fail(error, metrics=metrics)
    self.on_fail(error)

  def cancel(self) -> None:
    """Transition non-terminal -> cancelled, then fire on_cancel hook."""
    super().cancel()
    self.on_cancel()

  def advance_epoch(self, metrics: dict[str, Any] | None = None) -> None:
    """Increment epoch counter only. Does NOT call on_epoch_complete.

    The EpochLoop is the single caller of on_epoch_complete. advance_epoch
    only increments the epoch counter via super().
    """
    super().advance_epoch(metrics)

  def on_start(self) -> None:
    """Hook fired after start() transitions to running. Override in subclasses."""

  def on_epoch_complete(self, epoch: int, metrics: dict[str, Any], **kwargs: Any) -> None:
    """Hook called by EpochLoop after each epoch. Override in subclasses."""

  def on_validation_complete(self, epoch: int, metrics: dict[str, Any], **kwargs: Any) -> None:
    """Hook called by EpochLoop after validation.

    Accepts arbitrary kwargs (e.g. metric_metadata). Override in subclasses for
    custom validation handling.
    """

  def on_complete(self) -> None:
    """Hook fired after complete() transitions to completed. Override in subclasses."""

  def on_fail(self, error: str | None) -> None:
    """Hook fired after fail() transitions to failed. Override in subclasses."""

  def on_cancel(self) -> None:
    """Hook fired after cancel() transitions to cancelled. Override in subclasses."""

  def build_result(self) -> dict[str, float]:
    """Build metrics dict for complete(). Returns self.metrics by default.

    Returns:
      Metrics mapping forwarded to :meth:`~Experiment.complete`.
    """
    return self.metrics
