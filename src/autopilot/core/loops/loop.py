"""Composable optimization loops: Loop base and LoopConfig.

LoopConfig is a dataclass carrying all configuration for a single Loop.run() call.
Loop is the abstract base; subclass and override run() for custom loop behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopConfig:
  """Configuration for a single loop run. Built by Trainer.fit().

  Attributes:
    max_epochs             -- maximum number of epochs
    min_epoch              -- zero-based first epoch to run (used for checkpoint resume)
    dry_run                -- skip actual computation
    ctx                    -- caller-provided context dict (from fit(ctx=...))
    train_loader           -- training data loader
    val_loader             -- validation data loader (optional)
    loss                   -- Loss instance discovered from module tree
    optimizer              -- Optimizer from configure_optimizers()
    metrics                -- dict of {name: Metric} from module tree
    accumulate_grad_batches -- micro-batch count before optimizer step
    experiment             -- Experiment for lifecycle hooks
    metric_metadata        -- {metric_name: higher_is_better} for comparison logic
  """

  max_epochs: int = 10
  min_epoch: int = 0
  dry_run: bool = False
  ctx: dict[str, Any] | None = None
  train_loader: Any | None = None
  val_loader: Any | None = None
  loss: Any | None = None
  optimizer: Any | None = None
  metrics: dict[str, Any] = field(default_factory=dict)
  accumulate_grad_batches: int = 1
  experiment: Any | None = None
  metric_metadata: dict[str, bool] = field(default_factory=dict)


class Loop(ABC):
  """Abstract optimization loop. Subclass and override run().

  run(trainer, config) -> dict drives the epoch iteration. The returned dict
  is the loop result passed to on_loop_end callbacks.
  Built-in: EpochLoop (core/loops/epoch.py), EpochOrchestrator (core/loops/orchestrator.py).
  """

  @abstractmethod
  def run(self, trainer: Any, config: LoopConfig) -> dict[str, Any]:
    """Execute the loop for the given trainer and configuration.

    Args:
      trainer: Trainer owning module, callbacks, and policy.
      config: Loop configuration for this invocation.

    Returns:
      Loop result dict consumed by ``on_loop_end`` callbacks.
    """
    ...

  def __repr__(self) -> str:
    """Return a simple constructor-style representation."""
    return f'{type(self).__name__}()'
