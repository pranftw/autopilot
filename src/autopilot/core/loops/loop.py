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

  Fields:
    max_epochs             -- maximum number of epochs
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
  dry_run: bool = False
  ctx: dict[str, Any] | None = None
  train_loader: Any = None
  val_loader: Any = None
  loss: Any = None
  optimizer: Any = None
  metrics: dict[str, Any] = field(default_factory=dict)
  accumulate_grad_batches: int = 1
  experiment: Any = None
  metric_metadata: dict[str, bool] = field(default_factory=dict)


class Loop(ABC):
  """Abstract optimization loop. Subclass and override run().

  run(trainer, config) -> dict drives the epoch iteration. The returned dict
  is the loop result passed to experiment.on_loop_complete() and on_loop_end.
  Built-in: EpochLoop (core/loops/epoch.py), EpochOrchestrator (core/loops/orchestrator.py).
  """

  @abstractmethod
  def run(self, trainer: Any, config: LoopConfig) -> dict[str, Any]: ...

  def __repr__(self) -> str:
    return f'{type(self).__name__}()'
