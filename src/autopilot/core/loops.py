"""Composable optimization loops."""

from abc import ABC, abstractmethod
from autopilot.core.models import GateResult, Result
from autopilot.core.module import AutoPilotModule
from dataclasses import dataclass
from typing import Any


@dataclass
class LoopConfig:
  max_epochs: int = 10
  dry_run: bool = False
  ctx: dict[str, Any] | None = None
  train_loader: Any = None
  val_loader: Any = None
  loss: Any = None
  optimizer: Any = None
  metrics: dict[str, Any] | None = None
  accumulate_grad_batches: int = 1


class Loop(ABC):
  """Abstract optimization loop. Subclass and override run()."""

  @abstractmethod
  def run(self, trainer: Any, config: LoopConfig) -> dict[str, Any]: ...

  def __repr__(self) -> str:
    return f'{type(self).__name__}()'


class EpochLoop(Loop):
  """Epoch-based optimization loop with lifecycle hooks.

  Each epoch dispatches on_epoch_start/on_epoch_end hooks.
  Stops early if any callback signals 'stop' or policy gate fails.
  """

  def run(self, trainer: Any, config: LoopConfig) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
      if trainer.should_stop_at(trainer.on_epoch_start, epoch=epoch):
        break
      epoch_result = self._run_epoch(trainer, epoch, config)
      results.append(epoch_result)
      cb_result = Result(
        metrics=epoch_result.get('metrics', {}),
        passed=not epoch_result.get('stopped', False),
      )
      trainer.on_epoch_end(epoch=epoch, result=cb_result)
      if epoch_result.get('stopped'):
        break

    return {'epochs': results, 'total_epochs': len(results)}

  def _should_step(
    self,
    batch_idx: int,
    is_last_batch: bool,
    accumulate: int,
  ) -> bool:
    if is_last_batch:
      return True
    return (batch_idx + 1) % accumulate == 0

  def _run_epoch(
    self,
    trainer: Any,
    epoch: int,
    config: LoopConfig,
  ) -> dict[str, Any]:
    trainer.regression_detected = False

    module = trainer.module
    loss_fn = config.loss
    optimizer = config.optimizer
    metrics = config.metrics or {}
    accumulate = config.accumulate_grad_batches

    if config.dry_run:
      trainer._dispatch('on_train_epoch_start', epoch=epoch)
      trainer._dispatch('on_train_epoch_end', epoch=epoch)
      return {
        'dry_run': True,
        'epoch': epoch,
        'planned_epochs': config.max_epochs,
        'components': {
          'loss': loss_fn is not None,
          'optimizer': optimizer is not None,
          'store': trainer._store is not None,
          'metrics': bool(metrics),
          'train_loader': config.train_loader is not None,
          'val_loader': config.val_loader is not None,
        },
      }

    if isinstance(module, AutoPilotModule):
      module.on_train_start()
    module.train()
    trainer._dispatch('on_train_epoch_start', epoch=epoch)

    if config.train_loader is not None:
      batches = list(enumerate(config.train_loader))
      total = len(batches)
      for batch_idx, batch in batches:
        trainer._dispatch('on_train_batch_start', batch_idx=batch_idx)
        data = module.training_step(batch) if isinstance(module, AutoPilotModule) else module(batch)
        if loss_fn:
          loss_fn(data, batch)
        for m in metrics.values():
          m.update(data)
        trainer._dispatch('on_train_batch_end', batch_idx=batch_idx, data=data)

        is_last = batch_idx == total - 1
        if self._should_step(batch_idx, is_last, accumulate):
          if loss_fn:
            trainer._dispatch('on_before_backward')
            loss_fn.backward()
            trainer._dispatch('on_after_backward')
          if optimizer:
            trainer._dispatch('on_before_optimizer_step')
            optimizer.step()
            trainer._dispatch('on_before_zero_grad')
            optimizer.zero_grad()
          if loss_fn:
            loss_fn.reset()

    metric_values: dict[str, float] = {}
    for m in metrics.values():
      metric_values.update(m.compute())

    if trainer._policy:
      result = Result(metrics=metric_values)
      gate_result = trainer._policy(result)
      if gate_result == GateResult.FAIL:
        if trainer._store:
          trainer._store.checkout(trainer._best_epoch)
        return {'epoch': epoch, 'metrics': metric_values, 'stopped': True}

    trainer._best_epoch = epoch

    for m in metrics.values():
      m.reset()

    val_metrics: dict[str, float] = {}
    if config.val_loader:
      module.eval()
      if isinstance(module, AutoPilotModule):
        module.on_validation_start()
      trainer._dispatch('on_validation_epoch_start', epoch=epoch)
      for batch in config.val_loader:
        if isinstance(module, AutoPilotModule):
          val_data = module.validation_step(batch)
        else:
          val_data = module(batch)
        for m in metrics.values():
          m.update(val_data)
      for m in metrics.values():
        val_metrics.update(m.compute())
      trainer._last_val_metrics = val_metrics
      trainer._dispatch('on_validation_epoch_end', epoch=epoch)
      if isinstance(module, AutoPilotModule):
        module.on_validation_end()
      module.train()

    for m in metrics.values():
      m.reset()

    trainer._dispatch('on_train_epoch_end', epoch=epoch)
    if isinstance(module, AutoPilotModule):
      module.on_train_end()

    result_dict: dict[str, Any] = {'epoch': epoch, 'metrics': metric_values}
    if val_metrics:
      result_dict['val_metrics'] = val_metrics
    return result_dict
