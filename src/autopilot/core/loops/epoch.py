"""Epoch-based optimization loop with lifecycle hooks.

Drives the forward -> loss -> backward -> optimizer.step() protocol
per epoch, with gradient accumulation, metric collection, policy gating,
and optional validation.
"""

from autopilot.core.loops.loop import Loop, LoopConfig
from autopilot.core.models import Result
from autopilot.core.module import AutoPilotModule
from autopilot.core.types import GateResult
from typing import Any


class EpochLoop(Loop):
  """Epoch-based optimization loop with lifecycle hooks.

  _run_epoch() flow (train path):
    1. Fire on_train_start, module.train(), on_train_epoch_start
    2. Per train batch: training_step (or module(batch)), loss(data, batch),
       metric.update(data)
    3. When _should_step is true: loss.backward(), optimizer.step(),
       optimizer.zero_grad(), loss.reset()
    4. Compute metrics, call experiment.on_epoch_complete()
    5. Policy evaluation: on FAIL, rollback and stop
    6. Validation pass: module.eval(), validation_step per batch,
       experiment.on_validation_complete()
    7. Reset metrics, fire on_train_epoch_end, on_train_end

  Gradient accumulation:
    _should_step(batch_idx, is_last_batch, accumulate) returns True when
    (batch_idx + 1) % accumulate == 0 OR is_last_batch. This gates
    backward/step/zero_grad so multiple micro-batches accumulate before
    an optimizer step.

  IterableDataset support:
    Batches are materialized via list(enumerate(loader)) to handle datasets
    without __len__. Accumulation works without known total.

  Callback Result:
    _build_callback_result() merges val_metrics with 'val_' prefix into
    a Result for on_epoch_end callback dispatch.
  """

  def _build_callback_result(self, epoch_result: dict[str, Any]) -> Result:
    """Build a Result for callback dispatch, merging val_metrics with val_ prefix."""
    merged_metrics = dict(epoch_result.get('metrics', {}))
    val = epoch_result.get('val_metrics', {})
    if val:
      merged_metrics.update({f'val_{k}': v for k, v in val.items()})
    return Result(
      metrics=merged_metrics,
      passed=not epoch_result.get('stopped', False),
    )

  def run(self, trainer: Any, config: LoopConfig) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for epoch in range(1, config.max_epochs + 1):
      if trainer.should_stop_at(trainer.on_epoch_start, epoch=epoch):
        break
      epoch_result = self._run_epoch(trainer, epoch, config)
      results.append(epoch_result)
      cb_result = self._build_callback_result(epoch_result)
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
    experiment = config.experiment
    if experiment:
      experiment.should_rollback = False

    module = trainer.module
    loss_fn = config.loss
    optimizer = config.optimizer
    metrics = config.metrics
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
          'store': experiment.store is not None if experiment else False,
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

    if experiment:
      experiment.on_epoch_complete(epoch, metric_values)

    if trainer.policy:
      result = Result(metrics=metric_values)
      gate_result = trainer.policy(result)
      if gate_result == GateResult.FAIL:
        if experiment and experiment.store:
          experiment.rollback(experiment.best_epoch)
        return {'epoch': epoch, 'metrics': metric_values, 'stopped': True}

    if experiment:
      experiment.best_epoch = epoch

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
      trainer._dispatch('on_validation_epoch_end', epoch=epoch)
      if experiment:
        experiment.on_validation_complete(
          epoch,
          val_metrics,
          metric_metadata=config.metric_metadata,
        )
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
