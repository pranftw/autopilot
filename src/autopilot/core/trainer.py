"""Trainer: orchestrates experiment optimization with callbacks."""

from autopilot.core.callbacks import Callback
from autopilot.core.logger import Logger
from autopilot.core.loops import EpochLoop, Loop, LoopConfig
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.models import Result
from autopilot.core.module import AutoPilotModule, Module
from autopilot.core.optimizer import Optimizer
from autopilot.core.store import Store
from autopilot.policy.policy import Policy
from typing import Any


class Trainer:
  """Orchestrates experiment optimization.

  Owns the optimization loop and callback dispatch. Takes an optional logger
  directly (like Lightning). Module is passed to fit(), not the constructor.

  Example::

    trainer = Trainer(callbacks=[MetricsLogger()], logger=json_logger)
    trainer.fit(module, max_epochs=5)
  """

  def __init__(
    self,
    callbacks: list[Callback] | None = None,
    loop: Loop | None = None,
    dry_run: bool = False,
    logger: Logger | None = None,
    policy: Policy | None = None,
    store: Store | None = None,
    accumulate_grad_batches: int = 1,
  ) -> None:
    self._callbacks = list(callbacks or [])
    self._loop = loop or EpochLoop()
    self._dry_run = dry_run
    self._logger = logger
    self._policy = policy
    self._store = store
    self._accumulate_grad_batches = accumulate_grad_batches
    self._module: Module | None = None
    self._best_epoch = 0
    self._optimizer: Optimizer | None = None
    self.regression_detected: bool = False

  @property
  def module(self) -> Module | None:
    return self._module

  @property
  def callbacks(self) -> list[Callback]:
    return self._callbacks

  @property
  def loop(self) -> Loop:
    return self._loop

  @property
  def dry_run(self) -> bool:
    return self._dry_run

  @property
  def logger(self) -> Logger | None:
    return self._logger

  @property
  def policy(self) -> Policy | None:
    return self._policy

  @property
  def store(self) -> Store | None:
    return self._store

  @property
  def accumulate_grad_batches(self) -> int:
    return self._accumulate_grad_batches

  @property
  def optimizer(self) -> Optimizer | None:
    return self._optimizer

  def fit(
    self,
    module: Module,
    train_dataloaders: Any = None,
    val_dataloaders: Any = None,
    datamodule: Any = None,
    max_epochs: int = 10,
    ctx: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    self._module = module
    fit_ctx = ctx or {}

    if isinstance(module, AutoPilotModule):
      object.__setattr__(module, '_trainer', self)

    datamodule = datamodule
    if datamodule is not None:
      if hasattr(datamodule, 'prepare_data'):
        datamodule.prepare_data()
      if hasattr(datamodule, 'setup'):
        datamodule.setup('fit')

    train_loader = train_dataloaders
    val_loader = val_dataloaders
    if train_loader is None and datamodule is not None and hasattr(datamodule, 'train_dataloader'):
      train_loader = datamodule.train_dataloader()
    if val_loader is None and datamodule is not None and hasattr(datamodule, 'val_dataloader'):
      val_loader = datamodule.val_dataloader()

    optimizer: Any = None
    if isinstance(module, AutoPilotModule):
      module.setup()
      opt_cfg = module.configure_optimizers()
      if isinstance(opt_cfg, Optimizer):
        optimizer = opt_cfg
      elif isinstance(opt_cfg, dict):
        optimizer = opt_cfg.get('optimizer')
      else:
        optimizer = opt_cfg

    self._optimizer = optimizer

    loss_fn = next((m for m in module.modules() if isinstance(m, Loss)), None)
    metrics = {
      name: m
      for name, m in module.named_modules()
      if isinstance(m, Metric) and not isinstance(m, Loss)
    }

    loop_config = LoopConfig(
      max_epochs=max_epochs,
      dry_run=self._dry_run,
      ctx=fit_ctx,
      train_loader=train_loader,
      val_loader=val_loader,
      loss=loss_fn,
      optimizer=optimizer,
      metrics=metrics,
      accumulate_grad_batches=self._accumulate_grad_batches,
    )

    self._dispatch('on_fit_start')
    self.on_loop_start(max_epochs=max_epochs)
    result = self._loop.run(self, loop_config)
    self.on_loop_end(result=result)
    self._dispatch('on_fit_end')

    if isinstance(module, AutoPilotModule):
      module.teardown()
    if datamodule is not None and hasattr(datamodule, 'teardown'):
      datamodule.teardown('fit')

    return result

  def on_epoch_start(self, epoch: int) -> list[Any]:
    return self._dispatch('on_epoch_start', epoch=epoch)

  def on_epoch_end(
    self,
    epoch: int,
    result: Result | dict[str, Any] | None = None,
  ) -> list[Any]:
    return self._dispatch('on_epoch_end', epoch=epoch, result=result)

  def on_loop_start(self, max_epochs: int) -> list[Any]:
    return self._dispatch('on_loop_start', max_epochs=max_epochs)

  def on_loop_end(self, result: dict[str, Any]) -> list[Any]:
    return self._dispatch('on_loop_end', result=result)

  def should_stop_at(self, hook_method: Any, **kwargs: Any) -> bool:
    return 'stop' in hook_method(**kwargs)

  def _dispatch(self, hook_name: str, **kwargs: Any) -> list[Any]:
    results: list[Any] = []
    for cb in self._callbacks:
      method = getattr(cb, hook_name, None)
      if method and callable(method):
        result = method(trainer=self, **kwargs)
        if result is not None:
          results.append(result)
    return results

  def __repr__(self) -> str:
    return f'Trainer(dry_run={self._dry_run})'
