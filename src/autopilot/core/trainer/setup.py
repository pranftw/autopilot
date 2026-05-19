"""Fit preparation: datamodule, loaders, optimizer, metrics, environment bind.

Domain module for Trainer setup concerns. All functions take
a ``trainer`` instance (typed as ``Any`` to avoid circular imports) as
their first argument.
"""

from autopilot.ai.environment import bind_path_parameters
from autopilot.ai.parameter import PathParameter
from autopilot.core.callbacks.context import ContextLogCallback
from autopilot.core.errors import ConfigError
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.loss import Loss
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import Module
from autopilot.core.optimizer import Optimizer
from autopilot.core.scheduler import Scheduler
from autopilot.core.trainer.eval import collect_module_metrics
from autopilot.data.datamodule import DataModule, Stage, ensure_stage
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


def prepare_datamodule_fit(datamodule: DataModule | None) -> None:
  """Call datamodule prepare_data/setup when present (fit stage).

  Args:
    datamodule: Optional ``DataModule`` to prepare for the fit stage.
  """
  if datamodule is None:
    return
  datamodule.prepare_data()
  datamodule.setup(ensure_stage(Stage.fit))


def resolve_fit_loaders(
  train_dataloaders: Any,
  val_dataloaders: Any,
  datamodule: DataModule | None,
  test_dataloaders: Any | None = None,
) -> tuple[Any, Any, Any]:
  """Explicit loaders win; otherwise pull from datamodule when available.

  Returns:
    Tuple of ``(train_loader, val_loader, test_loader)``, each possibly
    ``None``.
  """
  train_loader = train_dataloaders
  val_loader = val_dataloaders
  test_loader = test_dataloaders
  if train_loader is None and datamodule is not None:
    train_loader = datamodule.train_dataloader()
  if val_loader is None and datamodule is not None:
    try:
      val_loader = datamodule.val_dataloader()
    except NotImplementedError:
      val_loader = None
  if test_loader is None and datamodule is not None:
    try:
      test_loader = datamodule.test_dataloader()
    except NotImplementedError:
      test_loader = None
  return train_loader, val_loader, test_loader


def configure_optimizer_and_metrics(
  trainer: Any,
  module: AutoPilotModule,
) -> tuple[Any, Any | None, dict[str, Metric], dict[str, bool]]:
  """Setup module; resolve optimizer, scheduler, loss leaf, metrics dict, metadata.

  When ``configure_optimizers()`` returns a dict with a ``'scheduler'`` key,
  validates that the scheduler's optimizer is the same object as the resolved
  optimizer.

  Args:
    trainer: ``Trainer`` instance.
    module: ``AutoPilotModule`` to configure.

  Returns:
    Tuple of ``(optimizer, loss_fn, metrics, metric_metadata)``.

  Raises:
    ConfigError: When the scheduler's optimizer is not the same instance as
      the resolved optimizer from the dict return.
  """
  optimizer: Any = None
  scheduler: Scheduler | None = None
  module.setup()
  opt_cfg = module.configure_optimizers()
  if isinstance(opt_cfg, Optimizer):
    optimizer = opt_cfg
  elif isinstance(opt_cfg, dict):
    optimizer = opt_cfg['optimizer']
    scheduler = opt_cfg.get('scheduler')
  else:
    optimizer = opt_cfg

  if scheduler is not None and scheduler.optimizer is not optimizer:
    msg = (
      'scheduler.optimizer is not the same object as the resolved optimizer. '
      'The scheduler must be constructed with the same optimizer instance returned '
      'in configure_optimizers(). Create the scheduler with the optimizer you return: '
      "{'optimizer': opt, 'scheduler': LambdaScheduler(opt, lr_lambda=...)}"
    )
    raise ConfigError(msg)

  trainer._optimizer = optimizer
  trainer._scheduler = scheduler

  loss_fn = next((m for m in module.modules() if isinstance(m, Loss)), None)

  metrics, metric_metadata = collect_module_metrics(module)

  return optimizer, loss_fn, metrics, metric_metadata


def setup_fit_context(
  trainer: Any,
  module: AutoPilotModule,
  datamodule: DataModule | None,
  train_dataloaders: Any | None,
  val_dataloaders: Any | None,
  ctx: dict[str, Any] | None,
  test_dataloaders: Any | None = None,
) -> tuple[Any, Any, Any, Any, Any | None, dict[str, Metric], dict[str, bool]]:
  """Prepare module, datamodule, loaders, and optimizer for fit.

  Args:
    trainer: ``Trainer`` instance.
    module: ``AutoPilotModule`` to train.
    datamodule: Optional ``DataModule``.
    train_dataloaders: Explicit training data iterable.
    val_dataloaders: Explicit validation data iterable.
    ctx: Optional context dict.
    test_dataloaders: Explicit test data iterable.

  Returns:
    Tuple of (train_loader, val_loader, test_loader, optimizer, loss_fn,
    metrics, metric_metadata).
  """
  trainer._module = module
  fit_ctx = ctx if ctx is not None else {}
  trainer._fit_ctx = fit_ctx
  module.trainer = trainer
  prepare_datamodule_fit(datamodule)
  train_loader, val_loader, test_loader = resolve_fit_loaders(
    train_dataloaders,
    val_dataloaders,
    datamodule,
    test_dataloaders,
  )
  optimizer, loss_fn, metrics, metric_metadata = configure_optimizer_and_metrics(trainer, module)
  return train_loader, val_loader, test_loader, optimizer, loss_fn, metrics, metric_metadata


def build_loop_config(
  trainer: Any,
  module: AutoPilotModule,
  *,
  train_loader: Any,
  val_loader: Any,
  max_epochs: int,
  min_epoch: int = 0,
  fit_ctx: dict[str, Any],
  optimizer: Any,
  loss_fn: Any | None,
  metrics: dict[str, Metric],
  metric_metadata: dict[str, bool],
) -> LoopConfig:
  """Assemble LoopConfig from resolved components.

  Args:
    trainer: ``Trainer`` instance.
    module: ``AutoPilotModule`` being trained.
    train_loader: Training data iterable.
    val_loader: Validation data iterable (or None).
    max_epochs: Maximum number of training epochs.
    min_epoch: Starting epoch (for checkpoint resume).
    fit_ctx: Caller-provided context dict.
    optimizer: Resolved optimizer.
    loss_fn: Resolved loss function (or None).
    metrics: Metric dict from module.
    metric_metadata: higher_is_better hints.

  Returns:
    Config object consumed by ``Loop.run``.
  """
  return LoopConfig(
    max_epochs=max_epochs,
    min_epoch=min_epoch,
    dry_run=trainer._dry_run,
    ctx=fit_ctx,
    train_loader=train_loader,
    val_loader=val_loader,
    loss=loss_fn,
    optimizer=optimizer,
    metrics=metrics,
    accumulate_grad_batches=trainer._accumulate_grad_batches,
    experiment=trainer._experiment,
    metric_metadata=metric_metadata,
  )


def bind_path_params(trainer: Any, module: Module, wt: Path) -> list[PathParameter]:
  """Bind PathParameters to worktree-relative paths.

  Delegates to :func:`autopilot.ai.environment.bind_path_parameters` --
  the single canonical implementation for path binding (DRY with
  ``tree switch --bind``).

  Args:
    trainer: ``Trainer`` instance.
    module: Module whose parameters are scanned for PathParameter instances.
    wt: Worktree path from environment activation.

  Returns:
    List of PathParameters that were bound (for unbind in finally).
  """
  cfg_root = Path(trainer.config.root) if trainer.config is not None else Path.cwd()
  return bind_path_parameters(module, cfg_root, wt)


def ensure_agent_optimizer_context(trainer: Any, module: Module) -> None:
  """Wire Trainer context into optimizer when context was omitted.

  Called after configure_optimizers() and PathParameter bind, so the
  optimizer sees working_root paths and can derive feedback_dir.

  Also auto-sets ``feedback_dir`` from ``config.root`` when the
  optimizer exposes a ``feedback_dir`` property with no explicit value,
  preventing ``ConfigError`` in agentic mode.

  The ``'trainer'`` key is injected into the context dict for optional
  traceability: ``AgentOptimizer`` may call ``trainer.emit_context()``
  after successful agentic steps. Optimizers must tolerate absence of
  ``'trainer'`` in non-Trainer contexts.

  Args:
    trainer: ``Trainer`` instance.
    module: Module being trained (for future context enrichment).
  """
  opt = trainer._optimizer
  if opt is not None and hasattr(opt, 'context'):
    current_ctx = opt.context
    if current_ctx is None or not current_ctx:
      opt.context = make_optimizer_context(trainer, module)
  if hasattr(opt, 'feedback_dir') and opt.feedback_dir is None and trainer._config is not None:
    opt.feedback_dir = str(trainer._config.root / '.optimization')


def make_optimizer_context(trainer: Any, module: Module) -> dict[str, Any]:
  """Build the initial context dict for an AgentOptimizer.

  Args:
    trainer: ``Trainer`` instance.
    module: Module being trained.

  Returns:
    Context with experiment, config, and initial epoch info.
  """
  ctx: dict[str, Any] = {'epoch': 0, 'trainer': trainer}
  if trainer._experiment is not None:
    ctx['experiment_id'] = trainer._experiment.id
  if trainer.config is not None:
    ctx['root'] = str(trainer.config.root)
  return ctx


def attach_default_callbacks(trainer: Any) -> None:
  """Auto-attach ``ContextLogCallback`` following Lightning's enable_* pattern.

  Behavior matrix:
    - ``enable_context_log=True`` + no user context callback + experiment present:
      append default ``ContextLogCallback()``.
    - ``enable_context_log=True`` + user provides ``_is_context_log_callback=True``
      callback: skip default silently (user replacement).
    - ``enable_context_log=False`` + no user context callback: no-op (opt-out).
    - ``enable_context_log=False`` + user context callback present: raise
      ``ConfigError`` (conflicting configuration).
    - No experiment: no-op regardless of flag.

  Uses the ``_is_context_log_callback`` class-level flag (DRY-06) for
  detection -- no ``isinstance`` on the concrete type.

  Args:
    trainer: ``Trainer`` instance.

  Raises:
    ConfigError: When ``enable_context_log=False`` but a context log callback
      is present in the callbacks list (conflicting configuration).
  """
  has_context_cb = any(getattr(cb, '_is_context_log_callback', False) for cb in trainer._callbacks)
  if has_context_cb and not trainer._enable_context_log:
    msg = (
      'Trainer configured with enable_context_log=False'
      ' but found a context log callback in the callbacks list.'
      ' Remove the callback or set enable_context_log=True.'
    )
    raise ConfigError(msg)
  if has_context_cb:
    return
  if not trainer._enable_context_log:
    return
  if trainer._experiment is None:
    return
  trainer._callbacks.append(ContextLogCallback())
