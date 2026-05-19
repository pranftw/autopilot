"""Environment activation, sanity validation, and full ``fit()`` driver."""

from autopilot.core.checkpoint import CheckpointIO
from autopilot.core.enums import Status
from autopilot.core.errors import ConfigError
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.trainer import checkpoint as trainer_ckpt
from autopilot.core.trainer import eval as trainer_eval
from autopilot.core.trainer import journal
from autopilot.core.trainer import setup as trainer_setup
from autopilot.data.datamodule import DataModule, Stage, ensure_stage
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def run_sanity_check(trainer: Any, module: AutoPilotModule, loop_config: Any) -> None:
  """Run a capped validation pass before the first training epoch.

  Args:
    trainer: Trainer instance.
    module: Module to validate.
    loop_config: ``LoopConfig`` providing ``val_loader``, ``metrics``, and ``dry_run``.
  """
  if trainer._num_sanity_val_steps <= 0:
    return
  if loop_config.dry_run:
    return

  val_loader = loop_config.val_loader
  if val_loader is None:
    return

  trainer._sanity_checking = True
  try:
    trainer.dispatch_callbacks('on_sanity_check_start')
    with journal.profile_store_section(trainer, 'sanity_check'):
      trainer.run_eval_phase(
        module,
        val_loader,
        step_method='validation_step',
        hook_prefix='validation',
        max_batches=trainer._num_sanity_val_steps,
        epoch_arg=0,
      )
    trainer.dispatch_callbacks('on_sanity_check_end')
  finally:
    for m in loop_config.metrics.values():
      m.reset()
    trainer._sanity_checking = False


def run_fit_loop(trainer: Any, max_epochs: int, loop_config: Any) -> dict[str, Any]:
  """Activate environment, bind PathParameters, then run the epoch loop.

  Args:
    trainer: Trainer instance.
    max_epochs: Planned epoch budget (forwarded to ``on_loop_start``).
    loop_config: Config for ``trainer._loop.run``.

  Returns:
    Loop result dict with per-epoch telemetry.

  Raises:
    ConfigError: When ``Config.environment`` is set but no experiment is configured.
  """
  module = trainer._module
  assert module is not None
  experiment = trainer._experiment
  env = trainer.config.environment if trainer.config is not None else None
  if env is not None and experiment is None:
    env_name = type(env).__name__
    msg = f'{env_name} requires an experiment; pass experiment= to Trainer'
    raise ConfigError(msg)
  if env is not None:
    assert experiment is not None
    ctx_mgr = env.activate(experiment, trainer.store, module)
  else:
    ctx_mgr = nullcontext(Path.cwd())
  with ctx_mgr as wt_path:
    rebound = trainer_setup.bind_path_params(trainer, module, Path(wt_path))
    try:
      trainer.dispatch_callbacks('setup', stage=Stage.fit)
      trainer_setup.ensure_agent_optimizer_context(trainer, module)
      trainer.dispatch_callbacks('on_fit_start')
      trainer.on_loop_start(max_epochs=max_epochs)
      trainer._run_sanity_check(module, loop_config)
      return trainer._loop.run(trainer, loop_config)
    finally:
      for param in rebound:
        param.unbind()


def teardown_fit(
  module: AutoPilotModule,
  datamodule: DataModule | None,
) -> None:
  """Teardown module and datamodule after fit completes or fails.

  Args:
    module: Module to tear down.
    datamodule: Optional datamodule to tear down for the fit stage.
  """
  module.teardown()
  if datamodule is not None:
    datamodule.teardown(ensure_stage(Stage.fit))


def execute_fit(
  trainer: Any,
  module: AutoPilotModule,
  train_dataloaders: Any | None,
  val_dataloaders: Any | None,
  datamodule: DataModule | None,
  max_epochs: int,
  ctx: dict[str, Any] | None,
  *,
  ckpt_path: Path | str | None = None,
  checkpoint_io: CheckpointIO | None = None,
  test_dataloaders: Any | None = None,
) -> dict[str, Any]:
  """Run training: setup, checkpoint resume, loop, success/failure handling, teardown.

  Args:
    trainer: Trainer instance.
    module: Module to train.
    train_dataloaders: Training data iterable or None.
    val_dataloaders: Validation iterable or None.
    datamodule: Optional ``DataModule``.
    max_epochs: Epoch budget.
    ctx: Optional ``fit_context`` dict.
    ckpt_path: Resume path or token.
    checkpoint_io: Checkpoint storage backend.
    test_dataloaders: Optional test iterable.

  Returns:
    Loop result dict, optionally including ``test_results``.

  Raises:
    ConfigError: When the experiment has been invalidated.
  """
  trainer._datamodule = datamodule
  (train_loader, val_loader, test_loader, optimizer, loss_fn, metrics, metric_metadata) = (
    trainer_setup.setup_fit_context(
      trainer,
      module,
      datamodule,
      train_dataloaders,
      val_dataloaders,
      ctx,
      test_dataloaders,
    )
  )
  trainer_setup.attach_default_callbacks(trainer)
  min_epoch = trainer_ckpt.resolve_checkpoint_resume(
    trainer,
    trainer_ckpt.resolve_ckpt_path_token(trainer, ckpt_path),
    checkpoint_io,
    module,
  )
  if trainer._experiment is not None and trainer._experiment.status == Status.invalidated:
    msg = (
      f'cannot resume training on invalidated experiment '
      f'{trainer._experiment.id!r}. Create a new experiment or revert '
      f'the invalidation before resuming.'
    )
    raise ConfigError(msg)
  loop_config = trainer_setup.build_loop_config(
    trainer,
    module,
    train_loader=train_loader,
    val_loader=val_loader,
    max_epochs=max_epochs,
    min_epoch=min_epoch,
    fit_ctx=trainer._fit_ctx,
    optimizer=optimizer,
    loss_fn=loss_fn,
    metrics=metrics,
    metric_metadata=metric_metadata,
  )
  exp_ctx = trainer._experiment if trainer._experiment is not None else nullcontext()
  with exp_ctx:
    try:
      loop_result = run_fit_loop(trainer, max_epochs, loop_config)
      result = trainer._fit_success_path(loop_result, loop_config)
      test_results = trainer_eval.run_test_phase(trainer, module, datamodule, test_loader)
      if test_results is not None:
        result['test_results'] = test_results
    except Exception as exc:
      trainer.dispatch_callbacks('on_exception', exception=exc)
      trainer._fit_failure_path(exc)
      raise
    else:
      return result
    finally:
      trainer._teardown_fit(module, datamodule)
      journal.write_profiler_summary(trainer)
      trainer.dispatch_callbacks('teardown', stage=Stage.fit)
      forest = trainer.forest
      if forest is not None:
        forest.save()
