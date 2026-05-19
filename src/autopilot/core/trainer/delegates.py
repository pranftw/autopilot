"""Trainer hook implementations delegated from :class:`~autopilot.core.trainer.trainer.Trainer`.

This mixin holds thin wrappers around ``checkpoint``, ``eval``, ``fit_loop``, ``journal``,
and ``setup`` helpers so :class:`~autopilot.core.trainer.trainer.Trainer` stays a slim shell.
"""

from autopilot.core.checkpoint import CheckpointIO
from autopilot.core.loops.loop import LoopConfig
from autopilot.core.metric import Metric
from autopilot.core.module.autopilot_module import AutoPilotModule
from autopilot.core.module.module import Module
from autopilot.core.trainer import checkpoint as trainer_ckpt
from autopilot.core.trainer import eval as trainer_eval
from autopilot.core.trainer import fit_loop as trainer_fit_loop
from autopilot.core.trainer import journal
from autopilot.core.trainer import setup as trainer_setup
from autopilot.data.datamodule import DataModule
from pathlib import Path
from typing import Any


class TrainerDelegates:
  """Mixin of ``Trainer`` methods that forward to sibling helper modules."""

  def save_checkpoint(self, path: Path, checkpoint_io: CheckpointIO | None = None) -> None:
    """Persist assembled trainer state via ``checkpoint.save_checkpoint``."""
    trainer_ckpt.save_checkpoint(self, path, checkpoint_io)

  def emit_context(
    self,
    reason: str,
    *,
    source: str | None = None,
    metadata: dict[str, Any] | None = None,
  ) -> None:
    """Emit context via ``journal.emit_context``."""
    journal.emit_context(self, reason, source=source, metadata=metadata)

  def capture_gradient_summaries(self) -> None:
    """Snapshot gradients via ``journal.capture_gradient_summaries``."""
    journal.capture_gradient_summaries(self)

  def run_eval_phase(
    self,
    module: AutoPilotModule,
    dataloader: Any,
    *,
    step_method: str,
    hook_prefix: str,
    max_batches: int | None = None,
    epoch_arg: int = 0,
  ) -> dict[str, float]:
    """Run one eval phase; see ``eval.run_eval_phase``."""
    return trainer_eval.run_eval_phase(
      self,
      module,
      dataloader,
      step_method=step_method,
      hook_prefix=hook_prefix,
      max_batches=max_batches,
      epoch_arg=epoch_arg,
    )

  def validate(
    self,
    module: AutoPilotModule,
    dataloaders: Any | None = None,
    datamodule: DataModule | None = None,
  ) -> dict[str, float]:
    """Standalone validation; see ``eval.validate``."""
    return trainer_eval.validate(self, module, dataloaders, datamodule)

  def test(
    self,
    module: AutoPilotModule,
    dataloaders: Any | None = None,
    datamodule: DataModule | None = None,
  ) -> dict[str, float]:
    """Standalone test; see ``eval.test``."""
    return trainer_eval.test(self, module, dataloaders, datamodule)

  def predict(
    self,
    module: AutoPilotModule,
    dataloaders: Any | None = None,
    datamodule: DataModule | None = None,
  ) -> list[Any]:
    """Prediction loop; see ``eval.predict``."""
    return trainer_eval.predict(self, module, dataloaders, datamodule)

  def _prepare_datamodule_fit(self, datamodule: DataModule | None) -> None:
    """Delegate to ``setup.prepare_datamodule_fit``."""
    trainer_setup.prepare_datamodule_fit(datamodule)

  def _resolve_fit_loaders(
    self,
    train_dataloaders: Any,
    val_dataloaders: Any,
    datamodule: DataModule | None,
    test_dataloaders: Any | None = None,
  ) -> tuple[Any, Any, Any]:
    """Delegate to ``setup.resolve_fit_loaders``."""
    return trainer_setup.resolve_fit_loaders(
      train_dataloaders, val_dataloaders, datamodule, test_dataloaders
    )

  def _collect_module_metrics(self, module: Module) -> tuple[dict[str, Metric], dict[str, bool]]:
    """Delegate to ``eval.collect_module_metrics``."""
    return trainer_eval.collect_module_metrics(module)

  def _configure_optimizer_and_metrics(
    self, module: AutoPilotModule
  ) -> tuple[Any, Any | None, dict[str, Metric], dict[str, bool]]:
    """Delegate to ``setup.configure_optimizer_and_metrics``."""
    return trainer_setup.configure_optimizer_and_metrics(self, module)

  def _setup_fit_context(
    self,
    module: AutoPilotModule,
    datamodule: DataModule | None,
    train_dataloaders: Any | None,
    val_dataloaders: Any | None,
    ctx: dict[str, Any] | None,
    test_dataloaders: Any | None = None,
  ) -> tuple[Any, Any, Any, Any, Any | None, dict[str, Metric], dict[str, bool]]:
    """Delegate to ``setup.setup_fit_context``."""
    return trainer_setup.setup_fit_context(
      self, module, datamodule, train_dataloaders, val_dataloaders, ctx, test_dataloaders
    )

  def _build_loop_config(
    self,
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
    """Delegate to ``setup.build_loop_config``."""
    return trainer_setup.build_loop_config(
      self,
      module,
      train_loader=train_loader,
      val_loader=val_loader,
      max_epochs=max_epochs,
      min_epoch=min_epoch,
      fit_ctx=fit_ctx,
      optimizer=optimizer,
      loss_fn=loss_fn,
      metrics=metrics,
      metric_metadata=metric_metadata,
    )

  def _ensure_agent_optimizer_context(self, module: Module) -> None:
    """Delegate to ``setup.ensure_agent_optimizer_context``."""
    trainer_setup.ensure_agent_optimizer_context(self, module)

  def _make_optimizer_context(self, module: Module) -> dict[str, Any]:
    """Delegate to ``setup.make_optimizer_context``."""
    return trainer_setup.make_optimizer_context(self, module)

  def _attach_default_callbacks(self) -> None:
    """Delegate to ``setup.attach_default_callbacks``."""
    trainer_setup.attach_default_callbacks(self)

  def _build_checkpoint_state(self) -> dict[str, Any]:
    """Delegate to ``checkpoint.build_checkpoint_state``."""
    return trainer_ckpt.build_checkpoint_state(self)

  def _restore_from_checkpoint(self, state: dict[str, Any], module: AutoPilotModule) -> None:
    """Delegate to ``checkpoint.restore_from_checkpoint``."""
    trainer_ckpt.restore_from_checkpoint(self, state, module)

  def _restore_path_parameter_files(self, state: dict[str, Any], module: AutoPilotModule) -> None:
    """Delegate to ``checkpoint.restore_path_parameter_files``."""
    trainer_ckpt.restore_path_parameter_files(self, state, module)

  def _resolve_checkpoint_resume(
    self,
    ckpt_path: Path | None,
    checkpoint_io: CheckpointIO | None,
    module: AutoPilotModule,
  ) -> int:
    """Delegate to ``checkpoint.resolve_checkpoint_resume``."""
    return trainer_ckpt.resolve_checkpoint_resume(self, ckpt_path, checkpoint_io, module)

  def _resolve_ckpt_path_token(self, ckpt_path: Path | str | None) -> Path | None:
    """Delegate to ``checkpoint.resolve_ckpt_path_token``."""
    return trainer_ckpt.resolve_ckpt_path_token(self, ckpt_path)

  def _find_checkpoint_callbacks(self) -> list:
    """Delegate to ``checkpoint.find_checkpoint_callbacks``."""
    return trainer_ckpt.find_checkpoint_callbacks(self)

  def _resolve_last_checkpoint(self) -> Path:
    """Delegate to ``checkpoint._resolve_last_checkpoint``."""
    return trainer_ckpt._resolve_last_checkpoint(self)

  def _resolve_best_checkpoint(self) -> Path:
    """Delegate to ``checkpoint._resolve_best_checkpoint``."""
    return trainer_ckpt._resolve_best_checkpoint(self)

  def _complete_experiment_success(self, loop_result: dict[str, Any]) -> None:
    """Delegate to ``journal.complete_experiment_success``."""
    journal.complete_experiment_success(self, loop_result)

  def _dispatch_module_eval_hook(self, module: Module, hook_prefix: str, phase: str) -> None:
    """Delegate to ``eval.dispatch_module_eval_hook``."""
    trainer_eval.dispatch_module_eval_hook(module, hook_prefix, phase)

  def _run_test_phase(
    self,
    module: AutoPilotModule,
    datamodule: DataModule | None,
    test_loader: Any,
  ) -> dict[str, float] | None:
    """Delegate to ``eval.run_test_phase``."""
    return trainer_eval.run_test_phase(self, module, datamodule, test_loader)

  def _predict_loop(self, module: AutoPilotModule, predict_loader: Any) -> list[Any]:
    """Delegate to ``eval.predict_loop``."""
    return trainer_eval.predict_loop(self, module, predict_loader)

  def emit_epoch_gradient_journal(self, *, epoch: int) -> None:
    """Delegate to ``journal.emit_epoch_gradient_journal``."""
    journal.emit_epoch_gradient_journal(self, epoch=epoch)

  def _emit_gradient_journal(self) -> None:
    """Delegate to ``journal._emit_gradient_journal``."""
    journal._emit_gradient_journal(self)

  def _write_profiler_summary(self) -> None:
    """Delegate to ``journal.write_profiler_summary``."""
    journal.write_profiler_summary(self)

  def _profile_store_section(self, action: str):
    """Delegate to ``journal.profile_store_section``."""
    return journal.profile_store_section(self, action)

  def _attach_dataset_fingerprint(self) -> None:
    """Delegate to ``journal._attach_dataset_fingerprint``."""
    journal._attach_dataset_fingerprint(self)

  def _bind_path_parameters(self, module: Module, wt: Path) -> list:
    """Delegate to ``setup.bind_path_params``."""
    return trainer_setup.bind_path_params(self, module, wt)

  def _run_sanity_check(self, module: AutoPilotModule, loop_config: LoopConfig) -> None:
    """Capped pre-train validation; see ``fit_loop.run_sanity_check``."""
    trainer_fit_loop.run_sanity_check(self, module, loop_config)

  def _run_fit_loop(self, max_epochs: int, loop_config: LoopConfig) -> dict[str, Any]:
    """Environment bind + epoch loop; see ``fit_loop.run_fit_loop``."""
    return trainer_fit_loop.run_fit_loop(self, max_epochs, loop_config)

  def _fit_success_path(
    self,
    loop_result: dict[str, Any],
    loop_config: LoopConfig,
  ) -> dict[str, Any]:
    """Success journaling and experiment completion; see ``journal.fit_success_path``."""
    return journal.fit_success_path(self, loop_result, loop_config)

  def _fit_failure_path(self, exc: Exception) -> None:
    """Failure journaling; see ``journal.fit_failure_path``."""
    journal.fit_failure_path(self, exc)

  def _teardown_fit(self, module: AutoPilotModule, datamodule: DataModule | None) -> None:
    """Module/datamodule teardown after ``fit``."""
    trainer_fit_loop.teardown_fit(module, datamodule)
