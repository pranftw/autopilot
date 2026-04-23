"""EpochOrchestrator -- extends EpochLoop with stop conditions and failure recovery.

ONLY overrides run() -- does NOT re-implement _run_epoch().
Each epoch delegates to super()._run_epoch() for batch logic,
callback dispatch, metric computation, and validation.
"""

from autopilot.core.errors import OrchestratorError
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import LoopConfig
from dataclasses import dataclass
from typing import Any


@dataclass
class OrchestratorConfig:
  """Configuration for EpochOrchestrator stop conditions and rollback."""

  auto_rollback: bool = True
  plateau_threshold: float = 0.01
  plateau_window: int = 3
  strategy: str = 'conservative'
  monitor: str | None = None


class EpochOrchestrator(EpochLoop):
  """Extends EpochLoop with stop conditions and failure recovery.

  ONLY overrides run() -- does NOT re-implement _run_epoch().
  """

  def __init__(self, config: OrchestratorConfig | None = None) -> None:
    self._config = config or OrchestratorConfig()
    self._last_good_epoch = 0
    self._metric_history: list[dict[str, float]] = []

  def run(self, trainer: Any, config: LoopConfig) -> dict[str, Any]:
    if config.dry_run:
      return self._build_dry_run_plan(config)

    experiment = config.experiment
    results: list[dict[str, Any]] = []
    stop_reason: str | None = None

    for epoch in range(1, config.max_epochs + 1):
      if trainer.should_stop_at(trainer.on_epoch_start, epoch=epoch):
        stop_reason = 'callback_stop'
        break

      epoch_result = self._run_epoch(trainer, epoch, config)
      results.append(epoch_result)

      cb_result = self._build_callback_result(epoch_result)
      trainer.on_epoch_end(epoch=epoch, result=cb_result)

      if epoch_result.get('stopped'):
        stop_reason = 'policy_fail'
        break

      if experiment and experiment.should_rollback:
        self._handle_rollback(experiment, epoch)
      else:
        self._last_good_epoch = epoch
        metrics = epoch_result.get('metrics', {})
        if metrics:
          self._metric_history.append(metrics)

      stop_reason = self._check_stop_conditions(epoch_result, epoch)
      if stop_reason:
        break

    return {
      'epochs': results,
      'total_epochs': len(results),
      'stop_reason': stop_reason,
      'last_good_epoch': self._last_good_epoch,
    }

  def _check_stop_conditions(self, _epoch_result: dict, _epoch: int) -> str | None:
    if self._detect_plateau(self._metric_history):
      return 'plateau'
    return None

  def _detect_plateau(self, metric_history: list[dict[str, float]]) -> bool:
    window = self._config.plateau_window
    if len(metric_history) < window:
      return False

    monitor = self._config.monitor
    if not monitor:
      return False

    recent = metric_history[-window:]
    values = [m.get(monitor) for m in recent]
    if any(v is None for v in values):
      return False

    max_val = max(values)
    min_val = min(values)
    if max_val == 0:
      return abs(max_val - min_val) < self._config.plateau_threshold
    return (max_val - min_val) / abs(max_val) < self._config.plateau_threshold

  def _handle_rollback(self, experiment: Any, _epoch: int) -> None:
    if not self._config.auto_rollback:
      return
    if experiment.store and self._last_good_epoch > 0:
      try:
        experiment.rollback(self._last_good_epoch)
      except Exception as exc:
        raise OrchestratorError(
          f'rollback to epoch {self._last_good_epoch} failed: {exc}',
        ) from exc

  def _build_dry_run_plan(self, config: LoopConfig) -> dict[str, Any]:
    return {
      'dry_run': True,
      'planned_epochs': config.max_epochs,
      'orchestrator_config': {
        'auto_rollback': self._config.auto_rollback,
        'plateau_threshold': self._config.plateau_threshold,
        'plateau_window': self._config.plateau_window,
        'strategy': self._config.strategy,
        'monitor': self._config.monitor,
      },
      'epochs': [],
      'total_epochs': 0,
    }

  def __repr__(self) -> str:
    return f'EpochOrchestrator(strategy={self._config.strategy!r})'
