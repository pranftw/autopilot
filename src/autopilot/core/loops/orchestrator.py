"""EpochOrchestrator -- extends EpochLoop with stop conditions and failure recovery.

Delegates to ``EpochLoop.run()`` via ``super()`` and overrides template hooks
(``_pre_run``, ``_should_stop_before_epoch``, ``_should_stop_after_epoch``,
``_build_run_result``) instead of duplicating the epoch iteration skeleton.
Does NOT re-implement ``_run_epoch()``.

Epochs are 0-based (0 to max_epochs-1), matching Lightning's current_epoch
convention.
"""

from autopilot.core.decision import DecisionEntry
from autopilot.core.errors import ConfigError, ExperimentError, OrchestratorError, StoreError
from autopilot.core.loops.epoch import EpochLoop
from autopilot.core.loops.loop import LoopConfig
from dataclasses import dataclass
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
  """Configuration for EpochOrchestrator stop conditions and rollback.

  ``monitor`` is required when ``plateau_window > 0`` (plateau detection
  enabled). A ``ConfigError`` is raised at construction time when
  ``plateau_window > 0`` and ``monitor is None``. When ``plateau_window == 0``
  (plateau disabled), ``monitor`` may safely be ``None``.

  The ``monitor`` key must match the **actual key present** in per-epoch
  metrics. When Trainer merges train/val splits, keys are prefixed
  (``train_*`` / ``val_*``), so use the prefixed name (e.g.
  ``'val_accuracy'``, not ``'accuracy'``). When only one split exists,
  metrics are unprefixed.

  Raises:
    ConfigError: When ``plateau_window > 0`` and ``monitor is None``.
  """

  auto_rollback: bool = True
  plateau_threshold: float = 0.01
  plateau_window: int = 3
  monitor: str | None = None

  def __post_init__(self) -> None:
    """Validate that ``monitor`` is set when plateau detection is enabled.

    Raises:
      ConfigError: When ``plateau_window > 0`` and ``monitor is None``.
    """
    if self.plateau_window > 0 and self.monitor is None:
      msg = (
        f'OrchestratorConfig.monitor is required when plateau_window > 0 '
        f'(got plateau_window={self.plateau_window}). Set monitor to a metric '
        f"name (e.g. 'val_accuracy', 'train_loss') or set plateau_window=0 "
        f'to disable plateau detection.'
      )
      raise ConfigError(msg)


class EpochOrchestrator(EpochLoop):
  """Extends ``EpochLoop`` with plateau detection, rollback, and stop-reason tracking.

  State (``_metric_history``, ``_last_good_epoch``) is reset at the start of
  each ``run()`` call via ``_pre_run``, so consecutive ``fit()`` calls on the
  same Trainer/Orchestrator instance get clean plateau detection.
  ``_last_good_epoch`` starts at -1 (since 0 is a valid epoch in 0-based
  convention).

  Overrides only the ``EpochLoop`` template hooks and ``run()`` (for dry-run
  short-circuit) -- does NOT re-implement ``_run_epoch()``.

  When plateau detection triggers, emits a context entry via
  ``trainer.emit_context()`` with ``source='plateau'`` and metadata produced
  by ``DecisionEntry.plateau_stop()``. Filter entries by
  ``entry.metadata.get('_type') == DecisionEntry.PLATEAU_STOP_TYPE``.

  ``stop_reason`` tokens (read-only property):

  - ``'callback_stop'``: a callback signaled early termination before an epoch.
  - ``'policy_fail'``: the policy gate rejected an epoch.
  - ``'plateau'``: the monitored metric plateaued within threshold/window.
  - ``None``: no early stop occurred (normal completion or not yet run).
  """

  def __init__(self, config: OrchestratorConfig | None = None) -> None:
    """Create orchestrator with optional stop-condition configuration.

    Args:
      config: Plateau and rollback settings; defaults when omitted.
    """
    self._config = config or OrchestratorConfig(plateau_window=0)
    self._last_good_epoch = -1
    self._metric_history: list[dict[str, float]] = []
    self._stop_reason: str | None = None

  @property
  def stop_reason(self) -> str | None:
    """The reason the loop stopped early, or ``None`` for normal completion.

    Recognized values: ``'callback_stop'``, ``'policy_fail'``, ``'plateau'``,
    or ``None`` (not yet run / completed all epochs without early stop).

    Returns:
      Stop reason string or ``None``.
    """
    return self._stop_reason

  def run(self, trainer: Any, config: LoopConfig) -> dict[str, Any]:
    """Run epochs with plateau / rollback orchestration and enriched loop result.

    Dry-run mode is handled here to return a plan without entering the epoch
    loop. All other cases delegate to ``super().run()`` which drives the
    shared iteration skeleton.

    Args:
      trainer: Trainer driving the inner epoch loop.
      config: Loop configuration shared with EpochLoop.

    Returns:
      Dict with ``epochs``, ``total_epochs``, ``stop_reason``, and ``last_good_epoch``.
    """
    if config.dry_run:
      return self._build_dry_run_plan(config)
    return super().run(trainer, config)

  def _pre_run(self, trainer: Any, config: LoopConfig) -> None:
    """Reset orchestrator state before the epoch loop begins.

    Args:
      trainer: Trainer driving the loop.
      config: Loop configuration for this run.
    """
    self._metric_history.clear()
    self._last_good_epoch = -1
    self._stop_reason = None

    if self._config.plateau_window == 0:
      logger.info('plateau detection is disabled (plateau_window=0).')

  def _should_stop_before_epoch(
    self,
    trainer: Any,
    epoch: int,
    config: LoopConfig,
  ) -> bool:
    """Check callback-based stop signal and record 'callback_stop' reason.

    Args:
      trainer: Trainer providing callback-based stop signals.
      epoch: Epoch index about to start.
      config: Loop configuration for this run.

    Returns:
      Whether to stop before this epoch.
    """
    if super()._should_stop_before_epoch(trainer, epoch, config):
      self._stop_reason = 'callback_stop'
      return True
    return False

  def _should_stop_after_epoch(
    self,
    trainer: Any,
    epoch: int,
    epoch_result: dict[str, Any],
    config: LoopConfig,
  ) -> bool:
    """Check policy failure, rollback, and plateau conditions after each epoch.

    Args:
      trainer: Trainer driving the loop.
      epoch: Epoch index that just completed.
      epoch_result: Result dict from ``_run_epoch``.
      config: Loop configuration for this run.

    Returns:
      Whether to stop after this epoch.
    """
    if super()._should_stop_after_epoch(trainer, epoch, epoch_result, config):
      self._stop_reason = 'policy_fail'
      return True

    experiment = config.experiment
    if experiment and experiment.should_rollback:
      self._handle_rollback(experiment, epoch)
    else:
      self._last_good_epoch = epoch
      metrics = epoch_result.get('metrics')
      if metrics:
        self._metric_history.append(metrics)

    stop_reason = self._check_stop_conditions(epoch_result, epoch)
    if stop_reason == 'plateau':
      monitor = self._config.monitor
      assert monitor is not None  # plateau_window > 0 guarantees monitor at config time
      values = self._plateau_window_values()
      trainer.emit_context(
        f'plateau detected after epoch {epoch}',
        source='plateau',
        metadata=DecisionEntry.plateau_stop(
          monitor,
          epoch,
          plateau_window=self._config.plateau_window,
          plateau_threshold=self._config.plateau_threshold,
          values=values,
        ),
      )
      self._stop_reason = stop_reason
      return True
    if stop_reason:
      self._stop_reason = stop_reason
      return True
    return False

  def _build_run_result(self, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the final result dict with orchestrator-specific fields.

    Args:
      results: Per-epoch result dicts collected during the loop.

    Returns:
      Dict with ``epochs``, ``total_epochs``, ``stop_reason``, ``last_good_epoch``.
    """
    base = super()._build_run_result(results)
    base['stop_reason'] = self._stop_reason
    base['last_good_epoch'] = self._last_good_epoch
    return base

  def _plateau_window_values(self) -> list[float]:
    """Return monitored metric values for the current plateau window."""
    window = self._config.plateau_window
    monitor = self._config.monitor
    if not monitor:
      return []
    recent = self._metric_history[-window:]
    return [m[monitor] for m in recent if monitor in m]

  def _check_stop_conditions(self, _epoch_result: dict, _epoch: int) -> str | None:
    """Return a stop reason string if any stop condition is met, else ``None``."""
    if self._detect_plateau(self._metric_history):
      return 'plateau'
    return None

  def _detect_plateau(self, metric_history: list[dict[str, float]]) -> bool:
    """Return True when the monitored metric has plateaued over the window.

    The ``monitor`` key must match post-merge metric names (``train_*`` /
    ``val_*`` prefixed when Trainer merges splits). Missing ``monitor``
    in any window epoch causes that check to return False (no plateau).
    """
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
    """Roll back to ``_last_good_epoch`` when auto-rollback is enabled.

    Raises:
      OrchestratorError: When the underlying ``experiment.rollback()`` fails
        due to a store or experiment error. Other exceptions propagate unwrapped.
    """
    if not self._config.auto_rollback:
      return
    if experiment.store and self._last_good_epoch >= 0:
      try:
        experiment.rollback(self._last_good_epoch)
      except (StoreError, ExperimentError) as exc:
        msg = f'rollback to epoch {self._last_good_epoch} failed: {exc}'
        raise OrchestratorError(msg) from exc

  def _build_dry_run_plan(self, config: LoopConfig) -> dict[str, Any]:
    """Return orchestrator-specific dry-run metadata without entering the loop."""
    return {
      'dry_run': True,
      'planned_epochs': config.max_epochs,
      'orchestrator_config': {
        'auto_rollback': self._config.auto_rollback,
        'plateau_threshold': self._config.plateau_threshold,
        'plateau_window': self._config.plateau_window,
        'monitor': self._config.monitor,
      },
      'epochs': [],
      'total_epochs': 0,
    }

  def __repr__(self) -> str:
    """Return representation including monitor and rollback settings."""
    return (
      f'EpochOrchestrator(monitor={self._config.monitor!r}, '
      f'auto_rollback={self._config.auto_rollback})'
    )
