"""CheckpointCallback: saves assembled trainer state at epoch end.

Persists full trainer/component state (experiment, module, optimizer,
callbacks) for training resumption via ``Trainer.fit(ckpt_path=...)``.

This is distinct from ``StoreCheckpointCallback`` which versions parameters
via ``Store.snapshot`` at epoch boundaries. ``CheckpointCallback`` targets
full training-loop resumption; ``StoreCheckpointCallback`` targets parameter
version control.

Resume token support:
  ``Trainer.fit(module, ckpt_path='last')`` resolves to the latest-epoch
  checkpoint saved by this callback.
  ``Trainer.fit(module, ckpt_path='best')`` resolves to the checkpoint with
  the highest monitored metric value (requires ``monitor=`` on construction).

Metric comparison uses higher-is-better semantics (matching Lightning's
default). Ties are broken by epoch index (later epoch wins).
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.checkpoint import CheckpointIO, JSONCheckpointIO
from pathlib import Path
from typing import Any

_CHECKPOINT_DIRECTORY_SENTINEL = object()


class CheckpointCallback(Callback):
  """Saves assembled trainer state after each training epoch.

  Uses ``Trainer.save_checkpoint`` to persist the full checkpoint dict
  (experiment, module, optimizer, callbacks) to a per-epoch JSON file.

  Runs on the ``on_train_epoch_end`` hook, which fires after validation
  in this framework's hook ordering.

  ``directory`` is required; omitting it raises ``TypeError`` with guidance.

  Resume token tracking:
    - ``last_checkpoint_path``: path to the checkpoint saved at the highest
      completed epoch index (deterministic, not filesystem mtime).
    - ``best_checkpoint_path``: path to the checkpoint with the highest
      monitored metric value (requires ``monitor`` to be set). Ties are
      broken by epoch index (later epoch wins).

  Args:
    directory: Directory for checkpoint files (parents created by atomic write).
    checkpoint_io: Optional storage backend (default ``JSONCheckpointIO``).
    monitor: Metric key to track for ``'best'`` checkpoint selection.
      Uses higher-is-better comparison. When ``None``, ``best_checkpoint_path``
      is always ``None`` and ``ckpt_path='best'`` raises ``ConfigError``.
  """

  def __init__(
    self,
    directory: Path = _CHECKPOINT_DIRECTORY_SENTINEL,  # type: ignore[assignment]  # ty: ignore[invalid-parameter-default]
    checkpoint_io: CheckpointIO | None = None,
    monitor: str | None = None,
  ) -> None:
    """Initialize checkpoint callback.

    Args:
      directory: Required directory for checkpoint files. ``None`` is rejected.
      checkpoint_io: Optional storage backend (default JSONCheckpointIO).
      monitor: Metric key to track for best-checkpoint selection.
        Higher values are considered better. When not set, ``'best'``
        resume token is unavailable.

    Raises:
      TypeError: When ``directory`` is omitted or ``None``.
    """
    if directory is _CHECKPOINT_DIRECTORY_SENTINEL or directory is None:
      msg = (
        'CheckpointCallback() missing required argument: directory. '
        'Pass directory=Path(...) to specify the checkpoint output directory.'
      )
      raise TypeError(msg)
    self._directory = directory
    self._checkpoint_io = checkpoint_io or JSONCheckpointIO()
    self._monitor = monitor
    self._last_checkpoint_path: Path | None = None
    self._best_checkpoint_path: Path | None = None
    self._best_metric_value: float | None = None

  @property
  def directory(self) -> Path:
    """Directory where epoch checkpoint files are written."""
    return self._directory

  @property
  def monitor(self) -> str | None:
    """Metric key being tracked for best checkpoint selection."""
    return self._monitor

  @property
  def last_checkpoint_path(self) -> Path | None:
    """Path to the most recently saved checkpoint (highest epoch index)."""
    return self._last_checkpoint_path

  @property
  def best_checkpoint_path(self) -> Path | None:
    """Path to the checkpoint with the highest monitored metric value."""
    return self._best_checkpoint_path

  @property
  def best_metric_value(self) -> float | None:
    """Best monitored metric value seen so far."""
    return self._best_metric_value

  def on_train_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    """Save a checkpoint after epoch ``epoch`` completes.

    Updates ``last_checkpoint_path`` to the newly saved file. When
    ``monitor`` is set, compares the metric value against the current best
    and updates ``best_checkpoint_path`` / ``best_metric_value`` if the new
    value is greater than or equal to the previous best (ties favor later
    epochs).

    Args:
      trainer: Active trainer with ``save_checkpoint`` method.
      module: Training module (unused; API symmetry with Lightning).
      epoch: Zero-based epoch index.
    """
    path = self._directory / f'epoch-{epoch:04d}.json'
    trainer.save_checkpoint(path, self._checkpoint_io)
    self._last_checkpoint_path = path
    self._update_best(trainer, epoch, path)

  def _update_best(self, trainer: Any, epoch: int, path: Path) -> None:
    """Update best checkpoint tracking based on monitored metric.

    Higher-is-better comparison. Ties (same value at different epochs) are
    broken by epoch index -- later epoch wins (deterministic).

    Args:
      trainer: Trainer providing access to loop result metrics.
      epoch: Current epoch index.
      path: Path to the just-saved checkpoint.
    """
    if self._monitor is None:
      return
    metric_value = self._extract_metric(trainer, epoch)
    if metric_value is None:
      return
    if self._best_metric_value is None or metric_value >= self._best_metric_value:
      self._best_metric_value = metric_value
      self._best_checkpoint_path = path

  def _extract_metric(self, trainer: Any, epoch: int) -> float | None:
    """Extract the monitored metric from the trainer's loop state.

    Looks into the loop's epoch results for the monitored key. Falls back
    to checking merged train/val metrics available via loop internals.

    Args:
      trainer: Active trainer.
      epoch: Current epoch index.

    Returns:
      The metric value as a float, or ``None`` if not found.
    """
    loop = getattr(trainer, 'loop', None)
    if loop is None:
      return None
    last_result = getattr(loop, '_last_epoch_metrics', None)
    if last_result is not None and self._monitor in last_result:
      return float(last_result[self._monitor])
    return None

  def state_dict(self) -> dict[str, Any]:
    """Serialize checkpoint tracking state for cross-process resume.

    Returns:
      Dict with best/last path and metric value. Empty when nothing tracked yet.
    """
    state: dict[str, Any] = {}
    if self._last_checkpoint_path is not None:
      state['last_checkpoint_path'] = str(self._last_checkpoint_path)
    if self._best_checkpoint_path is not None:
      state['best_checkpoint_path'] = str(self._best_checkpoint_path)
    if self._best_metric_value is not None:
      state['best_metric_value'] = self._best_metric_value
    return state

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Restore checkpoint tracking state from a persisted checkpoint.

    Args:
      state_dict: Dict produced by ``state_dict()``.
    """
    if 'last_checkpoint_path' in state_dict:
      self._last_checkpoint_path = Path(state_dict['last_checkpoint_path'])
    if 'best_checkpoint_path' in state_dict:
      self._best_checkpoint_path = Path(state_dict['best_checkpoint_path'])
    if 'best_metric_value' in state_dict:
      self._best_metric_value = state_dict['best_metric_value']
