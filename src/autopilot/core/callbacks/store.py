"""Store checkpoint callback: snapshots store at each epoch end.

Path resolution goes through Store (which uses Config internally).
Epoch number comes from the callback's epoch parameter passed by the
training loop, not from experiment.epoch or advance_epoch().

Context strings passed to ``store.snapshot(context=...)`` include the epoch
index and up to three metrics (sorted lexicographically by key, numeric values
only) for reflog human readability and agent traceability.
"""

from autopilot.core.callbacks.callback import Callback
from autopilot.core.models import Result
from typing import Any
import contextlib

MAX_CONTEXT_METRICS = 3


def _build_snapshot_context(epoch: int, result: Result | None) -> str:
  """Build a human-readable snapshot context string with epoch and metrics.

  Metrics are sorted lexicographically by key; only numeric values (int/float)
  are included. At most three entries are formatted as ``key=value`` tokens.

  Args:
    epoch: Zero-based epoch index.
    result: Optional Result containing metrics dict.

  Returns:
    Context string like ``'epoch 0 checkpoint (metrics: acc=0.9, loss=0.1)'``.
  """
  summary_parts: list[str] = []
  if result is not None and result.metrics:
    for key in sorted(result.metrics.keys()):
      value = result.metrics[key]
      if isinstance(value, (int, float)):
        summary_parts.append(f'{key}={value}')
      if len(summary_parts) == MAX_CONTEXT_METRICS:
        break
  summary = ', '.join(summary_parts) if summary_parts else ''
  if summary:
    return f'epoch {epoch} checkpoint (metrics: {summary})'
  return f'epoch {epoch} checkpoint'


class StoreCheckpointCallback(Callback):
  """Snapshots experiment parameters at each epoch end.

  Uses the epoch parameter from on_epoch_end(), NOT experiment.epoch.
  Gets store from trainer.store (public property).
  If trainer.store is None, skips silently.

  The snapshot ``context`` string carries the epoch index and up to three
  metrics (lexicographic key order, numeric values only) formatted as
  ``key=value`` tokens separated by ``, ``.
  """

  def __init__(self) -> None:
    """Initialize last-seen epoch bookkeeping for checkpoint restore."""
    self._last_epoch: int | None = None

  def on_epoch_end(
    self,
    trainer: Any,
    module: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    """Snapshot store parameters for the experiment at the given epoch.

    Skips silently when the experiment has ``should_rollback=True`` to
    avoid re-persisting rolled-back state.

    Args:
      trainer: Active trainer providing experiment and store.
      module: Training module (unused; API symmetry).
      epoch: Zero-based epoch index from the loop.
      result: Optional gate/epoch result with metrics for context string.
    """
    experiment = trainer.experiment
    if experiment is None:
      return
    if experiment.should_rollback:
      return
    store = trainer.store
    if store is None:
      return
    profiler = trainer.profiler
    if profiler is not None:
      with contextlib.suppress(ValueError, RuntimeError, OSError):
        profiler.start('store_snapshot')
    context = _build_snapshot_context(epoch, result)
    try:
      store.snapshot(
        experiment.id,
        epoch,
        experiment=experiment,
        force=True,
        context=context,
      )
    finally:
      if profiler is not None:
        with contextlib.suppress(ValueError, RuntimeError, OSError):
          profiler.stop('store_snapshot')
    self._last_epoch = epoch

  def state_dict(self) -> dict[str, Any]:
    """Expose last snapshotted epoch for checkpoint round-trips.

    Returns:
      Dict with key ``last_epoch`` when known, else None.
    """
    return {'last_epoch': self._last_epoch}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """Restore last-seen epoch from a checkpoint payload.

    Args:
      state_dict: Mapping containing ``last_epoch``.
    """
    self._last_epoch = state_dict['last_epoch']
