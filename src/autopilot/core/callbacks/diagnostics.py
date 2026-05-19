"""DiagnosticsCallback: produces diagnostic artifacts from batch data.

Uses store.resolve_path() for output paths instead of experiment-relative paths only.
Path resolution chain: Callback -> Store -> Config.
"""

from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.callbacks.callback import Callback
from autopilot.core.diagnostics import Diagnostics
from typing import Any

_DIAGNOSTICS_SENTINEL = object()


class DiagnosticsCallback(Callback):
  """Produces diagnostic artifacts from batch data.

  Composes a Diagnostics instance. Override Diagnostics hooks
  to customize analysis; the callback just orchestrates timing.

  Resolves paths via store.resolve_path(experiment.id) when a store
  is available on the trainer. Falls back to the diagnostics instance's
  configured directory otherwise.

  ``diagnostics`` is required; omitting it raises ``TypeError`` with guidance.
  """

  def __init__(self, diagnostics: Diagnostics = _DIAGNOSTICS_SENTINEL) -> None:  # type: ignore[assignment]  # ty: ignore[invalid-parameter-default]
    """Attach a Diagnostics instance for artifact production.

    Args:
      diagnostics: Analyzer and writer invoked at epoch end.

    Raises:
      TypeError: When ``diagnostics`` is omitted.
    """
    if diagnostics is _DIAGNOSTICS_SENTINEL:
      msg = (
        'DiagnosticsCallback() missing required argument: diagnostics. '
        'Pass a Diagnostics instance '
        '(from autopilot.core.diagnostics.Diagnostics).'
      )
      raise TypeError(msg)
    self._diagnostics = diagnostics

  @property
  def diagnostics(self) -> Diagnostics:
    """Diagnostics backend used by this callback.

    Returns:
      The composed Diagnostics instance.
    """
    return self._diagnostics

  def on_train_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    """Read batch data, analyze, and write diagnostic artifacts for the epoch.

    Args:
      trainer: Active trainer (store path resolution).
      module: Training module (unused; API symmetry with Lightning).
      epoch: Zero-based epoch index.
    """
    base_dir = self._resolve_dir(trainer)
    data = DataArtifact().read_raw(base_dir, epoch=epoch)
    if not data:
      return
    result = self._diagnostics.analyze(data, epoch)
    self._diagnostics.output_dir = base_dir
    self._diagnostics.write(result)

  def _resolve_dir(self, trainer: Any) -> Any:
    """Resolve the output directory via store or diagnostics fallback.

    Returns:
      Experiment output path from the store when available, else diagnostics.output_dir.
    """
    experiment = trainer.experiment
    store = trainer.store
    if store is not None and experiment is not None:
      return store.resolve_path(experiment.id)
    return self._diagnostics.output_dir

  def state_dict(self) -> dict[str, Any]:
    """Diagnostics state is not checkpointed.

    Returns:
      Empty dict for trainer checkpoint assembly.
    """
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """No-op: diagnostics callback keeps no serializable state.

    Args:
      state_dict: Ignored; this callback has no persistent state.
    """
