"""RunStateCallback: persists run state for crash detection and stop-reason forensics.

Uses Status enum values, not raw strings. Resolves paths via
store.resolve_path(experiment.id) when available.
"""

from autopilot.core.artifacts.experiment import RunStateArtifact
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.callbacks.callback import Callback
from autopilot.core.enums import Status
from autopilot.core.errors import ConfigError
from autopilot.core.models import Result
from pathlib import Path
from typing import Any
import time


class RunStateCallback(ArtifactOwner, Callback):
  """Persists run state for crash detection and stop-reason forensics.

  Writes run_state.json with status=Status.running on each epoch end.
  On loop end, updates with status=Status.completed and stop_reason.
  If the process dies, run_state.json will still say 'running'.

  Resolves paths via store.resolve_path(experiment.id) when available,
  falls back to path constructor argument.
  """

  def __init__(self, path: Path | None = None) -> None:
    """Initialize run state writer and artifact binding.

    Args:
      path: Fallback experiment directory when the trainer has no store.
    """
    self.init_artifacts()
    self._dir = path
    self.run_state_artifact = RunStateArtifact()

  def on_epoch_end(
    self,
    trainer: Any,
    module: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    """Write ``running`` run_state after each epoch completes.

    Args:
      trainer: Active trainer (store path resolution).
      module: Training module (unused; API symmetry).
      epoch: Zero-based epoch index.
      result: Optional epoch result dict passed by the loop.

    Raises:
      ConfigError: When run state directory is not available.
    """
    base_dir = self._resolve_dir(trainer)
    if base_dir is None:
      msg = (
        'run state directory not available;'
        ' attach RunStateCallback only when an experiment directory is configured'
      )
      raise ConfigError(msg)
    self.run_state_artifact.write(
      {
        'epoch': epoch,
        'timestamp': time.time(),
        'status': Status.running.value,
      },
      base_dir,
    )

  def on_loop_end(self, trainer: Any, module: Any, result: dict[str, Any]) -> None:
    """Finalize run_state with ``completed`` status and stop metadata.

    Args:
      trainer: Active trainer (store path resolution).
      module: Training module (unused; API symmetry).
      result: Loop result containing total epochs, stop reason, and last good epoch.

    Raises:
      ConfigError: When run state directory is not available.
    """
    base_dir = self._resolve_dir(trainer)
    if base_dir is None:
      msg = (
        'run state directory not available;'
        ' attach RunStateCallback only when an experiment directory is configured'
      )
      raise ConfigError(msg)
    # total_epochs: always present from EpochLoop and EpochOrchestrator
    total_epochs = result['total_epochs']
    # last_good_epoch: orchestrator-only; absent when loop ended without orchestrator metadata
    last_good_epoch = result.get('last_good_epoch')
    # stop_reason: orchestrator-only
    stop_reason = result.get('stop_reason')
    self.run_state_artifact.write(
      {
        'epoch': total_epochs,
        'timestamp': time.time(),
        'status': Status.completed.value,
        'stop_reason': stop_reason,
        'last_good_epoch': last_good_epoch,
      },
      base_dir,
    )

  def _resolve_dir(self, trainer: Any) -> Path | None:
    experiment = trainer.experiment
    store = trainer.store
    if store is not None and experiment is not None:
      return store.resolve_path(experiment.id)
    return self._dir

  def state_dict(self) -> dict[str, Any]:
    """Run state artifact holds durable state; callback has nothing extra.

    Returns:
      Empty dict for trainer checkpoint assembly.
    """
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    """No-op: nothing to restore beyond filesystem artifacts.

    Args:
      state_dict: Ignored; this callback has no persistent state.
    """
