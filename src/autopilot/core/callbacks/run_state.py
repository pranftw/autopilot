"""RunStateCallback: persists run state for crash detection and stop-reason forensics."""

from autopilot.core.artifacts.experiment import RunStateArtifact
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.callbacks.callback import Callback
from autopilot.core.models import Result
from pathlib import Path
from typing import Any
import time


class RunStateCallback(ArtifactOwner, Callback):
  """Persists run state for crash detection and stop-reason forensics.

  Writes run_state.json with status='running' on each epoch end.
  On loop end, updates with status='completed' and stop_reason.
  If the process dies, run_state.json will still say 'running'.
  """

  def __init__(self, experiment_dir: Path) -> None:
    self.__init_artifacts__()
    self._dir = experiment_dir
    self.run_state_artifact = RunStateArtifact()

  def on_epoch_end(
    self,
    trainer: Any,
    epoch: int,
    result: Result | None = None,
  ) -> None:
    self.run_state_artifact.write(
      {
        'epoch': epoch,
        'timestamp': time.time(),
        'status': 'running',
      },
      self._dir,
    )

  def on_loop_end(self, trainer: Any, result: dict[str, Any]) -> None:
    self.run_state_artifact.write(
      {
        'epoch': result.get('total_epochs', 0),
        'timestamp': time.time(),
        'status': 'completed',
        'stop_reason': result.get('stop_reason'),
        'last_good_epoch': result.get('last_good_epoch', 0),
      },
      self._dir,
    )

  def state_dict(self) -> dict[str, Any]:
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    pass
