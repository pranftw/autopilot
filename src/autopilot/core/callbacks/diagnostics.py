"""DiagnosticsCallback: produces diagnostic artifacts from batch data."""

from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.callbacks.callback import Callback
from autopilot.core.diagnostics import Diagnostics
from typing import Any


class DiagnosticsCallback(Callback):
  """Produces diagnostic artifacts from batch data.

  Composes a Diagnostics instance. Override Diagnostics hooks
  to customize analysis; the callback just orchestrates timing.
  """

  def __init__(self, diagnostics: Diagnostics) -> None:
    self._diagnostics = diagnostics

  @property
  def diagnostics(self) -> Diagnostics:
    return self._diagnostics

  def on_train_epoch_end(self, trainer: Any, epoch: int) -> None:
    data = DataArtifact().read_raw(self._diagnostics._dir, epoch=epoch)
    if not data:
      return
    result = self._diagnostics.analyze(data, epoch)
    self._diagnostics.write(result)

  def state_dict(self) -> dict[str, Any]:
    return {}

  def load_state_dict(self, state_dict: dict[str, Any]) -> None:
    pass
