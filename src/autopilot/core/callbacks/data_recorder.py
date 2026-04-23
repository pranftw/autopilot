"""DataRecorderCallback: records per-batch data to data.jsonl via artifact."""

from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.callbacks.callback import Callback
from autopilot.core.types import Datum
from pathlib import Path
from typing import Any


class DataRecorderCallback(ArtifactOwner, Callback):
  """Records per-batch data to data.jsonl via artifact.

  Override serialize_item() for custom serialization.
  """

  def __init__(self, experiment_dir: Path) -> None:
    self.__init_artifacts__()
    self._dir = experiment_dir
    self._batch_data: list[dict] = []
    self._current_epoch = 0
    self.data_artifact = DataArtifact()

  def serialize_item(self, data: Any) -> dict | None:
    """Override for custom serialization. Return None to skip."""
    if isinstance(data, Datum):
      return data.to_dict()
    if isinstance(data, dict):
      return data
    return None

  def on_train_epoch_start(self, trainer: Any, epoch: int) -> None:
    self._current_epoch = epoch
    self._batch_data = []

  def on_train_batch_end(
    self,
    trainer: Any,
    batch_idx: int = 0,
    data: Any = None,
  ) -> None:
    if data is not None:
      serialized = self.serialize_item(data)
      if serialized is not None:
        self._batch_data.append(serialized)

  def on_train_epoch_end(self, trainer: Any, epoch: int) -> None:
    for item in self._batch_data:
      self.data_artifact.append(item, self._dir, epoch=epoch)
