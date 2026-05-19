"""DataRecorderCallback: records per-batch data to data.jsonl via artifact.

Uses store.resolve_path(experiment.id) for output paths when available.
Path resolution chain: Callback -> Store -> Config.
"""

from autopilot.core.artifacts.epoch import DataArtifact
from autopilot.core.artifacts.owner import ArtifactOwner
from autopilot.core.callbacks.callback import Callback
from autopilot.core.types import Datum
from pathlib import Path
from typing import Any


class DataRecorderCallback(ArtifactOwner, Callback):
  """Records per-batch data to data.jsonl via artifact.

  Override serialize_item() for custom serialization.
  Resolves paths via store.resolve_path(experiment.id) when available,
  falls back to path constructor argument.
  """

  def __init__(self, path: Path | None = None) -> None:
    """Initialize recorder and epoch-scoped data artifact binding.

    Args:
      path: Fallback experiment directory when the trainer has no store.
    """
    self.init_artifacts()
    self._dir = path
    self._batch_data: list[dict] = []
    self._current_epoch = 0
    self.data_artifact = DataArtifact()

  def serialize_item(self, data: Any) -> dict | None:
    """Override for custom serialization.

    EvalDatum and other Datum subclasses serialize via to_dict(), which
    includes all subclass-specific fields (e.g. success, metrics, metadata).

    Args:
      data: Batch item from the training loop.

    Returns:
      JSON-serializable dict for recording, or None to skip this item.
    """
    if isinstance(data, Datum):
      return data.to_dict()
    if isinstance(data, dict):
      return data
    return None

  def on_train_epoch_start(self, trainer: Any, module: Any, epoch: int) -> None:
    """Reset batch buffer at the start of a training epoch.

    Args:
      trainer: Active trainer (unused; API symmetry).
      module: Training module (unused; API symmetry).
      epoch: Zero-based epoch index.
    """
    self._current_epoch = epoch
    self._batch_data = []

  def on_train_batch_end(
    self,
    trainer: Any,
    module: Any,
    batch_idx: int = 0,
    data: Any = None,
  ) -> None:
    """Append serialized batch data when the trainer supplies ``data``.

    Args:
      trainer: Active trainer (unused; API symmetry).
      module: Training module (unused; API symmetry).
      batch_idx: Index of the batch within the epoch.
      data: Optional batch payload passed through ``serialize_item``.
    """
    if data is not None:
      serialized = self.serialize_item(data)
      if serialized is not None:
        self._batch_data.append(serialized)

  def on_train_epoch_end(self, trainer: Any, module: Any, epoch: int) -> None:
    """Flush buffered batch rows to ``data.jsonl`` for the epoch.

    Args:
      trainer: Active trainer (store path resolution).
      module: Training module (unused; API symmetry).
      epoch: Zero-based epoch index.
    """
    base_dir = self._resolve_dir(trainer)
    if base_dir is None:
      return
    for item in self._batch_data:
      self.data_artifact.append(item, base_dir, epoch=epoch)

  def _resolve_dir(self, trainer: Any) -> Path | None:
    experiment = trainer.experiment
    store = trainer.store
    if store is not None and experiment is not None:
      return store.resolve_path(experiment.id)
    return self._dir
