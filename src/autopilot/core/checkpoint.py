"""Training checkpoint storage backends (CheckpointIO pattern).

Handles how trainer checkpoint dicts are persisted and retrieved.
``CheckpointIO`` is the abstract storage backend (like Lightning's CheckpointIO
plugin); ``JSONCheckpointIO`` is the default JSON-file builtin.

This is distinct from ``StoreCheckpointCallback`` (which versions parameters
via ``Store.snapshot`` at epoch boundaries) and from
``autopilot.ai.evaluation.checkpoints.CheckpointIO`` (which handles eval-pipeline
JSONL progress files). Full import paths disambiguate.
"""

from autopilot.tracking.io import atomic_write_json, read_json_dict
from pathlib import Path
from typing import Any


class CheckpointIO:
  """Storage backend for training checkpoint persistence.

  Like Lightning's CheckpointIO plugin: handles how checkpoint dicts are
  stored and retrieved. Subclass for remote storage, compression, etc.

  Methods:
    save(state, path)  -- write checkpoint state to storage
    load(path)         -- read checkpoint state from storage
    remove(path)       -- delete a checkpoint
    exists(path)       -- probe whether a checkpoint exists
  """

  def save(self, state: dict[str, Any], path: Path) -> None:
    """Write checkpoint state to storage.

    Args:
      state: Assembled trainer state (JSON-serializable dict).
      path: Destination path chosen by the trainer or callback.

    Raises:
      NotImplementedError: If the subclass does not implement persistence.
    """
    raise NotImplementedError

  def load(self, path: Path) -> dict[str, Any]:
    """Read checkpoint state from storage.

    Args:
      path: Source path.

    Returns:
      Checkpoint dict previously passed to :meth:`save`.

    Raises:
      NotImplementedError: If the subclass does not implement loading.
    """
    raise NotImplementedError

  def remove(self, path: Path) -> None:
    """Delete checkpoint at path.

    Args:
      path: Path to remove.

    Raises:
      NotImplementedError: If the subclass does not implement removal.
    """
    raise NotImplementedError

  def exists(self, path: Path) -> bool:
    """Return whether a checkpoint exists at path.

    Args:
      path: Path to probe.

    Raises:
      NotImplementedError: If the subclass does not implement the probe.
    """
    raise NotImplementedError


class JSONCheckpointIO(CheckpointIO):
  """JSON file persistence for training checkpoints (default builtin).

  Uses ``atomic_write_json`` for crash-safe writes and ``read_json_dict``
  for validated loading. Checkpoint files are plain JSON objects.
  """

  def save(self, state: dict[str, Any], path: Path) -> None:
    """Atomically write ``state`` as JSON.

    Delegates to ``atomic_write_json``; propagates ``TrackingError`` when
    the file cannot be written.

    Args:
      state: Checkpoint dict to persist.
      path: Destination file path.
    """
    atomic_write_json(path, state)

  def load(self, path: Path) -> dict[str, Any]:
    """Load and validate a JSON object dict from ``path``.

    Delegates to ``read_json_dict`` which raises ``TrackingError`` when
    the file is missing, corrupt, non-UTF-8, or not a JSON object.

    Args:
      path: Source file path.

    Returns:
      Parsed checkpoint dict.
    """
    return read_json_dict(path, 'checkpoint')

  def remove(self, path: Path) -> None:
    """Delete the checkpoint file.

    Args:
      path: File to remove. No-op if already absent.
    """
    path.unlink(missing_ok=True)

  def exists(self, path: Path) -> bool:
    """Return True when ``path`` is an existing regular file.

    Args:
      path: Path to probe.

    Returns:
      Whether the checkpoint file exists.
    """
    return path.is_file()
