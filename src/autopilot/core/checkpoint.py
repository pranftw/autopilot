"""Checkpoint base class and JSONCheckpoint. Like Lightning's CheckpointIO."""

from autopilot.core.models import Manifest
from autopilot.tracking.manifest import load_manifest as _load_manifest
from autopilot.tracking.manifest import save_manifest as _save_manifest
from pathlib import Path


class Checkpoint:
  """Base checkpoint persistence. Subclass for different backends."""

  def save_manifest(self, experiment_dir: Path, manifest: Manifest) -> None:
    raise NotImplementedError

  def load_manifest(self, experiment_dir: Path) -> Manifest | None:
    """Returns None if no manifest exists. Never raises for missing file."""
    raise NotImplementedError

  def exists(self, experiment_dir: Path) -> bool:
    raise NotImplementedError


class JSONCheckpoint(Checkpoint):
  """JSON file persistence. Default implementation."""

  def save_manifest(self, experiment_dir: Path, manifest: Manifest) -> None:
    _save_manifest(experiment_dir, manifest)

  def load_manifest(self, experiment_dir: Path) -> Manifest | None:
    path = experiment_dir / 'manifest.json'
    if not path.exists():
      return None
    return _load_manifest(experiment_dir)

  def exists(self, experiment_dir: Path) -> bool:
    return (experiment_dir / 'manifest.json').exists()
