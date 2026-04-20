"""Checkpoint loading for experiments.

Reads manifest from experiment directory.
"""

from autopilot.core.config import load_json
from autopilot.core.models import Manifest
from pathlib import Path


def load_checkpoint(experiment_dir: Path) -> Manifest | None:
  """Load the current manifest as a checkpoint, or None if missing."""
  manifest_path = experiment_dir / 'manifest.json'
  if not manifest_path.exists():
    return None
  data = load_json(manifest_path)
  return Manifest.from_dict(data)
