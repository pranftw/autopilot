"""Canonical load/save for experiment manifests."""

from autopilot.core.errors import TrackingError
from autopilot.core.models import Manifest
from autopilot.tracking.io import atomic_write_json
from pathlib import Path
import json


def _manifest_path(experiment_dir: Path) -> Path:
  return experiment_dir / 'manifest.json'


def load_manifest(experiment_dir: Path) -> Manifest:
  path = _manifest_path(experiment_dir)
  if not path.is_file():
    raise TrackingError(f'manifest not found: {path}')
  try:
    raw = path.read_text(encoding='utf-8')
    data = json.loads(raw)
  except OSError as exc:
    raise TrackingError(f'failed to read manifest at {path}: {exc}') from exc
  except json.JSONDecodeError as exc:
    raise TrackingError(f'invalid manifest JSON at {path}: {exc}') from exc
  if not isinstance(data, dict):
    raise TrackingError(f'manifest must be a JSON object at {path}')
  try:
    return Manifest.from_dict(data)
  except (TypeError, ValueError, KeyError) as exc:
    raise TrackingError(f'invalid manifest fields at {path}: {exc}') from exc


def save_manifest(experiment_dir: Path, manifest: Manifest) -> None:
  path = _manifest_path(experiment_dir)
  atomic_write_json(path, manifest.to_dict())
