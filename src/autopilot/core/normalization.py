"""Result normalization across splits.

Single codepath for converting observations into split-specific
summary files ({split}_summary.json). Split names are project-defined.
"""

from autopilot.core.errors import TrackingError
from autopilot.core.models import Datum
from pathlib import Path
from typing import Any
import json


def normalize_observation(observation: Datum) -> dict[str, Any]:
  """Convert an observation into a normalized summary dict."""
  return {
    'split': observation.split,
    'epoch': observation.epoch,
    'success': observation.success,
    'error_message': observation.error_message,
    'metrics': dict(observation.metrics),
    'metadata': dict(observation.metadata),
  }


def normalize_to_split_summary(
  observation: Datum,
  split: str,
  extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Create a split summary from an observation, merging extra metadata."""
  summary = normalize_observation(observation)
  summary['split'] = split
  if extra:
    summary.update(extra)
  return summary


def save_split_summary(
  experiment_dir: Path,
  split: str,
  summary: dict[str, Any],
) -> Path:
  """Write a split summary to the experiment directory."""
  path = experiment_dir / f'{split}_summary.json'
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    path.write_text(
      json.dumps(summary, indent=2) + '\n',
      encoding='utf-8',
    )
  except OSError as exc:
    raise TrackingError(f'failed to write summary at {path}: {exc}') from exc
  return path


def load_split_summary(
  experiment_dir: Path,
  split: str,
) -> dict[str, Any]:
  """Load a split summary from the experiment directory."""
  path = experiment_dir / f'{split}_summary.json'
  if not path.is_file():
    raise TrackingError(f'summary not found: {path}')
  try:
    return json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as exc:
    raise TrackingError(f'invalid summary JSON at {path}: {exc}') from exc


def save_result(
  experiment_dir: Path,
  result_dict: dict[str, Any],
) -> Path:
  """Write the result.json file."""
  path = experiment_dir / 'result.json'
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    path.write_text(
      json.dumps(result_dict, indent=2) + '\n',
      encoding='utf-8',
    )
  except OSError as exc:
    raise TrackingError(f'failed to write result at {path}: {exc}') from exc
  return path


def load_result(experiment_dir: Path) -> dict[str, Any] | None:
  """Load result.json from an experiment directory, or None if missing."""
  path = experiment_dir / 'result.json'
  if not path.is_file():
    return None
  try:
    return json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError as exc:
    raise TrackingError(f'invalid result JSON at {path}: {exc}') from exc


def save_promotion(
  experiment_dir: Path,
  promotion_dict: dict[str, Any],
) -> Path:
  """Write the promotion.json file."""
  path = experiment_dir / 'promotion.json'
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    path.write_text(
      json.dumps(promotion_dict, indent=2) + '\n',
      encoding='utf-8',
    )
  except OSError as exc:
    raise TrackingError(f'failed to write promotion at {path}: {exc}') from exc
  return path
