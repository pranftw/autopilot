"""Dataset split validation and snapshotting.

Single codepath for all dataset operations.
"""

from autopilot.core.errors import DatasetError
from autopilot.core.models import DatasetEntry, DatasetSnapshot
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import autopilot.core.paths as paths


def resolve_split_path(
  workspace: Path,
  split: str,
  filename: str,
) -> Path:
  """Resolve the filesystem path for a dataset split file."""
  return paths.dataset_split(workspace, split, filename)


def validate_split_file(path: Path) -> DatasetEntry:
  """Validate that a split file exists and return its entry metadata."""
  if not path.exists():
    raise DatasetError(f'split file not found: {path}')
  content = path.read_bytes()
  lines = content.strip().split(b'\n') if content.strip() else []
  return DatasetEntry(
    name=path.stem,
    split=path.parent.name,
    path=str(path),
    format=path.suffix.lstrip('.') or 'jsonl',
    rows=len(lines),
    content_hash=sha256(content).hexdigest()[:16],
  )


def validate_dataset(
  workspace: Path,
  profile_config: dict[str, Any],
) -> list[DatasetEntry]:
  """Validate all dataset splits referenced by a profile config."""
  entries: list[DatasetEntry] = []
  splits_config = profile_config.get('datasets', {}).get('splits', {})
  for split_name, filename in splits_config.items():
    path = resolve_split_path(workspace, split_name, filename)
    entry = validate_split_file(path)
    entries.append(entry)
  return entries


def create_dataset_snapshot(entries: list[DatasetEntry]) -> DatasetSnapshot:
  """Create an immutable snapshot of dataset state."""
  return DatasetSnapshot(
    created_at=datetime.now(timezone.utc).isoformat(),
    entries=entries,
  )


def hash_split_file(path: Path) -> str:
  """Compute a truncated SHA-256 hash of a split file."""
  if not path.exists():
    raise DatasetError(f'file not found for hashing: {path}')
  return sha256(path.read_bytes()).hexdigest()[:16]
