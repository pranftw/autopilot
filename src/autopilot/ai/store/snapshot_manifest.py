"""Snapshot manifest I/O and per-branch epoch listing for FileStore."""

from autopilot.ai.store.store_io import atomic_write_json_safe, reraise_as_store_error
from autopilot.core.config import EPOCH_SNAPSHOT_RE
from autopilot.core.errors import StoreError, TrackingError
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.types import SnapshotEntry
from autopilot.tracking.io import read_json_dict
from pathlib import Path
from typing import Any


def enumerate_snapshot_epochs(exp_dir: Path) -> list[int]:
  """List existing epoch numbers from snapshot files, sorted ascending.

  Uses the same ``epoch_<n>.json`` regex as stabilize to ignore
  non-epoch files.

  Returns:
    Sorted list of epoch integers found on disk.
  """
  epochs: list[int] = []
  for path in exp_dir.iterdir():
    if not path.is_file():
      continue
    match = EPOCH_SNAPSHOT_RE.match(path.name)
    if match is not None:
      epochs.append(int(match.group(1)))
  return sorted(epochs)


def load_snapshot_manifest(store: Any, experiment_id: str, epoch: int) -> SnapshotManifest:
  """Loads a snapshot manifest from disk.

  Args:
    store: FileStore instance.
    experiment_id: Branch that owns the snapshot.
    epoch: Epoch index to load.

  Returns:
    Deserialized SnapshotManifest.

  Raises:
    StoreError: If the snapshot file is missing.
  """
  path = store._snapshots_dir / experiment_id / f'epoch_{epoch}.json'
  if not path.exists():
    msg = f'snapshot not found: {experiment_id} epoch {epoch}'
    raise StoreError(msg)
  try:
    data = read_json_dict(path, f'snapshot {experiment_id} epoch {epoch}')
  except TrackingError as exc:
    reraise_as_store_error(exc, str(exc))
  return SnapshotManifest.from_dict(data)


def persist_snapshot_manifest(
  store: Any,
  experiment_id: str,
  epoch: int,
  manifest: SnapshotManifest,
) -> None:
  """Write a snapshot manifest to disk at the given epoch path.

  Args:
    store: FileStore instance.
    experiment_id: Branch that owns the snapshot.
    epoch: Epoch index for the manifest.
    manifest: Manifest to persist.
  """
  path = store._snapshots_dir / experiment_id / f'epoch_{epoch}.json'
  atomic_write_json_safe(path, manifest.to_dict())


def experiment_log(store: Any, experiment_id: str) -> list[SnapshotEntry]:
  """Return chronological snapshot history for an experiment branch.

  Args:
    store: FileStore instance.
    experiment_id: Branch to list history for.

  Returns:
    List of SnapshotEntry for each existing epoch.

  Raises:
    StoreError: When the branch does not exist in refs.
  """
  refs = store.load_refs()
  branches = refs.get('branches', {})
  if experiment_id not in branches:
    msg = f'experiment {experiment_id!r} not found'
    raise StoreError(msg)

  exp_dir = store._snapshots_dir / experiment_id
  if not exp_dir.exists():
    return []

  epochs = enumerate_snapshot_epochs(exp_dir)
  entries: list[SnapshotEntry] = []
  for ep in epochs:
    snap = load_snapshot_manifest(store, experiment_id, ep)
    entries.append(
      SnapshotEntry(
        epoch=snap.epoch,
        timestamp=snap.timestamp,
        file_count=len(snap.entries),
      )
    )
  return entries
