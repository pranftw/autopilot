"""Reflog, stash, tags, worktrees, and copy_epoch operations for FileStore."""

from autopilot.ai.store import store_reflog as _store_reflog
from autopilot.ai.store.snapshot import build_snapshot, group_by_param
from autopilot.ai.store.store_io import atomic_write_json_safe
from autopilot.ai.store_lock import hash_content
from autopilot.core.errors import StoreError, TrackingError
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.types import TagEntry, validate_tag_name
from autopilot.tracking.io import (
  read_json_dict,
  utc_now_iso,
)
from pathlib import Path
from typing import Any
import json

append_reflog = _store_reflog.append_reflog
expire_reflog = _store_reflog.expire_reflog
iter_reflog = _store_reflog.iter_reflog
recover_from_reflog = _store_reflog.recover_from_reflog

STASH_INDEX_PAD_WIDTH = 4


def compute_manifest_digest(manifest: SnapshotManifest) -> str:
  """Compute SHA-256 digest of a manifest's canonical JSON representation."""
  canonical = json.dumps(manifest.to_dict(), sort_keys=True, separators=(',', ':'))
  return hash_content(canonical)


# --- stash ---


def stash(store: Any, context: str | None = None) -> SnapshotManifest:
  """Capture current registered parameter state as a numbered stash manifest.

  Args:
    store: FileStore instance.
    context: Optional reason string recorded on the manifest.

  Returns:
    The stash manifest that was persisted.
  """
  store._acquire_lock()
  try:
    manifest = build_snapshot(store, context=context)
    manifest.epoch = -1

    store._stash_dir.mkdir(parents=True, exist_ok=True)
    next_index = _next_stash_index(store)
    filename = f'{next_index:0{STASH_INDEX_PAD_WIDTH}d}.json'
    stash_path = store._stash_dir / filename
    atomic_write_json_safe(stash_path, manifest.to_dict())
    append_reflog(store, 'stash', '_', old_epoch=None, new_epoch=next_index, context=context)
  finally:
    store._release_lock()
  return manifest


def stash_list(store: Any) -> list[SnapshotManifest]:
  """Return stash manifests ordered oldest to newest (by stash index).

  Args:
    store: FileStore instance.

  Returns:
    List of SnapshotManifest instances, ordered by stash index ascending.

  Raises:
    StoreError: When a stash manifest file is corrupt or unparseable.
  """
  if not store._stash_dir.exists():
    return []
  indices = _scan_stash_indices(store)
  result: list[SnapshotManifest] = []
  for idx in indices:
    path = _stash_file_path(store, idx)
    try:
      data = read_json_dict(path, f'stash {idx}')
      manifest = SnapshotManifest.from_dict(data)
      result.append(manifest)
    except (TrackingError, KeyError, ValueError) as exc:
      msg = f'corrupt stash manifest at {path}: {exc}'
      raise StoreError(msg) from exc
  return result


def stash_pop(
  store: Any, index: int | None = None, *, context: str | None = None
) -> SnapshotManifest:
  """Restore stash to working parameters and remove the stash file.

  Args:
    store: FileStore instance.
    index: Explicit stash index. When None, pop the newest stash (LIFO).
    context: Optional audit provenance string recorded in the reflog entry
      for this stash_pop operation.

  Returns:
    The manifest that was popped and restored.

  Raises:
    StoreError: If the stash stack is empty, index is out of range, or
      registered parameters are missing from the stash manifest.
  """
  store._acquire_lock()
  try:
    indices = _scan_stash_indices(store)
    if not indices:
      msg = 'stash stack is empty; nothing to pop'
      raise StoreError(msg)

    if index is None:
      pop_index = indices[-1]
    else:
      if index not in indices:
        msg = f'stash index {index} not found; available indices: {indices}'
        raise StoreError(msg)
      pop_index = index

    pop_path = _stash_file_path(store, pop_index)
    data = read_json_dict(pop_path, f'stash {pop_index}')
    manifest = SnapshotManifest.from_dict(data)

    grouped = group_by_param(store, manifest)
    missing_params = set(store._param_names) - set(grouped.keys())
    if missing_params:
      msg = (
        f'stash_pop: registered parameters {sorted(missing_params)!r} have no entry '
        f'in stash manifest. These parameters were likely registered after the stash '
        f'was created.'
      )
      raise StoreError(msg)
    for name, param in store._param_names.items():
      param.restore(grouped.get(name, {}))

    pop_path.unlink()

    _renumber_stash_files(store, pop_index)
    remaining = _scan_stash_indices(store)
    remaining_tip = remaining[-1] if remaining else None
    append_reflog(
      store, 'stash_pop', '_', old_epoch=pop_index, new_epoch=remaining_tip, context=context
    )
  finally:
    store._release_lock()
  return manifest


def _next_stash_index(store: Any) -> int:
  """Compute the next stash index from existing files."""
  indices = _scan_stash_indices(store)
  if not indices:
    return 0
  return indices[-1] + 1


def _scan_stash_indices(store: Any) -> list[int]:
  """Parse stash filenames into sorted integer indices."""
  if not store._stash_dir.exists():
    return []
  indices: list[int] = []
  for path in store._stash_dir.glob('*.json'):
    try:
      indices.append(int(path.stem))
    except ValueError:
      continue
  return sorted(indices)


def _stash_file_path(store: Any, index: int) -> Path:
  """Build the filesystem path for a stash file by index."""
  filename = f'{index:0{STASH_INDEX_PAD_WIDTH}d}.json'
  return store._stash_dir / filename


def _renumber_stash_files(store: Any, popped_index: int) -> None:
  """Renumber remaining stash files to stay dense after a pop."""
  indices = _scan_stash_indices(store)
  for old_idx in indices:
    if old_idx <= popped_index:
      continue
    new_idx = old_idx - 1
    old_path = _stash_file_path(store, old_idx)
    new_path = _stash_file_path(store, new_idx)
    old_path.rename(new_path)


# --- copy_epoch ---


def copy_epoch(
  store: Any,
  source_experiment_id: str,
  source_epoch: int,
  target_experiment_id: str,
  *,
  context: str | None = None,
) -> SnapshotManifest:
  """Copy snapshot manifest entries from source epoch to target branch next epoch.

  Content-addressed blobs are shared by digest -- no byte duplication.

  Args:
    store: FileStore instance.
    source_experiment_id: Branch to read from.
    source_epoch: Epoch to copy.
    target_experiment_id: Branch receiving the new epoch.
    context: Optional audit string stored on the new manifest.

  Returns:
    SnapshotManifest persisted at the new target epoch.

  Raises:
    StoreError: When branches or epochs are missing, blobs are missing,
      or epoch sequence is violated.
  """
  store._require_branch(source_experiment_id)
  store._require_branch(target_experiment_id)

  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})

    if source_experiment_id not in branches:
      msg = f'experiment {source_experiment_id!r} not found'
      raise StoreError(msg)
    if target_experiment_id not in branches:
      msg = f'experiment {target_experiment_id!r} not found'
      raise StoreError(msg)

    source_snap = store.load_snapshot(source_experiment_id, source_epoch)

    missing_digests = [
      entry.digest
      for entry in source_snap.entries.values()
      if not store._backend.object_exists(entry.digest)
    ]
    if missing_digests:
      msg = (
        f'cannot copy epoch: {len(missing_digests)} blob(s) missing from object store: '
        f'{missing_digests}'
      )
      raise StoreError(msg)

    target_branch = branches[target_experiment_id]
    old_epoch = target_branch['latest_epoch']
    next_epoch = old_epoch + 1

    effective_context = context or (f'copy_epoch from {source_experiment_id}@{source_epoch}')

    new_manifest = SnapshotManifest(
      epoch=next_epoch,
      timestamp=utc_now_iso(),
      entries=dict(source_snap.entries),
      schema=source_snap.schema,
      context=effective_context,
    )

    store._save_snapshot(target_experiment_id, next_epoch, new_manifest)

    reflog_old = None if old_epoch == -1 else old_epoch
    target_branch['latest_epoch'] = next_epoch
    refs['HEAD'] = target_experiment_id
    store.save_refs(refs)
    append_reflog(
      store,
      'copy_epoch',
      target_experiment_id,
      reflog_old,
      next_epoch,
      context=effective_context,
      source_experiment_id=source_experiment_id,
      source_epoch=source_epoch,
    )
  finally:
    store._release_lock()

  return new_manifest


# --- tags ---


def tag(
  store: Any,
  name: str,
  experiment_id: str,
  epoch: int,
  context: str | None = None,
) -> None:
  """Create an immutable tag pointing to a specific experiment and epoch.

  Args:
    store: FileStore instance.
    name: Tag name (alphanumeric, '-', '_', '.' only; max 128 chars).
    experiment_id: Branch the tag points to.
    epoch: Epoch the tag points to.
    context: Optional reason string for audit traceability.

  Raises:
    StoreError: If the tag name is invalid, already exists, or the
      branch/epoch does not exist.
  """
  validate_tag_name(name)

  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})
    if experiment_id not in branches:
      msg = f'experiment {experiment_id!r} not found'
      raise StoreError(msg)

    snap_path = store._snapshots_dir / experiment_id / f'epoch_{epoch}.json'
    if not snap_path.exists():
      msg = (
        f'epoch {epoch} does not exist for experiment {experiment_id!r}; '
        f'snapshot at that epoch before tagging'
      )
      raise StoreError(msg)

    tags = refs.get('tags', {})
    if name in tags:
      msg = f'tag {name!r} already exists; tags are immutable'
      raise StoreError(msg)

    manifest = store.load_snapshot(experiment_id, epoch)
    digest = compute_manifest_digest(manifest)

    tags[name] = {
      'experiment_id': experiment_id,
      'epoch': epoch,
      'context': context,
      'timestamp': utc_now_iso(),
      'manifest_digest': digest,
    }
    refs['tags'] = tags
    store.save_refs(refs)
    append_reflog(
      store,
      'tag',
      experiment_id,
      old_epoch=None,
      new_epoch=epoch,
      context=context,
    )
  finally:
    store._release_lock()


def get_tag(store: Any, name: str) -> TagEntry | None:
  """Look up a tag by name.

  Args:
    store: FileStore instance.
    name: Tag name to look up.

  Returns:
    TagEntry if found, None otherwise.
  """
  refs = store.load_refs()
  tags = refs.get('tags', {})
  tag_data = tags.get(name)
  if tag_data is None:
    return None
  return TagEntry(
    name=name,
    experiment_id=tag_data['experiment_id'],
    epoch=tag_data['epoch'],
    context=tag_data.get('context'),
    timestamp=tag_data.get('timestamp'),
    manifest_digest=tag_data.get('manifest_digest'),
  )


def list_tags(store: Any) -> list[TagEntry]:
  """List all tags, sorted by name.

  Args:
    store: FileStore instance.

  Returns:
    List of TagEntry instances sorted alphabetically by name.
  """
  refs = store.load_refs()
  tags = refs.get('tags', {})
  result: list[TagEntry] = []
  for name in sorted(tags):
    tag_data = tags[name]
    result.append(
      TagEntry(
        name=name,
        experiment_id=tag_data['experiment_id'],
        epoch=tag_data['epoch'],
        context=tag_data.get('context'),
        timestamp=tag_data.get('timestamp'),
        manifest_digest=tag_data.get('manifest_digest'),
      )
    )
  return result


def verify_tag(store: Any, name: str) -> dict[str, Any]:
  """Verify a tag's manifest digest against the current on-disk manifest.

  Args:
    store: FileStore instance.
    name: Tag name to verify.

  Returns:
    Verification result dict.

  Raises:
    StoreError: If the tag does not exist.
  """
  tag_entry = get_tag(store, name)
  if tag_entry is None:
    msg = (
      f"tag {name!r} not found. Use store.list_tags() or 'store tag list' to see available tags."
    )
    raise StoreError(msg)

  if tag_entry.manifest_digest is None:
    return {'verified': False, 'reason': 'no digest available'}

  manifest = store.load_snapshot(tag_entry.experiment_id, tag_entry.epoch)
  actual_digest = compute_manifest_digest(manifest)

  if actual_digest == tag_entry.manifest_digest:
    return {'verified': True}

  return {
    'verified': False,
    'reason': 'digest mismatch',
    'expected': tag_entry.manifest_digest,
    'actual': actual_digest,
  }
