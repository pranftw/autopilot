"""Refs, branch lifecycle, and forest.json persistence for FileStore.

``refs.json`` holds ``HEAD``, per-branch metadata (``latest_epoch``,
``parent_id``, ``parent_epoch``, ``merge_parents``), tags, worktrees, and
format ``version``. See ``FileStore`` module documentation for the full
on-disk layout contract.
"""

from autopilot.ai.store import snapshot as snapshot_mod
from autopilot.ai.store.store_io import atomic_write_json_safe, reraise_as_store_error
from autopilot.core.errors import StoreError, TrackingError
from autopilot.core.snapshot import SnapshotManifest
from autopilot.tracking.io import read_json
from pathlib import Path
from typing import Any


def load_refs(store: Any) -> dict[str, Any]:
  """Loads the refs structure (branches, HEAD, worktrees).

  Args:
    store: FileStore instance.

  Returns:
    Parsed refs dict, or empty dict if no refs file exists.
  """
  if not store._refs_file.exists():
    return {}
  try:
    data = read_json(store._refs_file)
  except TrackingError as exc:
    reraise_as_store_error(exc, f'failed to load refs: {exc}')
  if data is None:
    return {}
  if not isinstance(data, dict):
    msg = f'refs must contain a JSON object at {store._refs_file}, got {type(data).__name__}'
    raise StoreError(msg)
  return data


def save_refs(store: Any, refs: dict[str, Any]) -> None:
  """Persist the refs structure atomically.

  Args:
    store: FileStore instance.
    refs: Complete refs dict to write.
  """
  atomic_write_json_safe(store._refs_file, refs)


def require_branch(store: Any, experiment_id: str) -> dict[str, Any]:
  """Validate branch exists, return branch info.

  Args:
    store: FileStore instance.
    experiment_id: Branch to resolve.

  Returns:
    Branch entry dict from refs.

  Raises:
    StoreError: When the branch is missing.
  """
  refs = store.load_refs()
  branches = refs.get('branches', {})
  if experiment_id not in branches:
    msg = f'experiment {experiment_id!r} not found'
    raise StoreError(msg)
  return branches[experiment_id]


def require_branch_absent(store: Any, experiment_id: str) -> None:
  """Fast-fail check that a branch does NOT exist yet.

  Args:
    store: FileStore instance.
    experiment_id: Candidate branch name.

  Raises:
    StoreError: If the branch already exists.
  """
  refs = store.load_refs()
  branches = refs.get('branches', {})
  if experiment_id in branches:
    msg = f'experiment {experiment_id!r} already exists'
    raise StoreError(msg)


def branch(store: Any, experiment_id: str) -> None:
  """Creates a new branch forking from the current HEAD.

  Args:
    store: FileStore instance.
    experiment_id: Name for the new branch.

  Raises:
    StoreError: If the branch already exists or no HEAD is set.
  """
  require_branch_absent(store, experiment_id)

  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})

    if experiment_id in branches:
      msg = f'experiment {experiment_id!r} already exists'
      raise StoreError(msg)

    head = refs.get('HEAD')
    if not head or head not in branches:
      msg = 'no HEAD set; create an initial experiment first'
      raise StoreError(msg)

    head_branch = branches[head]
    head_epoch = head_branch['latest_epoch']

    snap = snapshot_mod.load_snapshot_manifest(store, head, head_epoch)

    snapshot_mod.persist_snapshot_manifest(
      store,
      experiment_id,
      0,
      SnapshotManifest(
        epoch=0,
        timestamp=snap.timestamp,
        entries=dict(snap.entries),
        schema=snap.schema,
        context=snap.context,
      ),
    )

    branches[experiment_id] = {
      'latest_epoch': 0,
      'parent_id': head,
      'parent_epoch': head_epoch,
    }
    refs['branches'] = branches
    store.save_refs(refs)
    store._append_reflog('branch', experiment_id, None, 0)
  finally:
    store._release_lock()


def reset_branch(store: Any, experiment_id: str) -> None:
  """Reset a branch's latest_epoch to -1, enabling re-run from epoch 0.

  Args:
    store: FileStore instance.
    experiment_id: Branch to reset.

  Raises:
    StoreError: If the branch does not exist.
  """
  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})
    if experiment_id not in branches:
      msg = f'experiment {experiment_id!r} not found'
      raise StoreError(msg)
    old_epoch = branches[experiment_id]['latest_epoch']
    reflog_old = None if old_epoch == -1 else old_epoch
    branches[experiment_id]['latest_epoch'] = -1
    store.save_refs(refs)
    store._append_reflog('reset_branch', experiment_id, reflog_old, -1)
  finally:
    store._release_lock()


def save_forest_state(store: Any, state: dict) -> None:
  """Persists the forest state dict to ``forest.json`` atomically.

  Args:
    store: FileStore instance.
    state: Serialized forest/tree structure to write.
  """
  atomic_write_json_safe(store._config.forest_file, state)


def load_forest_state(store: Any) -> dict | None:
  """Loads the forest state dict from ``forest.json``.

  Args:
    store: FileStore instance.

  Returns:
    Parsed dict, or None if the file does not exist.

  Raises:
    StoreError: If the file exists but cannot be parsed.
  """
  forest_file: Path = store._config.forest_file
  if not forest_file.is_file():
    return None
  try:
    data = read_json(forest_file)
  except TrackingError as exc:
    reraise_as_store_error(exc, str(exc))
  if data is None:
    return None
  if not isinstance(data, dict):
    msg = f'forest must contain a JSON object at {forest_file}, got {type(data).__name__}'
    raise StoreError(msg)
  return data
