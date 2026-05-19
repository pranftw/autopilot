"""Worktree directory create/list/remove for FileStore."""

from autopilot.core.errors import StoreError, TrackingError
from autopilot.tracking.io import exclusive_create
from pathlib import Path
from typing import Any
import shutil


def create_worktree(store: Any, experiment_id: str) -> Path:
  """Creates a worktree directory for ``experiment_id``.

  Args:
    store: FileStore instance.
    experiment_id: Branch to create a worktree for.

  Returns:
    Absolute path to the created worktree directory.

  Raises:
    StoreError: If the branch is missing or a lock is held.
  """
  store._require_branch(experiment_id)
  worktrees_dir = store._config.worktrees_path
  worktrees_dir.mkdir(parents=True, exist_ok=True)

  wt_path = worktrees_dir / experiment_id
  lock_path = wt_path.with_suffix('.lock')

  try:
    exclusive_create(lock_path)
  except FileExistsError:
    msg = f'worktree {experiment_id!r} is locked by another operation'
    raise StoreError(msg) from None
  except TrackingError as exc:
    raise StoreError(str(exc)) from exc

  try:
    wt_path.mkdir(parents=True, exist_ok=True)

    store._acquire_lock()
    try:
      refs = store.load_refs()
      worktrees = refs.get('worktrees', {})
      worktrees[experiment_id] = str(wt_path)
      refs['worktrees'] = worktrees
      store.save_refs(refs)
    finally:
      store._release_lock()
  finally:
    lock_path.unlink(missing_ok=True)

  return wt_path


def remove_worktree(store: Any, experiment_id: str) -> None:
  """Removes the worktree directory and deregisters from refs.

  Args:
    store: FileStore instance.
    experiment_id: Branch whose worktree should be removed.
  """
  worktrees_dir = store._config.worktrees_path
  wt_path = worktrees_dir / experiment_id

  if wt_path.exists():
    shutil.rmtree(wt_path)

  store._acquire_lock()
  try:
    refs = store.load_refs()
    worktrees = refs.get('worktrees', {})
    if experiment_id in worktrees:
      del worktrees[experiment_id]
      refs['worktrees'] = worktrees
      store.save_refs(refs)
  finally:
    store._release_lock()


def list_worktrees(store: Any) -> list[str]:
  """Returns sorted list of experiment IDs with registered worktrees.

  Args:
    store: FileStore instance.
  """
  refs = store.load_refs()
  worktrees = refs.get('worktrees', {})
  return sorted(worktrees)
