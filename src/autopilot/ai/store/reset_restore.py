"""Atomic branch reset + working-tree restore for FileStore.

Provides ``reset_and_restore`` which, under a single lock scope:
  1. Resets the branch tip (``latest_epoch``) to the target epoch (or -1).
  2. Syncs working-tree parameter files: restores from the target snapshot
     or clears tracked files when resetting to empty state.
  3. Appends two reflog entries: ``reset_branch`` then ``checkout``.

This module exists separately from ``refs.py`` (refs-only mutations) and
``snapshot.py`` (checkout/restore logic) to keep lock scope and file-restore
orchestration in one place without bloating either sibling module.
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.snapshot_helpers import group_by_param, remove_extraneous_files
from autopilot.core.errors import StoreError
from typing import Any


def reset_and_restore(
  store: Any,
  experiment_id: str,
  epoch: int | None = None,
  *,
  context: str | None = None,
) -> None:
  """Reset branch tip and sync working-tree files under a single lock.

  Validation (read-only, before lock):
    - Branch must exist.
    - When ``epoch`` is not ``None``: snapshot must exist at ``epoch``;
      ``epoch`` must be ``<=`` current ``latest_epoch``.

  Under lock:
    1. Read old tip from refs.
    2. Compute target tip: ``-1`` when ``epoch is None``, else ``epoch``.
    3. Write new ``latest_epoch`` and persist refs.
    4. Append ``reset_branch`` reflog entry.
    5. Sync working tree (clear or restore).
    6. Set HEAD and persist refs.
    7. Append ``checkout`` reflog entry.

  Args:
    store: FileStore instance.
    experiment_id: Branch to reset.
    epoch: Target tip epoch, or ``None`` for empty state (-1).
    context: Optional reason/provenance string recorded on both reflog
      entries for audit traceability.

  Raises:
    StoreError: If the branch does not exist, the epoch does not exist
      (when ``epoch`` is not ``None``), or ``epoch`` exceeds the current
      ``latest_epoch``.
  """
  branch_info = store._require_branch(experiment_id)
  current_tip = branch_info['latest_epoch']

  if epoch is not None:
    if epoch > current_tip:
      msg = (
        f'epoch {epoch} exceeds current latest_epoch {current_tip} '
        f'for experiment {experiment_id!r}; cannot fast-forward'
      )
      raise StoreError(msg)
    store.load_snapshot(experiment_id, epoch)

  target_tip = -1 if epoch is None else epoch

  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})
    old_tip = branches[experiment_id]['latest_epoch']

    branches[experiment_id]['latest_epoch'] = target_tip
    refs['branches'] = branches
    store.save_refs(refs)
    store._append_reflog(
      'reset_branch', experiment_id, _reflog_old(old_tip), target_tip, context=context
    )

    if target_tip == -1:
      _clear_working_tree(store)
    else:
      snap = store.load_snapshot(experiment_id, target_tip)
      grouped = group_by_param(store, snap)
      snapshot_keys = set(snap.entries)

      for name, param in store._param_names.items():
        param.restore(grouped.get(name, {}))
        if isinstance(param, PathParameter):
          entries_for_param = {k for k in snapshot_keys if k.startswith(f'{name}/')}
          if not entries_for_param:
            continue
          remove_extraneous_files(store, param, name, snapshot_keys)

    refs = store.load_refs()
    refs['HEAD'] = experiment_id
    store.save_refs(refs)
    store._append_reflog(
      'checkout', experiment_id, _reflog_old(old_tip), target_tip, context=context
    )
  finally:
    store._release_lock()


def _reflog_old(tip: int) -> int | None:
  """Normalize tip for reflog old_epoch (``-1`` -> ``None``).

  Args:
    tip: Branch tip value.

  Returns:
    ``None`` when tip is ``-1``, otherwise the tip value.
  """
  return None if tip == -1 else tip


def _clear_working_tree(store: Any) -> None:
  """Remove tracked parameter files from the working tree.

  For each registered parameter:
    - Call ``param.restore({})`` to clear in-memory state.
    - For ``PathParameter``: call ``remove_extraneous_files`` with an empty
      snapshot key set to delete all matched text files (binary and
      protected-prefix files preserved per ``snapshot_helpers``).

  Args:
    store: FileStore instance with registered parameters.
  """
  for name, param in store._param_names.items():
    param.restore({})
    if isinstance(param, PathParameter):
      remove_extraneous_files(store, param, name, set())
