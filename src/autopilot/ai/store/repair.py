"""Repair actions for FileStore doctor diagnostics."""

from autopilot.ai.store.snapshot import enumerate_snapshot_epochs
from autopilot.core.diagnostic import DiagnosticEntry
from pathlib import Path
from typing import Any


def repair_orphan_blob(store: Any, entry: DiagnosticEntry) -> None:
  """Remove an orphan blob from the object store.

  Args:
    store: FileStore instance.
    entry: Diagnostic entry with code 'orphan_blob' and path = digest.
  """
  digest = entry.path
  if digest is None:
    return
  blob_path = store._config.objects_path / digest[:2] / digest[2:]
  if blob_path.is_file():
    blob_path.unlink()
    parent = blob_path.parent
    if parent.exists() and not any(parent.iterdir()):
      parent.rmdir()


def repair_stale_lock(entry: DiagnosticEntry) -> None:
  """Remove a stale lock file.

  Args:
    entry: Diagnostic entry with code 'stale_lock' and path = lock file path.
  """
  if entry.path is None:
    return
  lock_path = Path(entry.path)
  if lock_path.is_file():
    lock_path.unlink()


def repair_broken_ref(store: Any, _entry: DiagnosticEntry) -> None:
  """Reset a broken branch ref to the last valid epoch.

  Args:
    store: FileStore instance.
    _entry: Diagnostic entry with code 'broken_ref' (unused; uniform dispatch API).
  """
  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})
    for exp_id, info in branches.items():
      latest = info.get('latest_epoch')
      if latest is None or latest < -1:
        info['latest_epoch'] = -1
        continue
      if latest == -1:
        continue
      snap_path = store._snapshots_dir / exp_id / f'epoch_{latest}.json'
      if not snap_path.exists():
        exp_dir = store._snapshots_dir / exp_id
        valid_epochs = enumerate_snapshot_epochs(exp_dir) if exp_dir.exists() else []
        info['latest_epoch'] = valid_epochs[-1] if valid_epochs else -1
    store.save_refs(refs)
  finally:
    store._release_lock()


def repair_reflog_gap(
  store: Any,
  entry: DiagnosticEntry,
  context: str | None,
) -> None:
  """Backfill a synthetic reflog entry for a branch missing from reflog.

  Args:
    store: FileStore instance.
    entry: Diagnostic entry with code 'reflog_gap' and path = branch id.
    context: Audit context string.
  """
  branch_id = entry.path
  if branch_id is None:
    return
  store._acquire_lock()
  try:
    store._append_reflog(
      'backfill',
      branch_id,
      old_epoch=None,
      new_epoch=None,
      context=context or 'doctor repair backfill',
    )
  finally:
    store._release_lock()
