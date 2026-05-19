"""Atomic multi-resource context manager for FileStore writes.

StoreTransaction coordinates refs.json, snapshot manifests, and reflog.jsonl
updates under a single lock scope. All writes are buffered in memory during
the transaction block. On successful exit the buffered writes are flushed to
disk via atomic helpers; on failure all buffered writes are discarded and the
lock is released without mutation.

Lifecycle::

  with store.transaction(context='merge apply') as txn:
    txn.write_manifest(branch, epoch, manifest)
    txn.set_refs(refs_dict)
    txn.append_reflog({'operation': 'merge_apply', ...})
  # all writes committed on successful exit
  # all writes rolled back on exception

Nesting is not supported: a second ``with store.transaction()`` while one is
active raises ``RuntimeError``.
"""

from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.base import Store
from autopilot.tracking.io import append_jsonl
from pathlib import Path
from typing import Any
import contextlib
import types


class StoreTransaction:
  """Atomic multi-resource context manager for FileStore writes.

  Acquires the store lock on entry, buffers manifest/refs/reflog writes
  in memory, and flushes them to disk on success. On failure, buffered
  writes are discarded and the lock is released without any durable mutation.

  Not reentrant: nesting raises ``RuntimeError``. Use the
  ``FileStore.transaction()`` factory to construct.

  Args:
    store: The Store instance to transact against.
    context: Optional reason/provenance string threaded to operations.
  """

  def __init__(self, store: Store, context: str | None = None) -> None:
    """Create a transaction bound to a Store.

    Args:
      store: The Store to transact against. Must support ``load_refs``,
        ``save_refs``, and snapshot persistence.
      context: Optional reason/provenance string for audit traceability.

    Raises:
      TypeError: If ``store`` is not a ``Store`` instance.
    """
    if not isinstance(store, Store):
      msg = f'StoreTransaction requires a Store instance, got {type(store).__name__}'
      raise TypeError(msg)
    self._store: Any = store
    self._context = context
    self._entered = False
    self._committed = False

    self._pending_manifests: list[tuple[str, int, SnapshotManifest]] = []
    self._pending_refs: dict[str, int] | None = None
    self._pending_refs_full: dict[str, Any] | None = None
    self._pending_reflog: list[dict[str, Any]] = []
    self._written_paths: list[Path] = []

  @property
  def context(self) -> str | None:
    """Optional provenance string for this transaction."""
    return self._context

  def __enter__(self) -> 'StoreTransaction':
    """Acquire the store lock and begin the transaction.

    Returns:
      This transaction instance.

    Raises:
      RuntimeError: If the store already has an active transaction.
    """
    if self._store.active_transaction is not None:
      msg = 'nested transactions are not supported; a transaction is already active on this store'
      raise RuntimeError(msg)
    self._store.acquire_transaction_lock()
    self._store.active_transaction = self
    self._entered = True
    return self

  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: types.TracebackType | None,
  ) -> None:
    """Commit on success or rollback on failure; always release the lock.

    Args:
      exc_type: Exception type if an error occurred during the block.
      exc_val: Exception value if an error occurred.
      exc_tb: Traceback if an error occurred.
    """
    try:
      if exc_type is None:
        self._commit()
      else:
        self._rollback()
    finally:
      self._store.active_transaction = None
      self._store.release_transaction_lock()

  def write_manifest(self, branch: str, epoch: int, manifest: SnapshotManifest) -> None:
    """Buffer a manifest write for atomic commit.

    Args:
      branch: Experiment/branch identifier.
      epoch: Epoch index for the manifest.
      manifest: The snapshot manifest to persist.
    """
    self._require_entered()
    self._pending_manifests.append((branch, epoch, manifest))

  def update_refs(self, branch: str, epoch: int) -> None:
    """Buffer a refs update for atomic commit.

    Updates the branch's ``latest_epoch`` and sets ``HEAD`` to the branch.
    Mutually exclusive with ``set_refs`` within a single transaction.

    Args:
      branch: Experiment/branch identifier.
      epoch: New latest epoch for the branch.

    Raises:
      RuntimeError: If ``set_refs`` was already called in this transaction.
    """
    self._require_entered()
    if self._pending_refs_full is not None:
      msg = 'cannot mix update_refs and set_refs in the same transaction'
      raise RuntimeError(msg)
    if self._pending_refs is None:
      self._pending_refs = {}
    self._pending_refs[branch] = epoch

  def set_refs(self, refs: dict[str, Any]) -> None:
    """Buffer a complete refs dict for atomic commit.

    Unlike ``update_refs`` which only sets ``latest_epoch`` for a branch,
    this buffers the entire refs dict (including merge_parents, HEAD, etc.)
    to be written atomically during commit. Mutually exclusive with
    ``update_refs`` within a single transaction.

    Args:
      refs: The complete refs dict to persist.

    Raises:
      RuntimeError: If ``update_refs`` was already called in this transaction.
    """
    self._require_entered()
    if self._pending_refs is not None:
      msg = 'cannot mix set_refs and update_refs in the same transaction'
      raise RuntimeError(msg)
    self._pending_refs_full = refs

  def append_reflog(self, entry: dict[str, Any]) -> None:
    """Buffer a reflog entry for atomic commit.

    Args:
      entry: Dict to append as a JSONL line to the reflog.
    """
    self._require_entered()
    self._pending_reflog.append(entry)

  def commit(self) -> None:
    """Explicit commit (normally handled by ``__exit__``)."""
    self._require_entered()
    self._commit()

  def rollback(self) -> None:
    """Explicit rollback (normally handled by ``__exit__`` on exception)."""
    self._require_entered()
    self._rollback()

  def _require_entered(self) -> None:
    """Guard against use outside a ``with`` block.

    Raises:
      RuntimeError: When the transaction has not been entered.
    """
    if not self._entered:
      msg = 'StoreTransaction must be used as a context manager'
      raise RuntimeError(msg)

  def _commit(self) -> None:
    """Persist all buffered writes atomically.

    Writes manifests and refs to temp files, then renames them into place.
    If any write fails, remaining temps are cleaned up.
    """
    if self._committed:
      return
    self._committed = True

    try:
      for branch, epoch, manifest in self._pending_manifests:
        self._store.persist_manifest(branch, epoch, manifest)
        snap_path = self._store.snapshot_manifest_path(branch, epoch)
        self._written_paths.append(snap_path)

      if self._pending_refs_full is not None:
        self._store.save_refs(self._pending_refs_full)
      elif self._pending_refs is not None:
        refs = self._store.load_refs()
        branches = refs.get('branches', {})
        last_branch = None
        for branch, epoch in self._pending_refs.items():
          if branch not in branches:
            branches[branch] = {
              'latest_epoch': -1,
              'parent_id': None,
              'parent_epoch': None,
            }
          branches[branch]['latest_epoch'] = epoch
          last_branch = branch
        refs['branches'] = branches
        if last_branch is not None:
          refs['HEAD'] = last_branch
        self._store.save_refs(refs)

      for entry in self._pending_reflog:
        reflog_path = self._store.reflog_path
        append_jsonl(reflog_path, entry)

    except Exception:
      self._rollback()
      raise

  def _rollback(self) -> None:
    """Discard all buffered writes and clean up persisted artifacts.

    Deletes manifest files that were already written to disk during
    a failed commit. Uses ``missing_ok=True`` for robustness.
    Best-effort: rollback never raises; the original commit error
    is surfaced by the caller.
    """
    for path in self._written_paths:
      with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
    self._written_paths.clear()
    self._pending_manifests.clear()
    self._pending_refs = None
    self._pending_refs_full = None
    self._pending_reflog.clear()
