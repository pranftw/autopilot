"""FileStoreCore: content-addressed file store implementation body.

The public ``FileStore`` wrapper and full package documentation live in
``autopilot.ai.store.file_store``.
"""

from autopilot.ai.store import merge as merge_mod
from autopilot.ai.store import refs as refs_mod
from autopilot.ai.store import reset_restore as reset_restore_mod
from autopilot.ai.store import snapshot as snapshot_mod
from autopilot.ai.store.file_store_core_peripherals import FileStoreCorePeripheralMixin
from autopilot.ai.store.snapshot import SchemaMatchResult
from autopilot.ai.store_lock import StorageBackend, hash_bytes
from autopilot.ai.transaction import StoreTransaction
from autopilot.core.branch import BranchHandle, RefsView
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.parameter import Parameter
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.base import Store
from autopilot.core.store.types import (
  DiffResult,
  MergeAnalysisResult,
  MergeIndex,
  MergeStrategy,
  StatusResult,
)
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


class FileStoreCore(FileStoreCorePeripheralMixin, Store):
  """Content-addressed file store core implementation."""

  STASH_INDEX_PAD_WIDTH = 4

  def __init__(self, config: AutoPilotConfig) -> None:
    """Initialize store layout and backing storage backend.

    Args:
      config: Paths for objects, snapshots, refs, forest, and worktrees.
    """
    self._config = config
    self._param_names: dict[str, Parameter] = {}

    self._backend = StorageBackend(config)
    self._snapshots_dir = config.snapshots_path
    self._refs_file = config.refs_file
    self.active_transaction: Any = None

    self._snapshots_dir.mkdir(parents=True, exist_ok=True)

  def register_parameters(self, parameters: dict[str, Parameter]) -> None:
    """Register named parameters for snapshotting.

    Keys become composite manifest prefixes (e.g. ``'prompts'`` produces
    entries like ``'prompts/system.txt'``). Call before snapshot/checkout/status.

    Args:
      parameters: Mapping from attribute name to Parameter instance.
    """
    self._param_names = dict(parameters)

  @property
  def config(self) -> AutoPilotConfig:
    """Returns the AutoPilotConfig instance governing all store paths."""
    return self._config

  @property
  def lock_timeout_s(self) -> float | None:
    """Effective lock timeout in seconds (``None`` = fail-fast)."""
    return self._backend.lock_timeout_s

  @lock_timeout_s.setter
  def lock_timeout_s(self, value: float | None) -> None:
    """Update the store lock timeout dynamically.

    Args:
      value: ``None`` = fail-fast, positive = wait seconds, ``-1.0`` = block forever.
    """
    self._backend.lock_timeout_s = value

  def transaction(self, *, context: str | None = None) -> 'StoreTransaction':
    """Create a transaction context manager for atomic multi-resource writes.

    Args:
      context: Optional reason/provenance string.

    Returns:
      A ``StoreTransaction`` context manager.
    """
    return StoreTransaction(self, context=context)

  def acquire_transaction_lock(self) -> None:
    """Acquire the store lock for a transaction scope."""
    self._acquire_lock()

  def release_transaction_lock(self) -> None:
    """Release the store lock after a transaction scope."""
    self._release_lock()

  def snapshot_manifest_path(self, experiment_id: str, epoch: int) -> Path:
    """Return the on-disk path for a snapshot manifest.

    Args:
      experiment_id: Branch that owns the snapshot.
      epoch: Epoch index.

    Returns:
      Path to the ``epoch_<N>.json`` manifest file.
    """
    return self._snapshots_dir / experiment_id / f'epoch_{epoch}.json'

  def persist_manifest(
    self,
    experiment_id: str,
    epoch: int,
    manifest: SnapshotManifest,
  ) -> None:
    """Write a snapshot manifest to disk.

    Args:
      experiment_id: Branch that owns the snapshot.
      epoch: Epoch index for the manifest.
      manifest: The manifest to persist.
    """
    self._save_snapshot(experiment_id, epoch, manifest)

  # branch handle and refs view

  def branch_handle(self, experiment_id: str) -> BranchHandle:
    """Return a curried ``BranchHandle`` for the given experiment branch.

    Args:
      experiment_id: Branch name to bind.

    Returns:
      ``BranchHandle`` bound to this store and the given experiment.
    """
    return BranchHandle(self, experiment_id)

  @property
  def refs_view(self) -> RefsView:
    """Return an iterable ``RefsView`` over all store branches.

    Returns:
      ``RefsView`` bound to this store.
    """
    return RefsView(self)

  # snapshot / checkout / diff / materialize (delegated to snapshot_mod)

  def snapshot(
    self,
    experiment_id: str,
    epoch: int,
    experiment: Experiment | None = None,
    context: str | None = None,
    *,
    force: bool = False,
  ) -> SnapshotManifest:
    """Persists parameter state for ``experiment_id`` at ``epoch``.

    Args:
      experiment_id: Branch / experiment identifier.
      epoch: Epoch index.
      experiment: Optional experiment to check completion status.
      context: Optional reason/provenance string.
      force: When True, always persist even if unchanged.

    Returns:
      Manifest written for this snapshot, or the prior manifest when skipped.

    Raises:
      StoreError: If no parameters are registered or the branch is missing.
      ExperimentError: If the experiment is already completed with strict mode.
    """
    return snapshot_mod.snapshot(self, experiment_id, epoch, experiment, context, force=force)

  def _snapshot_content_identical(
    self,
    prior: SnapshotManifest,
    current: SnapshotManifest,
  ) -> bool:
    """Check whether two manifests have identical file-entry content."""
    return snapshot_mod.snapshot_content_identical(prior, current)

  def checkout(
    self,
    experiment_id: str,
    epoch: int,
    *,
    context: str | None = None,
    strict_schema: bool = False,
  ) -> None:
    """Restores parameter state from the snapshot at ``epoch``.

    Args:
      experiment_id: Branch to checkout.
      epoch: Epoch to restore.
      context: Optional reason/provenance string recorded in the reflog.
      strict_schema: When True, raise StoreError on schema mismatch.
    """
    snapshot_mod.checkout(self, experiment_id, epoch, strict_schema=strict_schema, context=context)

  def validate_checkout(self, experiment_id: str, epoch: int) -> dict[str, Any]:
    """Perform all read-only validations for a checkout.

    Args:
      experiment_id: Branch to validate.
      epoch: Epoch to validate.

    Returns:
      Dict with validation results.
    """
    return snapshot_mod.validate_checkout(self, experiment_id, epoch)

  def diff(self, experiment_id: str, epoch_a: int, epoch_b: int) -> DiffResult:
    """Computes a unified diff between two epochs.

    Args:
      experiment_id: Branch to diff within.
      epoch_a: Earlier epoch.
      epoch_b: Later epoch.

    Returns:
      DiffResult with per-file entries.
    """
    return snapshot_mod.diff(self, experiment_id, epoch_a, epoch_b)

  def materialize(self, experiment_id: str, epoch: int) -> None:
    """Restores parameters to a historical epoch and resets the branch tip.

    Args:
      experiment_id: Branch to materialize within.
      epoch: Historical epoch to restore.
    """
    snapshot_mod.materialize(self, experiment_id, epoch)

  def status(self, experiment_id: str) -> StatusResult:
    """Compares current parameter state against the latest snapshot.

    Args:
      experiment_id: Branch to compare against.

    Returns:
      StatusResult with per-file status entries.
    """
    return snapshot_mod.status(self, experiment_id)

  # branch

  def branch(self, experiment_id: str) -> None:
    """Creates a new branch forking from the current HEAD.

    Args:
      experiment_id: Name for the new branch.

    Raises:
      StoreError: If the branch already exists or no HEAD is set.
    """
    refs_mod.branch(self, experiment_id)

  def reset_branch(self, experiment_id: str) -> None:
    """Reset a branch's latest_epoch to -1, enabling re-run from epoch 0.

    Args:
      experiment_id: Branch to reset.

    Raises:
      StoreError: If the branch does not exist.
    """
    refs_mod.reset_branch(self, experiment_id)

  def reset_and_restore(
    self,
    experiment_id: str,
    epoch: int | None = None,
    *,
    context: str | None = None,
  ) -> None:
    """Atomically reset branch tip and sync working-tree parameter files.

    Delegates to ``reset_restore_mod.reset_and_restore`` which performs
    both the refs tip mutation and working-tree file sync under a single
    lock scope, appending ``reset_branch`` and ``checkout`` reflog entries.

    Args:
      experiment_id: Branch to reset.
      epoch: Target tip epoch, or ``None`` for empty state (-1).
      context: Optional reason/provenance string recorded on both reflog
        entries for audit traceability.

    Raises:
      StoreError: If the branch does not exist, the epoch does not exist,
        or ``epoch`` exceeds the current ``latest_epoch``.
    """
    reset_restore_mod.reset_and_restore(self, experiment_id, epoch, context=context)

  # merge (delegated to merge_mod)

  def merge_analysis(
    self,
    experiment_id: str,
    from_experiment_id: str,
  ) -> MergeAnalysisResult:
    """Classify merge using refs and manifest key overlap.

    Args:
      experiment_id: Target branch (ours).
      from_experiment_id: Source branch (theirs).

    Returns:
      MergeAnalysisResult with classification.
    """
    return merge_mod.merge_analysis(self, experiment_id, from_experiment_id)

  def merge_preview(
    self,
    experiment_id: str,
    from_experiment_id: str,
    from_epoch: int | None = None,
    strategy: MergeStrategy = MergeStrategy.normal,
  ) -> MergeIndex:
    """Compute three-way merge into a MergeIndex staging area.

    Args:
      experiment_id: Target branch (ours).
      from_experiment_id: Source branch (theirs).
      from_epoch: Epoch to merge from.
      strategy: Resolution strategy.

    Returns:
      MergeIndex with conflicts and resolved entries.
    """
    return merge_mod.merge_preview(self, experiment_id, from_experiment_id, from_epoch, strategy)

  def merge_apply(self, merge_index: MergeIndex) -> SnapshotManifest:
    """Persist a resolved merge as a new epoch.

    Args:
      merge_index: Fully resolved MergeIndex.

    Returns:
      SnapshotManifest for the new merge epoch.
    """
    return merge_mod.merge_apply(self, merge_index)

  def merge_and_apply(
    self,
    experiment_id: str,
    from_experiment_id: str,
    from_epoch: int | None = None,
    strategy: MergeStrategy = MergeStrategy.normal,
  ) -> SnapshotManifest:
    """Run analysis + preview + apply.

    Args:
      experiment_id: Target branch (ours).
      from_experiment_id: Source branch (theirs).
      from_epoch: Epoch to merge from.
      strategy: Resolution strategy.

    Returns:
      SnapshotManifest for the new merge epoch.
    """
    return merge_mod.merge_and_apply(self, experiment_id, from_experiment_id, from_epoch, strategy)

  # internal helpers

  def _require_branch(self, experiment_id: str) -> dict[str, Any]:
    """Validate branch exists, return branch info."""
    return refs_mod.require_branch(self, experiment_id)

  def _require_branch_absent(self, experiment_id: str) -> None:
    """Fast-fail check that a branch does NOT exist yet.

    Raises:
      StoreError: If the branch already exists.
    """
    refs_mod.require_branch_absent(self, experiment_id)

  def _validate_schema(self, snap: SnapshotManifest) -> SchemaMatchResult:
    """Check manifest parameter names against registered parameters."""
    return snapshot_mod.validate_schema(self, snap)

  def _build_snapshot(self, context: str | None = None) -> SnapshotManifest:
    """Build a snapshot manifest from registered parameters."""
    return snapshot_mod.build_snapshot(self, context)

  def _resolve_original_path(self, param: Parameter, state_key: str) -> str | None:
    """Compute workspace-relative path for a parameter's state key."""
    return snapshot_mod.resolve_original_path(self, param, state_key)

  def _snapshot_all_params(self) -> dict[str, str]:
    """Snapshot all registered parameters into a flat dict."""
    return snapshot_mod.snapshot_all_params(self)

  def _group_by_param(self, snap: SnapshotManifest) -> dict[str, dict[str, str]]:
    """Group snapshot entries by parameter name."""
    return snapshot_mod.group_by_param(self, snap)

  def store_blob(self, content_hash: str, data: bytes) -> None:
    """Write a content-addressed blob to the object store.

    Args:
      content_hash: SHA-256 hex digest of the data.
      data: Raw bytes to store.
    """
    self._backend.store_object(content_hash, data)

  def _store_object_bytes(self, content_hash: str, data: bytes) -> None:
    self._backend.store_object(content_hash, data)

  def read_object(self, content_hash: str) -> bytes:
    """Reads a content-addressed blob by its SHA-256 hash.

    Args:
      content_hash: Full hex SHA-256 of the stored object.

    Returns:
      Raw bytes of the stored object.

    Raises:
      StoreError: If the stored bytes hash differs from ``content_hash``.
    """
    data = self._backend.read_object(content_hash)
    actual = hash_bytes(data)
    if actual != content_hash:
      msg = f'object {content_hash!r} corrupted: expected hash {content_hash!r}, got {actual!r}'
      raise StoreError(msg)
    return data

  def load_refs(self) -> dict[str, Any]:
    """Loads the refs structure (branches, HEAD, worktrees).

    Returns:
      Parsed refs dict, or empty dict if no refs file exists.
    """
    return refs_mod.load_refs(self)

  def save_refs(self, refs: dict[str, Any]) -> None:
    """Persist the refs structure atomically.

    Args:
      refs: Complete refs dict to write.
    """
    refs_mod.save_refs(self, refs)

  def load_snapshot(self, experiment_id: str, epoch: int) -> SnapshotManifest:
    """Loads a snapshot manifest from disk.

    Args:
      experiment_id: Branch that owns the snapshot.
      epoch: Epoch index to load.

    Returns:
      Deserialized SnapshotManifest.

    Raises:
      StoreError: If the snapshot file is missing.
    """
    return snapshot_mod.load_snapshot_manifest(self, experiment_id, epoch)

  def _save_snapshot(self, experiment_id: str, epoch: int, manifest: SnapshotManifest) -> None:
    snapshot_mod.persist_snapshot_manifest(self, experiment_id, epoch, manifest)

  def _enumerate_snapshot_epochs(self, exp_dir: Path) -> list[int]:
    """List existing epoch numbers from snapshot files, sorted ascending."""
    return snapshot_mod.enumerate_snapshot_epochs(exp_dir)

  def _remove_extraneous_files(
    self,
    param: Any,
    param_name: str,
    snapshot_keys: set[str],
  ) -> None:
    """Remove working-tree files absent from the snapshot overlay."""
    snapshot_mod.remove_extraneous_files(self, param, param_name, snapshot_keys)

  def _acquire_lock(self) -> None:
    self._backend.acquire_lock()

  def _release_lock(self) -> None:
    self._backend.release_lock()

  def _text_diff(self, key: str, old_content: bytes, new_content: bytes) -> str:
    return snapshot_mod.text_diff_content(key, old_content, new_content)

  def __repr__(self) -> str:
    """Returns a developer-friendly representation with store path and param count."""
    return f'FileStore(store_path={self._config.store_path}, parameters={len(self._param_names)})'
