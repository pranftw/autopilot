"""Peripheral FileStoreCore behaviors split out for module size budgets.

Handles snapshot logs, path resolution, worktrees, doctor/repair, forest state,
reflog, stash, epoch copy, and tags. Intended only as a mixin with FileStoreCore.
"""

from autopilot.ai.store import doctor as doctor_mod
from autopilot.ai.store import peripherals as peripherals_mod
from autopilot.ai.store import refs as refs_mod
from autopilot.ai.store import snapshot as snapshot_mod
from autopilot.ai.store import worktree as worktree_mod
from autopilot.core.config import AutoPilotConfig
from autopilot.core.diagnostic import DiagnosticEntry
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.types import SnapshotEntry, TagEntry
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any


class FileStoreCorePeripheralMixin:
  """Mixin delegating auxiliary store operations to peripheral modules."""

  _config: AutoPilotConfig

  # log

  def log(self, experiment_id: str) -> list[SnapshotEntry]:
    """Returns the chronological snapshot history for an experiment.

    Args:
      experiment_id: Branch to list history for.

    Returns:
      List of SnapshotEntry for each existing epoch.
    """
    return snapshot_mod.experiment_log(self, experiment_id)

  # path resolution

  def resolve_path(self, experiment_id: str, epoch: int | None = None) -> Path:
    """Returns the filesystem path for an experiment or specific epoch.

    Args:
      experiment_id: Branch to resolve path for.
      epoch: If given, returns the epoch-specific sub-path.

    Returns:
      Absolute path (created if missing).
    """
    if epoch is not None:
      path = self._config.store_epoch_path(experiment_id=experiment_id, epoch=epoch)
    else:
      path = self._config.store_experiment_path(experiment_id=experiment_id)
    path.mkdir(parents=True, exist_ok=True)
    return path

  # worktrees (delegated to worktree_mod)

  def create_worktree(self, experiment_id: str) -> Path:
    """Creates a worktree directory for ``experiment_id``.

    Args:
      experiment_id: Branch to create a worktree for.

    Returns:
      Absolute path to the created worktree directory.
    """
    return worktree_mod.create_worktree(self, experiment_id)

  def remove_worktree(self, experiment_id: str) -> None:
    """Removes the worktree directory and deregisters from refs.

    Args:
      experiment_id: Branch whose worktree should be removed.
    """
    worktree_mod.remove_worktree(self, experiment_id)

  def list_worktrees(self) -> list[str]:
    """Returns sorted list of experiment IDs with registered worktrees."""
    return worktree_mod.list_worktrees(self)

  # doctor / repair / prune (delegated to doctor_mod)

  def doctor(self) -> list[DiagnosticEntry]:
    """Detect and diagnose store health issues.

    Returns:
      List of DiagnosticEntry instances.
    """
    return doctor_mod.doctor(self)

  def doctor_report(self) -> dict[str, Any]:
    """Legacy dict-shaped doctor report.

    Returns:
      Dict with health status keys.
    """
    return doctor_mod.doctor_report(self)

  def diagnostics_to_report(self, entries: list[DiagnosticEntry]) -> dict[str, Any]:
    """Convert DiagnosticEntry list to legacy report dict.

    Args:
      entries: Diagnostic entries from ``doctor()``.

    Returns:
      Dict with legacy keys.
    """
    return doctor_mod.diagnostics_to_report(entries)

  def repair_diagnostics(
    self,
    entries: list[DiagnosticEntry],
    *,
    dry_run: bool = False,
    context: str | None = None,
  ) -> list[DiagnosticEntry]:
    """Apply repairs for repairable diagnostic entries.

    Args:
      entries: Diagnostic entries from ``doctor()``.
      dry_run: When True, no mutations are applied.
      context: Reason/provenance string.

    Returns:
      List of repaired entries.
    """
    return doctor_mod.repair_diagnostics(self, entries, dry_run=dry_run, context=context)

  def prune_orphans(self) -> list[str]:
    """Remove orphaned blobs not reachable from any snapshot manifest.

    Returns:
      List of removed blob digests.
    """
    return doctor_mod.prune_orphans(self)

  # forest persistence

  def save_state_dict(self, state: dict) -> None:
    """Persists the forest state dict to ``forest.json`` atomically.

    Args:
      state: Serialized forest/tree structure to write.
    """
    refs_mod.save_forest_state(self, state)

  def load_state_dict(self) -> dict | None:
    """Loads the forest state dict from ``forest.json``.

    Returns:
      Parsed dict, or None if the file does not exist.

    Raises:
      StoreError: If the file exists but cannot be parsed.
    """
    return refs_mod.load_forest_state(self)

  # reflog (delegated to peripherals_mod)

  @property
  def reflog_path(self) -> Path:
    """Path to the append-only reflog JSONL file."""
    return self._reflog_path

  @property
  def _reflog_path(self) -> Path:
    """Path to the append-only reflog JSONL file."""
    return self._config.store_path / 'reflog.jsonl'

  def _append_reflog(
    self,
    operation: str,
    experiment_id: str,
    old_epoch: int | None,
    new_epoch: int | None,
    context: str | None = None,
    **extra: Any,
  ) -> None:
    """Append a reflog entry after a successful refs mutation."""
    peripherals_mod.append_reflog(
      self, operation, experiment_id, old_epoch, new_epoch, context, **extra
    )

  def iter_reflog(self) -> Iterator[dict[str, Any]]:
    """Iterate reflog entries in chronological order (oldest first).

    Returns:
      Iterator yielding one dict per valid JSONL line.
    """
    return peripherals_mod.iter_reflog(self)

  def expire_reflog(self, older_than: timedelta) -> int:
    """Drop reflog entries older than the cutoff.

    Args:
      older_than: Age cutoff.

    Returns:
      Number of entries expired.
    """
    return peripherals_mod.expire_reflog(self, older_than)

  def recover_from_reflog(self, entry_index: int) -> None:
    """Restore branch tip metadata from reflog entry.

    Args:
      entry_index: 0-based index into valid reflog entries.
    """
    peripherals_mod.recover_from_reflog(self, entry_index)

  # stash (delegated to peripherals_mod)

  @property
  def _stash_dir(self) -> Path:
    """Path to the stash manifest directory."""
    return self._config.store_path / 'stash'

  def stash(self, context: str | None = None) -> SnapshotManifest:
    """Capture current registered parameter state as a stash.

    Args:
      context: Optional reason string.

    Returns:
      The stash manifest.
    """
    return peripherals_mod.stash(self, context)

  def stash_list(self) -> list[SnapshotManifest]:
    """Return stash manifests ordered oldest to newest.

    Returns:
      List of SnapshotManifest instances.
    """
    return peripherals_mod.stash_list(self)

  def stash_pop(self, index: int | None = None, *, context: str | None = None) -> SnapshotManifest:
    """Restore stash and remove the stash file.

    Args:
      index: Explicit stash index.
      context: Optional audit provenance string recorded in the reflog
        entry for this stash_pop operation.

    Returns:
      The manifest that was popped.
    """
    return peripherals_mod.stash_pop(self, index, context=context)

  # copy_epoch (delegated to peripherals_mod)

  def copy_epoch(
    self,
    source_experiment_id: str,
    source_epoch: int,
    target_experiment_id: str,
    *,
    context: str | None = None,
  ) -> SnapshotManifest:
    """Copy snapshot manifest from source to target as next epoch.

    Args:
      source_experiment_id: Branch to read from.
      source_epoch: Epoch to copy.
      target_experiment_id: Branch receiving the new epoch.
      context: Optional audit string.

    Returns:
      SnapshotManifest persisted at the new target epoch.
    """
    return peripherals_mod.copy_epoch(
      self, source_experiment_id, source_epoch, target_experiment_id, context=context
    )

  # tags (delegated to peripherals_mod)

  def tag(
    self,
    name: str,
    experiment_id: str,
    epoch: int,
    context: str | None = None,
  ) -> None:
    """Create an immutable tag.

    Args:
      name: Tag name.
      experiment_id: Branch the tag points to.
      epoch: Epoch the tag points to.
      context: Optional reason string.
    """
    peripherals_mod.tag(self, name, experiment_id, epoch, context)

  def get_tag(self, name: str) -> TagEntry | None:
    """Look up a tag by name.

    Args:
      name: Tag name.

    Returns:
      TagEntry if found, None otherwise.
    """
    return peripherals_mod.get_tag(self, name)

  def list_tags(self) -> list[TagEntry]:
    """List all tags, sorted by name.

    Returns:
      List of TagEntry instances.
    """
    return peripherals_mod.list_tags(self)

  def verify_tag(self, name: str) -> dict[str, Any]:
    """Verify a tag's manifest digest.

    Args:
      name: Tag name.

    Returns:
      Verification result dict.
    """
    return peripherals_mod.verify_tag(self, name)
