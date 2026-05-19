"""Unified persistence layer for experiment data, worktrees, and code versioning.

Store is the single persistence abstraction for the optimization framework.
Config-based path resolution: Store root and all internal paths come from
config (e.g. config.store_path). experiment.id is the branch name by
convention -- no separate StoreRef or WorktreeRef wrapper classes.

Built-in subclass: FileStore (ai/store/) with SHA-256 content addressing.
"""

from autopilot.core.config import AutoPilotConfig, Config
from autopilot.core.parameter import Parameter
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.types import (
  DiffResult,
  MergeAnalysisResult,
  MergeIndex,
  MergeStrategy,
  SnapshotEntry,
  StatusResult,
  TagEntry,
)
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any
import abc


class Store:
  """Unified persistence layer with Config-based path resolution.

  Constructor: ``Store(config)`` -- config-only. Parameters registered
  separately via ``register_parameters(dict)`` with named keys matching
  module attribute names. This base class declares the protocol;
  ``FileStore`` (ai/store/) is the built-in concrete implementation.

  Attributes:
    config: Path and storage configuration passed at construction.

  Merge lifecycle (three-step API):
    1. ``merge_analysis`` -- cheap classification (fast-forward, clean, conflict,
       up-to-date) using refs and manifest key overlap; no blob reads.
    2. ``merge_preview`` -- materialize ``MergeIndex`` with ``ConflictEntry``
       triples and auto-resolved entries per ``MergeStrategy``; sets a
       cryptographic ``preview_token``.
    3. ``merge_apply`` -- persist the resolved ``MergeIndex`` as a new epoch
       on the target experiment; validates ``preview_token`` freshness.

  Concurrency model: single-writer. FileStore uses ``O_EXCL`` file creation
  for branch locking (fail-fast, no retry). Concurrent snapshot() calls
  for different experiment_ids are safe. Concurrent snapshot() calls for
  the SAME experiment_id will fail with StoreError. Not thread-safe for
  reads during writes.   Public API: snapshot(), checkout(), log(), diff(),
  status(), merge_analysis(), merge_preview(), merge_apply(),
  load_snapshot(), read_object(), store_blob(), load_refs(),
  prune_orphans(), reset_branch(), tag(), get_tag(), list_tags(),
  copy_epoch().
  ``FileStore`` adds ``doctor()`` for read-only health diagnostics
  and ``save_refs()`` for persisting refs mutations.

  Lock recovery:
    If a ``.lock`` file is stale after a crashed process, remove it manually:
    ``rm .store/.lock`` (adjust path per your configured store root).

  Manages multiple experiments -- individual operations take experiment_id
  as parameter. experiment.id is the branch name by convention.

  See ``FileStore`` for the concrete content-addressed implementation details.

  Subclass and override to customize storage, hashing, or any operation.

  Example:
    Typical call flow on :class:`~autopilot.ai.store.file_store.FileStore` uses
    ``register_parameters``, ``branch``, ``snapshot``, ``checkout``, and ``diff``:

    >>> from pathlib import Path
    >>> from autopilot.ai.store.file_store import FileStore
    >>> from autopilot.core.config import AutoPilotConfig
    >>> from autopilot.core.parameter import ScalarParameter
    >>>
    >>> def _demo_store_workflow():
    ...   cfg = AutoPilotConfig(workspace=Path('/tmp/autopilot-doc-example'))
    ...   store = FileStore(cfg)
    ...   store.register_parameters({'prompt': ScalarParameter(value='hello')})
    ...   store.branch('exp-a')
    ...   manifest = store.snapshot('exp-a', 0, context='baseline')
    ...   store.checkout('exp-a', manifest.epoch)
    ...   return store.diff('exp-a', 0, manifest.epoch)
    >>> _demo_store_workflow()  # doctest: +SKIP
  """

  def __init__(self, config: Config) -> None:
    """Config-only constructor. Parameters registered separately via register_parameters.

    Args:
      config: Path and storage configuration.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  def register_parameters(self, parameters: dict[str, Parameter]) -> None:
    """Register named parameters for snapshotting.

    Keys are module attribute names (stable across reordering). Must be
    called before snapshot/checkout/status operations that require
    parameter content.

    Args:
      parameters: Mapping from attribute name to Parameter instance.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  @property
  def config(self) -> AutoPilotConfig:
    """Path and layout configuration for this store.

    Subclasses must override to return their concrete Config instance.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  @property
  def lock_timeout_s(self) -> float | None:
    """Effective lock timeout in seconds (``None`` = fail-fast).

    Subclasses override to expose their backend lock timeout.
    Base returns ``None`` (fail-fast).
    """
    return None

  @lock_timeout_s.setter
  def lock_timeout_s(self, value: float | None) -> None:
    """Update the lock timeout dynamically. No-op on base class."""

  # vcs operations

  def snapshot(
    self,
    experiment_id: str,
    epoch: int,
    experiment: Any = None,
    context: str | None = None,
    *,
    force: bool = False,
  ) -> SnapshotManifest:
    """Capture current state of all parameter files.

    By default, skips the snapshot when file content is identical to
    the latest epoch (idempotent).  Pass ``force=True`` to always
    persist a new epoch.

    Args:
      experiment_id: Branch name to snapshot.
      epoch: Zero-based epoch number.
      experiment: Optional ``Experiment`` instance for post-completion
        snapshot policy enforcement (see ``FileStore.snapshot``).
      context: Optional reason/provenance string recorded on the manifest
        for audit traceability.
      force: When True, always persist a new epoch even if file content
        is unchanged from the latest snapshot.

    Returns:
      Manifest describing the persisted snapshot, or the prior manifest
      when skipped (idempotent path).
    """
    raise NotImplementedError

  def checkout(
    self,
    experiment_id: str,
    epoch: int,
    *,
    context: str | None = None,
  ) -> None:
    """Restore all parameter files to a snapshot state.

    Implementations that persist refs must update HEAD to experiment_id here
    (see FileStore) so disk state matches the active branch pointer.

    Args:
      experiment_id: Branch to checkout.
      epoch: Epoch to restore.
      context: Optional reason/provenance string recorded in the reflog
        for audit traceability.
    """
    raise NotImplementedError

  def diff(self, experiment_id: str, epoch_a: int, epoch_b: int) -> DiffResult:
    """Compare two snapshots of the same experiment.

    Args:
      experiment_id: Branch containing both snapshots.
      epoch_a: Base epoch (earlier).
      epoch_b: Target epoch (later).

    Returns:
      Diff result with per-key change entries.
    """
    raise NotImplementedError

  def branch(self, experiment_id: str) -> None:
    """Create new branch from HEAD. Copies HEAD's latest snapshot as epoch 0."""
    raise NotImplementedError

  def reset_branch(self, experiment_id: str) -> None:
    """Reset a branch's latest_epoch to -1, enabling re-run from epoch 0.

    Existing snapshot manifests are retained (not deleted). HEAD is unchanged.
    The next ``snapshot()`` call for this branch succeeds at epoch 0.

    This is refs-only; use ``reset_and_restore`` when working-tree files
    must sync in the same operation.

    Args:
      experiment_id: Branch to reset.

    Raises:
      StoreError: If the branch does not exist.
    """
    raise NotImplementedError

  def reset_and_restore(
    self,
    experiment_id: str,
    epoch: int | None = None,
    *,
    context: str | None = None,
  ) -> None:
    """Atomically reset branch tip and sync working-tree parameter files.

    When ``epoch`` is ``None``: set ``latest_epoch`` to ``-1`` (empty branch
    tip) and remove tracked parameter files from the working tree.

    When ``epoch`` is an ``int``: set ``latest_epoch`` to that epoch and
    restore working-tree files from that epoch's snapshot.

    Appends two reflog entries under one lock: ``reset_branch`` then
    ``checkout``. Existing snapshot manifests are never deleted.

    Args:
      experiment_id: Branch to reset.
      epoch: Target tip epoch, or ``None`` for empty state (-1).
      context: Optional reason/provenance string recorded on both reflog
        entries for audit traceability.

    Raises:
      StoreError: If the branch does not exist, the epoch does not exist
        (when ``epoch`` is not ``None``), or ``epoch`` exceeds the current
        ``latest_epoch``.
    """
    raise NotImplementedError

  def merge_analysis(
    self,
    experiment_id: str,
    from_experiment_id: str,
  ) -> MergeAnalysisResult:
    """Classify merge: fast-forward, clean merge, conflicts, or up-to-date.

    Uses cheap refs + manifest key overlap (no blob reads). Same key touched
    on both sides since LCA implies ``has_conflicts=True``.

    Args:
      experiment_id: Target experiment branch receiving the merge.
      from_experiment_id: Source experiment branch to merge from.

    Returns:
      MergeAnalysisResult with classification and predicted conflict count.

    Raises:
      NotImplementedError: On this abstract base.
    """
    raise NotImplementedError

  def merge_preview(
    self,
    experiment_id: str,
    from_experiment_id: str,
    from_epoch: int | None = None,
    strategy: MergeStrategy = MergeStrategy.normal,
  ) -> MergeIndex:
    """Compute three-way merge into a MergeIndex staging area.

    Materializes ``ConflictEntry`` triples for each conflicted key and
    auto-resolves per strategy (ours/theirs clear conflicts into resolved).
    Sets ``preview_token`` from a stable hash of salient inputs.

    Args:
      experiment_id: Target experiment branch receiving the merge.
      from_experiment_id: Source experiment branch to merge from.
      from_epoch: Optional epoch in the source branch; defaults to latest.
      strategy: Resolution strategy for auto-resolving conflicts.

    Returns:
      MergeIndex staging area with conflicts and resolved entries.

    Raises:
      NotImplementedError: On this abstract base.
    """
    raise NotImplementedError

  def merge_apply(self, merge_index: MergeIndex) -> SnapshotManifest:
    """Persist a resolved merge as a new epoch on the target experiment.

    Validates ``preview_token`` matches current refs state (stale token
    raises ``StoreError``). Requires ``merge_index.is_resolved()`` for
    normal strategy; ours/theirs auto-resolve during preview.

    Args:
      merge_index: Fully resolved MergeIndex from ``merge_preview``.

    Returns:
      SnapshotManifest written for the new merge epoch.

    Raises:
      StoreError: If preview token is stale or conflicts remain unresolved.
      NotImplementedError: On this abstract base.
    """
    raise NotImplementedError

  def log(self, experiment_id: str) -> list[SnapshotEntry]:
    """History of all snapshots for an experiment."""
    raise NotImplementedError

  def status(self, experiment_id: str) -> StatusResult:
    """Compare current files on disk against latest snapshot."""
    raise NotImplementedError

  def materialize(self, experiment_id: str, epoch: int) -> None:
    """Materialize a snapshot as the current stable state."""
    raise NotImplementedError

  def prune_orphans(self) -> list[str]:
    """Remove orphaned object blobs not reachable from any snapshot manifest.

    Base implementation is a documented no-op. ``FileStore`` overrides with
    a filesystem walk against manifest edges.

    Returns:
      List of removed blob digests (empty on the base class).
    """
    return []

  # worktrees

  def create_worktree(self, experiment_id: str) -> Path:
    """Create an isolated working directory for parallel execution. Returns worktree path."""
    raise NotImplementedError

  def remove_worktree(self, experiment_id: str) -> None:
    """Remove a worktree and clean up its directory."""
    raise NotImplementedError

  def list_worktrees(self) -> list[str]:
    """Return experiment_ids of all active worktrees."""
    raise NotImplementedError

  # path resolution

  def resolve_path(self, experiment_id: str, epoch: int | None = None) -> Path:
    """Resolve path for experiment data. Epoch-scoped if epoch provided."""
    raise NotImplementedError

  # direct data access

  @abc.abstractmethod
  def load_snapshot(self, experiment_id: str, epoch: int) -> SnapshotManifest:
    """Load a snapshot manifest for the given experiment and epoch."""

  @abc.abstractmethod
  def read_object(self, content_hash: str) -> bytes:
    """Read raw object content by content hash."""

  @abc.abstractmethod
  def store_blob(self, content_hash: str, data: bytes) -> None:
    """Write a content-addressed blob to the object store.

    Args:
      content_hash: SHA-256 hex digest of the data.
      data: Raw bytes to store.
    """

  @abc.abstractmethod
  def load_refs(self) -> dict[str, Any]:
    """Load the refs structure (branches, HEAD string experiment id, etc.).

    HEAD here is FileStore-level, not Tree._head. Returns empty dict if no
    refs file exists.
    """

  # tags

  def tag(
    self,
    name: str,
    experiment_id: str,
    epoch: int,
    context: str | None = None,
  ) -> None:
    """Create an immutable tag pointing to a specific experiment and epoch.

    Args:
      name: Tag name (alphanumeric, '-', '_', '.' only; max 128 chars).
      experiment_id: Branch the tag points to.
      epoch: Epoch the tag points to.
      context: Optional reason string for audit traceability.

    Raises:
      StoreError: If the tag name is invalid, already exists, or the
        branch/epoch does not exist.
    """
    raise NotImplementedError

  def verify_tag(self, name: str) -> dict[str, Any]:
    """Verify a tag's manifest digest against the current on-disk manifest.

    Args:
      name: Tag name to verify.

    Returns:
      Verification result dict (see ``FileStore.verify_tag`` for shapes).

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  def get_tag(self, name: str) -> TagEntry | None:
    """Look up a tag by name.

    Args:
      name: Tag name to look up.

    Returns:
      TagEntry if found, None otherwise.
    """
    raise NotImplementedError

  def list_tags(self) -> list[TagEntry]:
    """List all tags, sorted by name.

    Returns:
      List of TagEntry instances sorted alphabetically by name.
    """
    raise NotImplementedError

  # copy-epoch

  def copy_epoch(
    self,
    source_experiment_id: str,
    source_epoch: int,
    target_experiment_id: str,
    *,
    context: str | None = None,
  ) -> SnapshotManifest:
    """Copy snapshot manifest entries from source epoch to target branch next epoch.

    Content-addressed backends share blobs by digest (no byte duplication).

    Args:
      source_experiment_id: Branch to read from.
      source_epoch: Epoch to copy.
      target_experiment_id: Branch receiving the new epoch.
      context: Optional audit string stored on the new manifest.

    Returns:
      SnapshotManifest persisted at the new target epoch.

    Raises:
      StoreError: When branches or epochs are missing, or epoch sequence violated.
    """
    raise NotImplementedError

  # stash

  def stash(self, context: str | None = None) -> SnapshotManifest:
    """Capture current registered parameter state as a numbered stash manifest.

    Stash manifests use ``epoch = -1`` as a sentinel (distinct from real
    epoch values which are >= 0). Working parameter files remain unchanged
    after stash (capture-only, unlike ``git stash``).

    Args:
      context: Optional reason string recorded on the manifest.

    Returns:
      The stash manifest that was persisted.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  def stash_list(self) -> list[SnapshotManifest]:
    """Return stash manifests ordered oldest to newest (by stash index).

    Corrupt stash JSON files are skipped with a warning. Non-numeric
    filenames in the stash directory are ignored.

    Returns:
      List of SnapshotManifest instances, ordered by stash index ascending.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  def stash_pop(self, index: int | None = None, *, context: str | None = None) -> SnapshotManifest:
    """Restore stash to working parameters and remove the stash file.

    After pop, remaining stash files are renumbered to stay dense (0000..N-1).

    Args:
      index: Explicit stash index. When None, pop the newest stash (LIFO).
      context: Optional audit provenance string recorded in the reflog entry
        for this stash_pop operation. Non-CLI callers may omit (defaults to
        None, producing a null/absent context field in the reflog).

    Returns:
      The manifest that was popped and restored.

    Raises:
      NotImplementedError: Always on this abstract base. Concrete
        implementations raise ``StoreError`` when the stack is empty or
        the index is out of range.
    """
    raise NotImplementedError

  # reflog lifecycle

  def iter_reflog(self) -> Iterator[dict[str, Any]]:
    """Yield reflog entries in chronological order (oldest first).

    Skips corrupt JSONL lines after emitting a message to stderr.

    Yields:
      One dict per valid line, matching existing persisted reflog record shape.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  def expire_reflog(self, older_than: timedelta) -> int:
    """Drop reflog entries older than the cutoff; returns count removed.

    Args:
      older_than: Entries whose timestamp is strictly before
        (now - older_than) are removed.

    Returns:
      Number of entries expired (removed).

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  def recover_from_reflog(self, entry_index: int) -> None:
    """Restore branch tip metadata from reflog entry at linear index.

    Args:
      entry_index: 0-based index into the sequence of valid reflog entries.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    raise NotImplementedError

  # branch handle and refs view

  def branch_handle(self, experiment_id: str) -> Any:
    """Return a curried ``BranchHandle`` for the given experiment branch.

    ``BranchHandle`` binds a store and experiment_id, providing shorthand
    methods (``snapshot``, ``checkout``, ``log``, ``diff``, ``latest_epoch``)
    that delegate to the corresponding ``Store`` methods without requiring
    the caller to pass ``experiment_id`` on every call.

    This is the **accessor** factory -- it does not create branches.
    Use ``Store.branch(experiment_id)`` to create new branches.

    Args:
      experiment_id: Branch name to bind.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    msg = (
      f'{type(self).__name__} does not implement branch_handle(). '
      f'Use a concrete Store subclass such as FileStore.'
    )
    raise NotImplementedError(msg)

  @property
  def refs_view(self) -> Any:
    """Return an iterable ``RefsView`` over all store branches.

    ``RefsView`` provides ``__getitem__`` (returns ``BranchHandle``),
    ``__contains__``, ``__iter__``, and ``__len__`` for branch enumeration.

    Raises:
      NotImplementedError: Always on this abstract base.
    """
    msg = (
      f'{type(self).__name__} does not implement refs_view. '
      f'Use a concrete Store subclass such as FileStore.'
    )
    raise NotImplementedError(msg)

  # forest persistence

  def save_state_dict(self, state: dict) -> None:
    """Persist forest structure (trees, nodes) to forest.json."""
    raise NotImplementedError

  def load_state_dict(self) -> dict | None:
    """Load forest structure from forest.json, or None if not found."""
    raise NotImplementedError
