"""Curried branch accessor and refs iteration for Store.

``BranchHandle`` binds a ``Store`` and ``experiment_id`` together, providing
a curried API for per-branch operations (snapshot, checkout, log, diff,
latest_epoch) without overloading ``Store.branch()``, which creates
branches.

``RefsView`` is a read-only iterable view over all branches known to the
store, supporting ``__getitem__``, ``__contains__``, ``__iter__``, and
``__len__``.

Relationship to ``Store.branch()``:

- ``Store.branch(experiment_id)`` -- **creates** a new branch.
- ``Store.branch_handle(experiment_id)`` -- returns a ``BranchHandle``
  that **curries** operations for an existing branch.
"""

from autopilot.core.errors import StoreError
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.base import Store
from autopilot.core.store.types import DiffResult, SnapshotEntry
from collections.abc import Iterator
from typing import Any


class BranchHandle:
  """Curried store operations for a single experiment branch.

  Delegates all operations to the underlying ``Store`` instance,
  binding ``experiment_id`` so callers do not repeat it on every call.

  Attributes:
    store: The store instance this handle delegates to.
    experiment_id: The branch name this handle is bound to.
  """

  def __init__(self, store: Store, experiment_id: str) -> None:
    """Bind a store and experiment_id for curried branch operations.

    Args:
      store: Store instance providing the underlying operations.
      experiment_id: Branch name to bind.
    """
    self.store = store
    self.experiment_id = experiment_id

  def snapshot(
    self,
    epoch: int,
    experiment: Any = None,
    context: str | None = None,
    *,
    force: bool = False,
  ) -> SnapshotManifest:
    """Capture parameter state for this branch at the given epoch.

    Delegates to ``store.snapshot(experiment_id, epoch, ...)``.

    Args:
      epoch: Zero-based epoch number.
      experiment: Optional experiment instance for strict completion checks.
      context: Optional reason/provenance string for audit traceability.
      force: When True, record a new epoch even when files are unchanged.

    Returns:
      Manifest describing the persisted snapshot.
    """
    return self.store.snapshot(
      self.experiment_id,
      epoch,
      experiment,
      context=context,
      force=force,
    )

  def checkout(self, epoch: int, context: str | None = None) -> None:
    """Restore parameter files to the snapshot at ``epoch`` for this branch.

    Delegates to ``store.checkout(experiment_id, epoch, context=context)``.

    Args:
      epoch: Epoch to restore.
      context: Optional reason/provenance string for audit traceability.
    """
    self.store.checkout(self.experiment_id, epoch, context=context)

  def log(self) -> list[SnapshotEntry]:
    """Return chronological snapshot history for this branch.

    Delegates to ``store.log(experiment_id)``.

    Returns:
      List of ``SnapshotEntry`` summaries ordered by epoch.
    """
    return self.store.log(self.experiment_id)

  def diff(self, epoch_a: int, epoch_b: int) -> DiffResult:
    """Compare two snapshots within this branch.

    Delegates to ``store.diff(experiment_id, epoch_a, epoch_b)``.

    Args:
      epoch_a: Base epoch (earlier).
      epoch_b: Target epoch (later).

    Returns:
      Diff result with per-key change entries.
    """
    return self.store.diff(self.experiment_id, epoch_a, epoch_b)

  def latest_epoch(self) -> int:
    """Read the latest epoch for this branch from store refs.

    Loads refs via ``store.load_refs()`` and reads
    ``['branches'][experiment_id]['latest_epoch']``.

    Returns:
      Latest epoch number (may be -1 after branch reset).

    Raises:
      StoreError: If the branch does not exist in refs.
    """
    refs = self.store.load_refs()
    branches: dict[str, Any] = refs.get('branches', {})
    if self.experiment_id not in branches:
      msg = (
        f'branch {self.experiment_id!r} not found in store refs. '
        f'Create the branch first with store.branch() or store.snapshot().'
      )
      raise StoreError(msg)
    return branches[self.experiment_id]['latest_epoch']


class RefsView:
  """Iterable read-only view over store branches.

  Provides dict-like access to ``BranchHandle`` instances by branch name,
  plus ``__contains__``, ``__iter__``, and ``__len__`` for enumeration.

  Attributes:
    store: The store instance providing branch data.
  """

  def __init__(self, store: Store) -> None:
    """Create a refs view over the given store.

    Args:
      store: Store instance to read branch data from.
    """
    self.store = store

  def _branches(self) -> dict[str, Any]:
    """Load current branches dict from store refs.

    Returns:
      Branches mapping from refs, or empty dict if no refs exist.
    """
    return self.store.load_refs().get('branches', {})

  def __getitem__(self, name: str) -> BranchHandle:
    """Get a ``BranchHandle`` for the named branch.

    Args:
      name: Branch / experiment id.

    Returns:
      BranchHandle bound to the named branch.

    Raises:
      StoreError: If the branch does not exist.
    """
    if name not in self._branches():
      msg = (
        f'branch {name!r} not found in store refs. '
        f'Create the branch first with store.branch() or store.snapshot().'
      )
      raise StoreError(msg)
    return BranchHandle(self.store, name)

  def __contains__(self, name: object) -> bool:
    """Check whether a branch name exists in refs.

    Args:
      name: Branch name to check.

    Returns:
      True if the branch exists.
    """
    if not isinstance(name, str):
      return False
    return name in self._branches()

  def __iter__(self) -> Iterator[str]:
    """Iterate over all branch names.

    Returns:
      Iterator of branch name strings.
    """
    return iter(self._branches())

  def __len__(self) -> int:
    """Return the number of branches.

    Returns:
      Count of branches in refs.
    """
    return len(self._branches())
