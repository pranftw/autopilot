"""LCA discovery and merge key classification for refs + snapshot manifests."""

from autopilot.core.errors import StoreError
from autopilot.core.snapshot import SnapshotManifest
from typing import Any


def find_lca(
  exp_a: str,
  exp_b: str,
  refs: dict[str, Any],
) -> tuple[str | None, int | None]:
  """Find lowest common ancestor using BFS on the refs DAG.

  Args:
    exp_a: First experiment id.
    exp_b: Second experiment id.
    refs: Full refs structure.

  Returns:
    ``(experiment_id, epoch)`` of the LCA, or ``(None, None)`` when
    no common ancestor exists.
  """
  branches = refs.get('branches', {})
  ancestors_a = ancestor_set(exp_a, branches)
  chain_b = ancestor_chain_ordered(exp_b, branches)

  for exp_id, epoch in chain_b:
    if (exp_id, epoch) in ancestors_a:
      return (exp_id, epoch)

  return (None, None)


def ancestor_set(
  experiment_id: str,
  branches: dict[str, Any],
) -> set[tuple[str, int]]:
  """Collect all ancestor (experiment_id, epoch) pairs via BFS.

  Returns:
    Set of ``(experiment_id, epoch)`` tuples reachable from the given
    experiment through parent and merge-parent links.
  """
  visited: set[tuple[str, int]] = set()
  info = branches.get(experiment_id)
  if info is None:
    return visited

  latest = info.get('latest_epoch', 0)
  queue = [(experiment_id, ep) for ep in range(latest + 1)]

  while queue:
    exp_id, epoch = queue.pop(0)
    if (exp_id, epoch) in visited:
      continue
    visited.add((exp_id, epoch))

    branch_info = branches.get(exp_id)
    if branch_info is None:
      continue

    enqueue_parent(branch_info, branches, visited, queue)
    enqueue_merge_parents(branch_info, branches, visited, queue)

  return visited


def enqueue_parent(
  branch_info: dict[str, Any],
  branches: dict[str, Any],
  visited: set[tuple[str, int]],
  queue: list[tuple[str, int]],
) -> None:
  """Add parent experiment epochs to the BFS queue if not yet visited."""
  parent_id = branch_info.get('parent_id')
  if parent_id is None or parent_id not in branches:
    return
  parent_epoch = branch_info.get('parent_epoch', 0)
  queue.extend((parent_id, ep) for ep in range(parent_epoch + 1) if (parent_id, ep) not in visited)


def enqueue_merge_parents(
  branch_info: dict[str, Any],
  branches: dict[str, Any],
  visited: set[tuple[str, int]],
  queue: list[tuple[str, int]],
) -> None:
  """Add merge-parent experiment epochs to the BFS queue if not yet visited."""
  for mp in branch_info.get('merge_parents', []):
    mp_exp = mp['experiment_id']
    mp_epoch = mp['epoch']
    if mp_exp not in branches:
      continue
    queue.extend((mp_exp, ep) for ep in range(mp_epoch + 1) if (mp_exp, ep) not in visited)


def ancestor_chain_ordered(
  experiment_id: str,
  branches: dict[str, Any],
) -> list[tuple[str, int]]:
  """Build an ordered ancestor chain for LCA search.

  Returns highest-epoch entries first within each experiment for
  greedy LCA matching.

  Returns:
    List of ``(experiment_id, epoch)`` pairs ordered for greedy LCA
    matching.
  """
  info = branches.get(experiment_id)
  if info is None:
    return []

  latest = info.get('latest_epoch', 0)
  chain = [(experiment_id, ep) for ep in range(latest, -1, -1)]

  current = experiment_id
  visited: set[str] = {current}
  while True:
    branch_info = branches.get(current)
    if branch_info is None:
      break
    parent_id = branch_info.get('parent_id')
    if parent_id is None or parent_id in visited:
      break
    visited.add(parent_id)
    parent_info = branches.get(parent_id)
    if parent_info is None:
      break
    parent_epoch = branch_info.get('parent_epoch', 0)
    chain.extend((parent_id, ep) for ep in range(parent_epoch, -1, -1))
    current = parent_id

  return chain


def load_ancestor_snapshot(
  store: Any,
  ancestor_exp: str | None,
  ancestor_epoch: int | None,
) -> SnapshotManifest:
  """Load the ancestor snapshot, returning empty manifest when no LCA.

  Returns:
    Snapshot at the LCA, or an empty manifest if no ancestor exists.
  """
  if ancestor_exp is not None and ancestor_epoch is not None:
    return store.load_snapshot(ancestor_exp, ancestor_epoch)
  return SnapshotManifest(epoch=0, timestamp='', entries={})


def require_both_branches(
  experiment_id: str,
  from_experiment_id: str,
  branches: dict[str, Any],
) -> None:
  """Validate both branches exist in refs.

  Raises:
    StoreError: If either experiment is missing from refs.
  """
  if experiment_id not in branches:
    msg = f'experiment {experiment_id!r} not found'
    raise StoreError(msg)
  if from_experiment_id not in branches:
    msg = f'experiment {from_experiment_id!r} not found'
    raise StoreError(msg)


def changed_keys(
  ancestor_snap: SnapshotManifest,
  current_snap: SnapshotManifest,
) -> set[str]:
  """Return manifest keys that differ between ancestor and current."""
  changed: set[str] = set()
  all_keys = set(ancestor_snap.entries) | set(current_snap.entries)
  for key in all_keys:
    anc = ancestor_snap.entries.get(key)
    cur = current_snap.entries.get(key)
    anc_hash = anc.digest if anc else None
    cur_hash = cur.digest if cur else None
    if anc_hash != cur_hash:
      changed.add(key)
  return changed


def divergent_keys(
  overlap: set[str],
  ours_snap: SnapshotManifest,
  theirs_snap: SnapshotManifest,
) -> set[str]:
  """Filter overlapping keys to those with divergent digests.

  Args:
    overlap: Keys changed on both sides relative to the ancestor.
    ours_snap: Snapshot from the target branch.
    theirs_snap: Snapshot from the source branch.

  Returns:
    Subset of overlap where ours and theirs have different digests.
  """
  divergent: set[str] = set()
  for key in overlap:
    ours_entry = ours_snap.entries.get(key)
    theirs_entry = theirs_snap.entries.get(key)
    ours_digest = ours_entry.digest if ours_entry else None
    theirs_digest = theirs_entry.digest if theirs_entry else None
    if ours_digest != theirs_digest:
      divergent.add(key)
  return divergent
