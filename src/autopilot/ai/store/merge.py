"""Merge pipeline, LCA algorithm, text merge, and preview tokens for FileStore."""

from autopilot.ai.store.merge_ancestry import (
  changed_keys,
  divergent_keys,
  find_lca,
  load_ancestor_snapshot,
  require_both_branches,
)
from autopilot.ai.store.merge_token import compute_preview_token, recompute_preview_token
from autopilot.ai.store.snapshot import FILE_ENTRY_MTIME_UNAVAILABLE
from autopilot.ai.store.text_merge import try_text_merge
from autopilot.ai.store_lock import hash_bytes
from autopilot.core.errors import StoreError
from autopilot.core.snapshot import FileEntry, SnapshotManifest
from autopilot.core.store.types import (
  ConflictEntry,
  MergeAnalysisResult,
  MergeClassification,
  MergeIndex,
  MergeStrategy,
)
from autopilot.tracking.io import utc_now_iso
from typing import Any

REFS_FORMAT_VERSION = 2


def merge_analysis(
  store: Any,
  experiment_id: str,
  from_experiment_id: str,
) -> MergeAnalysisResult:
  """Classify merge using refs and manifest key overlap (no blob reads).

  Args:
    store: FileStore instance.
    experiment_id: Target branch (ours).
    from_experiment_id: Source branch (theirs).

  Returns:
    MergeAnalysisResult with classification and predicted conflict count.
  """
  refs = store.load_refs()
  branches = refs.get('branches', {})
  require_both_branches(experiment_id, from_experiment_id, branches)

  ours_epoch = branches[experiment_id]['latest_epoch']
  theirs_epoch = branches[from_experiment_id]['latest_epoch']

  ancestor_exp, ancestor_epoch = find_lca(experiment_id, from_experiment_id, refs)

  if ancestor_exp == experiment_id and ancestor_epoch == ours_epoch:
    if ours_epoch == theirs_epoch and experiment_id == from_experiment_id:
      return MergeAnalysisResult(
        can_fast_forward=False,
        has_conflicts=False,
        conflict_count=0,
        ancestor_epoch=ancestor_epoch,
        classification=MergeClassification.up_to_date,
      )
    return MergeAnalysisResult(
      can_fast_forward=True,
      has_conflicts=False,
      conflict_count=0,
      ancestor_epoch=ancestor_epoch,
      classification=MergeClassification.fast_forward,
    )

  if ancestor_exp == from_experiment_id and ancestor_epoch == theirs_epoch:
    return MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=False,
      conflict_count=0,
      ancestor_epoch=ancestor_epoch,
      classification=MergeClassification.up_to_date,
    )

  ancestor_snap = load_ancestor_snapshot(store, ancestor_exp, ancestor_epoch)
  ours_snap = store.load_snapshot(experiment_id, ours_epoch)
  theirs_snap = store.load_snapshot(from_experiment_id, theirs_epoch)

  ours_changed = changed_keys(ancestor_snap, ours_snap)
  theirs_changed = changed_keys(ancestor_snap, theirs_snap)
  overlap = ours_changed & theirs_changed

  true_conflicts = divergent_keys(overlap, ours_snap, theirs_snap)

  if not theirs_changed:
    return MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=False,
      conflict_count=0,
      ancestor_epoch=ancestor_epoch,
      classification=MergeClassification.up_to_date,
    )
  if true_conflicts:
    return MergeAnalysisResult(
      can_fast_forward=False,
      has_conflicts=True,
      conflict_count=len(true_conflicts),
      ancestor_epoch=ancestor_epoch,
      classification=MergeClassification.conflict,
    )
  return MergeAnalysisResult(
    can_fast_forward=False,
    has_conflicts=False,
    conflict_count=0,
    ancestor_epoch=ancestor_epoch,
    classification=MergeClassification.clean,
  )


def merge_preview(
  store: Any,
  experiment_id: str,
  from_experiment_id: str,
  from_epoch: int | None = None,
  strategy: MergeStrategy = MergeStrategy.normal,
) -> MergeIndex:
  """Compute three-way merge into a MergeIndex staging area.

  Args:
    store: FileStore instance.
    experiment_id: Target branch (ours).
    from_experiment_id: Source branch (theirs).
    from_epoch: Epoch to merge from; defaults to latest epoch of source.
    strategy: Resolution strategy for auto-resolving conflicts.

  Returns:
    MergeIndex with conflicts and resolved entries.
  """
  refs = store.load_refs()
  branches = refs.get('branches', {})
  require_both_branches(experiment_id, from_experiment_id, branches)

  ours_epoch = branches[experiment_id]['latest_epoch']
  if from_epoch is None:
    from_epoch = branches[from_experiment_id]['latest_epoch']

  ancestor_exp, ancestor_epoch = find_lca(experiment_id, from_experiment_id, refs)
  ancestor_snap = load_ancestor_snapshot(store, ancestor_exp, ancestor_epoch)
  ours_snap = store.load_snapshot(experiment_id, ours_epoch)
  theirs_snap = store.load_snapshot(from_experiment_id, from_epoch)

  all_keys = set(ancestor_snap.entries) | set(ours_snap.entries) | set(theirs_snap.entries)
  conflicts: dict[str, ConflictEntry] = {}
  resolved: dict[str, FileEntry] = {}

  for key in sorted(all_keys):
    merge_key_three_way(store, key, ancestor_snap, ours_snap, theirs_snap, conflicts, resolved)

  if strategy in {MergeStrategy.ours, MergeStrategy.theirs}:
    auto_resolve_strategy(conflicts, resolved, strategy)
  elif strategy == MergeStrategy.union:
    auto_resolve_union(store, conflicts, resolved, ancestor_snap)

  token = compute_preview_token(
    experiment_id,
    ours_epoch,
    from_experiment_id,
    from_epoch,
    strategy,
    sorted(all_keys),
    ancestor_exp,
    ancestor_epoch,
  )

  return MergeIndex(
    conflicts=conflicts,
    resolved=resolved,
    experiment_id=experiment_id,
    source_experiment_id=from_experiment_id,
    strategy=strategy,
    preview_token=token,
  )


def merge_apply(store: Any, merge_index: MergeIndex) -> SnapshotManifest:
  """Persist a resolved merge as a new epoch on the target experiment.

  Validates preview_token freshness and resolution completeness, then
  writes the merged manifest, refs, and reflog atomically via
  ``StoreTransaction``. All three resources are buffered and flushed
  together on success; on failure none are persisted.

  Args:
    store: FileStore instance.
    merge_index: Fully resolved MergeIndex from ``merge_preview``.

  Returns:
    SnapshotManifest written for the new merge epoch.

  Raises:
    StoreError: If preview token is stale, experiments are missing, or
      conflicts remain unresolved.
  """
  if not merge_index.is_resolved():
    remaining = sorted(merge_index.conflicts)
    msg = (
      f'{len(remaining)} unresolved conflict(s) remain: {remaining}; '
      f'resolve all conflicts before calling merge_apply'
    )
    raise StoreError(msg)

  experiment_id = merge_index.experiment_id
  source_id = merge_index.source_experiment_id
  if experiment_id is None or source_id is None:
    msg = 'merge_index must have experiment_id and source_experiment_id set'
    raise StoreError(msg)

  with store.transaction(context=merge_index.strategy.value) as txn:
    refs = store.load_refs()
    branches = refs.get('branches', {})
    require_both_branches(experiment_id, source_id, branches)

    ours_epoch = branches[experiment_id]['latest_epoch']
    theirs_epoch = branches[source_id]['latest_epoch']

    expected_token = recompute_preview_token(
      store, experiment_id, source_id, merge_index.strategy, refs
    )
    if merge_index.preview_token != expected_token:
      msg = (
        'stale preview token: refs have changed since merge_preview was called; '
        'rerun merge_preview to get a fresh MergeIndex'
      )
      raise StoreError(msg)

    new_epoch = ours_epoch + 1
    manifest = SnapshotManifest(
      epoch=new_epoch,
      timestamp=utc_now_iso(),
      entries=dict(merge_index.resolved),
      schema=store.load_snapshot(experiment_id, ours_epoch).schema,
    )

    txn.write_manifest(experiment_id, new_epoch, manifest)

    branch = branches[experiment_id]
    branch['latest_epoch'] = new_epoch
    merge_parents = branch.get('merge_parents', [])
    merge_parents.append(
      {
        'experiment_id': source_id,
        'epoch': theirs_epoch,
      }
    )
    branch['merge_parents'] = merge_parents
    refs['branches'] = branches
    refs['HEAD'] = experiment_id
    refs['version'] = REFS_FORMAT_VERSION
    txn.set_refs(refs)
    txn.append_reflog(
      {
        'timestamp': utc_now_iso(),
        'operation': 'merge_apply',
        'experiment_id': experiment_id,
        'old_epoch': ours_epoch,
        'new_epoch': new_epoch,
        'source_experiment_id': source_id,
        'source_epoch': theirs_epoch,
        'context': merge_index.strategy.value,
      }
    )

  return manifest


def merge_and_apply(
  store: Any,
  experiment_id: str,
  from_experiment_id: str,
  from_epoch: int | None = None,
  strategy: MergeStrategy = MergeStrategy.normal,
) -> SnapshotManifest:
  """Run analysis + preview + apply when merge is fully resolved.

  Convenience for automation; interactive/agent flows should use explicit
  preview for conflict inspection. Raises ``StoreError`` when unresolved
  conflicts remain after preview (only possible with ``MergeStrategy.normal``
  and conflicting edits).

  Args:
    store: FileStore instance.
    experiment_id: Target branch (ours).
    from_experiment_id: Source branch (theirs).
    from_epoch: Epoch to merge from; defaults to latest epoch of source.
    strategy: Resolution strategy.

  Returns:
    SnapshotManifest for the new merge epoch.

  Raises:
    StoreError: If conflicts remain after preview with the given strategy.
  """
  analysis = merge_analysis(store, experiment_id, from_experiment_id)
  if analysis.classification == MergeClassification.up_to_date:
    return store.load_snapshot(
      experiment_id, store.load_refs()['branches'][experiment_id]['latest_epoch']
    )
  mi = merge_preview(store, experiment_id, from_experiment_id, from_epoch, strategy)
  if mi.conflicts and strategy == MergeStrategy.normal:
    conflict_keys = sorted(mi.conflicts)
    msg = (
      f'merge has {len(conflict_keys)} unresolved conflict(s): {conflict_keys}; '
      f'use a non-normal strategy or resolve conflicts manually via merge_preview'
    )
    raise StoreError(msg)
  return merge_apply(store, mi)


# --- merge internals ---


def merge_key_three_way(
  store: Any,
  key: str,
  ancestor_snap: SnapshotManifest,
  ours_snap: SnapshotManifest,
  theirs_snap: SnapshotManifest,
  conflicts: dict[str, ConflictEntry],
  resolved: dict[str, FileEntry],
) -> None:
  """Three-way merge a single manifest key into conflicts or resolved.

  Args:
    store: FileStore instance.
    key: Composite parameter key.
    ancestor_snap: Common ancestor snapshot.
    ours_snap: Target branch snapshot.
    theirs_snap: Source branch snapshot.
    conflicts: Mutable dict to record conflict entries.
    resolved: Mutable dict to record cleanly resolved entries.
  """
  ancestor_entry = ancestor_snap.entries.get(key)
  ours_entry = ours_snap.entries.get(key)
  theirs_entry = theirs_snap.entries.get(key)

  ancestor_hash = ancestor_entry.digest if ancestor_entry else None
  ours_hash = ours_entry.digest if ours_entry else None
  theirs_hash = theirs_entry.digest if theirs_entry else None

  if ours_hash == theirs_hash:
    if ours_entry is not None:
      resolved[key] = ours_entry
    return

  if ours_hash == ancestor_hash:
    if theirs_entry is not None:
      resolved[key] = theirs_entry
    return

  if theirs_hash == ancestor_hash:
    if ours_entry is not None:
      resolved[key] = ours_entry
    return

  if ours_entry is not None and theirs_entry is not None and ancestor_entry is not None:
    merged_entry = try_text_merge(store, key, ancestor_entry, ours_entry, theirs_entry)
    if merged_entry is not None:
      resolved[key] = merged_entry
      return

  conflicts[key] = ConflictEntry(
    key=key,
    ancestor=ancestor_entry,
    ours=ours_entry,
    theirs=theirs_entry,
  )


def auto_resolve_strategy(
  conflicts: dict[str, ConflictEntry],
  resolved: dict[str, FileEntry],
  strategy: MergeStrategy,
) -> None:
  """Auto-resolve all conflicts using ours or theirs strategy."""
  for key in list(conflicts):
    conflict = conflicts[key]
    chosen = conflict.ours if strategy == MergeStrategy.ours else conflict.theirs
    del conflicts[key]
    if chosen is not None:
      resolved[key] = chosen


def auto_resolve_union(
  store: Any,
  conflicts: dict[str, ConflictEntry],
  resolved: dict[str, FileEntry],
  ancestor_snap: SnapshotManifest,
) -> None:
  """Auto-resolve conflicts using union strategy where possible."""
  for key in list(conflicts):
    conflict = conflicts[key]
    if conflict.ours is None or conflict.theirs is None:
      continue
    if conflict.ancestor is not None:
      merged = try_text_merge(store, key, conflict.ancestor, conflict.ours, conflict.theirs)
      if merged is not None:
        del conflicts[key]
        resolved[key] = merged
        continue
    ours_bytes = store.read_object(conflict.ours.digest)
    theirs_bytes = store.read_object(conflict.theirs.digest)
    try:
      ours_text = ours_bytes.decode('utf-8')
      theirs_text = theirs_bytes.decode('utf-8')
    except UnicodeDecodeError:
      continue
    combined = ours_text + theirs_text
    combined_bytes = combined.encode('utf-8')
    combined_hash = hash_bytes(combined_bytes)
    store._store_object_bytes(combined_hash, combined_bytes)
    del conflicts[key]
    resolved[key] = FileEntry(
      digest=combined_hash, size=len(combined_bytes), mtime=FILE_ENTRY_MTIME_UNAVAILABLE
    )
