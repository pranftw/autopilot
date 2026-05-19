"""Doctor, repair, prune, and forest health for FileStore."""

from autopilot.ai.store import repair as repair_mod
from autopilot.ai.store.snapshot import enumerate_snapshot_epochs
from autopilot.core.diagnostic import DiagnosticEntry
from autopilot.core.errors import StoreError, TrackingError
from autopilot.core.snapshot import SnapshotManifest
from autopilot.tracking.io import read_json_dict, read_jsonl
from pathlib import Path
from typing import Any
import os


def doctor(store: Any) -> list[DiagnosticEntry]:
  """Detect and diagnose store health issues without mutating state.

  Runs the **detect** and **diagnose** phases of the three-phase pipeline.
  Does **not** mutate by default; use ``repair_diagnostics`` on the returned
  entries to apply safe repairs.

  Checks:
    - Every snapshot manifest parses and matches on-disk layout expectations.
    - Every manifest entry references a digest that exists under objects/.
    - Reachable digests from manifests vs orphan files.
    - refs.json branch entries reference known experiments / epochs.
    - forest.json structural validity.
    - Stale lock files from dead processes.
    - Reflog gaps (branches present in refs but absent from reflog).
    - Ghost epochs (manifest files beyond latest_epoch in refs).

  Args:
    store: FileStore instance.

  Returns:
    List of DiagnosticEntry instances describing each finding.
  """
  reachable, manifest_errors = walk_manifests(store)
  on_disk = collect_on_disk_blobs(store)

  diagnostics: list[DiagnosticEntry] = [
    DiagnosticEntry(
      code='manifest_error',
      severity='error',
      path=err.split(':')[0] if ':' in err else None,
      message=err,
      repairable=False,
    )
    for err in manifest_errors
  ]

  diagnostics.extend(
    DiagnosticEntry(
      code='missing_blob',
      severity='error',
      path=digest,
      message=f'blob {digest} referenced by manifest but not found in object store',
      repairable=False,
    )
    for digest in sorted(reachable - on_disk)
  )

  diagnostics.extend(
    DiagnosticEntry(
      code='orphan_blob',
      severity='warning',
      path=digest,
      message=(f'blob {digest} exists in object store but is not referenced by any manifest'),
      repairable=True,
      repair_action='delete',
    )
    for digest in sorted(on_disk - reachable)
  )

  refs_issues: list[str] = []
  check_refs_consistency(store, refs_issues)
  diagnostics.extend(
    DiagnosticEntry(
      code='broken_ref',
      severity='error',
      path=None,
      message=issue,
      repairable='tip epoch' in issue or 'latest_epoch' in issue,
      repair_action='reset' if ('tip epoch' in issue or 'latest_epoch' in issue) else None,
    )
    for issue in refs_issues
  )

  diagnostics.extend(check_forest_health(store))

  detect_stale_locks(store, diagnostics)
  detect_reflog_gaps(store, diagnostics)
  detect_ghost_epochs(store, diagnostics)

  return diagnostics


def doctor_report(store: Any) -> dict[str, Any]:
  """Legacy dict-shaped doctor report built from structured diagnostics.

  Returns:
    Dict with ``healthy``, ``manifest_errors``, ``missing_blobs``,
    ``orphan_blobs``, ``orphan_count``, ``refs_issues``, ``forest_errors``,
    ``reflog_gaps``, and ``diagnostics`` keys.
  """
  entries = doctor(store)
  return diagnostics_to_report(entries)


def diagnostics_to_report(entries: list[DiagnosticEntry]) -> dict[str, Any]:
  """Convert a list of DiagnosticEntry to the legacy report dict shape.

  Args:
    entries: Diagnostic entries from ``doctor()``.

  Returns:
    Dict with legacy keys for backward-compatible CLI and status output.
    Includes ``forest_missing`` bool (True when forest.json is absent).
  """
  manifest_errors = [e.message for e in entries if e.code == 'manifest_error']
  missing_blobs = [e.path for e in entries if e.code == 'missing_blob' and e.path is not None]
  orphan_blobs = [e.path for e in entries if e.code == 'orphan_blob' and e.path is not None]
  refs_issues = [e.message for e in entries if e.code == 'broken_ref']
  forest_errors = [e.message for e in entries if e.code == 'forest_corrupt']
  forest_missing = any(e.code == 'forest_missing' for e in entries)
  reflog_gaps = [e.path for e in entries if e.code == 'reflog_gap' and e.path is not None]

  healthy = not manifest_errors and not missing_blobs and not refs_issues and not forest_errors
  return {
    'healthy': healthy,
    'manifest_errors': manifest_errors,
    'missing_blobs': missing_blobs,
    'orphan_blobs': orphan_blobs,
    'orphan_count': len(orphan_blobs),
    'refs_issues': refs_issues,
    'forest_errors': forest_errors,
    'forest_missing': forest_missing,
    'reflog_gaps': reflog_gaps,
    'diagnostics': [e.to_dict() for e in entries],
  }


def repair_diagnostics(
  store: Any,
  entries: list[DiagnosticEntry],
  *,
  dry_run: bool = False,
  context: str | None = None,
) -> list[DiagnosticEntry]:
  """Apply repairs for repairable diagnostic entries.

  Args:
    store: FileStore instance.
    entries: Diagnostic entries from ``doctor()``.
    dry_run: When True, no mutations are applied.
    context: Reason/provenance string. Required when repairable entries
      exist and ``dry_run`` is False.

  Returns:
    List of DiagnosticEntry for items that were (or would be) repaired.

  Raises:
    StoreError: When ``context`` is None, ``dry_run`` is False, and
      repairable diagnostics exist.
  """
  repairable = [e for e in entries if e.repairable]
  if repairable and not dry_run and context is None:
    msg = (
      'repair_diagnostics requires context when repairable diagnostics '
      'exist; pass context="<reason>" for audit traceability'
    )
    raise StoreError(msg)

  repaired: list[DiagnosticEntry] = []
  for entry in repairable:
    if dry_run:
      repaired.append(entry)
      continue
    if entry.code == 'orphan_blob' and entry.repair_action == 'delete':
      repair_mod.repair_orphan_blob(store, entry)
      repaired.append(entry)
    elif entry.code == 'stale_lock' and entry.repair_action == 'delete':
      repair_mod.repair_stale_lock(entry)
      repaired.append(entry)
    elif entry.code == 'broken_ref' and entry.repair_action == 'reset':
      repair_mod.repair_broken_ref(store, entry)
      repaired.append(entry)
    elif entry.code == 'reflog_gap' and entry.repair_action == 'backfill':
      repair_mod.repair_reflog_gap(store, entry, context)
      repaired.append(entry)
    elif entry.code == 'ghost_epoch' and entry.repair_action == 'delete':
      if entry.path is not None:
        Path(entry.path).unlink(missing_ok=True)
      repaired.append(entry)

  return repaired


def detect_stale_locks(store: Any, diagnostics: list[DiagnosticEntry]) -> None:
  """Detect stale lock files from dead processes.

  Args:
    store: FileStore instance.
    diagnostics: Mutable list to append findings to.
  """
  lock_path = store._config.store_path / '.lock'
  if lock_path.is_file() and is_stale_lock(lock_path):
    diagnostics.append(
      DiagnosticEntry(
        code='stale_lock',
        severity='warning',
        path=str(lock_path),
        message=f'stale lock file at {lock_path} (owning process is dead)',
        repairable=True,
        repair_action='delete',
      )
    )

  worktrees_dir = store._config.worktrees_path
  if worktrees_dir.exists():
    diagnostics.extend(
      DiagnosticEntry(
        code='stale_lock',
        severity='warning',
        path=str(lock_file),
        message=f'stale lock file at {lock_file} (owning process is dead)',
        repairable=True,
        repair_action='delete',
      )
      for lock_file in worktrees_dir.glob('*.lock')
      if is_stale_lock(lock_file)
    )


def is_stale_lock(lock_path: Path) -> bool:
  """Check whether a lock file is stale (owning PID is dead).

  Args:
    lock_path: Path to the lock file.

  Returns:
    True only when the owning PID is demonstrably dead.
  """
  try:
    content = lock_path.read_text(encoding='utf-8').strip()
    if not content:
      return True
    pid = int(content)
  except (OSError, ValueError):
    return False
  try:
    os.kill(pid, 0)
  except ProcessLookupError:
    return True
  except PermissionError:
    return False
  else:
    return False


def detect_reflog_gaps(store: Any, diagnostics: list[DiagnosticEntry]) -> None:
  """Detect branches present in refs but absent from reflog.

  Args:
    store: FileStore instance.
    diagnostics: Mutable list to append findings to.
  """
  reflog_path = store._reflog_path
  if not reflog_path.exists():
    return
  entries = read_jsonl(reflog_path, strict=False)
  reflog_experiments = {entry['experiment_id'] for entry in entries if 'experiment_id' in entry}
  refs = store.load_refs()
  branches = refs.get('branches')
  if branches is None:
    return
  diagnostics.extend(
    DiagnosticEntry(
      code='reflog_gap',
      severity='info',
      path=bid,
      message=f'branch {bid!r} present in refs but absent from reflog',
      repairable=True,
      repair_action='backfill',
    )
    for bid in branches
    if bid not in reflog_experiments
  )


def detect_ghost_epochs(store: Any, diagnostics: list[DiagnosticEntry]) -> None:
  """Detect epoch manifest files beyond latest_epoch in refs.

  Args:
    store: FileStore instance.
    diagnostics: Mutable list to append findings to.
  """
  try:
    refs = store.load_refs()
  except StoreError:
    return
  branches = refs.get('branches', {})
  for exp_id, info in branches.items():
    latest = info.get('latest_epoch', -1)
    branch_dir = store._snapshots_dir / exp_id
    if not branch_dir.is_dir():
      continue
    on_disk = enumerate_snapshot_epochs(branch_dir)
    for epoch_num in on_disk:
      if epoch_num > latest:
        epoch_file = branch_dir / f'epoch_{epoch_num}.json'
        diagnostics.append(
          DiagnosticEntry(
            code='ghost_epoch',
            severity='warning',
            path=str(epoch_file),
            message=(
              f'epoch {epoch_num} exists on disk for branch {exp_id!r} '
              f'but latest_epoch is {latest}; '
              f'run store doctor --repair to delete the ghost manifest'
            ),
            repairable=True,
            repair_action='delete',
          )
        )


def walk_manifests(store: Any) -> tuple[set[str], list[str]]:
  """Walk all snapshot and stash manifests, returning reachable digests and errors.

  Args:
    store: FileStore instance.

  Returns:
    Tuple of (reachable digest set, error message list).
  """
  reachable: set[str] = set()
  errors: list[str] = []
  if store._snapshots_dir.exists():
    for exp_dir in store._snapshots_dir.iterdir():
      if not exp_dir.is_dir():
        continue
      for snap_file in exp_dir.iterdir():
        if not snap_file.name.endswith('.json'):
          continue
        walk_single_manifest(snap_file, reachable, errors)
  if store._stash_dir.exists():
    for stash_file in store._stash_dir.glob('*.json'):
      walk_single_manifest(stash_file, reachable, errors)
  return reachable, errors


def walk_single_manifest(
  path: Path,
  reachable: set[str],
  errors: list[str],
) -> None:
  """Parse one manifest file, adding digests to reachable or errors.

  Args:
    path: Manifest JSON file path.
    reachable: Mutable set to add discovered blob digests to.
    errors: Mutable list to append parse errors to.
  """
  try:
    data = read_json_dict(path, f'manifest {path}')
    manifest = SnapshotManifest.from_dict(data)
    reachable.update(entry.digest for entry in manifest.entries.values())
  except (TrackingError, StoreError, KeyError, ValueError) as exc:
    errors.append(f'{path}: {exc}')


def collect_on_disk_blobs(store: Any) -> set[str]:
  """Scan objects directory for blob digests on disk.

  Args:
    store: FileStore instance.

  Returns:
    Set of full digest strings found on disk.
  """
  blobs: set[str] = set()
  objects_dir = store._config.objects_path
  if not objects_dir.exists():
    return blobs
  for shard_dir in objects_dir.iterdir():
    if not shard_dir.is_dir():
      continue
    if len(shard_dir.name) != 2:
      continue
    try:
      int(shard_dir.name, 16)
    except ValueError:
      continue
    for blob_file in shard_dir.iterdir():
      if blob_file.is_file():
        blobs.add(shard_dir.name + blob_file.name)
  return blobs


def check_refs_consistency(store: Any, issues: list[str]) -> None:
  """Validate refs.json branch entries for basic consistency.

  Args:
    store: FileStore instance.
    issues: Mutable list to append refs issue strings to.
  """
  try:
    refs = store.load_refs()
  except StoreError as exc:
    issues.append(f'refs.json unreadable: {exc}')
    return
  branches = refs.get('branches', {})
  for exp_id, info in branches.items():
    latest = info.get('latest_epoch')
    if latest is None:
      issues.append(f'branch {exp_id!r}: missing latest_epoch')
      continue
    if latest < -1:
      issues.append(f'branch {exp_id!r}: invalid latest_epoch {latest}')
      continue
    if latest == -1:
      continue
    snap_path = store._snapshots_dir / exp_id / f'epoch_{latest}.json'
    if not snap_path.exists():
      issues.append(f'branch {exp_id!r}: tip epoch {latest} snapshot missing')


def check_forest_health(store: Any) -> list[DiagnosticEntry]:
  """Validate forest.json presence and structure.

  Returns:
    Diagnostic entries: ``forest_missing`` (info) when absent;
    ``forest_corrupt`` (error) when present but invalid.
  """
  forest_file = store._config.forest_file
  if not forest_file.exists():
    return [
      DiagnosticEntry(
        code='forest_missing',
        severity='info',
        path=str(forest_file),
        message='forest.json not found (no experiments created yet)',
        repairable=False,
      )
    ]

  try:
    data = read_json_dict(forest_file, 'forest.json')
  except (TrackingError, StoreError) as exc:
    return [
      DiagnosticEntry(
        code='forest_corrupt',
        severity='error',
        path=str(forest_file),
        message=f'forest.json is not valid JSON: {exc}',
        repairable=False,
      )
    ]

  errors: list[str] = []
  if not isinstance(data, dict):
    errors.append('forest.json root must be a JSON object')
  elif 'trees' not in data:
    errors.append('forest.json missing required "trees" key')
  elif not isinstance(data['trees'], list):
    errors.append('forest.json "trees" must be a list')
  else:
    for idx, tree_state in enumerate(data['trees']):
      validate_forest_tree(idx, tree_state, errors)

  return [
    DiagnosticEntry(
      code='forest_corrupt',
      severity='error',
      path=str(forest_file),
      message=err,
      repairable=False,
    )
    for err in errors
  ]


def validate_forest_tree(
  idx: int,
  tree_state: Any,
  errors: list[str],
) -> None:
  """Validate a single tree entry in forest.json.

  Args:
    idx: Index of the tree in the list.
    tree_state: Raw parsed tree dict from forest.json.
    errors: Mutable list to append validation errors to.
  """
  if not isinstance(tree_state, dict):
    errors.append(f'forest.json trees[{idx}] must be a dict')
    return
  if 'name' not in tree_state or not isinstance(tree_state.get('name'), str):
    errors.append(f'forest.json trees[{idx}] missing string "name"')
  if 'nodes' not in tree_state:
    errors.append(f'forest.json trees[{idx}] missing "nodes" key')
  elif not isinstance(tree_state['nodes'], list):
    errors.append(f'forest.json trees[{idx}] "nodes" must be a list')
  else:
    nodes_list: list[Any] = tree_state['nodes']
    for nidx, node in enumerate(nodes_list):
      if not isinstance(node, dict):
        errors.append(f'forest.json trees[{idx}].nodes[{nidx}] must be a dict')
      elif not isinstance(node.get('experiment'), str):
        errors.append(f'forest.json trees[{idx}].nodes[{nidx}] missing string "experiment"')


# --- orphan pruning ---


def prune_orphans(store: Any) -> list[str]:
  """Remove orphaned blobs not reachable from any snapshot manifest.

  Acquires the store lock for the full scan-and-remove cycle.
  Fail-closed: raises StoreError if any manifest is corrupt.

  Args:
    store: FileStore instance.

  Returns:
    List of removed blob digests.
  """
  store._acquire_lock()
  try:
    reachable = collect_reachable_digests(store)
    return remove_unreachable_blobs(store, reachable)
  finally:
    store._release_lock()


def collect_reachable_digests(store: Any) -> set[str]:
  """Walk all snapshot manifests and collect referenced blob digests.

  Raises:
    StoreError: If any manifest fails to parse.

  Returns:
    Set of blob digests reachable from at least one snapshot.
  """
  reachable, errors = walk_manifests(store)
  if errors:
    msg = (
      f'prune_orphans aborted: {len(errors)} corrupt manifest(s) found. '
      f'Fix or remove them before pruning:\n' + '\n'.join(errors)
    )
    raise StoreError(msg)
  return reachable


def remove_unreachable_blobs(store: Any, reachable: set[str]) -> list[str]:
  """Scan objects directory and remove blobs not in reachable set.

  Args:
    store: FileStore instance.
    reachable: Digests that should be kept.

  Returns:
    List of removed blob digests.
  """
  removed: list[str] = []
  objects_dir = store._config.objects_path
  if not objects_dir.exists():
    return removed
  for prefix_dir in objects_dir.iterdir():
    if not prefix_dir.is_dir():
      continue
    for blob_file in prefix_dir.iterdir():
      if not blob_file.is_file():
        continue
      digest = prefix_dir.name + blob_file.name
      if digest not in reachable:
        blob_file.unlink()
        removed.append(digest)
    if prefix_dir.exists() and not any(prefix_dir.iterdir()):
      prefix_dir.rmdir()
  return removed
