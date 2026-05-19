"""Snapshot, checkout, diff, and materialize operations for FileStore."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.snapshot_diff import diff, text_diff_content
from autopilot.ai.store.snapshot_helpers import (
  FILE_ENTRY_MTIME_UNAVAILABLE,
  PROTECTED_PREFIXES,
  SchemaMatchResult,
  build_snapshot,
  group_by_param,
  remove_extraneous_files,
  resolve_original_path,
  snapshot_all_params,
  validate_schema,
)
from autopilot.ai.store.snapshot_manifest import (
  enumerate_snapshot_epochs,
  experiment_log,
  load_snapshot_manifest,
  persist_snapshot_manifest,
)
from autopilot.ai.store_lock import hash_content
from autopilot.core.enums import Status
from autopilot.core.errors import ExperimentError, StoreError
from autopilot.core.experiment import Experiment
from autopilot.core.snapshot import SnapshotManifest
from autopilot.core.store.types import (
  DiffKind,
  StatusEntry,
  StatusResult,
)
from autopilot.tracking.io import utc_now_iso
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

DIGEST_DISPLAY_HEX_LEN = 12


def snapshot(
  store: Any,
  experiment_id: str,
  epoch: int,
  experiment: Experiment | None = None,
  context: str | None = None,
  *,
  force: bool = False,
) -> SnapshotManifest:
  """Persists parameter state for ``experiment_id`` at ``epoch``.

  Creates the branch at epoch ``0`` if missing; later epochs must be
  sequential. Reloads refs inside the backend lock to prevent TOCTOU
  races (plan 04).

  **Idempotent by default**: when the branch already has at least one
  snapshot and the file-entry digests of the new snapshot are identical
  to the latest epoch, the snapshot is skipped -- no new epoch is written,
  no reflog entry appended, and the prior manifest is returned with its
  original epoch number.  Pass ``force=True`` to persist a new epoch
  even when file content is unchanged (e.g. to record a context-only
  provenance marker).

  When ``experiment`` is provided and the experiment status is completed,
  the behavior depends on ``experiment.strict_snapshot_after_complete``:
  - When False (default): proceeds silently.
  - When True: raises ``ExperimentError`` to prevent silent writes after
    lifecycle end (BUG-046).

  Args:
    store: FileStore instance.
    experiment_id: Branch / experiment identifier.
    epoch: Epoch index (must equal ``latest_epoch + 1`` except for new branches).
    experiment: Optional experiment to check completion status against.
    context: Optional reason/provenance string recorded on the manifest
      for audit traceability.
    force: When True, always persist a new epoch even if file content is
      unchanged from the latest snapshot.

  Returns:
    Manifest written for this snapshot, or the prior manifest when skipped.

  Raises:
    StoreError: If epoch sequence or branch rules are violated.
    ExperimentError: If experiment is completed and strict mode is enabled.
  """
  if (
    experiment is not None
    and experiment.status == Status.completed
    and experiment.strict_snapshot_after_complete
  ):
    msg = (
      f'experiment {experiment_id!r} is completed; '
      f'snapshot rejected (strict_snapshot_after_complete=True)'
    )
    raise ExperimentError(msg)

  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})

    if experiment_id not in branches:
      if epoch != 0:
        msg = f'experiment {experiment_id!r} not found; first snapshot must be epoch 0'
        raise StoreError(msg)
      branches[experiment_id] = {
        'latest_epoch': -1,
        'parent_id': None,
        'parent_epoch': None,
      }

    branch = branches[experiment_id]
    expected = branch['latest_epoch'] + 1
    if epoch == 0 and branch['latest_epoch'] == 0 and branch.get('parent_id') is not None:
      pass
    elif epoch != expected:
      msg = f'epoch must be sequential: expected {expected}, got {epoch}'
      raise StoreError(msg)

    manifest = build_snapshot(store, context=context)
    manifest.epoch = epoch

    if not force and branch['latest_epoch'] >= 0:
      prior = store.load_snapshot(experiment_id, branch['latest_epoch'])
      if snapshot_content_identical(prior, manifest):
        return prior

    store._save_snapshot(experiment_id, epoch, manifest)

    prior_epoch = branch['latest_epoch']
    reflog_old = None if prior_epoch == -1 else prior_epoch
    branch['latest_epoch'] = epoch
    refs['branches'] = branches
    refs['HEAD'] = experiment_id
    store.save_refs(refs)
    store._append_reflog('snapshot', experiment_id, reflog_old, epoch, context=context)
  finally:
    store._release_lock()

  return manifest


def snapshot_content_identical(
  prior: SnapshotManifest,
  current: SnapshotManifest,
) -> bool:
  """Check whether two manifests have identical file-entry content.

  Compares sorted keys and per-key digests. Context is intentionally
  excluded from the identity check so that two snapshots with different
  context but identical files are considered the same.

  Args:
    prior: The existing latest snapshot.
    current: The newly built (not yet saved) snapshot.

  Returns:
    True when both manifests have the same keys with the same digests.
  """
  if set(prior.entries) != set(current.entries):
    return False
  return all(prior.entries[key].digest == current.entries[key].digest for key in prior.entries)


def checkout(
  store: Any,
  experiment_id: str,
  epoch: int,
  *,
  strict_schema: bool = False,
  context: str | None = None,
) -> None:
  """Restores parameter state from the snapshot at ``epoch``.

  Loads the snapshot for ``experiment_id`` at ``epoch``, validates schema
  against registered parameters (warning or error on mismatch), restores
  each parameter from the grouped entries, removes extraneous working-tree
  files not present in the snapshot (BUG-075), and updates HEAD in refs.

  Raises ``StoreError`` when the manifest has entries but no registered
  parameters match (BUG-003), preventing silent no-op checkouts.

  Acquires the backend lock to serialize concurrent ref mutations
  (BUG-036, BUG-037). Concurrent checkouts and checkout vs
  snapshot/branch/etc. serialize on the shared backend lock.

  Safety behaviors:

  - **Binary protection**: binary files (detected via null-byte heuristic)
    are never deleted during extraneous-file cleanup.
  - **Zero-entry guard**: when a parameter's snapshot contains no entries,
    extraneous-file cleanup is skipped silently to prevent accidental mass
    deletion.
  - **Permissions**: file permissions are NOT preserved through
    snapshot/checkout. Executable bits must be re-applied post-checkout.

  Args:
    store: FileStore instance.
    experiment_id: Branch to checkout.
    epoch: Epoch to restore.
    strict_schema: When True, raise StoreError on schema mismatch instead
      of logging a warning.
    context: Optional reason/provenance string recorded in the reflog
      for audit traceability.

  Raises:
    StoreError: If the experiment or epoch does not exist, or when
      the manifest has entries but no registered parameters match.
  """
  store._require_branch(experiment_id)
  snap = store.load_snapshot(experiment_id, epoch)
  schema_result = validate_schema(store, snap)

  if schema_result.mismatched and not schema_result.matched:
    msg = (
      f'no registered parameters match manifest keys '
      f'{sorted(snap.entries.keys())}; '
      f'registered: {sorted(store._param_names.keys())}. '
      'Register matching parameters before checkout.'
    )
    raise StoreError(msg)

  if schema_result.mismatched and strict_schema:
    msg = (
      f'schema mismatch: manifest has '
      f'{sorted(schema_result.matched | schema_result.mismatched)}, '
      f'registered parameters are {sorted(store._param_names)}'
    )
    raise StoreError(msg)

  if schema_result.mismatched:
    logger.warning(
      'schema mismatch: manifest has %s, registered parameters are %s',
      sorted(schema_result.matched | schema_result.mismatched),
      sorted(store._param_names),
    )

  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})
    branch_info = branches.get(experiment_id, {})
    tip_at_start = branch_info.get('latest_epoch')

    grouped = group_by_param(store, snap)
    snapshot_keys = set(snap.entries)

    for name, param in store._param_names.items():
      param.restore(grouped.get(name, {}))
      if isinstance(param, PathParameter):
        entries_for_param = {k for k in snapshot_keys if k.startswith(f'{name}/')}
        if not entries_for_param:
          continue
        remove_extraneous_files(store, param, name, snapshot_keys)

    refs['HEAD'] = experiment_id
    store.save_refs(refs)
    store._append_reflog('checkout', experiment_id, tip_at_start, epoch, context=context)
  finally:
    store._release_lock()


def validate_checkout(
  store: Any,
  experiment_id: str,
  epoch: int,
) -> dict[str, Any]:
  """Perform all read-only validations for a checkout without side effects.

  Verifies that the experiment branch exists, the snapshot at ``epoch``
  is loadable, and reports schema matching status. Used by CLI dry-run
  to validate before returning (BUG-017).

  Args:
    store: FileStore instance.
    experiment_id: Branch to validate.
    epoch: Epoch to validate.

  Returns:
    Dict with ``files_to_restore``, ``schema_match``, and
    ``schema_mismatch`` keys. Propagates ``StoreError`` from
    branch/snapshot lookup when experiment or epoch is invalid.
  """
  store._require_branch(experiment_id)
  manifest = store.load_snapshot(experiment_id, epoch)
  schema_result = validate_schema(store, manifest)
  return {
    'files_to_restore': len(manifest.entries),
    'schema_match': bool(schema_result.matched),
    'schema_mismatch': bool(schema_result.mismatched),
  }


def materialize(
  store: Any,
  experiment_id: str,
  epoch: int,
) -> None:
  """Restores parameters to a historical epoch and resets the branch tip.

  Validates no external modifications conflict with the expected state,
  then restores parameters and rewrites epoch 0 with the target snapshot.
  Updates HEAD and ``latest_epoch`` in refs (BUG-039). Reloads refs
  inside the backend lock to prevent TOCTOU races (plan 04).

  Args:
    store: FileStore instance.
    experiment_id: Branch to materialize within.
    epoch: Historical epoch to restore as the new tip.

  Raises:
    StoreError: If external modifications are detected or branch is missing.
  """
  store._require_branch(experiment_id)

  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.get('branches', {})
    latest_epoch = branches[experiment_id]['latest_epoch']

    snap = store.load_snapshot(experiment_id, epoch)
    current_snap = store.load_snapshot(experiment_id, latest_epoch)

    current_content = snapshot_all_params(store)
    for key, entry in snap.entries.items():
      if key in current_content:
        current_hash = hash_content(current_content[key])
        if current_hash != entry.digest:
          expected = current_snap.entries.get(key)
          if expected and current_hash != expected.digest:
            expected_hex = expected.digest[:DIGEST_DISPLAY_HEX_LEN]
            found_hex = current_hash[:DIGEST_DISPLAY_HEX_LEN]
            msg = (
              f'external modification detected for {key}: '
              f'expected {expected_hex}, found {found_hex}'
            )
            raise StoreError(msg)

    grouped = group_by_param(store, snap)
    for name, param in store._param_names.items():
      if name in grouped:
        param.restore(grouped[name])

    materialized = SnapshotManifest(
      epoch=0,
      timestamp=utc_now_iso(),
      entries=dict(snap.entries),
      schema=snap.schema,
      context=snap.context,
    )
    store._save_snapshot(experiment_id, 0, materialized)

    branches[experiment_id]['latest_epoch'] = epoch
    refs['HEAD'] = experiment_id
    store.save_refs(refs)
    store._append_reflog('materialize', experiment_id, latest_epoch, epoch)
  finally:
    store._release_lock()


def status(
  store: Any,
  experiment_id: str,
) -> StatusResult:
  """Compares current parameter state against the latest snapshot.

  Walks all registered parameters, hashes their current content, and
  compares against the stored snapshot to report added/modified/deleted
  /unchanged entries.

  Args:
    store: FileStore instance.
    experiment_id: Branch to compare against.

  Returns:
    StatusResult with per-file status entries.

  Raises:
    StoreError: If the experiment is not found.
  """
  refs = store.load_refs()
  branches = refs.get('branches', {})
  if experiment_id not in branches:
    msg = f'experiment {experiment_id!r} not found'
    raise StoreError(msg)

  latest_epoch = branches[experiment_id]['latest_epoch']
  snap = store.load_snapshot(experiment_id, latest_epoch)
  entries: list[StatusEntry] = []

  current_keys: set[str] = set()
  for name, param in store._param_names.items():
    content = param.snapshot()
    for state_key, text in content.items():
      full_key = f'{name}/{state_key}'
      current_keys.add(full_key)
      current_hash = hash_content(text)

      if full_key not in snap.entries:
        entries.append(StatusEntry(path=full_key, status=DiffKind.added))
      elif current_hash != snap.entries[full_key].digest:
        entries.append(StatusEntry(path=full_key, status=DiffKind.modified))
      else:
        entries.append(StatusEntry(path=full_key, status=DiffKind.unchanged))

  entries.extend(
    StatusEntry(path=key, status=DiffKind.deleted)
    for key in snap.entries
    if key not in current_keys
  )

  return StatusResult(entries=entries)
