"""Stable preview tokens for merge staging."""

from autopilot.ai.store.merge_ancestry import find_lca, load_ancestor_snapshot
from autopilot.core.store.types import MergeStrategy
from typing import Any
import hashlib


def compute_preview_token(
  target_exp: str,
  target_epoch: int,
  source_exp: str,
  source_epoch: int,
  strategy: MergeStrategy,
  sorted_keys: list[str],
  ancestor_exp: str | None,
  ancestor_epoch: int | None,
) -> str:
  """Compute a stable SHA-256 preview token from merge inputs.

  Returns:
    Hex-encoded SHA-256 digest.
  """
  parts = [
    target_exp,
    str(target_epoch),
    source_exp,
    str(source_epoch),
    strategy.value,
    ','.join(sorted_keys),
    str(ancestor_exp),
    str(ancestor_epoch),
  ]
  return hashlib.sha256('|'.join(parts).encode('utf-8')).hexdigest()


def recompute_preview_token(
  store: Any,
  experiment_id: str,
  source_id: str,
  strategy: MergeStrategy,
  refs: dict[str, Any],
) -> str:
  """Recompute preview token from current refs state for staleness check.

  Returns:
    Hex-encoded SHA-256 digest from current refs tips.
  """
  branches = refs.get('branches', {})
  ours_epoch = branches[experiment_id]['latest_epoch']
  theirs_epoch = branches[source_id]['latest_epoch']

  ancestor_exp, ancestor_epoch = find_lca(experiment_id, source_id, refs)
  ancestor_snap = load_ancestor_snapshot(store, ancestor_exp, ancestor_epoch)
  ours_snap = store.load_snapshot(experiment_id, ours_epoch)
  theirs_snap = store.load_snapshot(source_id, theirs_epoch)

  union_entries = set(ancestor_snap.entries) | set(ours_snap.entries) | set(theirs_snap.entries)
  all_keys = sorted(union_entries)
  return compute_preview_token(
    experiment_id,
    ours_epoch,
    source_id,
    theirs_epoch,
    strategy,
    all_keys,
    ancestor_exp,
    ancestor_epoch,
  )
