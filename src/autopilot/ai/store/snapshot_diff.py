"""Epoch-to-epoch diff computation for FileStore snapshots."""

from autopilot.core.store.types import DiffEntry, DiffKind, DiffResult
from typing import Any
import difflib


def diff(
  store: Any,
  experiment_id: str,
  epoch_a: int,
  epoch_b: int,
) -> DiffResult:
  """Computes a unified diff between two epochs of the same experiment.

  Args:
    store: FileStore instance.
    experiment_id: Branch to diff within.
    epoch_a: Earlier epoch (base).
    epoch_b: Later epoch (target).

  Returns:
    DiffResult with per-file added/deleted/modified entries and text diffs.
  """
  store._require_branch(experiment_id)
  snap_a = store.load_snapshot(experiment_id, epoch_a)
  snap_b = store.load_snapshot(experiment_id, epoch_b)

  all_keys = set(snap_a.entries) | set(snap_b.entries)
  entries: list[DiffEntry] = []

  for key in sorted(all_keys):
    in_a = key in snap_a.entries
    in_b = key in snap_b.entries

    if in_a and not in_b:
      entries.append(
        DiffEntry(
          path=key,
          status=DiffKind.deleted,
          old_hash=snap_a.entries[key].digest,
        )
      )
    elif not in_a and in_b:
      entries.append(
        DiffEntry(
          path=key,
          status=DiffKind.added,
          new_hash=snap_b.entries[key].digest,
        )
      )
    elif snap_a.entries[key].digest != snap_b.entries[key].digest:
      old_content = store.read_object(snap_a.entries[key].digest)
      new_content = store.read_object(snap_b.entries[key].digest)
      text_diff = text_diff_content(key, old_content, new_content)
      entries.append(
        DiffEntry(
          path=key,
          status=DiffKind.modified,
          old_hash=snap_a.entries[key].digest,
          new_hash=snap_b.entries[key].digest,
          text_diff=text_diff,
        )
      )

  return DiffResult(entries=entries)


def text_diff_content(key: str, old_content: bytes, new_content: bytes) -> str:
  """Compute unified text diff between two byte strings.

  Args:
    key: File key for diff headers.
    old_content: Old file content.
    new_content: New file content.

  Returns:
    Unified diff string or binary-differ marker.
  """
  try:
    old_lines = old_content.decode('utf-8').splitlines(keepends=True)
    new_lines = new_content.decode('utf-8').splitlines(keepends=True)
  except UnicodeDecodeError:
    return '(binary files differ)'
  return ''.join(difflib.unified_diff(old_lines, new_lines, fromfile=key, tofile=key))
