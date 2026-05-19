"""Line-level text merge helpers for three-way merge and union resolution."""

from autopilot.ai.store.snapshot import FILE_ENTRY_MTIME_UNAVAILABLE
from autopilot.ai.store_lock import hash_bytes
from autopilot.core.errors import StoreError
from autopilot.core.snapshot import FileEntry
from typing import Any
import difflib


def try_text_merge(
  store: Any,
  _key: str,
  ancestor_entry: FileEntry,
  ours_entry: FileEntry,
  theirs_entry: FileEntry,
) -> FileEntry | None:
  """Attempt line-level three-way text merge for a single key.

  Returns:
    Merged FileEntry on success; None if the merge conflicts.
  """
  try:
    base_content = store.read_object(ancestor_entry.digest)
    ours_content = store.read_object(ours_entry.digest)
    theirs_content = store.read_object(theirs_entry.digest)
  except StoreError:
    return None

  merged_text = three_way_merge_text(base_content, ours_content, theirs_content)
  if merged_text is None:
    return None

  merged_bytes = merged_text.encode('utf-8')
  merged_hash = hash_bytes(merged_bytes)
  store._store_object_bytes(merged_hash, merged_bytes)
  return FileEntry(digest=merged_hash, size=len(merged_bytes), mtime=FILE_ENTRY_MTIME_UNAVAILABLE)


def three_way_merge_text(
  base: bytes,
  ours: bytes,
  theirs: bytes,
) -> str | None:
  """Line-level three-way text merge.

  Returns:
    Merged text on clean merge, None on conflict.
  """
  try:
    base_lines = base.decode('utf-8').splitlines(keepends=True)
    ours_lines = ours.decode('utf-8').splitlines(keepends=True)
    theirs_lines = theirs.decode('utf-8').splitlines(keepends=True)
  except UnicodeDecodeError:
    return None

  ours_diff = list(difflib.unified_diff(base_lines, ours_lines))
  theirs_diff = list(difflib.unified_diff(base_lines, theirs_lines))

  if not ours_diff:
    return ''.join(theirs_lines)
  if not theirs_diff:
    return ''.join(ours_lines)

  ours_changes = extract_changed_line_numbers(base_lines, ours_lines)
  theirs_changes = extract_changed_line_numbers(base_lines, theirs_lines)

  if ours_changes.intersection(theirs_changes):
    return None

  return apply_non_overlapping_changes(base_lines, ours_lines, theirs_lines)


def apply_non_overlapping_changes(
  base_lines: list[str],
  ours_lines: list[str],
  theirs_lines: list[str],
) -> str:
  """Applies both ours and theirs non-overlapping edits onto the base.

  Args:
    base_lines: Common ancestor lines.
    ours_lines: Target branch lines.
    theirs_lines: Source branch lines.

  Returns:
    Merged file text.
  """
  ours_edits = collect_edits(base_lines, ours_lines)
  theirs_edits = collect_edits(base_lines, theirs_lines)
  all_edit_starts = sorted(set(ours_edits) | set(theirs_edits))
  result: list[str] = []
  skip_until = 0
  for idx, line in enumerate(base_lines):
    if idx < skip_until:
      continue
    if idx in ours_edits:
      tag, i2, replacement = ours_edits[idx]
      result.extend(replacement)
      if tag in {'replace', 'delete'}:
        skip_until = i2
        continue
    elif idx in theirs_edits:
      tag, i2, replacement = theirs_edits[idx]
      result.extend(replacement)
      if tag in {'replace', 'delete'}:
        skip_until = i2
        continue
    else:
      result.append(line)
  for start in all_edit_starts:
    if start >= len(base_lines):
      if start in ours_edits:
        result.extend(ours_edits[start][2])
      elif start in theirs_edits:
        result.extend(theirs_edits[start][2])
  return ''.join(result)


def collect_edits(
  base_lines: list[str],
  modified_lines: list[str],
) -> dict[int, tuple[str, int, list[str]]]:
  """Collect non-equal opcodes as ``{base_start: (tag, base_end, replacement_lines)}``.

  Returns:
    Edit map keyed by base line index.
  """
  edits: dict[int, tuple[str, int, list[str]]] = {}
  sm = difflib.SequenceMatcher(None, base_lines, modified_lines)
  for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag != 'equal':
      edits[i1] = (tag, i2, modified_lines[j1:j2])
  return edits


def extract_changed_line_numbers(
  base_lines: list[str],
  modified_lines: list[str],
) -> set[int]:
  """Extract base line numbers that differ from modified.

  Returns:
    Set of base line indices touched by non-equal opcodes.
  """
  changes: set[int] = set()
  sm = difflib.SequenceMatcher(None, base_lines, modified_lines)
  for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
    if tag != 'equal':
      changes.update(range(i1, max(i2, i1 + 1)))
      if tag == 'insert':
        changes.add(i1)
  return changes
