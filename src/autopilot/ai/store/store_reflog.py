"""Store reflog helpers for FileStore (append, iterate, expire, recover)."""

from autopilot.core.errors import StoreError, TrackingError
from autopilot.tracking.io import append_jsonl, parse_timestamp, utc_now_iso
from collections.abc import Iterator
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
import json
import sys


def append_reflog(
  store: Any,
  operation: str,
  experiment_id: str,
  old_epoch: int | None,
  new_epoch: int | None,
  context: str | None = None,
  **extra: Any,
) -> None:
  """Append a reflog entry after a successful refs mutation.

  Must be called while the store lock is already held.

  Args:
    store: FileStore instance.
    operation: Operation name.
    experiment_id: Primary branch affected.
    old_epoch: Branch tip before operation.
    new_epoch: Branch tip after operation.
    context: Optional caller reason string.
    **extra: Additional operation-specific keys.
  """
  record: dict[str, Any] = {
    'timestamp': utc_now_iso(),
    'operation': operation,
    'experiment_id': experiment_id,
    'old_epoch': old_epoch,
    'new_epoch': new_epoch,
    'context': context,
  }
  record.update(extra)
  with suppress(OSError, TrackingError):
    append_jsonl(store._reflog_path, record)


def iter_reflog(store: Any) -> Iterator[dict[str, Any]]:
  """Yield reflog entries in chronological order (oldest first).

  Skips corrupt JSONL lines after emitting a message to stderr.

  Args:
    store: FileStore instance.

  Yields:
    One dict per valid line.

  Raises:
    StoreError: When the reflog file cannot be read.
  """
  path = store._reflog_path
  if not path.is_file():
    return
  try:
    raw = path.read_text(encoding='utf-8')
  except OSError as exc:
    msg = f'cannot read reflog at {path}: {exc}'
    raise StoreError(msg) from exc
  for line_no, line in enumerate(raw.splitlines(), start=1):
    stripped = line.strip()
    if not stripped:
      continue
    try:
      data = json.loads(stripped)
    except json.JSONDecodeError:
      sys.stderr.write(f'reflog: skipping corrupt line {line_no} in {path}\n')
      continue
    if not isinstance(data, dict):
      sys.stderr.write(f'reflog: skipping non-object on line {line_no} in {path}\n')
      continue
    yield data


def expire_reflog(store: Any, older_than: timedelta) -> int:
  """Drop reflog entries older than the cutoff; returns count removed.

  Args:
    store: FileStore instance.
    older_than: Entries whose timestamp is strictly before
      (now - older_than) are removed.

  Returns:
    Number of entries expired (removed).

  Raises:
    StoreError: When the reflog cannot be read or rewritten safely.
  """
  cutoff = datetime.now(UTC) - older_than
  path = store._reflog_path
  if not path.is_file():
    return 0

  try:
    raw = path.read_text(encoding='utf-8')
  except OSError as exc:
    msg = f'cannot read reflog at {path}: {exc}'
    raise StoreError(msg) from exc

  kept_lines: list[str] = []
  removed = 0
  for line in raw.splitlines():
    stripped = line.strip()
    if not stripped:
      continue
    try:
      data = json.loads(stripped)
    except json.JSONDecodeError:
      kept_lines.append(stripped)
      continue
    if not isinstance(data, dict):
      kept_lines.append(stripped)
      continue
    ts_str = data.get('timestamp')
    if ts_str is None:
      kept_lines.append(stripped)
      continue
    try:
      ts = parse_timestamp(ts_str)
    except (ValueError, TypeError):
      kept_lines.append(stripped)
      continue
    if ts < cutoff:
      removed += 1
    else:
      kept_lines.append(stripped)

  store._acquire_lock()
  try:
    new_content = ''.join(line + '\n' for line in kept_lines)
    tmp = path.with_suffix('.jsonl.tmp')
    try:
      tmp.write_text(new_content, encoding='utf-8')
      tmp.replace(path)
    except OSError as exc:
      msg = f'cannot rewrite reflog at {path}: {exc}'
      raise StoreError(msg) from exc
  finally:
    store._release_lock()

  return removed


def recover_from_reflog(store: Any, entry_index: int) -> None:
  """Restore branch tip metadata from reflog entry at linear index.

  Args:
    store: FileStore instance.
    entry_index: 0-based index into the sequence of valid reflog entries.

  Raises:
    StoreError: Out-of-range index, malformed entry, or conflicting store state.
  """
  entries = list(iter_reflog(store))
  if entry_index < 0 or entry_index >= len(entries):
    if not entries:
      msg = (
        f'reflog entry index {entry_index} out of range: '
        f'reflog is empty. Use iter_reflog() to inspect available entries.'
      )
    else:
      msg = (
        f'reflog entry index {entry_index} out of range '
        f'(valid: 0..{len(entries) - 1}). '
        f'Use iter_reflog() to inspect available entries.'
      )
    raise StoreError(msg)

  entry = entries[entry_index]
  experiment_id = entry.get('experiment_id')
  new_epoch = entry.get('new_epoch')

  if experiment_id is None:
    msg = (
      f'reflog entry at index {entry_index} is missing "experiment_id". '
      f'Cannot recover branch tip from a malformed entry.'
    )
    raise StoreError(msg)
  if new_epoch is None:
    msg = (
      f'reflog entry at index {entry_index} is missing "new_epoch". '
      f'Cannot recover branch tip without a target epoch.'
    )
    raise StoreError(msg)

  store._acquire_lock()
  try:
    refs = store.load_refs()
    branches = refs.setdefault('branches', {})
    if experiment_id not in branches:
      branches[experiment_id] = {'latest_epoch': new_epoch}
    else:
      branches[experiment_id]['latest_epoch'] = new_epoch
    refs['HEAD'] = experiment_id
    store.save_refs(refs)
    append_reflog(store, 'recover', experiment_id, None, new_epoch, context='reflog recovery')
  finally:
    store._release_lock()
