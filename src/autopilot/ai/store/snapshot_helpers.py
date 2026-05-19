"""Shared helpers for snapshot manifests, grouping, and checkout hygiene."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store_lock import hash_content
from autopilot.core.errors import StoreError
from autopilot.core.parameter import Parameter
from autopilot.core.snapshot import (
  FileEntry,
  ParameterSchema,
  ParameterSchemaEntry,
  SnapshotManifest,
)
from autopilot.tracking.io import BINARY_SNIFF_BYTES, utc_now_iso
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTECTED_PREFIXES = frozenset({'.git', '.autopilot', 'node_modules', '.venv', '__pycache__'})
FILE_ENTRY_MTIME_UNAVAILABLE = 0.0


def is_probably_binary_file(path: Path) -> bool:
  """Heuristic: file is binary if it contains null bytes in first 8KB.

  Args:
    path: File path to check.

  Returns:
    True when the file appears to be binary; False for text or on read error.
  """
  try:
    chunk = path.read_bytes()[:BINARY_SNIFF_BYTES]
  except OSError:
    return False
  return b'\x00' in chunk


@dataclass(frozen=True)
class SchemaMatchResult:
  """Result of validating manifest param names against registered parameters.

  Attributes:
    matched: Manifest parameter names that are also registered.
    mismatched: Manifest parameter names with no registered counterpart.
  """

  matched: frozenset[str]
  mismatched: frozenset[str]


def validate_schema(store: Any, snap: SnapshotManifest) -> SchemaMatchResult:
  """Check manifest parameter names against registered parameters.

  Derives parameter names from the manifest entries (the prefix before the
  first ``/`` in each key) and compares them to registered parameter names.
  Falls back to ``ParameterSchema`` names when the manifest has no entries
  but does carry a schema.

  Args:
    store: FileStore instance.
    snap: Snapshot manifest to validate.

  Returns:
    SchemaMatchResult with matched and mismatched name sets.
  """
  if snap.entries:
    manifest_names = {key.partition('/')[0] for key in snap.entries}
  elif snap.schema is not None:
    manifest_names = {e.name for e in snap.schema.parameters}
  else:
    return SchemaMatchResult(matched=frozenset(), mismatched=frozenset())
  registered = set(store._param_names)
  return SchemaMatchResult(
    matched=frozenset(manifest_names & registered),
    mismatched=frozenset(manifest_names - registered),
  )


def build_snapshot(store: Any, context: str | None = None) -> SnapshotManifest:
  """Build a snapshot manifest from registered parameters.

  Args:
    store: FileStore instance.
    context: Optional reason/provenance string.

  Returns:
    SnapshotManifest with entries from all registered parameters.

  Raises:
    StoreError: If no parameters are registered.
  """
  if not store._param_names:
    msg = (
      'No parameters registered. Call FileStore.register_parameters(dict) '
      'before snapshot(). An empty snapshot is almost certainly a bug.'
    )
    raise StoreError(msg)
  entries: dict[str, FileEntry] = {}
  schema_entries: list[ParameterSchemaEntry] = []
  for name, param in store._param_names.items():
    content = param.snapshot()
    for state_key, text in content.items():
      full_key = f'{name}/{state_key}'
      sha = hash_content(text)
      store._store_object_bytes(sha, text.encode('utf-8'))
      original_path = resolve_original_path(store, param, state_key)
      entries[full_key] = FileEntry(
        digest=sha,
        size=len(text),
        mtime=FILE_ENTRY_MTIME_UNAVAILABLE,
        original_path=original_path,
      )
    entry = param.schema_entry()
    entry.name = name
    schema_entries.append(entry)
  schema = ParameterSchema(parameters=schema_entries) if schema_entries else None
  return SnapshotManifest(
    epoch=0, timestamp=utc_now_iso(), entries=entries, schema=schema, context=context
  )


def resolve_original_path(store: Any, param: Parameter, state_key: str) -> str | None:
  """Compute the workspace-relative path for a parameter's state key.

  Uses ``working_root`` so worktree-bound paths resolve correctly.

  Returns:
    Workspace-relative path string when resolvable; otherwise ``None``.
  """
  if not isinstance(param, PathParameter):
    return None
  param_path = Path(param.working_root).expanduser() / state_key
  try:
    return str(param_path.relative_to(store._config.workspace))
  except ValueError:
    return None


def snapshot_all_params(store: Any) -> dict[str, str]:
  """Snapshot all registered parameters into a flat dict.

  Args:
    store: FileStore instance.

  Returns:
    Dict mapping composite keys to text content.
  """
  result: dict[str, str] = {}
  for name, param in store._param_names.items():
    content = param.snapshot()
    for state_key, text in content.items():
      result[f'{name}/{state_key}'] = text
  return result


def group_by_param(
  store: Any,
  snap: SnapshotManifest,
) -> dict[str, dict[str, str]]:
  """Group snapshot entries by parameter name, reading object content.

  Args:
    store: FileStore instance.
    snap: Snapshot manifest to group.

  Returns:
    Dict mapping param name to {state_key: text_content}.
  """
  grouped: dict[str, dict[str, str]] = {}
  for full_key, entry in snap.entries.items():
    param_name, _, rel_key = full_key.partition('/')
    text = store.read_object(entry.digest).decode('utf-8')
    grouped.setdefault(param_name, {})[rel_key] = text
  return grouped


def remove_extraneous_files(
  _store: Any,
  param: PathParameter,
  param_name: str,
  snapshot_keys: set[str],
) -> None:
  """Remove working-tree files absent from the snapshot overlay (BUG-075).

  After checkout restores files from the snapshot, any file matched by the
  parameter's pattern that is NOT in the snapshot should be removed. This
  prevents stale files from leaking across checkouts.

  Files under protected directory prefixes (``PROTECTED_PREFIXES``) are
  never deleted, regardless of whether they appear in the snapshot.

  Binary files (detected via null-byte heuristic) are preserved even when
  not in the manifest, since they are skipped during snapshot and would
  otherwise be destroyed on checkout (BUG-012).

  Args:
    param: PathParameter whose working tree to clean.
    param_name: Parameter key prefix for snapshot entries.
    snapshot_keys: Set of composite keys present in the snapshot.
  """
  root = Path(param.working_root).expanduser()
  for matched in param.matched_files():
    try:
      rel = matched.relative_to(root)
    except ValueError:
      continue
    parts = rel.parts
    if parts and parts[0] in PROTECTED_PREFIXES:
      continue
    composite_key = f'{param_name}/{rel}'
    if composite_key not in snapshot_keys:
      if is_probably_binary_file(matched):
        continue
      matched.unlink(missing_ok=True)
