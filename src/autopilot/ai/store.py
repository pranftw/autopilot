"""FileStore: content-addressed file versioning for the optimization loop.

Pure Python implementation using SHA-256, 2-char prefix sharding, JSON
snapshot manifests, and atomic writes. Like .git/ but purpose-built for
experiment code versioning.

Idempotent constructor follows the Experiment.__init__ pattern:
if slug exists, loads existing state; otherwise initializes fresh.

Fully decoupled from concrete parameter types: operates exclusively
through param.snapshot() / param.restore().
"""

from autopilot.core.errors import StoreError, TrackingError
from autopilot.core.parameter import Parameter
from autopilot.core.store import (
  DiffEntry,
  DiffResult,
  FileEntry,
  MergeResult,
  SnapshotEntry,
  SnapshotManifest,
  StatusEntry,
  StatusResult,
  Store,
)
from autopilot.tracking.io import atomic_write_json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import difflib
import hashlib
import json
import os


def _hash_content(text: str) -> str:
  return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _hash_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(65536), b''):
      h.update(chunk)
  return h.hexdigest()


def _atomic_write_json_safe(path: Path, payload: dict[str, Any]) -> None:
  try:
    atomic_write_json(path, payload)
  except TrackingError as exc:
    raise StoreError(str(exc)) from exc


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat()


class FileStore(Store):
  """Content-addressed file store. One instance per experiment slug.

  Idempotent: if slug exists in refs.json, loads state.
  Otherwise snapshots parameters, stores objects, writes epoch_0 baseline.

  Storage layout:
    objects/<2-char-prefix>/<rest>          -- SHA-256 hashed blobs, deduplicated
    snapshots/<slug>/epoch_<N>.json        -- JSON manifest mapping composite keys to FileEntry
    refs.json                              -- slug -> {latest_epoch, parent_slug,
                                                parent_epoch} + HEAD

  Composite keys: snapshot entries use 'param_name/state_key' format
  (e.g. 'param_0/main.py'). Parameters are keyed by construction order
  (param_0, param_1, ...), not by any user-facing name.

  Locking: exclusive file lock via os.open(O_CREAT | O_EXCL) during snapshot
  writes. A crash leaves a stale .lock that must be manually removed.

  Fully decoupled from concrete parameter types: operates exclusively
  through param.snapshot() / param.restore().

  Gotchas:
    - FileEntry.mtime is always 0.0 for snapshot content (no filesystem timestamps).
    - promote() checks for external modifications and raises StoreError if files
      were edited outside the store between epochs.
    - Changing the parameter list between runs may misalign keys.
  """

  def __init__(self, path: str | Path, slug: str, parameters: list[Parameter]) -> None:
    self._path = Path(path)
    self._slug = slug
    self._param_names: dict[str, Parameter] = {}
    for idx, param in enumerate(parameters):
      self._param_names[f'param_{idx}'] = param

    self._objects_dir = self._path / 'objects'
    self._snapshots_dir = self._path / 'snapshots'
    self._refs_file = self._path / 'refs.json'
    self._lock_file = self._path / '.lock'

    refs = self._load_refs()
    if slug in refs and slug != 'HEAD':
      self._epoch = refs[slug]['latest_epoch']
    else:
      self._init_fresh(slug)

  def _init_fresh(self, slug: str) -> None:
    self._objects_dir.mkdir(parents=True, exist_ok=True)
    self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    manifest = self._build_snapshot()
    manifest.epoch = 0
    self._save_snapshot(slug, 0, manifest)

    refs = self._load_refs()
    refs[slug] = {
      'latest_epoch': 0,
      'parent_slug': None,
      'parent_epoch': None,
    }
    refs['HEAD'] = {'slug': slug, 'epoch': 0}
    self._save_refs(refs)
    self._epoch = 0

  @property
  def slug(self) -> str:
    return self._slug

  @property
  def epoch(self) -> int:
    return self._epoch

  @property
  def path(self) -> Path:
    return self._path

  # snapshot

  def snapshot(self, epoch: int) -> SnapshotManifest:
    expected = self._epoch + 1
    if epoch != expected:
      raise StoreError(f'epoch must be sequential: expected {expected}, got {epoch}')

    self._acquire_lock()
    try:
      manifest = self._build_snapshot()
      manifest.epoch = epoch
      self._save_snapshot(self._slug, epoch, manifest)

      refs = self._load_refs()
      refs[self._slug]['latest_epoch'] = epoch
      refs['HEAD'] = {'slug': self._slug, 'epoch': epoch}
      self._save_refs(refs)
      self._epoch = epoch
    finally:
      self._release_lock()

    return manifest

  # checkout

  def checkout(self, epoch: int) -> None:
    snap = self._load_snapshot(self._slug, epoch)
    grouped = self._group_by_param(snap)
    for name, param in self._param_names.items():
      if name in grouped:
        param.restore(grouped[name])

    refs = self._load_refs()
    refs['HEAD'] = {'slug': self._slug, 'epoch': epoch}
    self._save_refs(refs)

  # diff

  def diff(self, epoch_a: int, slug_b: str, epoch_b: int) -> DiffResult:
    snap_a = self._load_snapshot(self._slug, epoch_a)
    snap_b = self._load_snapshot(slug_b, epoch_b)

    all_keys = set(snap_a.entries) | set(snap_b.entries)
    entries: list[DiffEntry] = []

    for key in sorted(all_keys):
      in_a = key in snap_a.entries
      in_b = key in snap_b.entries

      if in_a and not in_b:
        entries.append(
          DiffEntry(
            path=key,
            status='deleted',
            old_hash=snap_a.entries[key].hash,
          )
        )
      elif not in_a and in_b:
        entries.append(
          DiffEntry(
            path=key,
            status='added',
            new_hash=snap_b.entries[key].hash,
          )
        )
      elif snap_a.entries[key].hash != snap_b.entries[key].hash:
        old_content = self._read_object(snap_a.entries[key].hash)
        new_content = self._read_object(snap_b.entries[key].hash)
        text_diff = self._text_diff(key, old_content, new_content)
        entries.append(
          DiffEntry(
            path=key,
            status='modified',
            old_hash=snap_a.entries[key].hash,
            new_hash=snap_b.entries[key].hash,
            text_diff=text_diff,
          )
        )

    return DiffResult(entries=entries)

  # branch

  def branch(self, new_slug: str, from_epoch: int) -> None:
    refs = self._load_refs()
    if new_slug in refs and new_slug != 'HEAD':
      raise StoreError(f'slug {new_slug!r} already exists')

    snap = self._load_snapshot(self._slug, from_epoch)
    self._save_snapshot(
      new_slug,
      0,
      SnapshotManifest(
        epoch=0,
        timestamp=snap.timestamp,
        entries=dict(snap.entries),
      ),
    )

    refs[new_slug] = {
      'latest_epoch': 0,
      'parent_slug': self._slug,
      'parent_epoch': from_epoch,
    }
    self._save_refs(refs)

  # merge

  def merge(self, from_slug: str, from_epoch: int | None = None) -> MergeResult:
    refs = self._load_refs()
    if from_slug not in refs or from_slug == 'HEAD':
      raise StoreError(f'slug {from_slug!r} not found')

    if from_epoch is None:
      from_epoch = refs[from_slug]['latest_epoch']

    base_snap = self._find_common_ancestor(self._slug, from_slug, refs)
    ours_snap = self._load_snapshot(self._slug, self._epoch)
    theirs_snap = self._load_snapshot(from_slug, from_epoch)

    all_keys = set(base_snap.entries) | set(ours_snap.entries) | set(theirs_snap.entries)
    merged_entries: dict[str, FileEntry] = {}
    conflicts: list[str] = []

    for key in sorted(all_keys):
      base_entry = base_snap.entries.get(key)
      ours_entry = ours_snap.entries.get(key)
      theirs_entry = theirs_snap.entries.get(key)

      base_hash = base_entry.hash if base_entry else None
      ours_hash = ours_entry.hash if ours_entry else None
      theirs_hash = theirs_entry.hash if theirs_entry else None

      if ours_hash == theirs_hash:
        if ours_entry:
          merged_entries[key] = ours_entry
      elif ours_hash == base_hash:
        if theirs_entry:
          merged_entries[key] = theirs_entry
      elif theirs_hash == base_hash:
        if ours_entry:
          merged_entries[key] = ours_entry
      else:
        if ours_entry and theirs_entry and base_entry:
          base_content = self._read_object(base_hash)
          ours_content = self._read_object(ours_hash)
          theirs_content = self._read_object(theirs_hash)
          merged_text = self._three_way_merge(key, base_content, ours_content, theirs_content)
          if merged_text is not None:
            merged_bytes = merged_text.encode('utf-8')
            merged_hash = _hash_bytes(merged_bytes)
            self._store_object_bytes(merged_hash, merged_bytes)
            merged_entries[key] = FileEntry(
              hash=merged_hash,
              size=len(merged_bytes),
              mtime=0.0,
            )
          else:
            conflicts.append(key)
        else:
          conflicts.append(key)

    merged_snap = SnapshotManifest(
      epoch=self._epoch,
      timestamp=_now_iso(),
      entries=merged_entries,
    )

    return MergeResult(
      merged=len(conflicts) == 0,
      conflicts=conflicts,
      merged_snapshot=merged_snap,
    )

  # log

  def log(self) -> list[SnapshotEntry]:
    refs = self._load_refs()
    if self._slug not in refs or self._slug == 'HEAD':
      raise StoreError(f'slug {self._slug!r} not found')

    latest = refs[self._slug]['latest_epoch']
    entries: list[SnapshotEntry] = []
    for ep in range(latest + 1):
      snap = self._load_snapshot(self._slug, ep)
      entries.append(
        SnapshotEntry(
          epoch=snap.epoch,
          timestamp=snap.timestamp,
          file_count=len(snap.entries),
        )
      )
    return entries

  # status

  def status(self) -> StatusResult:
    snap = self._load_snapshot(self._slug, self._epoch)
    entries: list[StatusEntry] = []

    current_keys: set[str] = set()
    for name, param in self._param_names.items():
      content = param.snapshot()
      for state_key, text in content.items():
        full_key = f'{name}/{state_key}'
        current_keys.add(full_key)
        current_hash = _hash_content(text)

        if full_key not in snap.entries:
          entries.append(StatusEntry(path=full_key, status='added'))
        elif current_hash != snap.entries[full_key].hash:
          entries.append(StatusEntry(path=full_key, status='modified'))
        else:
          entries.append(StatusEntry(path=full_key, status='unchanged'))

    for key in snap.entries:
      if key not in current_keys:
        entries.append(StatusEntry(path=key, status='deleted'))

    return StatusResult(entries=entries)

  # promote

  def promote(self, epoch: int) -> None:
    snap = self._load_snapshot(self._slug, epoch)
    current_snap = self._load_snapshot(self._slug, self._epoch)

    current_content = self._snapshot_all_params()
    for key, entry in snap.entries.items():
      if key in current_content:
        current_hash = _hash_content(current_content[key])
        if current_hash != entry.hash:
          expected = current_snap.entries.get(key)
          if expected and current_hash != expected.hash:
            raise StoreError(
              f'external modification detected for {key}: '
              f'expected {expected.hash[:12]}, found {current_hash[:12]}'
            )

    grouped = self._group_by_param(snap)
    for name, param in self._param_names.items():
      if name in grouped:
        param.restore(grouped[name])

    promoted_manifest = SnapshotManifest(
      epoch=0,
      timestamp=_now_iso(),
      entries=dict(snap.entries),
    )
    self._save_snapshot(self._slug, 0, promoted_manifest)

    refs = self._load_refs()
    refs['HEAD'] = {'slug': self._slug, 'epoch': epoch}
    self._save_refs(refs)

  # internal helpers

  def _build_snapshot(self) -> SnapshotManifest:
    entries: dict[str, FileEntry] = {}
    for name, param in self._param_names.items():
      content = param.snapshot()
      for state_key, text in content.items():
        full_key = f'{name}/{state_key}'
        sha = _hash_content(text)
        self._store_object_bytes(sha, text.encode('utf-8'))
        entries[full_key] = FileEntry(hash=sha, size=len(text), mtime=0.0)
    return SnapshotManifest(epoch=0, timestamp=_now_iso(), entries=entries)

  def _snapshot_all_params(self) -> dict[str, str]:
    """Get all current parameter content as {full_key: text}."""
    result: dict[str, str] = {}
    for name, param in self._param_names.items():
      content = param.snapshot()
      for state_key, text in content.items():
        result[f'{name}/{state_key}'] = text
    return result

  def _group_by_param(self, snap: SnapshotManifest) -> dict[str, dict[str, str]]:
    """Group snapshot entries by parameter name and load object content."""
    grouped: dict[str, dict[str, str]] = {}
    for full_key, entry in snap.entries.items():
      param_name, _, rel_key = full_key.partition('/')
      text = self._read_object(entry.hash).decode('utf-8')
      grouped.setdefault(param_name, {})[rel_key] = text
    return grouped

  def _store_object_bytes(self, content_hash: str, data: bytes) -> None:
    prefix = content_hash[:2]
    rest = content_hash[2:]
    obj_dir = self._objects_dir / prefix
    obj_path = obj_dir / rest
    if obj_path.exists():
      return
    obj_dir.mkdir(parents=True, exist_ok=True)
    tmp = obj_path.with_suffix('.tmp')
    try:
      tmp.write_bytes(data)
      tmp.replace(obj_path)
    except OSError as exc:
      raise StoreError(f'failed to store object {content_hash}: {exc}') from exc

  def _read_object(self, content_hash: str) -> bytes:
    prefix = content_hash[:2]
    rest = content_hash[2:]
    obj_path = self._objects_dir / prefix / rest
    if not obj_path.exists():
      raise StoreError(f'object {content_hash} not found')
    return obj_path.read_bytes()

  def _load_refs(self) -> dict[str, Any]:
    if not self._refs_file.exists():
      return {}
    try:
      text = self._refs_file.read_text(encoding='utf-8')
      return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
      raise StoreError(f'failed to load refs: {exc}') from exc

  def _save_refs(self, refs: dict[str, Any]) -> None:
    _atomic_write_json_safe(self._refs_file, refs)

  def _load_snapshot(self, slug: str, epoch: int) -> SnapshotManifest:
    path = self._snapshots_dir / slug / f'epoch_{epoch}.json'
    if not path.exists():
      raise StoreError(f'snapshot not found: {slug} epoch {epoch}')
    try:
      text = path.read_text(encoding='utf-8')
      data = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
      raise StoreError(f'failed to load snapshot {slug} epoch {epoch}: {exc}') from exc
    return SnapshotManifest.from_dict(data)

  def _save_snapshot(self, slug: str, epoch: int, manifest: SnapshotManifest) -> None:
    path = self._snapshots_dir / slug / f'epoch_{epoch}.json'
    _atomic_write_json_safe(path, manifest.to_dict())

  def _acquire_lock(self) -> None:
    self._path.mkdir(parents=True, exist_ok=True)
    try:
      fd = os.open(str(self._lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
      os.close(fd)
    except FileExistsError:
      raise StoreError('store is locked by another operation')

  def _release_lock(self) -> None:
    self._lock_file.unlink(missing_ok=True)

  def _text_diff(self, key: str, old_content: bytes, new_content: bytes) -> str:
    try:
      old_lines = old_content.decode('utf-8').splitlines(keepends=True)
      new_lines = new_content.decode('utf-8').splitlines(keepends=True)
    except UnicodeDecodeError:
      return '(binary files differ)'
    return ''.join(difflib.unified_diff(old_lines, new_lines, fromfile=key, tofile=key))

  def _three_way_merge(
    self,
    key: str,
    base: bytes,
    ours: bytes,
    theirs: bytes,
  ) -> str | None:
    try:
      base_lines = base.decode('utf-8').splitlines(keepends=True)
      ours_lines = ours.decode('utf-8').splitlines(keepends=True)
      theirs_lines = theirs.decode('utf-8').splitlines(keepends=True)
    except UnicodeDecodeError:
      return None

    merged: list[str] = []
    has_conflict = False

    ours_diff = list(difflib.unified_diff(base_lines, ours_lines))
    theirs_diff = list(difflib.unified_diff(base_lines, theirs_lines))

    if not ours_diff:
      return ''.join(theirs_lines)
    if not theirs_diff:
      return ''.join(ours_lines)

    ours_changes = self._extract_changed_line_numbers(base_lines, ours_lines)
    theirs_changes = self._extract_changed_line_numbers(base_lines, theirs_lines)

    if not ours_changes.intersection(theirs_changes):
      result_lines = list(ours_lines)
      offset = 0
      for lineno, content in sorted(self._extract_changes(base_lines, theirs_lines).items()):
        if content is None:
          if 0 <= lineno + offset < len(result_lines):
            result_lines.pop(lineno + offset)
            offset -= 1
        else:
          if lineno + offset < len(result_lines):
            result_lines[lineno + offset] = content
          else:
            result_lines.append(content)
      return ''.join(result_lines)

    for i, base_line in enumerate(base_lines):
      ours_line = ours_lines[i] if i < len(ours_lines) else None
      theirs_line = theirs_lines[i] if i < len(theirs_lines) else None

      if ours_line == theirs_line:
        if ours_line is not None:
          merged.append(ours_line)
      elif ours_line == base_line:
        if theirs_line is not None:
          merged.append(theirs_line)
      elif theirs_line == base_line:
        if ours_line is not None:
          merged.append(ours_line)
      else:
        has_conflict = True
        merged.append(f'<<<<<<< {self._slug}\n')
        if ours_line is not None:
          merged.append(ours_line)
        merged.append('=======\n')
        if theirs_line is not None:
          merged.append(theirs_line)
        merged.append('>>>>>>>\n')

    if has_conflict:
      return None

    return ''.join(merged)

  def _extract_changed_line_numbers(
    self,
    base_lines: list[str],
    modified_lines: list[str],
  ) -> set[int]:
    changes: set[int] = set()
    sm = difflib.SequenceMatcher(None, base_lines, modified_lines)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
      if tag != 'equal':
        for i in range(i1, i2):
          changes.add(i)
    return changes

  def _extract_changes(
    self,
    base_lines: list[str],
    modified_lines: list[str],
  ) -> dict[int, str | None]:
    changes: dict[int, str | None] = {}
    sm = difflib.SequenceMatcher(None, base_lines, modified_lines)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
      if tag == 'replace':
        for idx, j in enumerate(range(j1, j2)):
          changes[i1 + idx] = modified_lines[j]
      elif tag == 'delete':
        for i in range(i1, i2):
          changes[i] = None
      elif tag == 'insert':
        for j in range(j1, j2):
          changes[i1] = modified_lines[j]
    return changes

  def _find_common_ancestor(
    self,
    slug_a: str,
    slug_b: str,
    refs: dict[str, Any],
  ) -> SnapshotManifest:
    ancestors_a = set(self._ancestor_chain(slug_a, refs))
    ancestors_b = self._ancestor_chain(slug_b, refs)

    for slug, epoch in ancestors_b:
      if (slug, epoch) in ancestors_a:
        return self._load_snapshot(slug, epoch)

    return SnapshotManifest(epoch=0, timestamp=_now_iso(), entries={})

  def _ancestor_chain(
    self,
    slug: str,
    refs: dict[str, Any],
  ) -> list[tuple[str, int]]:
    """Walk parent chain collecting (slug, epoch) pairs from root to branch point."""
    chain: list[tuple[str, int]] = []
    current = slug
    visited: set[str] = set()
    while current and current != 'HEAD' and current not in visited:
      visited.add(current)
      info = refs.get(current)
      if not info:
        break
      for ep in range(info.get('latest_epoch', 0) + 1):
        chain.append((current, ep))
      parent = info.get('parent_slug')
      if parent:
        parent_epoch = info.get('parent_epoch', 0)
        for ep in range(parent_epoch + 1):
          chain.append((parent, ep))
      current = parent
    return chain

  def __repr__(self) -> str:
    return f'FileStore(slug={self._slug!r}, epoch={self._epoch}, path={self._path})'
