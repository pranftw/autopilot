"""Merge, diff, tag, and status model types for the Store persistence layer.

Portable dataclasses, enums, and constants consumed by both the abstract
``Store`` base (``core/store/base.py``) and its concrete ``FileStore``
implementation (``ai/store/``).
"""

from autopilot.core.errors import StoreError
from autopilot.core.serialization import DictMixin
from autopilot.core.snapshot import FileEntry, SnapshotManifest
from dataclasses import dataclass, field
from typing import Any
import enum
import re

TAG_NAME_MAX_LEN = 128
TAG_NAME_ALLOWED_RE = re.compile(r'^[a-zA-Z0-9._-]+$')


@dataclass(frozen=True)
class TagEntry(DictMixin):
  """Immutable tag pointing to a specific experiment branch and epoch.

  Attributes:
    name: Tag name (ASCII letters, digits, hyphen, underscore, dot only;
      max 128 chars; no slashes or path-like names).
    experiment_id: Branch the tag points to.
    epoch: Epoch the tag points to.
    context: Optional reason string for audit traceability.
    timestamp: ISO 8601 creation timestamp, or None when not available.
    manifest_digest: SHA-256 hex digest of the canonical manifest JSON at
      tag creation time. ``None`` for pre-attestation tags (created before
      digest computation was added). Deserialization tolerance: missing key
      in ``refs.json`` maps to ``None``.
  """

  name: str
  experiment_id: str
  epoch: int
  context: str | None = None
  timestamp: str | None = None
  manifest_digest: str | None = None


def validate_tag_name(name: str) -> None:
  """Validate a tag name against naming rules.

  Args:
    name: Tag name to validate.

  Raises:
    StoreError: If the name violates any naming rule.
  """
  if not name:
    msg = 'tag name must not be empty'
    raise StoreError(msg)
  if len(name) > TAG_NAME_MAX_LEN:
    msg = f'tag name exceeds {TAG_NAME_MAX_LEN} characters: {name!r}'
    raise StoreError(msg)
  if not TAG_NAME_ALLOWED_RE.match(name):
    msg = (
      f'tag name {name!r} contains invalid characters. '
      f'Allowed: ASCII letters (a-z, A-Z), digits (0-9), hyphen (-), '
      f'underscore (_), and dot (.). '
      f'Slashes (/), spaces, and other punctuation are not permitted.'
    )
    raise StoreError(msg)
  if name.startswith('.') or name.endswith('.'):
    msg = f'tag name {name!r} must not start or end with "."'
    raise StoreError(msg)
  if '..' in name:
    msg = f'tag name {name!r} must not contain ".."'
    raise StoreError(msg)


class DiffKind(enum.StrEnum):
  """Status values for diff and status entries.

  Members compare equal to plain strings (``DiffKind.added == 'added'``)
  so ``DiffEntry.status`` / ``StatusEntry.status`` can stay typed as ``str``
  while call sites compare against enum members.
  """

  added = 'added'
  modified = 'modified'
  deleted = 'deleted'
  unchanged = 'unchanged'


class MergeClassification(enum.StrEnum):
  """Merge analysis classification values.

  Members compare equal to plain strings so ``MergeAnalysisResult.classification``
  (typed ``MergeClassification``) serializes via ``.value`` in ``to_dict()``
  and reconstructs from strings in ``from_dict()``.
  """

  up_to_date = 'up_to_date'
  fast_forward = 'fast_forward'
  clean = 'clean'
  conflict = 'conflict'


@dataclass
class DiffEntry(DictMixin):
  """Single file change between two snapshots.

  Attributes:
    path: Manifest-relative file path (e.g. ``'prompt.txt'``).
    status: Change kind as a ``DiffKind`` string value (``'added'``,
      ``'modified'``, ``'deleted'``, ``'unchanged'``).
    old_hash: Content digest from the earlier snapshot, or None for added files.
    new_hash: Content digest from the later snapshot, or None for deleted files.
    text_diff: Optional unified diff text between old and new content.
  """

  path: str
  status: str
  old_hash: str | None = None
  new_hash: str | None = None
  text_diff: str | None = None


@dataclass
class DiffResult(DictMixin):
  """Diff between two snapshots: list of per-file changes.

  Attributes:
    entries: Per-file diff rows between the two snapshots.
  """

  entries: list[DiffEntry] = field(default_factory=list)

  def added(self) -> list[DiffEntry]:
    """Return diff rows with status ``added``."""
    return [e for e in self.entries if e.status == DiffKind.added]

  def modified(self) -> list[DiffEntry]:
    """Return diff rows with status ``modified``."""
    return [e for e in self.entries if e.status == DiffKind.modified]

  def deleted(self) -> list[DiffEntry]:
    """Return diff rows with status ``deleted``."""
    return [e for e in self.entries if e.status == DiffKind.deleted]

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'DiffResult':
    """Deserialize from dict, handling null -> empty container coercion for collection fields.

    Returns:
      ``DiffResult`` with a possibly empty ``entries`` list.
    """
    raw_list = data.get('entries')
    entry_seq = [] if raw_list is None else raw_list
    entries = [DiffEntry.from_dict(e) for e in entry_seq]
    return cls(entries=entries)


class MergeStrategy(enum.Enum):
  """Merge strategy for three-way merge operations.

  Attributes:
    normal: Standard three-way merge; conflicts require manual resolution.
    ours: Auto-resolve conflicts by choosing the target branch side.
    theirs: Auto-resolve conflicts by choosing the source branch side.
    union: Deterministic concatenation for non-overlapping additive text; overlapping
      regions fall back to normal conflict behavior.
  """

  normal = 'normal'
  ours = 'ours'
  theirs = 'theirs'
  union = 'union'


@dataclass
class ConflictEntry(DictMixin):
  """Single-key merge conflict with optional ancestor/ours/theirs sides.

  Each side is a ``FileEntry`` when the key exists on that branch, or ``None``
  when the key is absent (e.g. a delete-vs-modify conflict has ``ours=None``).

  Attributes:
    key: Composite manifest key (``param_name/state_key``).
    ancestor: Entry from the common ancestor snapshot; None if added since LCA.
    ours: Entry from the target branch; None if deleted or absent.
    theirs: Entry from the source branch; None if deleted or absent.
  """

  key: str
  ancestor: FileEntry | None = None
  ours: FileEntry | None = None
  theirs: FileEntry | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ConflictEntry':
    """Deserialize from dict with nested FileEntry parsing.

    Args:
      data: Raw dict with ``key`` and optional ``ancestor``/``ours``/``theirs``.

    Returns:
      ConflictEntry instance.
    """
    return cls(
      key=data['key'],
      ancestor=FileEntry.from_dict(data['ancestor']) if data.get('ancestor') else None,
      ours=FileEntry.from_dict(data['ours']) if data.get('ours') else None,
      theirs=FileEntry.from_dict(data['theirs']) if data.get('theirs') else None,
    )


@dataclass
class MergeAnalysisResult(DictMixin):
  """Pre-flight merge classification.

  Cheap heuristic using refs and manifest key overlap (no blob reads).
  Same key touched on both sides since LCA implies ``has_conflicts=True``.

  Attributes:
    can_fast_forward: True when target has not diverged from ancestor.
    has_conflicts: True when key-overlap heuristic predicts conflicts.
    conflict_count: Predicted number of conflicted keys.
    ancestor_epoch: LCA epoch on the ancestor experiment, or None.
    classification: Merge classification as a ``MergeClassification`` enum member.
  """

  can_fast_forward: bool
  has_conflicts: bool
  conflict_count: int
  ancestor_epoch: int | None = None
  classification: MergeClassification = MergeClassification.up_to_date

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'MergeAnalysisResult':
    """Deserialize from dict, reconstructing the MergeClassification enum.

    Args:
      data: Raw dict with merge analysis fields.

    Returns:
      MergeAnalysisResult with a proper MergeClassification enum value.

    Raises:
      StoreError: When classification value is not a valid MergeClassification member.
    """
    classification_raw = data.get('classification', MergeClassification.up_to_date.value)
    try:
      classification = MergeClassification(classification_raw)
    except ValueError:
      valid = [m.value for m in MergeClassification]
      msg = f'unknown merge classification {classification_raw!r}; valid values: {valid}'
      raise StoreError(msg) from None
    return cls(
      can_fast_forward=data['can_fast_forward'],
      has_conflicts=data['has_conflicts'],
      conflict_count=data['conflict_count'],
      ancestor_epoch=data.get('ancestor_epoch'),
      classification=classification,
    )


@dataclass
class MergeIndex(DictMixin):
  """Staging area for merge resolution before apply.

  Populated by ``merge_preview``; conflicts must be resolved (moved into
  ``resolved``) before ``merge_apply`` will accept the index. Strategies
  ``ours``/``theirs`` auto-resolve all conflicts during preview.

  Attributes:
    conflicts: Unresolved conflict entries keyed by manifest key.
    resolved: Resolved entries keyed by manifest key (includes clean merges).
    experiment_id: Target experiment receiving the merge.
    source_experiment_id: Source experiment being merged in.
    strategy: Strategy used to produce this index.
    preview_token: Cryptographic token binding this index to specific ref tips;
      ``merge_apply`` rejects stale tokens.
  """

  conflicts: dict[str, ConflictEntry] = field(default_factory=dict)
  resolved: dict[str, FileEntry] = field(default_factory=dict)
  experiment_id: str | None = None
  source_experiment_id: str | None = None
  strategy: MergeStrategy = MergeStrategy.normal
  preview_token: str | None = None

  def resolve(self, key: str, entry: FileEntry) -> None:
    """Record explicit resolution for a conflicted key.

    Moves ``key`` from ``conflicts`` to ``resolved`` with the provided entry.

    Args:
      key: Manifest key to resolve.
      entry: The FileEntry to use as the resolved value.

    Raises:
      StoreError: If ``key`` is not in ``conflicts``.
    """
    if key not in self.conflicts:
      msg = f'key {key!r} is not in conflicts; cannot resolve'
      raise StoreError(msg)
    del self.conflicts[key]
    self.resolved[key] = entry

  def resolve_ours(self, key: str) -> None:
    """Resolve using the ours side from the ConflictEntry.

    Args:
      key: Manifest key to resolve.

    Raises:
      StoreError: If ``key`` is not in conflicts or ours side is None.
    """
    if key not in self.conflicts:
      msg = f'key {key!r} is not in conflicts; cannot resolve_ours'
      raise StoreError(msg)
    conflict = self.conflicts[key]
    if conflict.ours is None:
      msg = f'ours side is None for conflict key {key!r}; cannot resolve_ours'
      raise StoreError(msg)
    self.resolve(key, conflict.ours)

  def resolve_theirs(self, key: str) -> None:
    """Resolve using the theirs side from the ConflictEntry.

    Args:
      key: Manifest key to resolve.

    Raises:
      StoreError: If ``key`` is not in conflicts or theirs side is None.
    """
    if key not in self.conflicts:
      msg = f'key {key!r} is not in conflicts; cannot resolve_theirs'
      raise StoreError(msg)
    conflict = self.conflicts[key]
    if conflict.theirs is None:
      msg = f'theirs side is None for conflict key {key!r}; cannot resolve_theirs'
      raise StoreError(msg)
    self.resolve(key, conflict.theirs)

  def is_resolved(self) -> bool:
    """True when all conflicts have been resolved.

    Returns:
      True when ``len(self.conflicts) == 0``.
    """
    return len(self.conflicts) == 0

  def to_snapshot(self) -> SnapshotManifest:
    """Build a SnapshotManifest from resolved entries.

    Returns:
      SnapshotManifest containing only ``resolved`` entries. Epoch and
      timestamp are placeholders (0, empty) -- ``merge_apply`` overwrites.

    Raises:
      StoreError: If unresolved conflicts remain.
    """
    if not self.is_resolved():
      remaining = sorted(self.conflicts)
      msg = f'cannot build snapshot with {len(remaining)} unresolved conflict(s): {remaining}'
      raise StoreError(msg)
    return SnapshotManifest(
      epoch=0,
      timestamp='',
      entries=dict(self.resolved),
    )

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'MergeIndex':
    """Deserialize from dict with nested ConflictEntry/FileEntry parsing.

    Args:
      data: Raw dict from ``to_dict()`` output.

    Returns:
      MergeIndex with properly typed nested objects.
    """
    raw_conflicts = data.get('conflicts')
    conflicts_map = {} if raw_conflicts is None else raw_conflicts
    conflicts = {k: ConflictEntry.from_dict(v) for k, v in conflicts_map.items()}
    raw_resolved = data.get('resolved')
    resolved_map = {} if raw_resolved is None else raw_resolved
    resolved = {k: FileEntry.from_dict(v) for k, v in resolved_map.items()}
    raw_strategy = data.get('strategy', 'normal')
    strategy = MergeStrategy(raw_strategy)
    return cls(
      conflicts=conflicts,
      resolved=resolved,
      experiment_id=data.get('experiment_id'),
      source_experiment_id=data.get('source_experiment_id'),
      strategy=strategy,
      preview_token=data.get('preview_token'),
    )


@dataclass
class StatusEntry(DictMixin):
  """Single file status relative to a snapshot.

  Attributes:
    path: Manifest-relative file path.
    status: Status kind as a ``DiffKind`` string value (e.g. ``'modified'``).
  """

  path: str
  status: str


@dataclass
class StatusResult(DictMixin):
  """Status of all tracked files relative to a snapshot.

  Attributes:
    entries: Per-file status rows for tracked paths.
  """

  entries: list[StatusEntry] = field(default_factory=list)

  def modified(self) -> list[StatusEntry]:
    """Return status rows tagged ``modified``."""
    return [e for e in self.entries if e.status == DiffKind.modified]

  def added(self) -> list[StatusEntry]:
    """Return status rows tagged ``added``."""
    return [e for e in self.entries if e.status == DiffKind.added]

  def deleted(self) -> list[StatusEntry]:
    """Return status rows tagged ``deleted``."""
    return [e for e in self.entries if e.status == DiffKind.deleted]

  def unchanged(self) -> list[StatusEntry]:
    """Return status rows tagged ``unchanged``."""
    return [e for e in self.entries if e.status == DiffKind.unchanged]

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'StatusResult':
    """Deserialize from dict, handling null -> empty container coercion for collection fields.

    Returns:
      ``StatusResult`` with a possibly empty ``entries`` list.
    """
    raw_list = data.get('entries')
    entry_seq = [] if raw_list is None else raw_list
    entries = [StatusEntry.from_dict(e) for e in entry_seq]
    return cls(entries=entries)


@dataclass
class SnapshotEntry(DictMixin):
  """Summary of a snapshot for log output.

  Attributes:
    epoch: Snapshot epoch index.
    timestamp: Snapshot timestamp string.
    file_count: Number of files recorded in the snapshot.
  """

  epoch: int
  timestamp: str
  file_count: int
