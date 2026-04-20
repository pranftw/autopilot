"""Store base class and supporting data models for content-addressed code versioning.

Store is the VCS equivalent for experiment code. Experiment slugs are branches.
Epochs are commits. Subclass and override to customize storage, hashing, or
any operation.

Same extension pattern as Loss -> JudgeLoss, Optimizer -> AgentOptimizer,
Checkpoint -> JSONCheckpoint.
"""

from autopilot.core.parameter import Parameter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileEntry:
  """Single file in a snapshot: content hash, size, and mtime."""

  hash: str
  size: int
  mtime: float

  def to_dict(self) -> dict[str, Any]:
    return {'hash': self.hash, 'size': self.size, 'mtime': self.mtime}

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'FileEntry':
    return cls(hash=data['hash'], size=data['size'], mtime=data['mtime'])


@dataclass
class SnapshotManifest:
  """Snapshot of all parameter files at a given epoch.

  entries maps 'param_name::relative_path' to FileEntry.
  """

  epoch: int
  timestamp: str
  entries: dict[str, FileEntry] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return {
      'epoch': self.epoch,
      'timestamp': self.timestamp,
      'entries': {k: v.to_dict() for k, v in self.entries.items()},
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'SnapshotManifest':
    entries = {k: FileEntry.from_dict(v) for k, v in data.get('entries', {}).items()}
    return cls(epoch=data['epoch'], timestamp=data['timestamp'], entries=entries)


@dataclass
class DiffEntry:
  """Single file change between two snapshots."""

  path: str
  status: str
  old_hash: str | None = None
  new_hash: str | None = None
  text_diff: str = ''

  def to_dict(self) -> dict[str, Any]:
    return {
      'path': self.path,
      'status': self.status,
      'old_hash': self.old_hash,
      'new_hash': self.new_hash,
      'text_diff': self.text_diff,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'DiffEntry':
    return cls(**data)


@dataclass
class DiffResult:
  """Diff between two snapshots: list of per-file changes."""

  entries: list[DiffEntry] = field(default_factory=list)

  def added(self) -> list[DiffEntry]:
    return [e for e in self.entries if e.status == 'added']

  def modified(self) -> list[DiffEntry]:
    return [e for e in self.entries if e.status == 'modified']

  def deleted(self) -> list[DiffEntry]:
    return [e for e in self.entries if e.status == 'deleted']

  def to_dict(self) -> dict[str, Any]:
    return {'entries': [e.to_dict() for e in self.entries]}

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'DiffResult':
    entries = [DiffEntry.from_dict(e) for e in data.get('entries', [])]
    return cls(entries=entries)


@dataclass
class MergeResult:
  """Result of a three-way merge."""

  merged: bool
  conflicts: list[str] = field(default_factory=list)
  merged_snapshot: SnapshotManifest | None = None

  def to_dict(self) -> dict[str, Any]:
    return {
      'merged': self.merged,
      'conflicts': self.conflicts,
      'merged_snapshot': self.merged_snapshot.to_dict() if self.merged_snapshot else None,
    }

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'MergeResult':
    snap = data.get('merged_snapshot')
    return cls(
      merged=data['merged'],
      conflicts=data.get('conflicts', []),
      merged_snapshot=SnapshotManifest.from_dict(snap) if snap else None,
    )


@dataclass
class StatusEntry:
  """Single file status relative to a snapshot."""

  path: str
  status: str

  def to_dict(self) -> dict[str, Any]:
    return {'path': self.path, 'status': self.status}

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'StatusEntry':
    return cls(**data)


@dataclass
class StatusResult:
  """Status of all tracked files relative to a snapshot."""

  entries: list[StatusEntry] = field(default_factory=list)

  def modified(self) -> list[StatusEntry]:
    return [e for e in self.entries if e.status == 'modified']

  def added(self) -> list[StatusEntry]:
    return [e for e in self.entries if e.status == 'added']

  def deleted(self) -> list[StatusEntry]:
    return [e for e in self.entries if e.status == 'deleted']

  def unchanged(self) -> list[StatusEntry]:
    return [e for e in self.entries if e.status == 'unchanged']

  def to_dict(self) -> dict[str, Any]:
    return {'entries': [e.to_dict() for e in self.entries]}

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'StatusResult':
    entries = [StatusEntry.from_dict(e) for e in data.get('entries', [])]
    return cls(entries=entries)


@dataclass
class SnapshotEntry:
  """Summary of a snapshot for log output."""

  epoch: int
  timestamp: str
  file_count: int

  def to_dict(self) -> dict[str, Any]:
    return {'epoch': self.epoch, 'timestamp': self.timestamp, 'file_count': self.file_count}

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'SnapshotEntry':
    return cls(**data)


class Store:
  """Abstract base class for content-addressed code versioning.

  Idempotent constructor: if slug exists, loads existing state.
  If slug is new, scans parameters and writes epoch_0 baseline.
  Like Experiment.__init__.

  Subclass and override all methods for custom backends.
  """

  def __init__(self, path: Path, slug: str, parameters: list[Parameter]) -> None:
    raise NotImplementedError

  @property
  def slug(self) -> str:
    raise NotImplementedError

  @property
  def epoch(self) -> int:
    raise NotImplementedError

  def snapshot(self, epoch: int) -> SnapshotManifest:
    """Capture current state of all parameter files. Like git commit."""
    raise NotImplementedError

  def checkout(self, epoch: int) -> None:
    """Restore all parameter files to a snapshot state. Like git checkout."""
    raise NotImplementedError

  def diff(self, epoch_a: int, slug_b: str, epoch_b: int) -> DiffResult:
    """Compare self.slug at epoch_a with slug_b at epoch_b. Like git diff."""
    raise NotImplementedError

  def branch(self, new_slug: str, from_epoch: int) -> None:
    """Fork this experiment's state into a new slug. Like git checkout -b."""
    raise NotImplementedError

  def merge(self, from_slug: str, from_epoch: int | None = None) -> MergeResult:
    """Three-way merge from another slug into self. Like git merge."""
    raise NotImplementedError

  def log(self) -> list[SnapshotEntry]:
    """History of all snapshots for this slug. Like git log."""
    raise NotImplementedError

  def status(self) -> StatusResult:
    """Compare current files on disk against latest snapshot. Like git status."""
    raise NotImplementedError

  def promote(self, epoch: int) -> None:
    """Permanently update the baseline to a given epoch. Like merging to main."""
    raise NotImplementedError
