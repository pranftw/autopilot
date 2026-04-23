"""Store base class and supporting data models for content-addressed code versioning.

Store is the VCS equivalent for experiment code. Experiment slugs are branches.
Epochs are commits. Subclass and override to customize storage, hashing, or
any operation.

Same extension pattern as Loss -> JudgeLoss, Optimizer -> AgentOptimizer,
Checkpoint -> JSONCheckpoint.
"""

from autopilot.core.parameter import Parameter
from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FileEntry(DictMixin):
  """Single file in a snapshot: content hash, size, and mtime."""

  hash: str
  size: int
  mtime: float


@dataclass
class SnapshotManifest(DictMixin):
  """Snapshot of all parameter files at a given epoch.

  entries maps 'param_name/state_key' to FileEntry.
  """

  epoch: int
  timestamp: str
  entries: dict[str, FileEntry] = field(default_factory=dict)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'SnapshotManifest':
    entries = {k: FileEntry.from_dict(v) for k, v in data.get('entries', {}).items()}
    return cls(epoch=data['epoch'], timestamp=data['timestamp'], entries=entries)


@dataclass
class DiffEntry(DictMixin):
  """Single file change between two snapshots."""

  path: str
  status: str
  old_hash: str | None = None
  new_hash: str | None = None
  text_diff: str | None = None


@dataclass
class DiffResult(DictMixin):
  """Diff between two snapshots: list of per-file changes."""

  entries: list[DiffEntry] = field(default_factory=list)

  def added(self) -> list[DiffEntry]:
    return [e for e in self.entries if e.status == 'added']

  def modified(self) -> list[DiffEntry]:
    return [e for e in self.entries if e.status == 'modified']

  def deleted(self) -> list[DiffEntry]:
    return [e for e in self.entries if e.status == 'deleted']

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'DiffResult':
    entries = [DiffEntry.from_dict(e) for e in data.get('entries', [])]
    return cls(entries=entries)


@dataclass
class MergeResult(DictMixin):
  """Result of a three-way merge."""

  merged: bool
  conflicts: list[str] = field(default_factory=list)
  merged_snapshot: SnapshotManifest | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'MergeResult':
    snap = data.get('merged_snapshot')
    return cls(
      merged=data['merged'],
      conflicts=data.get('conflicts', []),
      merged_snapshot=SnapshotManifest.from_dict(snap) if snap else None,
    )


@dataclass
class StatusEntry(DictMixin):
  """Single file status relative to a snapshot."""

  path: str
  status: str


@dataclass
class StatusResult(DictMixin):
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

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'StatusResult':
    entries = [StatusEntry.from_dict(e) for e in data.get('entries', [])]
    return cls(entries=entries)


@dataclass
class SnapshotEntry(DictMixin):
  """Summary of a snapshot for log output."""

  epoch: int
  timestamp: str
  file_count: int


class Store:
  """Abstract base class for content-addressed code versioning.

  Experiment slugs are branches. Epochs are commits. One instance per slug.

  Idempotent constructor: if slug exists, loads existing state.
  If slug is new, scans parameters and writes epoch_0 baseline.

  Operations (all git-analogous):
    snapshot(epoch)                        -- capture current state (git commit)
    checkout(epoch)                        -- restore to snapshot (git checkout)
    diff(epoch_a, slug_b, epoch_b)         -- compare snapshots (git diff)
    branch(new_slug, from_epoch)           -- fork into new slug (git checkout -b)
    merge(from_slug, from_epoch)           -- three-way merge (git merge)
    log()                                  -- snapshot history (git log)
    status()                               -- current vs latest snapshot (git status)
    promote(epoch)                         -- update baseline (merge to main)

  Sequential epoch invariant: snapshot epochs must be self.epoch + 1.
  Out-of-order snapshots raise StoreError.

  Parameter decoupling: Store interacts with parameters exclusively through
  param.snapshot() / param.restore(). It never imports concrete parameter types
  and never probes for domain-specific attributes.

  Built-in subclass: FileStore (ai/store.py) with SHA-256 content addressing.
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
