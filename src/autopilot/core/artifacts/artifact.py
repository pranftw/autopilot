"""Experiment artifact system: typed, self-describing file artifacts.

Three layers:
  Artifact -- base protocol (structure, operations, resolution)
  JSONArtifact / JSONLArtifact / TextArtifact -- file-format layer
  Domain artifacts -- structure and typed I/O (in sibling modules)
"""

from autopilot.tracking.io import append_jsonl, atomic_write_json, read_json, read_jsonl
from pathlib import Path
from typing import Any


class Artifact:
  """Base experiment artifact. Owns its file, structure, and all I/O.

  Like Parameter on Module: assigned as attributes on any ArtifactOwner,
  auto-registered via ArtifactOwner.__setattr__ into owner._artifacts.
  """

  def __init__(self, filename: str, scope: str = 'experiment') -> None:
    self._filename = filename
    self._scope = scope

  @property
  def filename(self) -> str:
    return self._filename

  @property
  def scope(self) -> str:
    return self._scope

  def schema(self) -> dict | None:
    """Describe the expected data structure."""
    return None

  def validate(self, data: Any) -> None:
    """Validate data before write/update/append. Raise on invalid."""

  def serialize(self, data: Any) -> Any:
    """Convert typed domain data to file-ready format."""
    return data

  def deserialize(self, raw: Any) -> Any:
    """Convert file-ready format back to typed domain data."""
    return raw

  def write(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Write data to artifact file. Full replace."""
    raise NotImplementedError

  def update(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Partial update: merge data into existing artifact."""
    raise NotImplementedError

  def append(self, record: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Append a record to the artifact."""
    raise NotImplementedError

  def read(self, base_dir: Path, epoch: int | None = None) -> Any:
    """Read artifact and return typed data via deserialize()."""
    raise NotImplementedError

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> Any:
    """Read artifact without deserialization."""
    raise NotImplementedError

  def resolve_path(self, base_dir: Path, epoch: int | None = None) -> Path:
    """Resolve the full file path."""
    if self._scope == 'epoch':
      if epoch is None:
        raise ValueError(f'epoch required for epoch-scoped artifact {self._filename!r}')
      return base_dir / f'epoch_{epoch}' / self._filename
    return base_dir / self._filename

  def exists(self, base_dir: Path, epoch: int | None = None) -> bool:
    return self.resolve_path(base_dir, epoch).exists()

  def clear(self, base_dir: Path, epoch: int | None = None) -> None:
    path = self.resolve_path(base_dir, epoch)
    if path.exists():
      path.unlink()

  def __repr__(self) -> str:
    return f'{type(self).__name__}({self._filename!r}, scope={self._scope!r})'


class JSONArtifact(Artifact):
  """JSON file format. Atomic write, dict-based merge for update()."""

  def merge(self, existing: dict, new: dict) -> dict:
    """Merge strategy for update(). Default: shallow merge."""
    return {**existing, **new}

  def write(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    self.validate(data)
    serialized = self.serialize(data)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, serialized)
    return path

  def update(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    self.validate(data)
    existing = self.read_raw(base_dir, epoch) or {}
    merged = self.merge(existing, self.serialize(data))
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, merged)
    return path

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> dict | None:
    return read_json(self.resolve_path(base_dir, epoch))

  def read(self, base_dir: Path, epoch: int | None = None) -> Any:
    raw = self.read_raw(base_dir, epoch)
    if raw is None:
      return None
    return self.deserialize(raw)


class JSONLArtifact(Artifact):
  """JSONL file format. Append-only, record-at-a-time."""

  def append(self, record: Any, base_dir: Path, epoch: int | None = None) -> Path:
    self.validate(record)
    serialized = self.serialize(record)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path, serialized)
    return path

  def write(self, records: list, base_dir: Path, epoch: int | None = None) -> Path:
    """Full replace: truncate and write all records as JSONL."""
    for r in records:
      self.validate(r)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
      path.unlink()
    for r in records:
      append_jsonl(path, self.serialize(r))
    return path

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> list[dict]:
    path = self.resolve_path(base_dir, epoch)
    if not path.exists():
      return []
    return read_jsonl(path)

  def read(self, base_dir: Path, epoch: int | None = None) -> list:
    return [self.deserialize(r) for r in self.read_raw(base_dir, epoch)]


class TextArtifact(Artifact):
  """Text/markdown file format."""

  def write(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    self.validate(data)
    text = self.serialize(data)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path

  def append(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    self.validate(data)
    text = self.serialize(data)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
      f.write(text)
    return path

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> str | None:
    path = self.resolve_path(base_dir, epoch)
    if not path.exists():
      return None
    return path.read_text(encoding='utf-8')

  def read(self, base_dir: Path, epoch: int | None = None) -> Any:
    raw = self.read_raw(base_dir, epoch)
    if raw is None:
      return None
    return self.deserialize(raw)
