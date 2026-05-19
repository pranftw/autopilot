"""Experiment artifact system: typed, self-describing file artifacts.

Three layers:
  Artifact -- base protocol (structure, operations, resolution)
  JSONArtifact / JSONLArtifact / TextArtifact -- file-format layer
  Domain artifacts -- structure and typed I/O (in sibling modules)

Path resolution: callers pass a base_dir obtained from Config or
Store (e.g. store.resolve_path(experiment_id)). Artifacts never
resolve paths themselves -- they only join base_dir with filename.
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
    """Bind this artifact to a relative filename and storage scope.

    Args:
      filename: Relative path/filename under the experiment directory.
      scope: ``'experiment'`` for a single file per run, or ``'epoch'`` for
        per-epoch subdirectories.
    """
    self._filename = filename
    self._scope = scope

  @property
  def filename(self) -> str:
    """Return the artifact's relative filename."""
    return self._filename

  @property
  def scope(self) -> str:
    """Return the storage scope (``'experiment'`` or ``'epoch'``)."""
    return self._scope

  def schema(self) -> dict | None:
    """Describe the expected data structure for documentation and tooling.

    Returns:
      A schema description dict, or ``None`` when unspecified.
    """
    return None

  def validate(self, data: Any) -> None:
    """Validate data before write/update/append. Raise on invalid."""

  def serialize(self, data: Any) -> Any:
    """Convert typed domain data to file-ready format.

    Args:
      data: Domain object or structure to serialize.

    Returns:
      Payload suitable for the underlying file format.
    """
    return data

  def deserialize(self, raw: Any) -> Any:
    """Convert file-ready format back to typed domain data.

    Args:
      raw: Decoded structure from disk.

    Returns:
      Domain-level value produced from ``raw``.
    """
    return raw

  def write(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Write data to artifact file. Full replace.

    Raises:
      NotImplementedError: Subclasses must implement this method.
    """
    raise NotImplementedError

  def update(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Partial update: merge data into existing artifact.

    Raises:
      NotImplementedError: Subclasses must implement this method.
    """
    raise NotImplementedError

  def append(self, record: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Append a record to the artifact.

    Raises:
      NotImplementedError: Subclasses must implement this method.
    """
    raise NotImplementedError

  def read(self, base_dir: Path, epoch: int | None = None) -> Any:
    """Read artifact and return typed data via deserialize().

    Raises:
      NotImplementedError: Subclasses must implement this method.
    """
    raise NotImplementedError

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> Any:
    """Read artifact without deserialization.

    Raises:
      NotImplementedError: Subclasses must implement this method.
    """
    raise NotImplementedError

  def resolve_path(self, base_dir: Path, epoch: int | None = None) -> Path:
    """Resolve the absolute path for this artifact under ``base_dir``.

    Args:
      base_dir: Experiment or store root directory.
      epoch: Required when ``scope`` is ``'epoch'``; selects ``epoch_N`` dir.

    Returns:
      Full path to the artifact file.

    Raises:
      ValueError: When ``scope`` is ``'epoch'`` but ``epoch`` is ``None``.
    """
    if self._scope == 'epoch':
      if epoch is None:
        msg = f'epoch required for epoch-scoped artifact {self._filename!r}'
        raise ValueError(msg)
      return base_dir / f'epoch_{epoch}' / self._filename
    return base_dir / self._filename

  def exists(self, base_dir: Path, epoch: int | None = None) -> bool:
    """Return whether the artifact file exists on disk."""
    return self.resolve_path(base_dir, epoch).exists()

  def clear(self, base_dir: Path, epoch: int | None = None) -> None:
    """Remove the artifact file if it exists."""
    path = self.resolve_path(base_dir, epoch)
    if path.exists():
      path.unlink()

  def __repr__(self) -> str:
    """Return a debug string including filename and scope."""
    return f'{type(self).__name__}({self._filename!r}, scope={self._scope!r})'


class JSONArtifact(Artifact):
  """JSON file format. Atomic write, dict-based merge for update()."""

  def merge(self, existing: dict, new: dict) -> dict:
    """Merge strategy for update(). Default: shallow merge.

    Args:
      existing: Current on-disk dict after ``read_raw``.
      new: Serialized payload for the incoming update.

    Returns:
      Dict to write atomically (default: shallow merge, ``new`` wins keys).
    """
    return {**existing, **new}

  def write(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Validate, serialize, and atomically write JSON to disk.

    Returns:
      Path to the written artifact file.
    """
    self.validate(data)
    serialized = self.serialize(data)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, serialized)
    return path

  def update(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Merge serialized data into existing JSON and write atomically.

    Returns:
      Path to the written artifact file.
    """
    self.validate(data)
    existing_raw = self.read_raw(base_dir, epoch)
    existing: dict = {} if not isinstance(existing_raw, dict) else existing_raw
    merged = self.merge(existing, self.serialize(data))
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, merged)
    return path

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> dict | list[dict] | None:
    """Load JSON from disk without applying ``deserialize``.

    Returns:
      Parsed JSON (dict, list of dicts), or ``None`` if missing/empty per I/O.
    """
    return read_json(self.resolve_path(base_dir, epoch))

  def read(self, base_dir: Path, epoch: int | None = None) -> Any:
    """Read JSON and return typed data via ``deserialize``.

    Returns:
      ``None`` when no raw payload exists; otherwise deserialized value.
    """
    raw = self.read_raw(base_dir, epoch)
    if raw is None:
      return None
    return self.deserialize(raw)


class JSONLArtifact(Artifact):
  """JSONL file format. Append-only, record-at-a-time."""

  def append(self, record: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Validate, serialize, and append one JSONL record.

    Returns:
      Path to the JSONL file after append.
    """
    self.validate(record)
    serialized = self.serialize(record)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(path, serialized)
    return path

  def write(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Full replace: truncate and write all records as JSONL.

    Returns:
      Path to the JSONL file after writing all records.
    """
    records = data
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
    """Load all JSONL lines as dicts.

    Returns:
      List of row dicts, or an empty list when the file is missing.
    """
    path = self.resolve_path(base_dir, epoch)
    if not path.exists():
      return []
    return read_jsonl(path)

  def read(self, base_dir: Path, epoch: int | None = None) -> list:
    """Read all lines and deserialize each record.

    Returns:
      List of deserialized domain values.
    """
    return [self.deserialize(r) for r in self.read_raw(base_dir, epoch)]


class TextArtifact(Artifact):
  """Text/markdown file format."""

  def write(self, data: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Validate, serialize to text, and write the file (full replace).

    Returns:
      Path to the written text file.
    """
    self.validate(data)
    text = self.serialize(data)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path

  def append(self, record: Any, base_dir: Path, epoch: int | None = None) -> Path:
    """Append serialized text to the file (open in append mode).

    Returns:
      Path to the text file after append.
    """
    self.validate(record)
    text = self.serialize(record)
    path = self.resolve_path(base_dir, epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a', encoding='utf-8') as f:
      f.write(text)
    return path

  def read_raw(self, base_dir: Path, epoch: int | None = None) -> str | None:
    """Read file contents as UTF-8 text without ``deserialize``.

    Returns:
      File text, or ``None`` if the file does not exist.
    """
    path = self.resolve_path(base_dir, epoch)
    if not path.exists():
      return None
    return path.read_text(encoding='utf-8')

  def read(self, base_dir: Path, epoch: int | None = None) -> Any:
    """Read text and return typed data via ``deserialize``.

    Returns:
      ``None`` when the file is missing; otherwise deserialized value.
    """
    raw = self.read_raw(base_dir, epoch)
    if raw is None:
      return None
    return self.deserialize(raw)
