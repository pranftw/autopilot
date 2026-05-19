"""Experiment-level key-value configuration metadata.

``MetadataArtifact`` stores durable key-value pairs for experiments,
distinct from ``dataset_meta`` (which tracks data lineage). Use cases:
human labels, run tags, external IDs, environment markers.

On-disk layout: ``{experiment_path}/metadata.json`` -- a single JSON object.
Missing or empty file semantics: ``show()`` returns ``{}``, ``get()`` returns
``None``. Empty keys are rejected with ``ValueError``.
"""

from autopilot.core.artifacts.artifact import JSONArtifact
from autopilot.core.errors import TrackingError
from pathlib import Path
from typing import Any

METADATA_FILENAME = 'metadata.json'


class MetadataArtifact(JSONArtifact):
  """Experiment-level key-value configuration metadata.

  Extends ``JSONArtifact`` for atomic reads/writes of a flat JSON dict.
  Path resolution deferred to callers who supply the experiment base_dir.

  Single-writer: ``set()`` reads, modifies, and writes without file locking.
  CLI dispatch serializes commands, but concurrent programmatic access
  on the same file is not safe.

  Storage: ``{experiment_path}/metadata.json``.

  Examples:
    >>> artifact = MetadataArtifact()
    >>> artifact.set('env', 'staging', base_dir=exp_dir)
    >>> artifact.get('env', base_dir=exp_dir)
    'staging'
    >>> artifact.show(base_dir=exp_dir)
    {'env': 'staging'}
  """

  def __init__(self) -> None:
    """Initialize with fixed filename ``metadata.json``."""
    super().__init__(METADATA_FILENAME)

  def set(self, key: str, value: Any, base_dir: Path) -> None:
    """Set a metadata key-value pair and persist atomically.

    Performs a read-modify-write cycle without file locking. Safe for
    single-writer use (e.g. CLI dispatch, which serializes commands).
    Concurrent writers on the same file may lose updates.

    Args:
      key: Non-empty string key.
      value: Any JSON-serializable value.
      base_dir: Experiment directory for path resolution.

    Raises:
      ValueError: When ``key`` is empty.
    """
    if not key:
      msg = 'Metadata key must not be empty. Provide a descriptive key name.'
      raise ValueError(msg)
    data = self._load_dict(base_dir)
    data[key] = value
    self.write(data, base_dir)

  def get(self, key: str, base_dir: Path) -> Any | None:
    """Return value for *key*, or None if missing.

    Args:
      key: Metadata key to look up.
      base_dir: Experiment directory for path resolution.

    Returns:
      Stored value, or ``None`` when the key is absent.
    """
    return self._load_dict(base_dir).get(key)

  def show(self, base_dir: Path) -> dict[str, Any]:
    """Return all metadata as a shallow-copy dict.

    Callers may mutate the returned dict without affecting stored state.

    Args:
      base_dir: Experiment directory for path resolution.

    Returns:
      Shallow copy of the metadata dict (empty dict when file missing).
    """
    return dict(self._load_dict(base_dir))

  def _load_dict(self, base_dir: Path) -> dict[str, Any]:
    """Load the metadata JSON as a dict, defaulting to empty.

    Returns empty dict when the file is missing, contains non-dict JSON,
    or contains unparseable content. Corrupt files degrade gracefully
    rather than crashing -- the same behavior as a missing file.

    Degradation table:
      - Missing file: ``{}`` (``OSError``)
      - Invalid JSON: ``{}`` (``TrackingError`` from ``read_json``)
      - Valid JSON but non-dict (e.g. ``null``, ``"string"``): ``{}``
      - Empty file: ``{}`` (``TrackingError``)
      - Valid ``{}`` JSON: ``{}``

    Args:
      base_dir: Experiment directory for path resolution.

    Returns:
      Dict from disk, or ``{}`` when file is missing, empty, or corrupt.
    """
    try:
      raw = self.read_raw(base_dir)
    except (TrackingError, OSError):
      raw = None
    if not isinstance(raw, dict):
      return {}
    return raw
