"""Snapshot data models for parameter file versioning.

FileEntry represents a single file in a store snapshot (content digest,
size, mtime, original path). SnapshotManifest groups multiple FileEntry
objects under a single epoch+timestamp. ParameterSchemaEntry and
ParameterSchema embed per-parameter metadata in manifests for named-key
stores.

These data classes are intentionally decoupled from Config and Store to
allow import from modules that Config depends on (e.g. config.stabilize
uses SnapshotManifest without creating a circular import through store.py).
"""

from autopilot.core.serialization import DictMixin
from dataclasses import dataclass, field
from dataclasses import fields as dc_fields
from typing import Any


@dataclass
class FileEntry(DictMixin):
  """Single file in a snapshot: content digest, size, mtime, and original path.

  Attributes:
    digest: Content-addressed hash of file bytes.
    size: File size in bytes.
    mtime: Source file modification time.
    original_path: Project-relative path of the original file on disk, or
      None. Required for stabilize to copy files back to the correct location.
  """

  digest: str
  size: int
  mtime: float
  original_path: str | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'FileEntry':
    """Build a ``FileEntry`` from a persistence dict.

    Args:
      data: Raw dict with ``digest``, ``size``, ``mtime``, and optional
        ``original_path`` keys.

    Returns:
      ``FileEntry`` instance.
    """
    names = {f.name for f in dc_fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class ParameterSchemaEntry(DictMixin):
  """Schema entry describing one parameter in a snapshot.

  Embedded in ParameterSchema to record type and provenance metadata
  for each named parameter at snapshot time.

  Attributes:
    name: Parameter registration key (module attribute name).
    type_name: Concrete class name (e.g. ``'PathParameter'``).
    source: Filesystem source path for PathParameter; None for other types.
    pattern: Glob pattern for PathParameter; None for other types.
  """

  name: str
  type_name: str
  source: str | None = None
  pattern: str | None = None


@dataclass
class ParameterSchema(DictMixin):
  """Schema metadata embedded in SnapshotManifest.

  Records per-parameter type and provenance at snapshot time so checkout
  can validate the registered parameters match the stored schema.

  Attributes:
    parameters: Ordered list of schema entries, one per registered parameter.
  """

  parameters: list[ParameterSchemaEntry] = field(default_factory=list)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'ParameterSchema':
    """Deserialize from dict with nested ParameterSchemaEntry parsing.

    Args:
      data: Raw dict with ``parameters`` key containing entry dicts.

    Returns:
      ParameterSchema with deserialized entries.
    """
    raw_params = data.get('parameters')
    params_list = [] if raw_params is None else raw_params
    entries = [ParameterSchemaEntry.from_dict(e) for e in params_list]
    return cls(parameters=entries)


@dataclass
class SnapshotManifest(DictMixin):
  """Snapshot of all parameter files at a given epoch.

  Attributes:
    epoch: Snapshot epoch index.
    timestamp: Snapshot creation time string.
    entries: Map from ``param_name/state_key`` to ``FileEntry``.
    schema: Embedded parameter schema, or None for legacy schema-less manifests.
    context: Optional human- or machine-readable reason/provenance string for
      audit traceability.
  """

  epoch: int
  timestamp: str
  entries: dict[str, FileEntry] = field(default_factory=dict)
  schema: ParameterSchema | None = None
  context: str | None = None

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> 'SnapshotManifest':
    """Deserialize from dict, handling null -> empty container coercion for collection fields.

    Returns:
      Manifest with parsed ``FileEntry`` values under ``entries`` and
      optional ``ParameterSchema``.
    """
    raw_entries = data.get('entries')
    entry_map = {} if raw_entries is None else raw_entries
    entries = {k: FileEntry.from_dict(v) for k, v in entry_map.items()}
    raw_schema = data.get('schema')
    schema = ParameterSchema.from_dict(raw_schema) if raw_schema is not None else None
    return cls(
      epoch=data['epoch'],
      timestamp=data['timestamp'],
      entries=entries,
      schema=schema,
      context=data.get('context'),
    )
