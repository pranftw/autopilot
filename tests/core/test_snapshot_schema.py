"""Tests for ParameterSchema, ParameterSchemaEntry, and SnapshotManifest schema field."""

from autopilot.core.parameter import Parameter
from autopilot.core.snapshot import (
  FileEntry,
  ParameterSchema,
  ParameterSchemaEntry,
  SnapshotManifest,
)


class TestParameterSchemaEntry:
  """Round-trip and default tests for ParameterSchemaEntry."""

  def test_parameter_schema_entry_round_trip(self) -> None:
    entry = ParameterSchemaEntry(
      name='prompts',
      type_name='PathParameter',
      source='/tmp/prompts',
      pattern='**/*.md',
    )
    data = entry.to_dict()
    restored = ParameterSchemaEntry.from_dict(data)
    assert restored.name == entry.name
    assert restored.type_name == entry.type_name
    assert restored.source == entry.source
    assert restored.pattern == entry.pattern

  def test_parameter_schema_entry_defaults(self) -> None:
    entry = ParameterSchemaEntry(name='p', type_name='Parameter')
    assert entry.source is None
    assert entry.pattern is None
    data = entry.to_dict()
    assert data['source'] is None
    assert data['pattern'] is None


class TestParameterSchema:
  """Round-trip tests for ParameterSchema."""

  def test_parameter_schema_round_trip(self) -> None:
    entries = [
      ParameterSchemaEntry(name='prompts', type_name='PathParameter', source='/p', pattern='*'),
      ParameterSchemaEntry(name='config', type_name='Parameter'),
    ]
    schema = ParameterSchema(parameters=entries)
    data = schema.to_dict()
    restored = ParameterSchema.from_dict(data)
    assert len(restored.parameters) == 2
    assert restored.parameters[0].name == 'prompts'
    assert restored.parameters[0].source == '/p'
    assert restored.parameters[1].name == 'config'
    assert restored.parameters[1].source is None

  def test_parameter_schema_empty(self) -> None:
    schema = ParameterSchema()
    data = schema.to_dict()
    restored = ParameterSchema.from_dict(data)
    assert restored.parameters == []


class TestSnapshotManifestWithSchema:
  """SnapshotManifest with and without schema."""

  def test_snapshot_manifest_with_schema(self) -> None:
    schema = ParameterSchema(parameters=[ParameterSchemaEntry(name='p', type_name='PathParameter')])
    manifest = SnapshotManifest(
      epoch=1,
      timestamp='2026-01-01T00:00:00Z',
      entries={'p/file.txt': FileEntry(digest='abc', size=3, mtime=0.0)},
      schema=schema,
    )
    data = manifest.to_dict()
    assert 'schema' in data
    assert data['schema']['parameters'][0]['name'] == 'p'

    restored = SnapshotManifest.from_dict(data)
    assert restored.schema is not None
    assert len(restored.schema.parameters) == 1
    assert restored.schema.parameters[0].type_name == 'PathParameter'

  def test_snapshot_manifest_without_schema(self) -> None:
    manifest = SnapshotManifest(epoch=0, timestamp='2026-01-01T00:00:00Z')
    assert manifest.schema is None
    data = manifest.to_dict()
    restored = SnapshotManifest.from_dict(data)
    assert restored.schema is None

  def test_snapshot_manifest_schema_none_in_dict(self) -> None:
    data = {'epoch': 2, 'timestamp': 't', 'entries': {}, 'schema': None}
    restored = SnapshotManifest.from_dict(data)
    assert restored.schema is None


class TestParameterBaseSchemaEntry:
  """Parameter.schema_entry() base behavior."""

  def test_parameter_base_schema_entry(self) -> None:
    param = Parameter()
    entry = param.schema_entry()
    assert entry.type_name == 'Parameter'
    assert not entry.name
    assert entry.source is None
    assert entry.pattern is None
