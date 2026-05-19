"""Tests for SnapshotManifest.context field and Store.snapshot context threading."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.snapshot import FileEntry, ParameterSchema, SnapshotManifest
from pathlib import Path


def test_snapshot_manifest_context_field_default_none() -> None:
  """Manifest without context has context is None."""
  manifest = SnapshotManifest(epoch=0, timestamp='2025-01-01T00:00:00Z')
  assert manifest.context is None


def test_snapshot_manifest_context_field_set() -> None:
  """SnapshotManifest(..., context='reason') stores the value."""
  manifest = SnapshotManifest(
    epoch=5,
    timestamp='2025-06-01T12:00:00Z',
    entries={'k': FileEntry(digest='abc', size=3, mtime=0.0)},
    schema=ParameterSchema(parameters=[]),
    context='optimizer applied gradient',
  )
  assert manifest.context == 'optimizer applied gradient'


def test_snapshot_manifest_to_dict_includes_context() -> None:
  """to_dict() includes 'context' key."""
  manifest = SnapshotManifest(epoch=0, timestamp='t', context='why')
  d = manifest.to_dict()
  assert 'context' in d
  assert d['context'] == 'why'

  manifest_none = SnapshotManifest(epoch=0, timestamp='t')
  d_none = manifest_none.to_dict()
  assert 'context' in d_none
  assert d_none['context'] is None


def test_snapshot_manifest_from_dict_with_context() -> None:
  """from_dict restores string context."""
  data = {
    'epoch': 2,
    'timestamp': '2025-01-01T00:00:00Z',
    'entries': {},
    'context': 'policy gate accepted',
  }
  manifest = SnapshotManifest.from_dict(data)
  assert manifest.context == 'policy gate accepted'

  data_none = {
    'epoch': 0,
    'timestamp': 't',
    'entries': {},
    'context': None,
  }
  manifest_none = SnapshotManifest.from_dict(data_none)
  assert manifest_none.context is None


def test_snapshot_manifest_roundtrip_with_context() -> None:
  """Full to_dict / from_dict with non-None context."""
  original = SnapshotManifest(
    epoch=3,
    timestamp='2025-03-15T10:30:00Z',
    entries={'p/file.txt': FileEntry(digest='deadbeef', size=10, mtime=1.0)},
    schema=ParameterSchema(parameters=[]),
    context='epoch 3 accepted by policy',
  )
  restored = SnapshotManifest.from_dict(original.to_dict())
  assert restored.context == original.context
  assert restored.epoch == original.epoch
  assert restored.timestamp == original.timestamp


def test_snapshot_manifest_roundtrip_without_context() -> None:
  """Omit key in input dict; round-trip yields None."""
  data = {
    'epoch': 0,
    'timestamp': 't',
    'entries': {},
  }
  manifest = SnapshotManifest.from_dict(data)
  assert manifest.context is None
  roundtripped = SnapshotManifest.from_dict(manifest.to_dict())
  assert roundtripped.context is None


def _make_config(tmp_path: Path) -> AutoPilotConfig:
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  return config


def _make_source(tmp_path: Path) -> Path:
  src = tmp_path / 'src'
  src.mkdir(parents=True, exist_ok=True)
  (src / 'prompt.txt').write_text('hello world')
  return src


def test_store_snapshot_accepts_context(tmp_path: Path) -> None:
  """FileStore fixture with registered params; snapshot returns manifest with context."""
  src = _make_source(tmp_path)
  config = _make_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'prompts': param})

  manifest = store.snapshot('exp-001', 0, context='initial epoch reason')
  assert manifest.context == 'initial epoch reason'

  loaded = store.load_snapshot('exp-001', 0)
  assert loaded.context == 'initial epoch reason'


def test_store_snapshot_without_context(tmp_path: Path) -> None:
  """Snapshot without context returns manifest with None."""
  src = _make_source(tmp_path)
  config = _make_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'prompts': param})

  manifest = store.snapshot('exp-001', 0)
  assert manifest.context is None

  loaded = store.load_snapshot('exp-001', 0)
  assert loaded.context is None


def test_store_branch_preserves_parent_context(tmp_path: Path) -> None:
  """Branch copies snap.context from the parent snapshot."""
  src = _make_source(tmp_path)
  config = _make_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'prompts': param})

  store.snapshot('exp-001', 0, context='parent context')
  store.branch('exp-002')
  loaded = store.load_snapshot('exp-002', 0)
  assert loaded.context == 'parent context'


def test_store_materialize_preserves_context(tmp_path: Path) -> None:
  """Materialize preserves the target epoch's context."""
  src = _make_source(tmp_path)
  config = _make_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'prompts': param})

  store.snapshot('exp-001', 0, context='epoch zero context')
  store.snapshot('exp-001', 1, context='epoch one context')
  store.materialize('exp-001', 0)
  loaded = store.load_snapshot('exp-001', 0)
  assert loaded.context == 'epoch zero context'
