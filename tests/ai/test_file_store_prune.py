"""Tests for Store.prune_orphans and FileStore.prune_orphans.

Covers:
  - Base Store.prune_orphans is a no-op
  - FileStore removes unreachable blobs
  - FileStore keeps referenced blobs
  - FileStore aborts on corrupt manifests (fail-closed, BUG-002)
  - store_blob is declared as abstract on Store base (BUG-009)
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from autopilot.core.store.base import Store
from pathlib import Path
import json
import pytest


class _StubStore(Store):
  """Minimal concrete Store to test base prune_orphans."""

  def __init__(self) -> None:
    pass


class TestStorePruneOrphansBaseNoOp:
  def test_base_returns_empty(self) -> None:
    """Base Store.prune_orphans returns empty list (no-op)."""
    store = _StubStore()
    result = store.prune_orphans()
    assert result == []


class TestFileStorePruneOrphansRemovesUnreachable:
  def test_removes_unreachable_blob(self, tmp_path: Path) -> None:
    """Orphan blob is removed while referenced blob is kept."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('hello world')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    manifest = store.snapshot('exp-001', 0)
    referenced_digest = next(iter(manifest.entries.values())).digest

    orphan_digest = 'ff' + 'a1' * 31
    orphan_dir = config.objects_path / orphan_digest[:2]
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / orphan_digest[2:]).write_text('orphan content')

    removed = store.prune_orphans()
    assert orphan_digest in removed

    ref_path = config.objects_path / referenced_digest[:2] / referenced_digest[2:]
    assert ref_path.exists()

  def test_empty_store_returns_empty(self, tmp_path: Path) -> None:
    """Empty store with no objects returns empty."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})
    removed = store.prune_orphans()
    assert removed == []


class TestFileStorePruneOrphansCorruptManifest:
  """BUG-002: prune_orphans must fail closed on corrupt manifests."""

  def test_aborts_on_corrupt_manifest(self, tmp_path: Path) -> None:
    """Corrupt manifest causes StoreError; referenced blob stays on disk."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('hello world')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    manifest = store.snapshot('exp-001', 0)
    referenced_digest = next(iter(manifest.entries.values())).digest

    snapshots_dir = config.store_path / 'snapshots' / 'exp-001'
    snap_files = list(snapshots_dir.glob('*.json'))
    assert len(snap_files) == 1
    snap_files[0].write_text(json.dumps({}))

    with pytest.raises(StoreError, match='corrupt'):
      store.prune_orphans()

    ref_path = config.objects_path / referenced_digest[:2] / referenced_digest[2:]
    assert ref_path.exists()

  def test_aborts_on_invalid_json_manifest(self, tmp_path: Path) -> None:
    """Invalid JSON in manifest causes StoreError."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('hello world')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    store.snapshot('exp-001', 0)

    snapshots_dir = config.store_path / 'snapshots' / 'exp-001'
    snap_files = list(snapshots_dir.glob('*.json'))
    assert len(snap_files) == 1
    snap_files[0].write_text('not valid json at all')

    with pytest.raises(StoreError, match='corrupt'):
      store.prune_orphans()

  def test_succeeds_when_all_manifests_valid(self, tmp_path: Path) -> None:
    """Happy path: valid manifests allow pruning; orphan removed, referenced kept."""
    prompts_dir = tmp_path / 'prompts'
    prompts_dir.mkdir()
    (prompts_dir / 'main.txt').write_text('hello world')

    config = AutoPilotConfig(workspace=tmp_path)
    param = PathParameter(source=str(prompts_dir), pattern='*.txt')
    store = FileStore(config)
    store.register_parameters({'source': param})

    manifest = store.snapshot('exp-001', 0)
    referenced_digest = next(iter(manifest.entries.values())).digest

    orphan_digest = 'ff' + 'a1' * 31
    orphan_dir = config.objects_path / orphan_digest[:2]
    orphan_dir.mkdir(parents=True, exist_ok=True)
    (orphan_dir / orphan_digest[2:]).write_text('orphan content')

    removed = store.prune_orphans()
    assert orphan_digest in removed
    ref_path = config.objects_path / referenced_digest[:2] / referenced_digest[2:]
    assert ref_path.exists()


class TestStoreBlobAbstract:
  """BUG-009: store_blob is declared as abstract on Store base."""

  def test_store_blob_is_abstract_method(self) -> None:
    """Store.store_blob has __isabstractmethod__ flag set."""
    assert hasattr(Store.store_blob, '__isabstractmethod__')
    assert Store.store_blob.__isabstractmethod__ is True

  def test_file_store_store_blob_writes_object(self, tmp_path: Path) -> None:
    """FileStore.store_blob writes data retrievable via read_object."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    store.register_parameters({})

    data = b'test blob content'
    from autopilot.ai.store_lock import hash_bytes

    digest = hash_bytes(data)

    store.store_blob(digest, data)
    result = store.read_object(digest)
    assert result == data
