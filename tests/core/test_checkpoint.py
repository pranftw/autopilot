"""Tests for Checkpoint base class and JSONCheckpoint."""

from autopilot.core.checkpoint import Checkpoint, JSONCheckpoint
from autopilot.core.models import Manifest
from pathlib import Path
import pytest


class TestCheckpointBase:
  def test_save_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Checkpoint().save_manifest(Path('/tmp'), Manifest(slug='x'))

  def test_load_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Checkpoint().load_manifest(Path('/tmp'))

  def test_exists_raises(self) -> None:
    with pytest.raises(NotImplementedError):
      Checkpoint().exists(Path('/tmp'))


class TestJSONCheckpoint:
  def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
    cp = JSONCheckpoint()
    manifest = Manifest(slug='test-1', title='Test', idea='try it')
    cp.save_manifest(tmp_path, manifest)
    loaded = cp.load_manifest(tmp_path)
    assert loaded is not None
    assert loaded.slug == 'test-1'
    assert loaded.idea == 'try it'

  def test_load_returns_none_when_missing(self, tmp_path: Path) -> None:
    cp = JSONCheckpoint()
    assert cp.load_manifest(tmp_path) is None

  def test_exists_true(self, tmp_path: Path) -> None:
    cp = JSONCheckpoint()
    manifest = Manifest(slug='test-1')
    cp.save_manifest(tmp_path, manifest)
    assert cp.exists(tmp_path) is True

  def test_exists_false(self, tmp_path: Path) -> None:
    cp = JSONCheckpoint()
    assert cp.exists(tmp_path) is False

  def test_metadata_preserved(self, tmp_path: Path) -> None:
    cp = JSONCheckpoint()
    manifest = Manifest(
      slug='test-1',
      metadata={'profile': 'default', 'environment': 'staging'},
    )
    cp.save_manifest(tmp_path, manifest)
    loaded = cp.load_manifest(tmp_path)
    assert loaded.metadata['profile'] == 'default'
    assert loaded.metadata['environment'] == 'staging'


class TestCustomCheckpoint:
  def test_subclass_works(self) -> None:
    class MemCheckpoint(Checkpoint):
      def __init__(self):
        self.store: dict[str, Manifest] = {}

      def save_manifest(self, experiment_dir, manifest):
        self.store[str(experiment_dir)] = manifest

      def load_manifest(self, experiment_dir):
        return self.store.get(str(experiment_dir))

      def exists(self, experiment_dir):
        return str(experiment_dir) in self.store

    cp = MemCheckpoint()
    m = Manifest(slug='x')
    cp.save_manifest(Path('/tmp/a'), m)
    assert cp.exists(Path('/tmp/a'))
    assert cp.load_manifest(Path('/tmp/a')).slug == 'x'
    assert not cp.exists(Path('/tmp/b'))
