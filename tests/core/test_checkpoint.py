"""Tests for CheckpointIO ABC and JSONCheckpointIO."""

from autopilot.core.checkpoint import CheckpointIO, JSONCheckpointIO
from autopilot.core.errors import TrackingError
from pathlib import Path
import autopilot.core.checkpoint as _ckpt_mod
import pytest


class TestCheckpointIOBase:
  """CheckpointIO ABC raises NotImplementedError for all methods."""

  def test_save_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError):
      CheckpointIO().save({'k': 'v'}, Path('/tmp/x.json'))

  def test_load_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError):
      CheckpointIO().load(Path('/tmp/x.json'))

  def test_remove_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError):
      CheckpointIO().remove(Path('/tmp/x.json'))

  def test_exists_not_implemented(self) -> None:
    with pytest.raises(NotImplementedError):
      CheckpointIO().exists(Path('/tmp/x.json'))


class TestJSONCheckpointIORoundtrip:
  """Save -> load -> assert equality."""

  def test_simple_dict(self, tmp_path: Path) -> None:
    io = JSONCheckpointIO()
    state = {'lr': 0.01, 'epoch': 3}
    path = tmp_path / 'ckpt.json'
    io.save(state, path)
    loaded = io.load(path)
    assert loaded == state

  def test_nested_dict(self, tmp_path: Path) -> None:
    io = JSONCheckpointIO()
    state = {
      'experiment': {'id': 'e1', 'status': 'running', 'metrics': {'acc': 0.95}},
      'module': {'params': [1, 2, 3]},
      'optimizer': {'lr': 0.1, 'blocked_strategies': ['a', 'b']},
    }
    path = tmp_path / 'ckpt.json'
    io.save(state, path)
    loaded = io.load(path)
    assert loaded == state

  def test_empty_dict(self, tmp_path: Path) -> None:
    io = JSONCheckpointIO()
    path = tmp_path / 'ckpt.json'
    io.save({}, path)
    loaded = io.load(path)
    assert loaded == {}


class TestJSONCheckpointIOLoadErrors:
  """Load raises TrackingError for bad files."""

  def test_load_missing_file(self, tmp_path: Path) -> None:
    io = JSONCheckpointIO()
    path = tmp_path / 'missing.json'
    with pytest.raises(TrackingError):
      io.load(path)

  def test_load_not_dict_json(self, tmp_path: Path) -> None:
    path = tmp_path / 'array.json'
    path.write_text('[1, 2, 3]')
    io = JSONCheckpointIO()
    with pytest.raises(TrackingError, match='checkpoint'):
      io.load(path)

  def test_load_corrupt_json(self, tmp_path: Path) -> None:
    path = tmp_path / 'corrupt.json'
    path.write_text('{not json')
    io = JSONCheckpointIO()
    with pytest.raises(TrackingError):
      io.load(path)

  def test_load_empty_file(self, tmp_path: Path) -> None:
    path = tmp_path / 'empty.json'
    path.write_text('')
    io = JSONCheckpointIO()
    with pytest.raises(TrackingError):
      io.load(path)


class TestJSONCheckpointIOExistsAndRemove:
  """exists/remove lifecycle."""

  def test_exists_false_before_save(self, tmp_path: Path) -> None:
    io = JSONCheckpointIO()
    path = tmp_path / 'ckpt.json'
    assert io.exists(path) is False

  def test_exists_true_after_save(self, tmp_path: Path) -> None:
    io = JSONCheckpointIO()
    path = tmp_path / 'ckpt.json'
    io.save({'a': 1}, path)
    assert io.exists(path) is True

  def test_remove_deletes_file(self, tmp_path: Path) -> None:
    io = JSONCheckpointIO()
    path = tmp_path / 'ckpt.json'
    io.save({'a': 1}, path)
    assert path.is_file()
    io.remove(path)
    assert not path.exists()

  def test_remove_missing_is_noop(self, tmp_path: Path) -> None:
    io = JSONCheckpointIO()
    path = tmp_path / 'nonexistent.json'
    io.remove(path)


class TestJSONCheckpointIOBinaryFile:
  """Binary/non-UTF-8 checkpoint file surfaces TrackingError."""

  def test_json_checkpoint_io_binary_file_raises_tracking_error(self, tmp_path: Path) -> None:
    path = tmp_path / 'binary.json'
    path.write_bytes(b'\x80\x81\x82\xff\xfe')
    io = JSONCheckpointIO()
    with pytest.raises(TrackingError, match='failed to read JSON'):
      io.load(path)


class TestOldCheckpointClassesRemoved:
  """Confirm Checkpoint and JSONCheckpoint are gone."""

  def test_checkpoint_not_exported(self) -> None:
    assert not hasattr(_ckpt_mod, 'Checkpoint'), 'old Checkpoint class should be deleted'

  def test_json_checkpoint_not_exported(self) -> None:
    assert not hasattr(_ckpt_mod, 'JSONCheckpoint'), 'old JSONCheckpoint class should be deleted'


class TestCustomCheckpointIO:
  """Subclass pattern works."""

  def test_subclass_roundtrip(self) -> None:
    class MemIO(CheckpointIO):
      def __init__(self) -> None:
        self._store: dict[str, dict] = {}

      def save(self, state, path):
        self._store[str(path)] = state

      def load(self, path):
        return self._store[str(path)]

      def remove(self, path):
        self._store.pop(str(path), None)

      def exists(self, path):
        return str(path) in self._store

    io = MemIO()
    state = {'epoch': 5}
    io.save(state, Path('/tmp/a'))
    assert io.exists(Path('/tmp/a'))
    assert io.load(Path('/tmp/a')) == state
    io.remove(Path('/tmp/a'))
    assert not io.exists(Path('/tmp/a'))
