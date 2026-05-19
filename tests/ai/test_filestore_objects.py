"""Tests for FileStore content-addressed object validation (BUG-042)."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.errors import StoreError
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
import pytest


def _make_store(
  tmp_path: Path,
  files: dict[str, str] | None = None,
) -> tuple[FileStore, Path, PathParameter]:
  if files is None:
    files = {'main.py': 'print("hello")\n'}
  src = make_source_dir(tmp_path, files=files)
  config = make_store_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot('root', 0)
  return store, src, param


def test_read_object_round_trip_bytes(tmp_path: Path) -> None:
  store, _src, _param = _make_store(tmp_path)
  manifest = store.load_snapshot('root', 0)
  digest = next(iter(manifest.entries.values())).digest
  data = store.read_object(digest)
  assert data.decode('utf-8') == 'print("hello")\n'


def test_read_object_corrupt_byte_raises_store_error(tmp_path: Path) -> None:
  store, _src, _param = _make_store(tmp_path)
  manifest = store.load_snapshot('root', 0)
  digest = next(iter(manifest.entries.values())).digest
  objects_dir = store.config.objects_path
  obj_path = objects_dir / digest[:2] / digest[2:]
  raw = bytearray(obj_path.read_bytes())
  raw[0] ^= 0xFF
  obj_path.write_bytes(bytes(raw))

  with pytest.raises(StoreError) as excinfo:
    store.read_object(digest)
  msg = str(excinfo.value)
  assert digest in msg
  assert 'expected hash' in msg
  assert 'got' in msg
