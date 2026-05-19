"""Tests for FileStore log() with sparse epoch snapshots (BUG-043)."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config


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


def test_log_sparse_epochs_returns_three_entries(tmp_path: Path) -> None:
  store, src, _param = _make_store(tmp_path)
  store.snapshot('exp', 0)
  (src / 'main.py').write_text('e1\n')
  store.snapshot('exp', 1)
  (src / 'main.py').write_text('e2\n')
  store.snapshot('exp', 2)
  (src / 'main.py').write_text('e3\n')
  store.snapshot('exp', 3)

  epoch_two = store.config.snapshots_path / 'exp' / 'epoch_2.json'
  epoch_two.unlink()

  entries = store.log('exp')
  assert len(entries) == 3


def test_log_sparse_epoch_values_include_only_present(tmp_path: Path) -> None:
  store, src, _param = _make_store(tmp_path)
  store.snapshot('exp', 0)
  (src / 'main.py').write_text('e1\n')
  store.snapshot('exp', 1)
  (src / 'main.py').write_text('e2\n')
  store.snapshot('exp', 2)
  (src / 'main.py').write_text('e3\n')
  store.snapshot('exp', 3)

  epoch_two = store.config.snapshots_path / 'exp' / 'epoch_2.json'
  epoch_two.unlink()

  entries = store.log('exp')
  epochs = {entry.epoch for entry in entries}
  assert epochs == {0, 1, 3}


def test_log_contiguous_matches_count(tmp_path: Path) -> None:
  store, src, _param = _make_store(tmp_path)
  store.snapshot('exp', 0)
  for epoch in range(1, 5):
    (src / 'main.py').write_text(f'content {epoch}\n')
    store.snapshot('exp', epoch)

  entries = store.log('exp')
  assert len(entries) == 5
  assert {entry.epoch for entry in entries} == {0, 1, 2, 3, 4}
