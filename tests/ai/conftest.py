"""Shared FileStore test builders for tests/ai/ subtree."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from pathlib import Path


def make_store_config(tmp_path: Path) -> AutoPilotConfig:
  """Build a minimal AutoPilotConfig rooted at *tmp_path*.

  Args:
    tmp_path: Temporary directory (pytest fixture).

  Returns:
    Config with ``store_path`` pointing at ``{tmp_path}/.autopilot``.
  """
  config = AutoPilotConfig(workspace=tmp_path)
  config.store_path = tmp_path / '.autopilot'
  return config


def make_source_dir(
  tmp_path: Path,
  name: str = 'src',
  files: dict[str, str] | None = None,
) -> Path:
  """Create a source directory with seed files.

  Args:
    tmp_path: Temporary directory (pytest fixture).
    name: Subdirectory name under *tmp_path*.
    files: Mapping of ``filename -> content``.  Defaults to a single
      ``main.py`` with ``print("hello")``.

  Returns:
    Path to the created source directory.
  """
  src = tmp_path / name
  src.mkdir(parents=True, exist_ok=True)
  for fname, content in (files or {'main.py': 'print("hello")'}).items():
    (src / fname).parent.mkdir(parents=True, exist_ok=True)
    (src / fname).write_text(content)
  return src


def make_file_store(
  tmp_path: Path,
  slug: str = 'exp-001',
  files: dict[str, str] | None = None,
  source_name: str = 'src',
) -> tuple[FileStore, Path, PathParameter]:
  """Create a FileStore with one PathParameter and an initial snapshot.

  Args:
    tmp_path: Temporary directory (pytest fixture).
    slug: Experiment id for the initial snapshot.
    files: Seed files for the source directory.
    source_name: Name of the source subdirectory.

  Returns:
    Tuple of ``(store, source_dir, param)``.
  """
  src = make_source_dir(tmp_path, name=source_name, files=files)
  config = make_store_config(tmp_path)
  param = PathParameter(source=str(src), pattern='*')
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot(slug, 0)
  return store, src, param
