"""Tests for PathParameter symlink safety during snapshot."""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store.snapshot_helpers import build_snapshot
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
import logging
import pytest


@pytest.fixture
def param_root(tmp_path: Path) -> Path:
  """Parameter root directory with a regular file."""
  root = tmp_path / 'params'
  root.mkdir()
  (root / 'normal.txt').write_text('hello', encoding='utf-8')
  return root


class TestPathParameterSymlinkSafety:
  """Tests for symlink handling during PathParameter.snapshot()."""

  def test_path_parameter_symlink_outside_root_skipped(
    self, tmp_path: Path, param_root: Path
  ) -> None:
    """Symlink under root pointing outside is excluded from snapshot."""
    outside = tmp_path / 'outside'
    outside.mkdir()
    secret = outside / 'secret.txt'
    secret.write_text('confidential', encoding='utf-8')

    link = param_root / 'escape.txt'
    link.symlink_to(secret)

    param = PathParameter(source=str(param_root), pattern='**/*')
    snap = param.snapshot()

    assert 'escape.txt' not in snap
    assert 'normal.txt' in snap
    assert snap['normal.txt'] == 'hello'

  def test_path_parameter_symlink_within_root_snapshotted(self, param_root: Path) -> None:
    """Symlink under root pointing to another file inside root is captured."""
    target = param_root / 'normal.txt'
    link = param_root / 'alias.txt'
    link.symlink_to(target)

    param = PathParameter(source=str(param_root), pattern='**/*')
    snap = param.snapshot()

    assert 'alias.txt' in snap
    assert snap['alias.txt'] == 'hello'
    assert 'normal.txt' in snap

  def test_symlink_broken_skipped(self, param_root: Path) -> None:
    """Broken symlink (dangling target) is skipped without error."""
    link = param_root / 'broken.txt'
    link.symlink_to(param_root / 'nonexistent.txt')

    param = PathParameter(source=str(param_root), pattern='**/*')
    snap = param.snapshot()

    assert 'broken.txt' not in snap
    assert 'normal.txt' in snap

  def test_symlink_to_directory_followed(self, tmp_path: Path, param_root: Path) -> None:
    """Symlink to a directory within root does not lose content.

    Files inside the target directory are captured via the canonical path.
    The glob may resolve the symlink directory to its real target, so we
    verify the content appears in the snapshot under some key.
    """
    subdir = param_root / 'subdir'
    subdir.mkdir()
    (subdir / 'inner.txt').write_text('inside', encoding='utf-8')

    link = param_root / 'linked_dir'
    link.symlink_to(subdir)

    param = PathParameter(source=str(param_root), pattern='**/*')
    snap = param.snapshot()

    assert any('inner.txt' in k for k in snap)
    inner_values = [v for k, v in snap.items() if 'inner.txt' in k]
    assert 'inside' in inner_values

  def test_regular_files_not_affected(self, param_root: Path) -> None:
    """Non-symlink files are snapshotted normally regardless of symlink checks."""
    (param_root / 'extra.txt').write_text('data', encoding='utf-8')
    param = PathParameter(source=str(param_root), pattern='**/*')
    snap = param.snapshot()
    assert 'normal.txt' in snap
    assert 'extra.txt' in snap

  def test_symlink_outside_root_emits_warning(
    self, tmp_path: Path, param_root: Path, caplog: pytest.LogCaptureFixture
  ) -> None:
    """Escape symlink triggers a warning log with 'outside_root' reason."""
    outside = tmp_path / 'outside'
    outside.mkdir()
    secret = outside / 'secret.txt'
    secret.write_text('confidential', encoding='utf-8')
    link = param_root / 'escape.txt'
    link.symlink_to(secret)

    param = PathParameter(source=str(param_root), pattern='**/*')
    with caplog.at_level(logging.WARNING, logger='autopilot.ai.parameter'):
      snap = param.snapshot()

    assert 'escape.txt' not in snap
    assert any('outside_root' in record.message for record in caplog.records)
    assert any('skipped symlink' in record.message for record in caplog.records)

  def test_symlink_broken_emits_warning(
    self, param_root: Path, caplog: pytest.LogCaptureFixture
  ) -> None:
    """Broken symlink triggers a warning log with 'broken' reason."""
    link = param_root / 'broken.txt'
    link.symlink_to(param_root / 'nonexistent.txt')

    param = PathParameter(source=str(param_root), pattern='**/*')
    with caplog.at_level(logging.WARNING, logger='autopilot.ai.parameter'):
      snap = param.snapshot()

    assert 'broken.txt' not in snap
    assert any('broken' in record.message for record in caplog.records)
    assert any('skipped symlink' in record.message for record in caplog.records)

  def test_symlink_macos_style_intermediate_resolve(self, tmp_path: Path, param_root: Path) -> None:
    """Resolve-based containment handles intermediate symlinks.

    Creates nested symlinks within the root (alias -> subdir) and an
    escape symlink (escape -> outside target). Verifies content through
    the alias is captured while the escape is excluded.
    """
    subdir = param_root / 'sub'
    subdir.mkdir()
    (subdir / 'inner.txt').write_text('inside', encoding='utf-8')

    alias_link = param_root / 'alias'
    alias_link.symlink_to(subdir)

    outside = tmp_path / 'outside_dir'
    outside.mkdir()
    (outside / 'secret.txt').write_text('leak', encoding='utf-8')
    escape_link = param_root / 'escape_link'
    escape_link.symlink_to(outside / 'secret.txt')

    param = PathParameter(source=str(param_root), pattern='**/*')
    snap = param.snapshot()

    assert any('inner.txt' in k for k in snap)
    inner_values = [v for k, v in snap.items() if 'inner.txt' in k]
    assert 'inside' in inner_values
    assert 'escape_link' not in snap

  def test_filestore_snapshot_excludes_escape_symlink(self, tmp_path: Path) -> None:
    """FileStore build_snapshot excludes escape symlinks from manifest entries."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    param_root = ws / 'params'
    param_root.mkdir()
    (param_root / 'good.txt').write_text('safe content', encoding='utf-8')

    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'secret.txt').write_text('confidential', encoding='utf-8')
    (param_root / 'escape.txt').symlink_to(outside / 'secret.txt')

    config = AutoPilotConfig(workspace=ws)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    param = PathParameter(source=str(param_root), pattern='**/*')
    store.register_parameters({'files': param})

    manifest = build_snapshot(store, context='test')

    entry_keys = set(manifest.entries.keys())
    assert 'files/good.txt' in entry_keys
    assert 'files/escape.txt' not in entry_keys
