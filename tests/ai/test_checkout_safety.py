"""Checkout safety tests: protected paths, schema mismatch, and dry-run validation.

Regression tests for BUG-001 (protected-path destruction), BUG-003 (silent
no-op checkout), and BUG-017 (dry-run skips validation).
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store.snapshot import PROTECTED_PREFIXES, SchemaMatchResult
from autopilot.core.errors import StoreError
from pathlib import Path
from tests.ai.conftest import make_source_dir, make_store_config
import pytest


def _make_store_with_snapshot(
  tmp_path: Path,
  slug: str = 'exp-001',
  pattern: str = '**/*',
  files: dict[str, str] | None = None,
) -> tuple[FileStore, Path, PathParameter]:
  """Create a store with one snapshot at epoch 0."""
  src = make_source_dir(tmp_path, files=files)
  config = make_store_config(tmp_path)
  param = PathParameter(source=str(src), pattern=pattern)
  store = FileStore(config)
  store.register_parameters({'source': param})
  store.snapshot(slug, 0)
  return store, src, param


# 4.1: protected paths (BUG-001)


class TestProtectedPaths:
  """Checkout must never delete files under protected directory prefixes."""

  def test_checkout_preserves_git_directory(self, tmp_path: Path) -> None:
    store, src, _ = _make_store_with_snapshot(tmp_path, pattern='**/*')
    git_dir = src / '.git'
    git_dir.mkdir()
    (git_dir / 'HEAD').write_text('ref: refs/heads/main')

    store.checkout('exp-001', 0)

    assert (git_dir / 'HEAD').exists()
    assert (git_dir / 'HEAD').read_text() == 'ref: refs/heads/main'

  def test_checkout_preserves_autopilot_directory(self, tmp_path: Path) -> None:
    store, src, _ = _make_store_with_snapshot(tmp_path, pattern='**/*')
    ap_dir = src / '.autopilot'
    ap_dir.mkdir()
    (ap_dir / 'config.json').write_text('{}')

    store.checkout('exp-001', 0)

    assert (ap_dir / 'config.json').exists()

  def test_checkout_preserves_venv_directory(self, tmp_path: Path) -> None:
    store, src, _ = _make_store_with_snapshot(tmp_path, pattern='**/*')
    venv_dir = src / '.venv'
    venv_dir.mkdir(parents=True)
    bin_dir = venv_dir / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'python').write_text('#!/usr/bin/env python')

    store.checkout('exp-001', 0)

    assert (bin_dir / 'python').exists()

  def test_checkout_removes_unprotected_files(self, tmp_path: Path) -> None:
    store, src, _ = _make_store_with_snapshot(
      tmp_path, pattern='**/*', files={'main.py': 'print("hello")'}
    )
    stale = src / 'stale.py'
    stale.write_text('old code')

    store.checkout('exp-001', 0)

    assert not stale.exists()
    assert (src / 'main.py').exists()

  def test_checkout_narrow_pattern_scopes_deletion(self, tmp_path: Path) -> None:
    store, src, _ = _make_store_with_snapshot(
      tmp_path, pattern='*.py', files={'main.py': 'print("hello")'}
    )
    txt_file = src / 'notes.txt'
    txt_file.write_text('some notes')
    extra_py = src / 'extra.py'
    extra_py.write_text('extra code')

    store.checkout('exp-001', 0)

    assert txt_file.exists(), 'txt file outside pattern should survive'
    assert not extra_py.exists(), 'py file not in snapshot should be removed'

  def test_all_protected_prefixes_are_covered(self) -> None:
    expected = {'.git', '.autopilot', 'node_modules', '.venv', '__pycache__'}
    assert expected == PROTECTED_PREFIXES

  def test_nested_protected_file_preserved(self, tmp_path: Path) -> None:
    """Deep files under protected prefixes are also preserved."""
    store, src, _ = _make_store_with_snapshot(tmp_path, pattern='**/*')
    deep = src / '.git' / 'objects' / 'pack' / 'data.pack'
    deep.parent.mkdir(parents=True)
    deep.write_text('pack data')

    store.checkout('exp-001', 0)

    assert deep.exists()

  def test_checkout_preserves_node_modules(self, tmp_path: Path) -> None:
    store, src, _ = _make_store_with_snapshot(tmp_path, pattern='**/*')
    pkg = src / 'node_modules' / 'pkg' / 'index.js'
    pkg.parent.mkdir(parents=True)
    pkg.write_text('module.exports = {}')

    store.checkout('exp-001', 0)

    assert pkg.exists()

  def test_checkout_preserves_pycache(self, tmp_path: Path) -> None:
    store, src, _ = _make_store_with_snapshot(tmp_path, pattern='**/*')
    cache = src / '__pycache__' / 'mod.cpython-311.pyc'
    cache.parent.mkdir(parents=True)
    cache.write_text('bytecode')

    store.checkout('exp-001', 0)

    assert cache.exists()


# 4.2: schema mismatch (BUG-003)


class TestSchemaMismatch:
  """Checkout must raise when manifest has entries but no params match."""

  def test_checkout_raises_on_manifest_param_mismatch(self, tmp_path: Path) -> None:
    src = make_source_dir(tmp_path, files={'main.py': 'print("hello")'})
    config = make_store_config(tmp_path)
    param = PathParameter(source=str(src), pattern='*.py')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-001', 0)

    store2 = FileStore(config)
    store2.register_parameters({'other_name': param})

    with pytest.raises(StoreError, match='no registered parameters match'):
      store2.checkout('exp-001', 0)

  def test_checkout_empty_manifest_succeeds(self, tmp_path: Path) -> None:
    src = tmp_path / 'empty_src'
    src.mkdir()
    config = make_store_config(tmp_path)
    param = PathParameter(source=str(src), pattern='*.py')
    store = FileStore(config)
    store.register_parameters({'source': param})
    store.snapshot('exp-001', 0)

    store.checkout('exp-001', 0)

  def test_checkout_matching_params_restores_files(self, tmp_path: Path) -> None:
    store, src, _ = _make_store_with_snapshot(tmp_path, files={'main.py': 'original'})
    (src / 'main.py').write_text('modified')
    assert (src / 'main.py').read_text() == 'modified'

    store.checkout('exp-001', 0)

    assert (src / 'main.py').read_text() == 'original'

  def test_schema_match_result_fields(self, tmp_path: Path) -> None:
    """SchemaMatchResult correctly reports matched and mismatched sets."""
    store, _src, _ = _make_store_with_snapshot(tmp_path, files={'main.py': 'code'})
    snap = store.load_snapshot('exp-001', 0)
    result = store._validate_schema(snap)

    assert isinstance(result, SchemaMatchResult)
    assert 'source' in result.matched
    assert not result.mismatched


# 4.3: dry-run validation (BUG-017)


class TestDryRunValidation:
  """Dry-run must validate experiment/epoch/snapshot before returning."""

  def test_checkout_dry_run_rejects_missing_experiment(self, tmp_path: Path) -> None:
    store, _, _ = _make_store_with_snapshot(tmp_path)

    with pytest.raises(StoreError, match='not found'):
      store.validate_checkout('nonexistent', 0)

  def test_checkout_dry_run_rejects_invalid_epoch(self, tmp_path: Path) -> None:
    store, _, _ = _make_store_with_snapshot(tmp_path)

    with pytest.raises(StoreError, match='snapshot not found'):
      store.validate_checkout('exp-001', 999)

  def test_checkout_dry_run_returns_structured_info(self, tmp_path: Path) -> None:
    store, _, _ = _make_store_with_snapshot(tmp_path, files={'main.py': 'code', 'util.py': 'util'})
    info = store.validate_checkout('exp-001', 0)

    assert info['files_to_restore'] == 2
    assert info['schema_match'] is True
    assert info['schema_mismatch'] is False
