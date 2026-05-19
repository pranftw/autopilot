"""Tests for store and parameter validation hardening (Plan 04).

Covers:
  - 2.2: FileStore.snapshot rejects empty registration (StoreError)
  - 2.3: FileStore.doctor forest health checks
  - 2.5: PathParameter.restore path traversal guard
  - 2.6: Binary file protection on checkout
  - 2.7: Empty snapshot guard against mass deletion
  - 2.8: File permission not preserved (documentation test)
"""

from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.ai.store.snapshot_helpers import is_probably_binary_file
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from pathlib import Path
import json
import pytest
import stat

# -- 2.2: Fail-fast empty register_parameters before snapshot --


class TestSnapshotRequiresRegisteredParameters:
  """FileStore.snapshot raises StoreError when no parameters are registered."""

  def test_snapshot_requires_registered_parameters(self, tmp_path: Path) -> None:
    """Calling snapshot without register_parameters raises StoreError."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    with pytest.raises(StoreError, match='register_parameters'):
      store.snapshot('exp-1', 0)

  def test_snapshot_passes_with_registered_params(self, tmp_path: Path) -> None:
    """Snapshot succeeds after registering at least one parameter."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    param_dir = tmp_path / 'params'
    param_dir.mkdir()
    (param_dir / 'file.txt').write_text('hello', encoding='utf-8')
    param = PathParameter(source=str(param_dir), pattern='**/*')
    store.register_parameters({'my_param': param})

    manifest = store.snapshot('exp-1', 0)
    assert manifest.epoch == 0

  def test_snapshot_empty_param_snapshot_allowed(self, tmp_path: Path) -> None:
    """Parameter returning {} from snapshot() is allowed when registered."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    param_dir = tmp_path / 'empty_params'
    param_dir.mkdir()
    param = PathParameter(source=str(param_dir), pattern='*.nonexistent')
    store.register_parameters({'empty': param})

    manifest = store.snapshot('exp-empty', 0)
    assert manifest.epoch == 0


# -- 2.3: Forest health checks inside FileStore.doctor() --


class TestDoctorForestMissing:
  """Doctor reports info (not error) when forest.json is missing."""

  def test_doctor_missing_forest_reports_info_not_error(self, tmp_path: Path) -> None:
    """Missing forest.json: healthy=True, forest_missing=True, info diagnostic."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    result = store.doctor_report()
    assert result['healthy'] is True
    assert result['forest_missing'] is True
    assert result['forest_errors'] == []
    info_entries = [
      d
      for d in result['diagnostics']
      if d.get('code') == 'forest_missing' and d.get('severity') == 'info'
    ]
    assert len(info_entries) == 1


class TestDoctorForestMalformedJson:
  """Doctor reports parse error for malformed JSON."""

  def test_doctor_malformed_forest_still_unhealthy(self, tmp_path: Path) -> None:
    """Corrupt JSON: healthy=False, forest_missing=False, forest_errors non-empty."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text('{not valid json!', encoding='utf-8')

    result = store.doctor_report()
    assert result['healthy'] is False
    assert result['forest_missing'] is False
    assert result['forest_errors']
    assert any('not valid JSON' in e for e in result['forest_errors'])
    corrupt_entries = [
      d
      for d in result['diagnostics']
      if d.get('code') == 'forest_corrupt' and d.get('severity') == 'error'
    ]
    assert len(corrupt_entries) >= 1

  def test_doctor_report_forest_missing_field(self, tmp_path: Path) -> None:
    """doctor_report()['forest_missing'] is True when file absent."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    result = store.doctor_report()
    assert 'forest_missing' in result
    assert result['forest_missing'] is True


class TestDoctorForestMissingTreesKey:
  """Doctor reports error when 'trees' key missing."""

  def test_doctor_forest_missing_trees(self, tmp_path: Path) -> None:
    """forest.json without 'trees' key produces explicit error."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text('{}', encoding='utf-8')

    result = store.doctor_report()
    assert any('"trees" key' in e for e in result['forest_errors'])


class TestDoctorForestTreesNotList:
  """Doctor reports error when 'trees' is not a list."""

  def test_doctor_forest_trees_not_list(self, tmp_path: Path) -> None:
    """forest.json with trees as non-list produces error."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text(json.dumps({'trees': 'oops'}), encoding='utf-8')

    result = store.doctor_report()
    assert any('"trees" must be a list' in e for e in result['forest_errors'])


class TestDoctorForestValidMinimal:
  """Doctor passes on a valid minimal forest."""

  def test_doctor_forest_valid_minimal(self, tmp_path: Path) -> None:
    """Valid minimal forest.json produces no forest_errors."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text(json.dumps({'trees': [], 'active': None}), encoding='utf-8')

    result = store.doctor_report()
    assert result['forest_errors'] == []


class TestDoctorForestTreeNodeValidation:
  """Doctor validates tree and node structure."""

  def test_doctor_forest_tree_missing_name(self, tmp_path: Path) -> None:
    """Tree without string name key produces error."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    data = {'trees': [{'nodes': []}]}
    config.forest_file.write_text(json.dumps(data), encoding='utf-8')

    result = store.doctor_report()
    assert any('missing string "name"' in e for e in result['forest_errors'])

  def test_doctor_forest_node_missing_experiment(self, tmp_path: Path) -> None:
    """Node without string experiment key produces error."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    data = {'trees': [{'name': 'main', 'nodes': [{'other': 'stuff'}]}]}
    config.forest_file.write_text(json.dumps(data), encoding='utf-8')

    result = store.doctor_report()
    assert any('missing string "experiment"' in e for e in result['forest_errors'])

  def test_doctor_forest_healthy_composite(self, tmp_path: Path) -> None:
    """Forest errors flip healthy=False even without manifest/blob issues."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text(json.dumps({'trees': 'bad'}), encoding='utf-8')

    result = store.doctor_report()
    assert result['healthy'] is False

  def test_doctor_forest_tree_not_a_dict(self, tmp_path: Path) -> None:
    """Tree entry that is not a dict produces error."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    data = {'trees': ['not-a-dict']}
    config.forest_file.write_text(json.dumps(data), encoding='utf-8')

    result = store.doctor_report()
    assert any('must be a dict' in e for e in result['forest_errors'])

  def test_doctor_forest_node_not_a_dict(self, tmp_path: Path) -> None:
    """Node entry that is not a dict produces error."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    data = {'trees': [{'name': 'main', 'nodes': [42]}]}
    config.forest_file.write_text(json.dumps(data), encoding='utf-8')

    result = store.doctor_report()
    assert any('nodes[0] must be a dict' in e for e in result['forest_errors'])

  def test_doctor_forest_missing_nodes_key(self, tmp_path: Path) -> None:
    """Tree without 'nodes' key produces error."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    data = {'trees': [{'name': 'main'}]}
    config.forest_file.write_text(json.dumps(data), encoding='utf-8')

    result = store.doctor_report()
    assert any('nodes' in e.lower() for e in result['forest_errors'])

  def test_doctor_forest_valid_with_nodes(self, tmp_path: Path) -> None:
    """Valid forest with realistic nodes produces no errors."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    config.forest_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
      'active': 'main',
      'trees': [
        {
          'name': 'main',
          'description': None,
          'head': 'exp-1',
          'nodes': [
            {
              'type': 'Node',
              'experiment': 'exp-1',
              'parent': None,
              'baseline': None,
            }
          ],
        }
      ],
    }
    config.forest_file.write_text(json.dumps(data), encoding='utf-8')

    result = store.doctor_report()
    assert result['forest_errors'] == []


# -- 2.5: Path traversal guard in PathParameter.restore() --


class TestRestorePathTraversal:
  """PathParameter.restore rejects traversal attempts."""

  def test_restore_path_traversal_dot_dot(self, tmp_path: Path) -> None:
    """'../outside.py' key raises ValueError with 'path traversal blocked'."""
    param_dir = tmp_path / 'root'
    param_dir.mkdir()
    param = PathParameter(source=str(param_dir))

    with pytest.raises(ValueError, match='path traversal blocked'):
      param.restore({'../outside.py': 'evil'})

  def test_restore_path_traversal_deeply_escaped(self, tmp_path: Path) -> None:
    """'../../deeply/escaped.py' key raises ValueError."""
    param_dir = tmp_path / 'root'
    param_dir.mkdir()
    param = PathParameter(source=str(param_dir))

    with pytest.raises(ValueError, match='path traversal blocked'):
      param.restore({'../../deeply/escaped.py': 'evil'})

  def test_restore_path_traversal_absolute_key(self, tmp_path: Path) -> None:
    """'/absolute/path.py' key raises ValueError with 'absolute or empty'."""
    param_dir = tmp_path / 'root'
    param_dir.mkdir()
    param = PathParameter(source=str(param_dir))

    with pytest.raises(ValueError, match='absolute or empty'):
      param.restore({'/absolute/path.py': 'evil'})

  def test_restore_path_traversal_empty_key(self, tmp_path: Path) -> None:
    """'' (empty) key raises ValueError."""
    param_dir = tmp_path / 'root'
    param_dir.mkdir()
    param = PathParameter(source=str(param_dir))

    with pytest.raises(ValueError, match='absolute or empty'):
      param.restore({'': 'evil'})

  def test_restore_path_traversal_still_inside_allowed(self, tmp_path: Path) -> None:
    """'subdir/../still_inside.py' resolves within root and is allowed."""
    param_dir = tmp_path / 'root'
    param_dir.mkdir()
    param = PathParameter(source=str(param_dir))

    param.restore({'subdir/../still_inside.py': 'ok'})
    assert (param_dir / 'still_inside.py').read_text(encoding='utf-8') == 'ok'

  def test_restore_path_traversal_normal_nested(self, tmp_path: Path) -> None:
    """'normal/nested/file.py' writes successfully."""
    param_dir = tmp_path / 'root'
    param_dir.mkdir()
    param = PathParameter(source=str(param_dir))

    param.restore({'normal/nested/file.py': 'content'})
    assert (param_dir / 'normal' / 'nested' / 'file.py').read_text(encoding='utf-8') == 'content'

  def test_restore_path_traversal_symlink_escape(self, tmp_path: Path) -> None:
    """Symlink under root pointing outside triggers traversal block."""
    param_dir = tmp_path / 'root'
    param_dir.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    link_target = param_dir / 'escape_link'
    link_target.symlink_to(outside)
    param = PathParameter(source=str(param_dir))

    with pytest.raises(ValueError, match='path traversal blocked'):
      param.restore({'escape_link/secret.txt': 'evil'})


# -- 2.6: Binary file protection on checkout --


class TestCheckoutBinaryProtection:
  """Binary files survive checkout even when not in manifest."""

  def test_is_binary_file_with_null_bytes(self, tmp_path: Path) -> None:
    """_is_binary_file returns True for files with null bytes."""
    f = tmp_path / 'binary.bin'
    f.write_bytes(b'hello\x00world')
    assert is_probably_binary_file(f) is True

  def test_is_binary_file_with_pure_text(self, tmp_path: Path) -> None:
    """_is_binary_file returns False for pure text."""
    f = tmp_path / 'text.txt'
    f.write_text('hello world', encoding='utf-8')
    assert is_probably_binary_file(f) is False

  def test_is_binary_file_with_unreadable(self, tmp_path: Path) -> None:
    """_is_binary_file returns False for unreadable files (fail-open)."""
    f = tmp_path / 'nope.bin'
    assert is_probably_binary_file(f) is False

  def test_checkout_binary_protection_survives(self, tmp_path: Path) -> None:
    """Binary file in glob range but not in manifest survives checkout."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    param_dir = tmp_path / 'src'
    param_dir.mkdir()
    (param_dir / 'code.py').write_text('print("hi")', encoding='utf-8')

    param = PathParameter(source=str(param_dir), pattern='**/*.py')
    store.register_parameters({'src': param})
    store.snapshot('exp-1', 0)

    (param_dir / 'extra.py').write_text('extra', encoding='utf-8')
    (param_dir / 'image.png').write_bytes(b'\x89PNG\x00\x00\x00')
    (param_dir / 'extra.bin').write_bytes(b'\x00binary\x00')

    store.checkout('exp-1', 0)

    assert not (param_dir / 'extra.py').exists()
    assert (param_dir / 'extra.bin').exists()
    assert (param_dir / 'image.png').exists()

  def test_checkout_binary_not_matching_glob_untouched(self, tmp_path: Path) -> None:
    """Binary file outside the glob pattern stays untouched during checkout."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    param_dir = tmp_path / 'src'
    param_dir.mkdir()
    (param_dir / 'app.py').write_text('print("app")', encoding='utf-8')
    (param_dir / 'data.db').write_bytes(b'\x00sqlite\x00data')

    param = PathParameter(source=str(param_dir), pattern='**/*.py')
    store.register_parameters({'src': param})
    store.snapshot('exp-1', 0)

    store.checkout('exp-1', 0)

    assert (param_dir / 'app.py').exists()
    assert (param_dir / 'data.db').exists()

  def test_checkout_text_not_in_manifest_deleted(self, tmp_path: Path) -> None:
    """Text file not in manifest is deleted on checkout."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    param_dir = tmp_path / 'src'
    param_dir.mkdir()
    (param_dir / 'keep.py').write_text('keep', encoding='utf-8')
    param = PathParameter(source=str(param_dir), pattern='**/*')
    store.register_parameters({'src': param})
    store.snapshot('exp-1', 0)

    (param_dir / 'stale.py').write_text('stale', encoding='utf-8')
    store.checkout('exp-1', 0)

    assert not (param_dir / 'stale.py').exists()
    assert (param_dir / 'keep.py').exists()


# -- 2.7: Empty snapshot guard against mass deletion --


class TestCheckoutEmptySnapshotGuard:
  """Zero-entry snapshot does NOT trigger extraneous file deletion."""

  def test_checkout_empty_snapshot_guard_files_survive(self, tmp_path: Path) -> None:
    """Files survive when snapshot has no entries for the parameter."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    param_dir = tmp_path / 'src'
    param_dir.mkdir()
    param = PathParameter(source=str(param_dir), pattern='*.nonexistent')
    store.register_parameters({'src': param})
    store.snapshot('exp-1', 0)

    (param_dir / 'added_later.py').write_text('code', encoding='utf-8')

    store.checkout('exp-1', 0)

    assert (param_dir / 'added_later.py').exists()

  def test_checkout_empty_snapshot_guard_normal_behavior(self, tmp_path: Path) -> None:
    """Non-empty snapshot still removes extra files (normal behavior)."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    param_dir = tmp_path / 'src'
    param_dir.mkdir()
    (param_dir / 'original.py').write_text('original', encoding='utf-8')
    param = PathParameter(source=str(param_dir), pattern='**/*.py')
    store.register_parameters({'src': param})
    store.snapshot('exp-1', 0)

    (param_dir / 'extra.py').write_text('extra', encoding='utf-8')
    store.checkout('exp-1', 0)

    assert not (param_dir / 'extra.py').exists()
    assert (param_dir / 'original.py').exists()

  def test_checkout_empty_snapshot_guard_multi_param(self, tmp_path: Path) -> None:
    """Multiple params: empty param files survive, populated param extras deleted."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    empty_dir = tmp_path / 'empty_param'
    empty_dir.mkdir()
    populated_dir = tmp_path / 'populated_param'
    populated_dir.mkdir()
    (populated_dir / 'keep.txt').write_text('keep', encoding='utf-8')

    empty_param = PathParameter(source=str(empty_dir), pattern='*.nonexistent')
    populated_param = PathParameter(source=str(populated_dir), pattern='**/*')
    store.register_parameters({'empty': empty_param, 'pop': populated_param})
    store.snapshot('exp-1', 0)

    (empty_dir / 'safe.txt').write_text('safe', encoding='utf-8')
    (populated_dir / 'stale.txt').write_text('stale', encoding='utf-8')

    store.checkout('exp-1', 0)

    assert (empty_dir / 'safe.txt').exists()
    assert not (populated_dir / 'stale.txt').exists()
    assert (populated_dir / 'keep.txt').exists()


# -- 2.8: Permission not preserved --


class TestPermissionsNotPreserved:
  """File permissions are not preserved through restore."""

  def test_permissions_not_preserved(self, tmp_path: Path) -> None:
    """Executable bit (755) is not restored after checkout; file remains 644."""
    config = AutoPilotConfig(workspace=tmp_path)
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)

    param_dir = tmp_path / 'scripts'
    param_dir.mkdir()
    script = param_dir / 'run.sh'
    script.write_text('#!/bin/bash\necho hi', encoding='utf-8')
    script.chmod(0o755)

    param = PathParameter(source=str(param_dir), pattern='**/*')
    store.register_parameters({'scripts': param})
    store.snapshot('exp-1', 0)

    script.chmod(0o644)
    store.checkout('exp-1', 0)

    mode = stat.S_IMODE(script.stat().st_mode)
    assert not (mode & stat.S_IXUSR)
