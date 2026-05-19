"""Tests for store doctor --repair and workspace doctor --repair CLI.

Covers:
  - workspace doctor --repair creates missing dirs (4.3 #13)
  - --repair --json output includes diagnostics (4.3 #14)
  - clean store/workspace --repair exits 0 (4.3 #15)
  - partial failure produces non-zero exit (4.3 #16)
  - all repairs succeed exits 0 (4.3 #17)
  - permission error produces structured failure (4.3 #18)
"""

from autopilot.ai.forest import FileForest
from autopilot.ai.parameter import PathParameter
from autopilot.ai.store.file_store import FileStore
from autopilot.core.config import AutoPilotConfig
from autopilot.core.errors import StoreError
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
from unittest.mock import patch
import pytest


def _init_workspace(tmp_path: Path) -> Path:
  """Create the full autopilot workspace layout.

  Returns:
    Path to the workspace root.
  """
  ws = tmp_path / 'project'
  ws.mkdir()
  ap = ws / '.autopilot'
  ap.mkdir()
  for subdir in ('experiments', 'records', 'datasets', 'projects'):
    (ap / subdir).mkdir()
  return ws


def _init_store_with_orphan(ws: Path) -> tuple[FileStore, str]:
  """Create a store with a snapshot and inject an orphan blob.

  Returns:
    Tuple of (store, orphan_digest).
  """
  prompts = ws / 'prompts'
  prompts.mkdir(exist_ok=True)
  (prompts / 'main.txt').write_text('hello')

  config = AutoPilotConfig(workspace=ws)
  store = FileStore(config)
  param = PathParameter(source=str(prompts), pattern='*.txt')
  store.register_parameters({'source': param})
  store.snapshot('exp-001', 0)
  FileForest(store).save()

  shard = config.objects_path / 'ff'
  shard.mkdir(parents=True, exist_ok=True)
  orphan_digest = 'ff' + 'a1b2c3d4e5'
  (shard / 'a1b2c3d4e5').write_bytes(b'orphan')
  return store, orphan_digest


class TestWorkspaceDoctorRepairMissingDirs:
  """test_workspace_doctor_repair_missing_dirs -- creates missing dirs."""

  def test_repair_creates_missing_experiments_dir(self, tmp_path: Path) -> None:
    """Missing experiments dir is recreated by workspace doctor --repair."""
    ws = _init_workspace(tmp_path)
    experiments_dir = ws / '.autopilot' / 'experiments'
    experiments_dir.rmdir()
    assert not experiments_dir.exists()

    result = run_cli(ws, ['workspace', 'doctor', '--repair'])
    assert result.get('ok', True)
    assert experiments_dir.exists()
    repaired = result.get('result', {}).get('repaired', [])
    assert 'experiments_dir' in repaired

  def test_repair_creates_multiple_missing_dirs(self, tmp_path: Path) -> None:
    """All missing workspace directories are recreated in one --repair call."""
    ws = _init_workspace(tmp_path)
    for subdir in ('experiments', 'records', 'datasets'):
      (ws / '.autopilot' / subdir).rmdir()

    result = run_cli(ws, ['workspace', 'doctor', '--repair'])
    repaired = result.get('result', {}).get('repaired', [])
    for name in ('experiments_dir', 'records_dir', 'datasets_dir'):
      assert name in repaired
    for subdir in ('experiments', 'records', 'datasets'):
      assert (ws / '.autopilot' / subdir).exists()


class TestRepairJsonOutput:
  """test_repair_json_output -- JSON includes diagnostics."""

  def test_store_doctor_repair_json_has_diagnostics(self, tmp_path: Path) -> None:
    """store doctor --repair --json includes diagnostics and repaired lists."""
    ws = _init_workspace(tmp_path)
    _init_store_with_orphan(ws)

    result = run_cli(ws, ['store', 'doctor', '--repair'])
    payload = result.get('result', {})
    assert 'diagnostics' in payload
    assert isinstance(payload['diagnostics'], list)
    assert 'repaired' in payload

    repaired_codes = [r['code'] for r in payload['repaired']]
    assert 'orphan_blob' in repaired_codes

  def test_workspace_doctor_repair_json_has_repaired(self, tmp_path: Path) -> None:
    """workspace doctor --repair --json includes repaired list."""
    ws = _init_workspace(tmp_path)
    (ws / '.autopilot' / 'experiments').rmdir()

    result = run_cli(ws, ['workspace', 'doctor', '--repair'])
    payload = result.get('result', {})
    assert 'repaired' in payload
    assert 'experiments_dir' in payload['repaired']


class TestRepairNoIssues:
  """test_repair_no_issues -- clean store/workspace exits 0."""

  def test_store_doctor_repair_clean(self, tmp_path: Path) -> None:
    """Clean store with --repair exits 0 with healthy=True."""
    ws = _init_workspace(tmp_path)
    prompts = ws / 'prompts'
    prompts.mkdir()
    (prompts / 'main.txt').write_text('hello')

    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)
    param = PathParameter(source=str(prompts), pattern='*.txt')
    store.register_parameters({'source': param})
    store.snapshot('exp-001', 0)
    FileForest(store).save()

    result = run_cli(ws, ['store', 'doctor', '--repair'])
    payload = result.get('result', {})
    assert payload.get('healthy') is True
    assert payload.get('repaired') == []

  def test_workspace_doctor_repair_clean(self, tmp_path: Path) -> None:
    """Clean workspace with --repair exits 0 with healthy=True."""
    ws = _init_workspace(tmp_path)
    result = run_cli(ws, ['workspace', 'doctor', '--repair'])
    payload = result.get('result', {})
    assert payload.get('healthy') is True
    assert payload.get('repaired') == []


class TestRepairExitCodeSuccess:
  """test_repair_exit_code_success -- all repairs succeed -> exit 0."""

  def test_all_repairs_succeed(self, tmp_path: Path) -> None:
    """When all repairable issues are fixed, the result indicates success."""
    ws = _init_workspace(tmp_path)
    _init_store_with_orphan(ws)

    result = run_cli(ws, ['store', 'doctor', '--repair'])
    assert result.get('ok') is True
    payload = result.get('result', {})
    assert len(payload.get('repaired', [])) >= 1


class TestRepairContextRequired:
  """test_repair_requires_context -- --repair without --context fails."""

  def test_store_doctor_repair_without_context_fails(self, tmp_path: Path) -> None:
    """store doctor --repair without --context produces an error."""
    ws = _init_workspace(tmp_path)
    prompts = ws / 'prompts'
    prompts.mkdir()
    (prompts / 'main.txt').write_text('hello')

    config = AutoPilotConfig(workspace=ws)
    store = FileStore(config)
    param = PathParameter(source=str(prompts), pattern='*.txt')
    store.register_parameters({'source': param})
    store.snapshot('exp-001', 0)
    FileForest(store).save()

    with pytest.raises(SystemExit):
      run_cli_no_context(ws, ['store', 'doctor', '--repair'])

  def test_workspace_doctor_repair_without_context_fails(self, tmp_path: Path) -> None:
    """workspace doctor --repair without --context produces an error."""
    ws = _init_workspace(tmp_path)
    (ws / '.autopilot' / 'experiments').rmdir()

    with pytest.raises(SystemExit):
      run_cli_no_context(ws, ['workspace', 'doctor', '--repair'])


class TestRepairDryRunCli:
  """test_repair_dry_run_cli -- --repair --dry-run previews without mutating."""

  def test_store_doctor_dry_run_does_not_delete(self, tmp_path: Path) -> None:
    """store doctor --repair --dry-run reports but does not delete orphans."""
    ws = _init_workspace(tmp_path)
    _, orphan_digest = _init_store_with_orphan(ws)

    config = AutoPilotConfig(workspace=ws)
    shard = config.objects_path / orphan_digest[:2]
    orphan_file = shard / orphan_digest[2:]
    assert orphan_file.exists()

    result = run_cli(ws, ['store', 'doctor', '--repair', '--dry-run'])
    payload = result.get('result', {})
    assert payload.get('dry_run') is True
    assert len(payload.get('repaired', [])) >= 1
    assert orphan_file.exists()

  def test_workspace_doctor_dry_run_does_not_create_dirs(self, tmp_path: Path) -> None:
    """workspace doctor --repair --dry-run does not create missing dirs."""
    ws = _init_workspace(tmp_path)
    experiments_dir = ws / '.autopilot' / 'experiments'
    experiments_dir.rmdir()

    result = run_cli(ws, ['workspace', 'doctor', '--repair', '--dry-run'])
    payload = result.get('result', {})
    assert payload.get('dry_run') is True
    assert 'experiments_dir' in payload.get('repaired', [])
    assert not experiments_dir.exists()


class TestRepairExitCodePartial:
  """test_repair_exit_code_partial -- non-zero when repair path fails."""

  def test_repair_fails_on_permission_error_in_store(self, tmp_path: Path) -> None:
    """StoreError during repair produces non-zero exit via ctx.fail."""
    ws = _init_workspace(tmp_path)
    _init_store_with_orphan(ws)

    with (
      patch(
        'autopilot.ai.store.file_store.FileStore.repair_diagnostics',
        side_effect=StoreError('simulated repair failure'),
      ),
      pytest.raises(SystemExit),
    ):
      run_cli(ws, ['store', 'doctor', '--repair'])


class TestRepairPermissionError:
  """test_repair_permission_error -- permission denied surfaced as failure."""

  def test_permission_error_surfaces_as_json_failure(self, tmp_path: Path) -> None:
    """Permission error during orphan repair surfaces as CLI failure."""
    ws = _init_workspace(tmp_path)
    _init_store_with_orphan(ws)

    with (
      patch(
        'autopilot.ai.store.file_store.FileStore.repair_diagnostics',
        side_effect=StoreError('permission denied: cannot delete orphan blob'),
      ),
      pytest.raises(SystemExit),
    ):
      run_cli(ws, ['store', 'doctor', '--repair'])
