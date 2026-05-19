"""End-to-end workspace lifecycle integration tests.

Covers: full init-to-deploy sequence, tree name validation, and empty
workspace status behavior.
"""

from autopilot.core.forest import validate_tree_name
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context
import contextlib
import io
import pytest


def test_full_workspace_lifecycle(tmp_path: Path) -> None:
  """Six-step lifecycle: init, tree create, experiment add, complete, deploy, query."""
  ws = tmp_path / 'ws'
  ws.mkdir()

  run_cli(ws, ['workspace', 'init'])

  run_cli(ws, ['tree', 'create', 'main'])

  run_cli(ws, ['experiment', 'add', '--id', 'lifecycle-exp', '--hypothesis', 'lifecycle test'])

  run_cli(
    ws,
    ['experiment', 'complete', 'lifecycle-exp', '--metrics', '{"accuracy": 0.85}'],
  )

  run_cli(ws, ['experiment', 'deploy', 'lifecycle-exp', '--as', 'production'])

  result = run_cli_no_context(ws, ['query', '--deployed'])
  assert result['ok']
  ids = {r['id'] for r in result['result']['experiments']}
  assert 'lifecycle-exp' in ids


def test_tree_create_unicode_rejected(tmp_path: Path) -> None:
  """Non-ASCII tree name is rejected by validate_tree_name."""
  with pytest.raises(ValueError, match='invalid characters'):
    validate_tree_name('t\u00ebst')


def test_tree_create_slash_rejected(tmp_path: Path) -> None:
  """Slash in tree name fails validation."""
  with pytest.raises(ValueError, match='invalid characters'):
    validate_tree_name('feat/branch')


def test_tree_create_unicode_rejected_cli(tmp_path: Path) -> None:
  """Non-ASCII tree name fails via CLI with SystemExit."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])

  with pytest.raises((SystemExit, ValueError)):
    run_cli(ws, ['tree', 'create', 't\u00ebst'])


def test_tree_create_slash_rejected_cli(tmp_path: Path) -> None:
  """Slash in tree name fails via CLI with SystemExit."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])

  with pytest.raises((SystemExit, ValueError)):
    run_cli(ws, ['tree', 'create', 'feat/branch'])


def test_empty_workspace_status(tmp_path: Path) -> None:
  """Uninitialized workspace yields unhealthy status with SystemExit(1), no traceback."""
  ws = tmp_path / 'empty_ws'
  ws.mkdir()

  stderr_buf = io.StringIO()
  with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stderr(stderr_buf):
    run_cli_no_context(ws, ['workspace', 'status'])

  assert exc_info.value.code == 1
  stderr_output = stderr_buf.getvalue()
  assert 'Traceback' not in stderr_output
