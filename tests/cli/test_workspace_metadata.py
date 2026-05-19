"""Tests for workspace and tree metadata (dogfood V3, sub-plan 06).

Covers:
- workspace.json persistence on ``workspace init``
- workspace status ``description`` field
- tree create description fallback from ``--context``
- workspace doctor forest.json validation
- workspace status exit code on unhealthy
"""

from autopilot.tracking.io import read_json_dict
from pathlib import Path
from tests.cli.conftest import run_cli, run_cli_no_context, run_cli_text
from typing import Any
import json
import pytest


def _status_payload(workspace: Path) -> dict[str, Any]:
  """Run workspace status and return the inner result payload."""
  envelope = run_cli_no_context(workspace, ['workspace', 'status'])
  return envelope.get('result', envelope)


def _doctor_payload(workspace: Path) -> dict[str, Any]:
  """Run workspace doctor and return the inner result payload."""
  envelope = run_cli_no_context(workspace, ['workspace', 'doctor'])
  return envelope.get('result', envelope)


# ---------------------------------------------------------------------------
# 2.1  Workspace metadata file and status
# ---------------------------------------------------------------------------


def test_workspace_init_persists_purpose(tmp_path: Path) -> None:
  """workspace init with --context writes .autopilot/workspace.json."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  wj = ws / '.autopilot' / 'workspace.json'
  assert wj.is_file()
  data = read_json_dict(wj, 'workspace.json')
  assert data['description'] == 'test'


def test_workspace_status_shows_purpose(tmp_path: Path) -> None:
  """After init with context, workspace status --json includes description."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  result = _status_payload(ws)
  assert result['description'] == 'test'


def test_workspace_status_no_purpose_null(tmp_path: Path) -> None:
  """Workspace without workspace.json yields description: null in status."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  (ws / '.autopilot').mkdir(parents=True)
  (ws / '.autopilot' / 'projects').mkdir()
  (ws / '.autopilot' / 'experiments').mkdir()
  (ws / '.autopilot' / 'records').mkdir()
  (ws / '.autopilot' / 'datasets').mkdir()
  result = _status_payload(ws)
  assert result['description'] is None


# ---------------------------------------------------------------------------
# 2.2  Tree create and context
# ---------------------------------------------------------------------------


def test_tree_create_context_as_description(tmp_path: Path) -> None:
  """tree create with --context and no --description uses context as description."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  run_cli(ws, ['tree', 'create', 'mytree'])
  envelope = run_cli_no_context(ws, ['tree', 'list'])
  result = envelope.get('result', envelope)
  trees = result['trees']
  assert len(trees) == 1
  assert trees[0]['name'] == 'mytree'
  assert trees[0]['description'] == 'test'


def test_tree_create_explicit_description_wins(tmp_path: Path) -> None:
  """--description overrides --context for tree description."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  run_cli(ws, ['tree', 'create', 'mytree', '--description', 'explicit desc'])
  envelope = run_cli_no_context(ws, ['tree', 'list'])
  result = envelope.get('result', envelope)
  trees = result['trees']
  assert len(trees) == 1
  assert trees[0]['description'] == 'explicit desc'


# ---------------------------------------------------------------------------
# 2.3  Doctor and exit codes
# ---------------------------------------------------------------------------


def test_workspace_doctor_corrupt_forest(tmp_path: Path) -> None:
  """Invalid JSON at forest_file yields healthy: false and forest_error."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  store_path = ws / '.autopilot' / 'store'
  store_path.mkdir(parents=True, exist_ok=True)
  forest_file = store_path / 'forest.json'
  forest_file.write_text('NOT VALID JSON', encoding='utf-8')
  result = _doctor_payload(ws)
  assert result['healthy'] is False
  assert 'forest_json' in result['issues']
  assert 'forest_error' in result


def test_workspace_doctor_valid_forest(tmp_path: Path) -> None:
  """Valid JSON object at forest_file passes the forest parse check."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  store_path = ws / '.autopilot' / 'store'
  store_path.mkdir(parents=True, exist_ok=True)
  forest_file = store_path / 'forest.json'
  forest_file.write_text(json.dumps({'trees': {}}), encoding='utf-8')
  result = _doctor_payload(ws)
  assert result['healthy'] is True
  assert 'forest_json' not in result['issues']
  assert result['checks']['forest_json'] is True


def test_workspace_status_exit_code_on_unhealthy(tmp_path: Path) -> None:
  """Unhealthy workspace status raises SystemExit(1)."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  with pytest.raises(SystemExit) as exc_info:
    run_cli_no_context(ws, ['workspace', 'status'])
  assert exc_info.value.code == 1


def test_workspace_status_exit_code_on_healthy(tmp_path: Path) -> None:
  """Healthy workspace exits 0 (no SystemExit raised)."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  result = _status_payload(ws)
  assert result['health']['workspace_doctor']['healthy'] is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_workspace_status_corrupt_workspace_json(tmp_path: Path) -> None:
  """Corrupt workspace.json does not break status; description is null."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  wj = ws / '.autopilot' / 'workspace.json'
  wj.write_text('BROKEN', encoding='utf-8')
  result = _status_payload(ws)
  assert result['description'] is None


def test_workspace_status_text_shows_purpose(tmp_path: Path) -> None:
  """Text output includes 'Purpose: ...' when description is set."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  text = run_cli_text(ws, ['workspace', 'status'])
  assert 'Purpose: test' in text


def test_workspace_doctor_forest_json_not_dict(tmp_path: Path) -> None:
  """forest.json that is valid JSON but not an object is flagged."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  store_path = ws / '.autopilot' / 'store'
  store_path.mkdir(parents=True, exist_ok=True)
  forest_file = store_path / 'forest.json'
  forest_file.write_text('["not", "a", "dict"]', encoding='utf-8')
  result = _doctor_payload(ws)
  assert result['healthy'] is False
  assert 'forest_json' in result['issues']
  assert 'forest_error' in result


def test_workspace_status_health_agrees_with_doctor_on_corrupt_forest(
  tmp_path: Path,
) -> None:
  """workspace status health section agrees with doctor on corrupt forest."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  run_cli(ws, ['workspace', 'init'])
  store_path = ws / '.autopilot' / 'store'
  store_path.mkdir(parents=True, exist_ok=True)
  forest_file = store_path / 'forest.json'
  forest_file.write_text('NOT VALID JSON', encoding='utf-8')

  doctor_result = _doctor_payload(ws)

  with pytest.raises(SystemExit):
    run_cli_no_context(ws, ['workspace', 'status'])

  assert doctor_result['healthy'] is False
