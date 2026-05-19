"""Tests for workspace status health-driven ``ok`` field (Plan 04).

Covers BFR-05: workspace status top-level ``ok`` reflects composite health
from workspace_doctor and store_doctor. Also covers 2.4: store_doctor
with forest errors propagates unhealthy through workspace status.
"""

from autopilot.ai.store.file_store import FileStore
from autopilot.cli.commands.workspace import _workspace_status_overall_ok
from autopilot.cli.context import build_context
from autopilot.cli.main import build_parser
from autopilot.core.config import AutoPilotConfig
from pathlib import Path
from typing import Any
import contextlib
import io
import json


def _status_envelope(workspace: Path) -> dict[str, Any]:
  """Run workspace status and return the full JSON envelope.

  Catches ``SystemExit`` raised by unhealthy workspaces so the captured
  stdout envelope is always returned.
  """
  parser = build_parser()
  full_argv = ['workspace', 'status', '--workspace', str(workspace), '--json']
  parsed = parser.parse_args(full_argv)
  ctx = build_context(parsed)

  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
    parsed.handler(ctx, parsed)

  output = buf.getvalue().strip()
  if output:
    return json.loads(output)
  return {}


def _status_payload(workspace: Path) -> dict[str, Any]:
  """Run workspace status and return the inner result payload."""
  envelope = _status_envelope(workspace)
  return envelope.get('result', envelope)


class TestWorkspaceStatusOverallOkHelper:
  """Unit tests for ``_workspace_status_overall_ok``."""

  def test_healthy_workspace_returns_true(self) -> None:
    """Fully healthy payload yields ok=True."""
    payload: dict[str, Any] = {
      'health': {
        'workspace_doctor': {'healthy': True, 'checks': {}},
      }
    }
    assert _workspace_status_overall_ok(payload) is True

  def test_unhealthy_workspace_doctor_returns_false(self) -> None:
    """Unhealthy workspace_doctor yields ok=False."""
    payload: dict[str, Any] = {
      'health': {
        'workspace_doctor': {'healthy': False, 'checks': {}},
      }
    }
    assert _workspace_status_overall_ok(payload) is False

  def test_unhealthy_store_doctor_returns_false(self) -> None:
    """Unhealthy store_doctor yields ok=False."""
    payload: dict[str, Any] = {
      'health': {
        'workspace_doctor': {'healthy': True, 'checks': {}},
        'store_doctor': {'healthy': False},
      }
    }
    assert _workspace_status_overall_ok(payload) is False

  def test_absent_store_doctor_does_not_fail(self) -> None:
    """Missing store_doctor does not cause ok=False."""
    payload: dict[str, Any] = {
      'health': {
        'workspace_doctor': {'healthy': True, 'checks': {}},
      }
    }
    assert _workspace_status_overall_ok(payload) is True

  def test_both_healthy_returns_true(self) -> None:
    """Both doctors healthy yields ok=True."""
    payload: dict[str, Any] = {
      'health': {
        'workspace_doctor': {'healthy': True, 'checks': {}},
        'store_doctor': {'healthy': True},
      }
    }
    assert _workspace_status_overall_ok(payload) is True

  def test_missing_health_section_returns_false(self) -> None:
    """Payload without health defaults to ok=False (no workspace_doctor)."""
    payload: dict[str, Any] = {}
    assert _workspace_status_overall_ok(payload) is False


class TestWorkspaceStatusOkFalseOnUnhealthyLayout:
  """Integration: missing .autopilot/ layout causes ok=False in JSON envelope."""

  def test_missing_layout_envelope_ok_false(self, tmp_path: Path) -> None:
    """Missing workspace layout causes JSON envelope ok=False."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    envelope = _status_envelope(ws)
    assert envelope.get('ok') is False

  def test_missing_layout_health_unhealthy(self, tmp_path: Path) -> None:
    """Missing workspace layout causes workspace_doctor.healthy=False."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    result = _status_payload(ws)
    assert result['health']['workspace_doctor']['healthy'] is False


class TestWorkspaceStatusOkTrueOnHealthyWorkspace:
  """Integration: healthy workspace has ok=True."""

  def test_healthy_workspace_envelope_ok_true(self, tmp_path: Path) -> None:
    """Fully initialized workspace reports ok=True."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.init_workspace()
    config.store_path.mkdir(parents=True, exist_ok=True)
    store = FileStore(config)
    from autopilot.ai.forest import FileForest

    forest = FileForest(store)
    forest.save()

    envelope = _status_envelope(ws)
    assert envelope.get('ok') is True


class TestStatusOkFalseWhenStoreDoctorUnhealthy:
  """Integration: corrupt forest.json makes store_doctor unhealthy -> ok=False."""

  def test_status_ok_false_when_store_doctor_unhealthy(self, tmp_path: Path) -> None:
    """Corrupt forest.json causes store_doctor.healthy=False, envelope ok=False."""
    ws = tmp_path / 'ws'
    ws.mkdir()
    config = AutoPilotConfig(workspace=ws)
    config.init_workspace()
    config.store_path.mkdir(parents=True, exist_ok=True)
    config.forest_file.write_text('{not valid json!!!', encoding='utf-8')

    envelope = _status_envelope(ws)
    assert envelope.get('ok') is False

    result = envelope.get('result', envelope)
    store_doctor = result['health'].get('store_doctor', {})
    assert store_doctor.get('healthy') is False
