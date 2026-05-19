"""Tests for debug executions list --context-contains filtering."""

from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.executions import (
  ExecutionRecord,
  log_execution,
)
from autopilot.tracking.io import utc_now_iso
from pathlib import Path
from tests.cli.conftest import run_cli_no_context
import pytest


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Workspace with executions.jsonl containing records with varied context."""
  ws = tmp_path / 'ws'
  ws.mkdir()
  config = AutoPilotConfig(workspace=ws)
  config.autopilot_path.mkdir(parents=True, exist_ok=True)

  records = [
    ExecutionRecord(
      timestamp=utc_now_iso(),
      command='optimize train',
      args=['--max-epochs', '5'],
      duration_ms=1200.0,
      exit_code=0,
      context='initial training run',
    ),
    ExecutionRecord(
      timestamp=utc_now_iso(),
      command='experiment add',
      args=['--hypothesis', 'test'],
      duration_ms=50.0,
      exit_code=0,
      context='adding baseline experiment',
    ),
    ExecutionRecord(
      timestamp=utc_now_iso(),
      command='debug collect',
      args=[],
      duration_ms=300.0,
      exit_code=0,
      context=None,
    ),
    ExecutionRecord(
      timestamp=utc_now_iso(),
      command='optimize train',
      args=['--max-epochs', '10'],
      duration_ms=2500.0,
      exit_code=1,
      context='retry training after failure',
    ),
  ]
  for record in records:
    log_execution(config.executions_path, record)

  return ws


class TestDebugExecutionsContextContains:
  """Tests for --context-contains flag on debug executions list."""

  def test_debug_executions_context_contains_filters(self, ws: Path) -> None:
    """Records with context containing substring are listed; others excluded."""
    result = run_cli_no_context(
      ws, ['debug', 'executions', 'list', '--context-contains', 'training']
    )
    rows = result['result']['executions']
    assert len(rows) == 2
    commands = [r['command'] for r in rows]
    assert 'optimize train' in commands

  def test_debug_executions_context_contains_no_match(self, ws: Path) -> None:
    """No record contains substring; empty listing."""
    result = run_cli_no_context(
      ws, ['debug', 'executions', 'list', '--context-contains', 'zzz-no-match']
    )
    rows = result['result']['executions']
    assert len(rows) == 0

  def test_debug_executions_without_flag_unchanged(self, ws: Path) -> None:
    """Omitting --context-contains preserves pre-change listing."""
    result = run_cli_no_context(ws, ['debug', 'executions', 'list'])
    rows = result['result']['executions']
    assert len(rows) == 4

  def test_context_contains_with_none_context_excluded(self, ws: Path) -> None:
    """Records with context=None are excluded when --context-contains is used."""
    result = run_cli_no_context(ws, ['debug', 'executions', 'list', '--context-contains', 'debug'])
    rows = result['result']['executions']
    assert len(rows) == 0

  def test_context_contains_combined_with_command_filter(self, ws: Path) -> None:
    """--context-contains composes with --command filter."""
    result = run_cli_no_context(
      ws,
      [
        'debug',
        'executions',
        'list',
        '--context-contains',
        'training',
        '--command',
        'optimize train',
      ],
    )
    rows = result['result']['executions']
    assert len(rows) == 2
    for row in rows:
      assert row['command'] == 'optimize train'
