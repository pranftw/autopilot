"""Tests for the ``autopilot track`` command (Plan 21).

The track command runs arbitrary shell commands under dispatch capture and
records one ``ExecutionRecord`` row in ``executions.jsonl``. All tests use
``unittest.mock.patch`` to replace ``subprocess.run`` -- no real subprocess
is ever spawned inside the test suite.
"""

from autopilot.cli.commands.track import TrackCommand
from autopilot.cli.main import AutoPilotCLI, build_parser
from autopilot.core.config import AutoPilotConfig
from autopilot.tracking.executions import create_execution_record, log_execution
from autopilot.tracking.io import parse_timestamp
from pathlib import Path
from tests.cli.conftest import make_ctx, run_cli_no_context
from unittest.mock import MagicMock, patch
import argparse
import pytest


@pytest.fixture
def ws(tmp_path: Path) -> Path:
  """Minimal workspace for track command tests."""
  workspace = tmp_path / 'ws'
  workspace.mkdir()
  return workspace


class TestTrackSimpleCommand:
  """2.1 — success path via patched subprocess.run."""

  def test_track_simple_command(self, ws: Path) -> None:
    """Handler calls subprocess.run with shell=False and correct argv, then raises SystemExit(0)."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with patch('autopilot.cli.commands.track.subprocess.run', return_value=mock_proc) as mock_run:
      cmd = TrackCommand()
      ctx = make_ctx(ws)
      ns = argparse.Namespace(user_argv=['--', 'echo', 'hello'])

      with pytest.raises(SystemExit) as exc_info:
        cmd.forward(ctx, ns)

      assert exc_info.value.code == 0
      mock_run.assert_called_once_with(
        ['echo', 'hello'],
        shell=False,
        check=False,
      )

  def test_track_preserves_failing_exit_code(self, ws: Path) -> None:
    """Nonzero child return code propagates as SystemExit(returncode)."""
    mock_proc = MagicMock()
    mock_proc.returncode = 42

    with patch('autopilot.cli.commands.track.subprocess.run', return_value=mock_proc) as mock_run:
      cmd = TrackCommand()
      ctx = make_ctx(ws)
      ns = argparse.Namespace(user_argv=['--', 'false'])

      with pytest.raises(SystemExit) as exc_info:
        cmd.forward(ctx, ns)

      assert exc_info.value.code == 42
      mock_run.assert_called_once_with(
        ['false'],
        shell=False,
        check=False,
      )

  def test_track_strips_leading_separator(self, ws: Path) -> None:
    """Leading ``--`` is stripped before passing to subprocess."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with patch('autopilot.cli.commands.track.subprocess.run', return_value=mock_proc) as mock_run:
      cmd = TrackCommand()
      ctx = make_ctx(ws)
      ns = argparse.Namespace(user_argv=['--', 'ls', '-la'])

      with pytest.raises(SystemExit):
        cmd.forward(ctx, ns)

      mock_run.assert_called_once_with(
        ['ls', '-la'],
        shell=False,
        check=False,
      )

  def test_track_no_command_fails(self, ws: Path) -> None:
    """Empty argv after stripping ``--`` calls ctx.fail."""
    cmd = TrackCommand()
    ctx = make_ctx(ws)
    ns = argparse.Namespace(user_argv=['--'])

    with pytest.raises(SystemExit):
      cmd.forward(ctx, ns)

  def test_track_empty_argv_fails(self, ws: Path) -> None:
    """Completely empty user_argv calls ctx.fail."""
    cmd = TrackCommand()
    ctx = make_ctx(ws)
    ns = argparse.Namespace(user_argv=[])

    with pytest.raises(SystemExit):
      cmd.forward(ctx, ns)

  def test_track_without_separator(self, ws: Path) -> None:
    """Argv without leading ``--`` is forwarded as-is."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with patch('autopilot.cli.commands.track.subprocess.run', return_value=mock_proc) as mock_run:
      cmd = TrackCommand()
      ctx = make_ctx(ws)
      ns = argparse.Namespace(user_argv=['git', 'status'])

      with pytest.raises(SystemExit):
        cmd.forward(ctx, ns)

      mock_run.assert_called_once_with(
        ['git', 'status'],
        shell=False,
        check=False,
      )


class TestTrackRequiresContext:
  """2.2 — context enforcement via CLI dispatch."""

  def test_track_requires_context(self, ws: Path) -> None:
    """Track without --context must fail with '--context is required' message.

    Context enforcement lives in ``CLI.dispatch``, so we verify the CLI
    recognizes track as requiring context and that the handler is not
    exempt.
    """
    cli = AutoPilotCLI()
    assert cli.requires_context('track')
    assert 'track' not in cli.context_exempt_commands

  def test_track_is_not_context_exempt(self) -> None:
    """Track must not appear in _BASE_CONTEXT_EXEMPT."""
    cli = AutoPilotCLI()
    assert cli.requires_context('track')


class TestTrackTimestampAndAudit:
  """2.3 — timestamp format and debug visibility."""

  def test_track_record_timestamp_is_iso_utc(self, ws: Path) -> None:
    """ExecutionRecord timestamp parses as valid ISO 8601 UTC."""
    record = create_execution_record(
      command='track',
      args=['echo', 'hello'],
      duration_ms=100.0,
      exit_code=0,
      context='test tracking',
    )
    ts = parse_timestamp(record.timestamp)
    assert ts.year >= 2024
    assert ts.tzinfo is not None

  def test_debug_executions_list_includes_track(self, ws: Path) -> None:
    """After writing a track record, debug executions list shows it."""
    config = AutoPilotConfig(workspace=ws)
    record = create_execution_record(
      command='track',
      args=['echo', 'hello'],
      duration_ms=50.0,
      exit_code=0,
      context='test track audit',
    )
    log_execution(config.executions_path, record)

    result = run_cli_no_context(ws, ['debug', 'executions', 'list'])
    assert 'result' in result
    executions = result['result']['executions']
    commands = [r['command'] for r in executions]
    assert 'track' in commands


class TestTrackRegistration:
  """Verify TrackCommand is properly wired into AutoPilotCLI."""

  def test_track_registered_on_cli(self) -> None:
    """AutoPilotCLI has a track command."""
    cli = AutoPilotCLI()
    assert hasattr(cli, 'track')
    assert isinstance(cli.track, TrackCommand)

  def test_track_in_parser(self) -> None:
    """build_parser includes 'track' as a valid subcommand."""
    parser = build_parser()
    with patch('autopilot.cli.commands.track.subprocess.run') as mock_run:
      mock_proc = MagicMock()
      mock_proc.returncode = 0
      mock_run.return_value = mock_proc

      args = parser.parse_args(
        [
          'track',
          '--',
          'echo',
          'test',
          '--workspace',
          '/tmp/fake',
          '--context',
          'test',
        ]
      )
      assert args.command == 'track'
      assert args.handler is not None

  def test_track_capture_output_not_nested(self, ws: Path) -> None:
    """Handler does not call capture_output (relies on dispatch wrapper)."""
    mock_proc = MagicMock()
    mock_proc.returncode = 0

    with (
      patch('autopilot.cli.commands.track.subprocess.run', return_value=mock_proc),
      patch('autopilot.tracking.executions.capture_output') as mock_capture,
    ):
      cmd = TrackCommand()
      ctx = make_ctx(ws)
      ns = argparse.Namespace(user_argv=['--', 'echo', 'hi'])

      with pytest.raises(SystemExit):
        cmd.forward(ctx, ns)

      mock_capture.assert_not_called()
