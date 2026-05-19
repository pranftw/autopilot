"""CLI consistency regression tests (plan 09).

Covers:
  - propose verify context enforcement and help text (P1#15)
  - proposal-id numeric guard (P1#17)
  - track JSON envelope (P2#24)
  - dry-run context exemption (P2#29)
"""

from autopilot.ai.proposal import ChangeProposal, record_proposal
from autopilot.cli.commands.propose import ProposeCommand
from autopilot.cli.context import CLIContext
from autopilot.cli.main import AutoPilotCLI
from autopilot.cli.output import Output
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import argparse
import contextlib
import io
import json
import pytest


def _make_propose_ctx(
  tmp_path: Path,
  *,
  experiment: str = 'test-exp',
  use_json: bool = True,
) -> MagicMock:
  """Build a mock CLIContext for propose tests."""
  ctx = MagicMock()
  ctx.experiment = experiment
  ctx.epoch = 1
  ctx.workspace = tmp_path
  ctx.project = None
  ctx.output = Output(use_json=use_json)
  ctx.fail = CLIContext.fail.__get__(ctx, type(ctx))
  exp_dir = tmp_path / experiment
  exp_dir.mkdir(parents=True, exist_ok=True)
  ctx.experiment_path.return_value = exp_dir
  return ctx


def _seed_proposal(
  exp_dir: Path,
  proposal_id: str = 'a1b2c3d4',
  epoch: int = 0,
) -> None:
  """Seed a proposal artifact on disk."""
  record_proposal(
    exp_dir,
    ChangeProposal(
      proposal_id=proposal_id,
      hypothesis='test hypothesis',
      target_node='accuracy',
      change_type='rule_change',
      epoch=epoch,
      status='proposed',
    ),
  )


def _run_dispatch_no_context(
  workspace: Path,
  argv: list[str],
) -> tuple[int, str]:
  """Run CLI through dispatch (exercises context enforcement) without --context.

  Global flags are prepended before command tokens so argparse
  REMAINDER (used by ``track``) does not swallow them.

  Returns:
    Tuple of (exit_code, captured stdout).
  """
  cli = AutoPilotCLI()
  parser = cli.build_parser()
  full_argv = ['--workspace', str(workspace), '--json', *argv]
  parsed = parser.parse_args(full_argv)
  ctx = cli.build_context(parsed)

  buf = io.StringIO()
  exit_code = 0
  with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
    try:
      cli.dispatch(ctx, parsed, argv=full_argv)
    except SystemExit as exc:
      exit_code = exc.code if isinstance(exc.code, int) else 1

  return exit_code, buf.getvalue()


def _run_dispatch_with_context(
  workspace: Path,
  argv: list[str],
) -> tuple[int, str]:
  """Run CLI through dispatch with --context 'test'.

  Global flags are prepended before command tokens so argparse
  REMAINDER (used by ``track``) does not swallow them.

  Returns:
    Tuple of (exit_code, captured stdout).
  """
  cli = AutoPilotCLI()
  parser = cli.build_parser()
  full_argv = ['--workspace', str(workspace), '--json', '--context', 'test', *argv]
  parsed = parser.parse_args(full_argv)
  ctx = cli.build_context(parsed)

  buf = io.StringIO()
  exit_code = 0
  with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
    try:
      cli.dispatch(ctx, parsed, argv=full_argv)
    except SystemExit as exc:
      exit_code = exc.code if isinstance(exc.code, int) else 1

  return exit_code, buf.getvalue()


class TestProposeVerifyContext:
  """P1#15: propose verify requires --context (mutating)."""

  def test_propose_verify_requires_context(self, tmp_path: Path) -> None:
    """Invoke propose verify without --context; expect rejection."""
    exp_dir = tmp_path / 'ws' / '.autopilot' / 'experiments' / 'exp-a'
    exp_dir.mkdir(parents=True)
    _seed_proposal(exp_dir)

    ws = tmp_path / 'ws'
    exit_code, output = _run_dispatch_no_context(
      ws,
      ['propose', 'verify', '--proposal-id', 'a1b2c3d4', '--experiment', 'exp-a', '--epoch', '0'],
    )
    assert exit_code != 0
    assert '--context' in output.lower() or 'context' in output.lower()

  def test_propose_verify_help_mentions_mutating(self) -> None:
    """Help text for propose verify mentions mutation and --context."""
    meta = getattr(ProposeCommand.verify, 'subcommand_meta', None)
    assert meta is not None
    help_text = meta.help
    assert help_text is not None
    assert 'mutating' in help_text
    assert '--context' in help_text


class TestProposalIdNumeric:
  """P1#17: numeric proposal-id gives a helpful error."""

  def test_proposal_id_numeric_helpful_error(self, tmp_path: Path) -> None:
    """Pass a pure-decimal --proposal-id; expect actionable error mentioning hex."""
    ctx = _make_propose_ctx(tmp_path)
    exp_dir = ctx.experiment_path()
    _seed_proposal(exp_dir)

    cmd = ProposeCommand()
    args = argparse.Namespace(
      proposal_id='0',
      higher_is_better=None,
      lower_is_better=None,
    )

    buf = io.StringIO()
    with pytest.raises(SystemExit) as exc_info, contextlib.redirect_stdout(buf):
      cmd.verify(ctx, args)
    assert exc_info.value.code != 0
    captured = buf.getvalue()
    assert 'numeric' in captured.lower() or 'hex' in captured.lower()
    assert 'propose list' in captured

  def test_proposal_id_hex_accepted(self, tmp_path: Path, capsys: Any) -> None:
    """Valid 8-char hex ID passes the numeric guard and reaches lookup."""
    ctx = _make_propose_ctx(tmp_path)
    exp_dir = ctx.experiment_path()
    _seed_proposal(exp_dir, proposal_id='a1b2c3d4', epoch=0)

    (exp_dir / 'epoch_0_metrics.json').write_text(json.dumps({'accuracy': 0.9}), encoding='utf-8')
    evaluation_dir = exp_dir / 'evaluation'
    evaluation_dir.mkdir(exist_ok=True)

    cmd = ProposeCommand()
    args = argparse.Namespace(
      proposal_id='a1b2c3d4',
      higher_is_better=None,
      lower_is_better=None,
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      cmd.verify(ctx, args)

    captured = buf.getvalue()
    envelope = json.loads(captured)
    assert envelope['ok'] is True
    assert envelope['result']['proposal_id'] == 'a1b2c3d4'


class TestTrackJsonEnvelope:
  """P2#24: track emits JSON envelope before SystemExit."""

  def test_track_json_envelope(self, tmp_path: Path) -> None:
    """track --json with a successful command emits ok=True envelope."""
    exit_code, output = _run_dispatch_with_context(
      tmp_path,
      ['track', '--', 'echo', 'hello'],
    )
    assert exit_code == 0
    lines = [line for line in output.strip().splitlines() if line.strip()]
    json_output = '\n'.join(lines)
    envelope = json.loads(json_output)
    assert envelope['ok'] is True
    result = envelope['result']
    assert result['exit_code'] == 0
    assert result['argv'] == ['echo', 'hello']

  def test_track_json_failure_envelope(self, tmp_path: Path) -> None:
    """track --json with a failing command emits ok=False envelope."""
    exit_code, output = _run_dispatch_with_context(
      tmp_path,
      ['track', '--', 'false'],
    )
    assert exit_code == 1
    lines = [line for line in output.strip().splitlines() if line.strip()]
    json_output = '\n'.join(lines)
    envelope = json.loads(json_output)
    assert envelope['ok'] is False
    result = envelope['result']
    assert result['exit_code'] == 1


class TestDryRunContextExemption:
  """P2#29: dry-run commands skip --context enforcement."""

  def test_checkout_dry_run_no_context(
    self, tmp_path: Path, workspace_with_store_and_forest: dict[str, Any]
  ) -> None:
    """store checkout --dry-run without --context should not fail on context."""
    ws_data = workspace_with_store_and_forest
    ws = ws_data['workspace']
    source_dir = ws / 'src'
    source_dir.mkdir(exist_ok=True)

    exit_code, output = _run_dispatch_no_context(
      ws,
      [
        'store',
        'checkout',
        '--experiment',
        'seed-exp',
        '--epoch',
        '0',
        '--source',
        str(source_dir),
        '--dry-run',
      ],
    )
    combined = output.lower()
    context_rejected = '--context' in combined and 'required' in combined
    assert not context_rejected, f'dry-run should skip context enforcement: {output}'
    if exit_code != 0:
      assert 'context' not in combined or 'required' not in combined, (
        f'dry-run failure must not be about --context enforcement: {output}'
      )

  def test_checkout_real_requires_context(
    self, tmp_path: Path, workspace_with_store_and_forest: dict[str, Any]
  ) -> None:
    """store checkout without --dry-run and without --context should fail."""
    ws_data = workspace_with_store_and_forest
    ws = ws_data['workspace']
    source_dir = ws / 'src'
    source_dir.mkdir(exist_ok=True)

    exit_code, output = _run_dispatch_no_context(
      ws,
      [
        'store',
        'checkout',
        '--experiment',
        'seed-exp',
        '--epoch',
        '0',
        '--source',
        str(source_dir),
      ],
    )
    assert exit_code != 0
    assert 'context' in output.lower()
