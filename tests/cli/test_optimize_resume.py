"""Tests for optimize resume CLI subcommand.

Covers checkpoint path forwarding, missing file error, and JSON envelope.
"""

from autopilot.cli.commands.optimize import OptimizeCommand, Resume
from pathlib import Path
from tests.cli.conftest import make_mock_cli_context
from unittest.mock import MagicMock, patch
import json
import pytest


def _make_ctx(tmp_path: Path, use_json: bool = True) -> MagicMock:
  """Build a mock CLIContext with resume-specific trainer defaults."""
  trainer = MagicMock()
  trainer.fit.return_value = {
    'epochs': [{'epoch': 0, 'metrics': {}}, {'epoch': 1, 'metrics': {}}],
    'total_epochs': 2,
  }
  return make_mock_cli_context(
    tmp_path,
    use_json=use_json,
    module=MagicMock(),
    datamodule=None,
    trainer=trainer,
  )


class TestOptimizeResumeCommand:
  def test_instantiates(self) -> None:
    cmd = OptimizeCommand()
    assert hasattr(cmd, 'resume')
    assert cmd.resume.name == 'resume'

  def test_resume_forwards_ckpt_path(self, tmp_path: Path, capsys) -> None:
    """Patched Trainer.fit receives ckpt_path pathlib argument."""
    ckpt = tmp_path / 'checkpoint.json'
    ckpt.write_text('{}')

    ctx = _make_ctx(tmp_path)
    args = MagicMock()
    args.ckpt = str(ckpt)
    args.max_epochs = 5

    cmd = Resume()
    with patch('autopilot.cli.commands.optimize.Path.cwd', return_value=tmp_path):
      cmd.forward(ctx, args)

    ctx.trainer.fit.assert_called_once()
    call_kwargs = ctx.trainer.fit.call_args
    assert call_kwargs.kwargs['ckpt_path'] == ckpt.resolve()
    assert call_kwargs.kwargs['max_epochs'] == 5

  def test_missing_checkpoint_fails(self, tmp_path: Path) -> None:
    """Exit non-zero / ctx.fail path for missing checkpoint."""
    ctx = _make_ctx(tmp_path)
    ctx.output = MagicMock()
    args = MagicMock()
    args.ckpt = str(tmp_path / 'nonexistent.json')
    args.max_epochs = 5

    cmd = Resume()
    with (
      patch('autopilot.cli.commands.optimize.Path.cwd', return_value=tmp_path),
      pytest.raises(SystemExit) as exc_info,
    ):
      cmd.forward(ctx, args)
    assert exc_info.value.code == 1

  def test_json_envelope(self, tmp_path: Path, capsys) -> None:
    """JSON envelope includes ok and result.resumed_from."""
    ckpt = tmp_path / 'ckpt.json'
    ckpt.write_text('{}')

    ctx = _make_ctx(tmp_path, use_json=True)
    args = MagicMock()
    args.ckpt = str(ckpt)
    args.max_epochs = 3

    cmd = Resume()
    with patch('autopilot.cli.commands.optimize.Path.cwd', return_value=tmp_path):
      cmd.forward(ctx, args)

    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert 'resumed_from' in envelope['result']
    assert str(ckpt.resolve()) in envelope['result']['resumed_from']
