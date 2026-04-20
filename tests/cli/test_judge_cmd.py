"""Tests for judge CLI command."""

from autopilot.cli.commands.judge_cmd import JudgeCommand
from autopilot.cli.output import Output
from autopilot.core.stage_io import append_epoch_artifact
from pathlib import Path
from unittest.mock import MagicMock
import json


def _make_ctx(tmp_path: Path, experiment: str = 'test-exp') -> MagicMock:
  ctx = MagicMock()
  ctx.experiment = experiment
  ctx.epoch = 1
  ctx.output = Output(use_json=True)
  ctx.judge = MagicMock()
  exp_dir = tmp_path / experiment
  exp_dir.mkdir(parents=True, exist_ok=True)
  ctx.experiment_dir.return_value = exp_dir
  return ctx


class TestJudgeCommand:
  def test_instantiates(self):
    cmd = JudgeCommand()
    assert cmd.name == 'judge'

  def test_run_no_judge(self):
    cmd = JudgeCommand()
    ctx = MagicMock()
    ctx.judge = None
    ctx.output = MagicMock()
    args = MagicMock(epoch=1, validate=False)
    cmd.run_judge(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_run_with_judge(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    cmd = JudgeCommand()
    args = MagicMock(epoch=1, validate=False)
    cmd.run_judge(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['status'] == 'completed'

  def test_distribution_no_epoch(self, tmp_path):
    ctx = _make_ctx(tmp_path)
    ctx.epoch = 0
    ctx.output = MagicMock()
    cmd = JudgeCommand()
    args = MagicMock(epoch=0)
    cmd.distribution(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_distribution_happy_path(self, tmp_path, capsys):
    ctx = _make_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    append_epoch_artifact(
      exp_dir,
      1,
      'data.jsonl',
      {
        'success': False,
        'metadata': {'failure_type': 'hallucination'},
      },
    )
    append_epoch_artifact(
      exp_dir,
      1,
      'data.jsonl',
      {
        'success': True,
        'metadata': {},
      },
    )
    cmd = JudgeCommand()
    args = MagicMock(epoch=1)
    cmd.distribution(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['total_items'] == 2
    assert envelope['result']['failure_distribution']['hallucination'] == 1
