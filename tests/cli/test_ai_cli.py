"""Tests for AI CLI command registration, argument parsing, and judge handlers."""

from autopilot.cli.commands.ai import GenerateRun, JudgeCommand, JudgeRun
from autopilot.cli.main import build_parser
from autopilot.cli.output import Output
from autopilot.core.artifacts.epoch import DataArtifact
from pathlib import Path
from unittest.mock import MagicMock
import json
import pytest


class TestGenerateParser:
  def test_run_requires_config(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'generate', 'run'])

  def test_run_accepts_all_flags(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
      [
        'ai',
        'generate',
        'run',
        '--config',
        'gen.json',
        '--total-count',
        '100',
        '--seed',
        '42',
        '--num-parallel',
        '5',
        '--max-rpm',
        '50',
      ]
    )
    assert args.ai_config == 'gen.json'
    assert args.total_count == 100
    assert args.seed == 42
    assert args.num_parallel == 5
    assert args.max_rpm == 50

  def test_resume_requires_checkpoint(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'generate', 'resume'])

  def test_dry_run_requires_config(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'generate', 'dry-run'])


class TestJudgeParser:
  def test_run_requires_input(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'judge', 'run'])

  def test_run_accepts_parallel_flags(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
      [
        'ai',
        'judge',
        'run',
        '--input',
        'items.jsonl',
        '--num-parallel',
        '3',
        '--max-rpm',
        '30',
      ]
    )
    assert args.judge_input == 'items.jsonl'
    assert args.num_parallel == 3
    assert args.max_rpm == 30

  def test_resume_requires_both(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'judge', 'resume', '--checkpoint', 'cp.jsonl'])

  def test_summarize_requires_input(self) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(['ai', 'judge', 'summarize'])

  def test_distribution_parses(self) -> None:
    parser = build_parser()
    args = parser.parse_args(['ai', 'judge', 'distribution', '--epoch', '1'])
    assert args.judge_action == 'distribution'


class TestSubcommandRouting:
  def test_generate_run_sets_handler(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
      [
        'ai',
        'generate',
        'run',
        '--config',
        'x.json',
      ]
    )
    assert isinstance(args.handler, GenerateRun)

  def test_judge_run_sets_handler(self) -> None:
    parser = build_parser()
    args = parser.parse_args(
      [
        'ai',
        'judge',
        'run',
        '--input',
        'items.jsonl',
      ]
    )
    assert isinstance(args.handler, JudgeRun)


def _make_judge_ctx(tmp_path: Path, experiment: str = 'test-exp') -> MagicMock:
  ctx = MagicMock()
  ctx.experiment = experiment
  ctx.epoch = 1
  ctx.output = Output(use_json=True)
  ctx.judge = MagicMock()
  exp_dir = tmp_path / experiment
  exp_dir.mkdir(parents=True, exist_ok=True)
  ctx.experiment_dir.return_value = exp_dir
  return ctx


class TestJudgeDistribution:
  def test_distribution_no_epoch(self, tmp_path: Path) -> None:
    ctx = _make_judge_ctx(tmp_path)
    ctx.epoch = 0
    ctx.output = MagicMock()
    cmd = JudgeCommand()
    args = MagicMock(epoch=0)
    cmd.distribution(ctx, args)
    ctx.output.error.assert_called_once()
    ctx.output.result.assert_not_called()

  def test_distribution_happy_path(self, tmp_path: Path, capsys) -> None:
    ctx = _make_judge_ctx(tmp_path)
    exp_dir = tmp_path / 'test-exp'
    data = DataArtifact()
    data.append(
      {'success': False, 'metadata': {'failure_type': 'hallucination'}},
      exp_dir,
      epoch=1,
    )
    data.append({'success': True, 'metadata': {}}, exp_dir, epoch=1)
    cmd = JudgeCommand()
    args = MagicMock(epoch=1)
    cmd.distribution(ctx, args)
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope['ok'] is True
    assert envelope['result']['total_items'] == 2
    assert envelope['result']['failure_distribution']['hallucination'] == 1
